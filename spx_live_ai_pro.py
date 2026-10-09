
# ============================================================
# SPX 0DTE AI ADVISOR v18.0
# REAL ^GSPC DATA / SPXW CONTRACT VALIDATION / DIAGNOSTIC MODE
# ============================================================
# Recommendations only. NO ORDER EXECUTION.
#
# SPX index data: Yahoo Finance ^GSPC
# Stock/context data: Alpaca
# SPXW contracts/quotes: Alpaca, if available to this account
#
# IMPORTANT:
# - Never substitute SPY x 10 for the actual SPX index.
# - Yahoo 5-minute data may be delayed or unavailable.
# - Indicative options quotes are not official OPRA quotes.
# - This script does not place orders.
# ============================================================

import os
import time
import warnings
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
import yfinance as yf

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

# ============================================================
# CONFIG
# ============================================================

VERSION = "v18.0-REAL-SPX-DIAGNOSTIC"

NY = ZoneInfo("America/New_York")

TIMEFRAME = "5Min"
YAHOO_INTERVAL = "5m"
YAHOO_PERIOD = "60d"

HISTORY_DAYS = 59
MIN_TRAIN_ROWS = 150
HORIZON = 6
ATR_TARGET = 0.50

MIN_PROBABILITY = 0.64
MIN_AUC = 0.55

MIN_SPX_SCORE = 78
MIN_CONFIRMATIONS = 2

MAX_SPREAD_PERCENT = 0.12
MIN_OPTION_PREMIUM = 0.50
MAX_SPX_STRIKE_DISTANCE = 35

SIGNAL_COOLDOWN_MINUTES = 20
POLL_SECONDS = 60

NO_TRADE_FIRST_MIN = 10
NO_TRADE_LAST_MIN = 30

MAX_DATA_AGE_MINUTES = 10
MAX_OPTION_QUOTE_AGE_MINUTES = 5

STOCKS = [
    "NVDA", "AAPL", "MSFT", "TSLA", "AMZN",
    "META", "GOOGL", "AMD", "AVGO", "NFLX",
]

MARKET_SYMBOLS = ["SPY", "QQQ"]
ALL_SYMBOLS = MARKET_SYMBOLS + STOCKS

# ============================================================
# CREDENTIALS / ENDPOINTS
# ============================================================

ALPACA_API_KEY = (
    os.getenv("ALPACA_API_KEY")
    or os.getenv("APCA_API_KEY_ID")
    or ""
)

ALPACA_SECRET_KEY = (
    os.getenv("ALPACA_SECRET_KEY")
    or os.getenv("APCA_API_SECRET_KEY")
    or os.getenv("APCA_SECRET_KEY")
    or ""
)

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

ALPACA_TRADING_URL = os.getenv(
    "ALPACA_TRADING_URL",
    "https://api.alpaca.markets"
).rstrip("/")

ALPACA_DATA_URL = "https://data.alpaca.markets"

DATA_FEED = os.getenv("ALPACA_DATA_FEED", "iex").lower()

# "indicative" may be available without OPRA.
# It is NOT the same as official OPRA market data.
OPTIONS_FEED = os.getenv(
    "ALPACA_OPTIONS_FEED",
    "indicative"
).lower()

OPTIONS_CONTRACTS_URL = (
    f"{ALPACA_TRADING_URL}/v2/options/contracts"
)

OPTIONS_QUOTES_URL = (
    f"{ALPACA_DATA_URL}/v1beta1/options/quotes/latest"
)

HEADERS = {
    "APCA-API-KEY-ID": ALPACA_API_KEY,
    "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY,
}

session = requests.Session()

STATE = {"last_sent": {}}


# ============================================================
# LOGGING
# ============================================================

def log(message):
    now = datetime.now(NY).strftime("%Y-%m-%d %H:%M:%S %Z")
    print(f"[{now}] {message}", flush=True)


def section(title):
    log("")
    log("=" * 68)
    log(title)
    log("=" * 68)


# ============================================================
# TELEGRAM
# ============================================================

def telegram_send(message):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        log("[TELEGRAM] Token/chat ID not configured")
        return False

    url = (
        f"https://api.telegram.org/"
        f"bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    )

    try:
        response = session.post(
            url,
            json={
                "chat_id": TELEGRAM_CHAT_ID,
                "text": message,
                "parse_mode": "HTML",
                "disable_web_page_preview": True,
            },
            timeout=15,
        )

        if response.ok:
            return True

        log(
            f"[TELEGRAM ERROR] HTTP {response.status_code}: "
            f"{response.text[:400]}"
        )
        return False

    except Exception as exc:
        log(f"[TELEGRAM EXCEPTION] {exc}")
        return False


# ============================================================
# TIME / MARKET HOURS
# ============================================================

def now_ny():
    return datetime.now(NY)


def today_ny():
    return now_ny().date().isoformat()


def to_utc_iso(dt):
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)

    return (
        dt.astimezone(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
    )


def market_open_now():
    now = now_ny()

    if now.weekday() >= 5:
        return False, "Weekend"

    minutes = now.hour * 60 + now.minute
    open_minutes = 9 * 60 + 30
    close_minutes = 16 * 60

    if minutes < open_minutes:
        return False, "Market has not opened"

    if minutes >= close_minutes:
        return False, "Market is closed"

    if minutes - open_minutes < NO_TRADE_FIRST_MIN:
        return False, "Opening protection window"

    if close_minutes - minutes <= NO_TRADE_LAST_MIN:
        return False, "Closing protection window"

    return True, "Market hours"


# ============================================================
# ALPACA HTTP
# ============================================================

def alpaca_get(url, params=None, timeout=30):
    if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
        log("[FATAL] Alpaca credentials are missing")
        return None

    try:
        response = session.get(
            url,
            headers=HEADERS,
            params=params,
            timeout=timeout,
        )

        if response.status_code != 200:
            log(f"[ALPACA ERROR] HTTP {response.status_code}")
            log(f"[ALPACA BODY] {response.text[:800]}")
            return None

        return response.json()

    except requests.exceptions.Timeout:
        log("[ALPACA ERROR] Request timed out")
        return None

    except Exception as exc:
        log(f"[ALPACA EXCEPTION] {type(exc).__name__}: {exc}")
        return None


# ============================================================
# ALPACA STOCK BARS
# ============================================================

def fetch_alpaca_bars(symbols):
    end_dt = datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(days=HISTORY_DAYS)

    collected = {symbol: [] for symbol in symbols}
    page_token = None

    while True:
        params = {
            "symbols": ",".join(symbols),
            "timeframe": TIMEFRAME,
            "start": to_utc_iso(start_dt),
            "end": to_utc_iso(end_dt),
            "limit": 10000,
            "feed": DATA_FEED,
            "sort": "asc",
        }

        if page_token:
            params["page_token"] = page_token

        data = alpaca_get(
            f"{ALPACA_DATA_URL}/v2/stocks/bars",
            params=params,
            timeout=45,
        )

        if data is None:
            log("[DATA] Alpaca bars request failed")
            break

        bars = data.get("bars") or {}

        for symbol in symbols:
            for bar in bars.get(symbol, []):
                try:
                    collected[symbol].append({
                        "timestamp": bar["t"],
                        "open": float(bar["o"]),
                        "high": float(bar["h"]),
                        "low": float(bar["l"]),
                        "close": float(bar["c"]),
                        "volume": float(bar.get("v", 0)),
                    })
                except (KeyError, TypeError, ValueError):
                    continue

        page_token = data.get("next_page_token")
        if not page_token:
            break

    frames = {}

    for symbol, rows in collected.items():
        if not rows:
            frames[symbol] = pd.DataFrame()
            continue

        df = pd.DataFrame(rows)
        df["timestamp"] = pd.to_datetime(
            df["timestamp"], utc=True, errors="coerce"
        )

        df = (
            df.dropna(subset=["timestamp", "close"])
            .drop_duplicates("timestamp")
            .sort_values("timestamp")
            .reset_index(drop=True)
        )

        frames[symbol] = df

    return frames


# ============================================================
# REAL SPX INDEX DATA FROM YAHOO FINANCE
# ============================================================

def fetch_real_spx():
    log("[SPX DATA] Requesting ^GSPC from Yahoo Finance")

    try:
        df = yf.download(
            tickers="^GSPC",
            period=YAHOO_PERIOD,
            interval=YAHOO_INTERVAL,
            auto_adjust=False,
            progress=False,
            threads=False,
            prepost=False,
            timeout=20,
        )

        if df is None or df.empty:
            log("[SPX DATA ERROR] Yahoo returned no index bars")
            return pd.DataFrame()

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df.columns = [
            str(column).lower().replace(" ", "_")
            for column in df.columns
        ]

        required = ["open", "high", "low", "close"]

        for column in required:
            if column not in df.columns:
                log(f"[SPX DATA ERROR] Missing column: {column}")
                return pd.DataFrame()

        df = df.reset_index()

        time_column = next(
            (
                column for column in df.columns
                if str(column).lower() in ("datetime", "date")
            ),
            None,
        )

        if time_column is None:
            log("[SPX DATA ERROR] Timestamp column not found")
            return pd.DataFrame()

        df = df.rename(columns={time_column: "timestamp"})

        timestamp_series = pd.to_datetime(
            df["timestamp"], errors="coerce"
        )

        if timestamp_series.dt.tz is None:
            timestamp_series = timestamp_series.dt.tz_localize(
                NY, ambiguous="NaT", nonexistent="NaT"
            )

        df["timestamp"] = timestamp_series.dt.tz_convert("UTC")

        if "volume" not in df.columns:
            df["volume"] = 0.0

        df["volume"] = pd.to_numeric(
            df["volume"], errors="coerce"
        ).fillna(0.0)

        for column in required:
            df[column] = pd.to_numeric(
                df[column], errors="coerce"
            )

        df = (
            df.dropna(subset=["timestamp"] + required)
            .drop_duplicates("timestamp")
            .sort_values("timestamp")
            .reset_index(drop=True)
        )

        if df.empty:
            log("[SPX DATA ERROR] No valid ^GSPC rows")
            return pd.DataFrame()

        last_timestamp = df["timestamp"].iloc[-1].to_pydatetime()
        age_minutes = (
            datetime.now(timezone.utc) - last_timestamp
        ).total_seconds() / 60

        log(
            f"[SPX DATA] ^GSPC bars={len(df)} | "
            f"last={last_timestamp} | age={age_minutes:.1f}m"
        )

        if age_minutes > MAX_DATA_AGE_MINUTES:
            log(
                "[SPX DATA REJECTED] Index data is stale. "
                "No SPX signal will be generated."
            )
            return pd.DataFrame()

        return df

    except Exception as exc:
        log(
            f"[SPX DATA EXCEPTION] "
            f"{type(exc).__name__}: {exc}"
        )
        return pd.DataFrame()


# ============================================================
# DATA HEALTH
# ============================================================

def report_data_health(frames):
    section("ALPACA STOCK DATA HEALTH")

    for symbol in ALL_SYMBOLS:
        df = frames.get(symbol)

        if df is None or df.empty:
            log(f"[DATA] {symbol}: NO DATA")
            continue

        last_timestamp = df["timestamp"].iloc[-1].to_pydatetime()
        age = (
            datetime.now(timezone.utc) - last_timestamp
        ).total_seconds() / 60

        status = (
            "FRESH"
            if age <= MAX_DATA_AGE_MINUTES
            else f"STALE {age:.1f}m"
        )

        log(
            f"[DATA] {symbol}: {len(df)} bars | "
            f"last={last_timestamp} | {status}"
        )


def data_is_fresh(df, max_age=MAX_DATA_AGE_MINUTES):
    if df is None or df.empty:
        return False

    last_timestamp = df["timestamp"].iloc[-1].to_pydatetime()
    age = (
        datetime.now(timezone.utc) - last_timestamp
    ).total_seconds() / 60

    return 0 <= age <= max_age


# ============================================================
# INDICATORS
# ============================================================

def rsi(series, period=14):
    delta = series.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean().replace(0, np.nan)

    rs = avg_gain / avg_loss

    return 100 - (100 / (1 + rs))


def atr(df, period=14):
    previous_close = df["close"].shift(1)

    true_range = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - previous_close).abs(),
            (df["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    return true_range.rolling(period).mean()


# ============================================================
# FEATURES / LABELS
# ============================================================

FEATURE_COLUMNS = [
    "ret_1", "ret_3", "ret_6", "ret_12",
    "rsi", "atr_pct", "range_pct", "volatility",
    "ma9_dist", "ma20_dist", "ma50_dist",
    "momentum", "acceleration", "volume_z",
    "market_ret_3", "market_ret_12", "relative_strength",
    "qqq_ret_3", "qqq_ret_12",
    "time_sin", "time_cos",
]


def make_features(df, market_df=None, qqq_df=None):
    x = df.copy().sort_values("timestamp").reset_index(drop=True)
    close = x["close"]

    x["ret_1"] = close.pct_change(1)
    x["ret_3"] = close.pct_change(3)
    x["ret_6"] = close.pct_change(6)
    x["ret_12"] = close.pct_change(12)

    x["rsi"] = rsi(close)
    x["atr"] = atr(x)
    x["atr_pct"] = x["atr"] / close
    x["range_pct"] = (x["high"] - x["low"]) / close
    x["volatility"] = x["ret_1"].rolling(20).std()

    for period in (9, 20, 50):
        moving_average = close.rolling(period).mean()
        x[f"ma{period}_dist"] = close / moving_average - 1

    x["momentum"] = close / close.shift(12) - 1
    x["acceleration"] = x["ret_3"] - x["ret_3"].shift(3)

    volume_mean = x["volume"].rolling(30).mean()
    volume_std = x["volume"].rolling(30).std().replace(0, np.nan)
    x["volume_z"] = (x["volume"] - volume_mean) / volume_std

    # Merge context data by timestamp; no future context rows are used.
    if market_df is not None and not market_df.empty:
        market = market_df[["timestamp", "close"]].rename(
            columns={"close": "market_close"}
        ).sort_values("timestamp")

        x = pd.merge_asof(
            x.sort_values("timestamp"),
            market,
            on="timestamp",
            direction="backward",
        )

        x["market_ret_3"] = x["market_close"].pct_change(3)
        x["market_ret_12"] = x["market_close"].pct_change(12)
        x["relative_strength"] = x["ret_3"] - x["market_ret_3"]
    else:
        x["market_ret_3"] = 0.0
        x["market_ret_12"] = 0.0
        x["relative_strength"] = 0.0

    if qqq_df is not None and not qqq_df.empty:
        qqq = qqq_df[["timestamp", "close"]].rename(
            columns={"close": "qqq_close"}
        ).sort_values("timestamp")

        x = pd.merge_asof(
            x.sort_values("timestamp"),
            qqq,
            on="timestamp",
            direction="backward",
        )

        x["qqq_ret_3"] = x["qqq_close"].pct_change(3)
        x["qqq_ret_12"] = x["qqq_close"].pct_change(12)
    else:
        x["qqq_ret_3"] = 0.0
        x["qqq_ret_12"] = 0.0

    local_time = x["timestamp"].dt.tz_convert(NY)
    minutes = local_time.dt.hour * 60 + local_time.dt.minute

    x["time_sin"] = np.sin(2 * np.pi * minutes / 1440)
    x["time_cos"] = np.cos(2 * np.pi * minutes / 1440)

    # Label only if one direction hits first/alone within the horizon.
    future_high = pd.concat(
        [x["high"].shift(-i) for i in range(1, HORIZON + 1)],
        axis=1,
    ).max(axis=1)

    future_low = pd.concat(
        [x["low"].shift(-i) for i in range(1, HORIZON + 1)],
        axis=1,
    ).min(axis=1)

    up_hit = future_high >= close + x["atr"] * ATR_TARGET
    down_hit = future_low <= close - x["atr"] * ATR_TARGET

    x["target"] = np.where(
        up_hit & ~down_hit,
        1,
        np.where(down_hit & ~up_hit, 0, np.nan),
    )

    return x


# ============================================================
# MODEL TRAINING - CHRONOLOGICAL HOLDOUT
# ============================================================

def train_model(feature_df):
    if feature_df is None or feature_df.empty:
        return None

    clean = feature_df.dropna(
        subset=FEATURE_COLUMNS + ["target"]
    ).copy()

    if len(clean) < MIN_TRAIN_ROWS:
        log(f"[MODEL] Not enough labeled rows: {len(clean)}")
        return None

    clean["target"] = clean["target"].astype(int)

    n = len(clean)
    train_end = int(n * 0.70)
    test_start = int(n * 0.80)

    train = clean.iloc[:train_end]
    test = clean.iloc[test_start:]

    if train["target"].nunique() < 2 or test["target"].nunique() < 2:
        log("[MODEL] Insufficient class diversity in train/test")
        return None

    models = []

    for seed in (17, 41, 83):
        model = HistGradientBoostingClassifier(
            max_iter=200,
            learning_rate=0.045,
            max_leaf_nodes=15,
            min_samples_leaf=25,
            l2_regularization=1.0,
            random_state=seed,
        )

        model.fit(train[FEATURE_COLUMNS], train["target"])
        models.append(model)

    probabilities = np.mean(
        [
            model.predict_proba(test[FEATURE_COLUMNS])[:, 1]
            for model in models
        ],
        axis=0,
    )

    try:
        auc = float(roc_auc_score(test["target"], probabilities))
    except Exception:
        auc = 0.50

    log(
        f"[MODEL TEST] rows={len(clean)} | "
        f"train={len(train)} | test={len(test)} | AUC={auc:.3f}"
    )

    return {
        "models": models,
        "auc": auc,
        "rows": len(clean),
    }


# ============================================================
# PREDICTION
# ============================================================

def predict(model_info, feature_df):
    if not model_info or feature_df is None or feature_df.empty:
        return None

    clean = feature_df.dropna(subset=FEATURE_COLUMNS)
    if clean.empty:
        return None

    latest = clean.iloc[-1]
    X = pd.DataFrame([latest[FEATURE_COLUMNS].to_dict()])

    probabilities = [
        model.predict_proba(X)[0][1]
        for model in model_info["models"]
    ]

    p_up = float(np.mean(probabilities))
    p_down = 1.0 - p_up

    if p_up >= MIN_PROBABILITY:
        signal = "CALL"
    elif p_down >= MIN_PROBABILITY:
        signal = "PUT"
    else:
        signal = "WAIT"

    vol = float(latest["volatility"])
    regime = (
        "HIGH VOL" if vol >= 0.003
        else "LOW VOL" if vol < 0.0015
        else "NORMAL"
    )

    return {
        "signal": signal,
        "p_up": p_up,
        "p_down": p_down,
        "confidence": max(p_up, p_down),
        "price": float(latest["close"]),
        "atr": float(latest["atr"]),
        "rsi": float(latest["rsi"]),
        "momentum": float(latest["momentum"]),
        "volatility": vol,
        "volume_z": float(latest["volume_z"]),
        "relative_strength": float(latest["relative_strength"]),
        "auc": model_info["auc"],
        "rows": model_info["rows"],
        "regime": regime,
    }


# ============================================================
# MARKET CONFIRMATION
# ============================================================

def calculate_confirmation(predictions):
    calls = 0
    puts = 0
    details = []

    for symbol in ALL_SYMBOLS:
        prediction = predictions.get(symbol)
        if not prediction:
            continue

        if prediction["signal"] == "CALL":
            calls += 1
            details.append(f"{symbol}:CALL")
        elif prediction["signal"] == "PUT":
            puts += 1
            details.append(f"{symbol}:PUT")

    total = calls + puts

    if total == 0 or calls == puts:
        bias = "NEUTRAL"
        strength = max(calls, puts) / total if total else 0.0
    elif calls > puts:
        bias = "BULLISH"
        strength = calls / total
    else:
        bias = "BEARISH"
        strength = puts / total

    return {
        "bias": bias,
        "strength": float(strength),
        "calls": calls,
        "puts": puts,
        "total": total,
        "details": details,
    }


def validate_spx_signal(prediction, confirmation):
    if not prediction:
        return False, "NO_SPX_PREDICTION"

    if prediction["signal"] not in ("CALL", "PUT"):
        return False, "ML_WAIT"

    if prediction["auc"] < MIN_AUC:
        return False, f"LOW_AUC_{prediction['auc']:.3f}"

    if prediction["confidence"] < MIN_PROBABILITY:
        return False, "LOW_CONFIDENCE"

    if not np.isfinite(prediction["momentum"]):
        return False, "INVALID_MOMENTUM"

    if abs(prediction["momentum"]) < 0.001:
        return False, "WEAK_MOMENTUM"

    desired_bias = (
        "BULLISH" if prediction["signal"] == "CALL" else "BEARISH"
    )

    if confirmation["bias"] != desired_bias:
        return False, "MARKET_NOT_ALIGNED"

    directional_votes = (
        confirmation["calls"]
        if desired_bias == "BULLISH"
        else confirmation["puts"]
    )

    if directional_votes < MIN_CONFIRMATIONS:
        return False, "NOT_ENOUGH_CONFIRMATIONS"

    if confirmation["strength"] < 0.50:
        return False, "WEAK_CONFIRMATION"

    return True, "OK"


# ============================================================
# SPXW CONTRACT DISCOVERY
# ============================================================

def get_spxw_contracts(option_type):
    params = {
        "underlying_symbols": "SPX",
        "root_symbol": "SPXW",
        "status": "active",
        "expiration_date": today_ny(),
        "type": option_type,
        "limit": 10000,
    }

    log(
        f"[SPXW CONTRACT REQUEST] "
        f"expiration={today_ny()} type={option_type}"
    )

    data = alpaca_get(
        OPTIONS_CONTRACTS_URL,
        params=params,
        timeout=30,
    )

    if data is None:
        log("[SPXW] Contract API request failed")
        return []

    contracts = (
        data.get("option_contracts")
        or data.get("contracts")
        or []
    )

    # Do not silently treat other roots as SPXW.
    contracts = [
        contract for contract in contracts
        if (
            "SPXW" in str(contract.get("symbol", "")).upper()
            or "SPXW" in str(contract.get("root_symbol", "")).upper()
        )
        and str(contract.get("underlying_symbol", "SPX")).upper() == "SPX"
        and contract.get("tradable", True)
    ]

    log(f"[SPXW] Valid contracts returned: {len(contracts)}")
    return contracts


# ============================================================
# OPTION QUOTES - FRESHNESS CHECK
# ============================================================

def get_option_quotes(symbols):
    result = {}

    if not symbols:
        return result

    for start in range(0, len(symbols), 100):
        batch = symbols[start:start + 100]

        data = alpaca_get(
            OPTIONS_QUOTES_URL,
            params={
                "symbols": ",".join(batch),
                "feed": OPTIONS_FEED,
            },
            timeout=30,
        )

        if data is None:
            log("[OPTION QUOTES] API request failed")
            continue

        for symbol, quote in (data.get("quotes") or {}).items():
            try:
                bid = float(quote.get("bp", 0))
                ask = float(quote.get("ap", 0))
                quote_time = pd.to_datetime(
                    quote.get("t"), utc=True, errors="coerce"
                )

                if (
                    bid <= 0
                    or ask <= 0
                    or ask < bid
                    or pd.isna(quote_time)
                ):
                    continue

                age_minutes = (
                    pd.Timestamp.now(tz="UTC") - quote_time
                ).total_seconds() / 60

                if age_minutes < -1 or age_minutes > MAX_OPTION_QUOTE_AGE_MINUTES:
                    log(
                        f"[QUOTE REJECT] {symbol}: "
                        f"stale/invalid quote age={age_minutes:.1f}m"
                    )
                    continue

                mid = (bid + ask) / 2
                if mid <= 0:
                    continue

                result[symbol] = {
                    "bid": bid,
                    "ask": ask,
                    "mid": mid,
                    "spread_pct": (ask - bid) / mid,
                    "timestamp": quote_time.isoformat(),
                    "age_minutes": age_minutes,
                }

            except (TypeError, ValueError, AttributeError):
                continue

    log(
        f"[OPTION QUOTES] valid={len(result)} | "
        f"feed={OPTIONS_FEED.upper()}"
    )

    return result


# ============================================================
# CONTRACT SELECTION
# ============================================================

def score_option(prediction, confirmation, quote, distance):
    score = 0.0

    score += min(
        35,
        max(0, (prediction["confidence"] - 0.50) * 100) * 0.875,
    )

    score += min(
        15,
        max(0, (prediction["auc"] - 0.50) * 100) * 0.75,
    )

    score += min(20, confirmation["strength"] * 20)

    spread = quote["spread_pct"]

    if spread <= 0.04:
        score += 15
    elif spread <= 0.06:
        score += 13
    elif spread <= 0.08:
        score += 10
    elif spread <= 0.10:
        score += 6
    elif spread <= MAX_SPREAD_PERCENT:
        score += 2
    else:
        return 0

    if distance <= 5:
        score += 10
    elif distance <= 10:
        score += 8
    elif distance <= 20:
        score += 6
    elif distance <= 30:
        score += 3
    else:
        score += 1

    momentum = abs(prediction["momentum"])

    if momentum >= 0.004:
        score += 5
    elif momentum >= 0.002:
        score += 3
    else:
        score += 1

    return int(min(100, max(0, round(score))))


def choose_spxw_contract(prediction, confirmation):
    signal = prediction["signal"]
    option_type = "call" if signal == "CALL" else "put"
    spx_price = prediction["price"]

    contracts = get_spxw_contracts(option_type)

    if not contracts:
        return None, "NO_SPXW_CONTRACTS_FROM_ALPACA"

    candidates = []

    for contract in contracts:
        try:
            strike = float(contract["strike_price"])
        except (KeyError, TypeError, ValueError):
            continue

        distance = abs(strike - spx_price)

        if distance <= MAX_SPX_STRIKE_DISTANCE:
            candidates.append((distance, contract))

    if not candidates:
        return None, "NO_STRIKE_NEAR_REAL_SPX_PRICE"

    candidates.sort(key=lambda item: item[0])
    selected = [contract for _, contract in candidates[:50]]

    symbols = [
        contract.get("symbol")
        for contract in selected
        if contract.get("symbol")
    ]

    quotes = get_option_quotes(symbols)

    if not quotes:
        return None, "NO_FRESH_OPTION_QUOTES"

    best = None
    best_score = -1
    spread_rejects = 0
    premium_rejects = 0

    for contract in selected:
        symbol = contract.get("symbol")
        quote = quotes.get(symbol)

        if not quote:
            continue

        if quote["mid"] < MIN_OPTION_PREMIUM:
            premium_rejects += 1
            continue

        if quote["spread_pct"] > MAX_SPREAD_PERCENT:
            spread_rejects += 1
            continue

        strike = float(contract["strike_price"])
        distance = abs(strike - spx_price)

        score = score_option(
            prediction,
            confirmation,
            quote,
            distance,
        )

        if score > best_score:
            best_score = score
            best = {
                "contract": contract,
                "quote": quote,
                "score": score,
                "distance": distance,
            }

    if best is None:
        log(
            f"[SPXW REJECT] no liquid quote | "
            f"premium_rejects={premium_rejects} | "
            f"spread_rejects={spread_rejects}"
        )
        return None, "NO_LIQUID_CONTRACT"

    if best["score"] < MIN_SPX_SCORE:
        return None, f"SCORE_TOO_LOW_{best['score']}"

    return best, "OK"


# ============================================================
# OPPORTUNITY / TELEGRAM MESSAGE
# ============================================================

def build_opportunity(prediction, confirmation):
    valid, reason = validate_spx_signal(prediction, confirmation)

    if not valid:
        return None, reason

    selected, reason = choose_spxw_contract(
        prediction,
        confirmation,
    )

    if selected is None:
        return None, reason

    contract = selected["contract"]
    quote = selected["quote"]
    entry = quote["mid"]

    return {
        "signal": prediction["signal"],
        "confidence": prediction["confidence"],
        "auc": prediction["auc"],
        "spx_price": prediction["price"],
        "score": selected["score"],
        "contract_symbol": contract.get("symbol", "UNKNOWN"),
        "strike": float(contract["strike_price"]),
        "expiration": contract.get("expiration_date", today_ny()),
        "entry": entry,
        "target": entry * 1.40,
        "stop": entry * 0.70,
        "bid": quote["bid"],
        "ask": quote["ask"],
        "spread_pct": quote["spread_pct"],
        "quote_age": quote["age_minutes"],
        "feed": OPTIONS_FEED.upper(),
        "confirmation": confirmation,
        "regime": prediction["regime"],
        "momentum": prediction["momentum"],
    }, "OK"


def format_alert(opportunity):
    signal = opportunity["signal"]
    emoji = "🟢" if signal == "CALL" else "🔴"
    confirmation = opportunity["confirmation"]

    return (
        f"🚨 <b>SPXW 0DTE {emoji} {signal}</b>\n"
        f"━━━━━━━━━━━━━━━━━━\n"
        f"🔥 Score: <b>{opportunity['score']}/100</b>\n"
        f"🧠 Model probability: <b>{opportunity['confidence']*100:.1f}%</b>\n"
        f"📊 Holdout AUC: <b>{opportunity['auc']:.3f}</b>\n"
        f"📈 Real Yahoo ^GSPC: <b>{opportunity['spx_price']:.2f}</b>\n"
        f"🎯 Strike: <b>{opportunity['strike']:.2f}</b>\n"
        f"📜 Contract: <code>{opportunity['contract_symbol']}</code>\n"
        f"💰 Midpoint reference: <b>${opportunity['entry']:.2f}</b>\n"
        f"🎯 Illustrative +40% level: ${opportunity['target']:.2f}\n"
        f"🛑 Illustrative -30% level: ${opportunity['stop']:.2f}\n"
        f"↔️ Spread: {opportunity['spread_pct']*100:.1f}%\n"
        f"🕒 Quote age: {opportunity['quote_age']:.1f} minutes\n"
        f"📡 Options feed: <b>{opportunity['feed']}</b>\n"
        f"🌡️ Regime: {opportunity['regime']}\n"
        f"📊 Confirmation: <b>{confirmation['bias']}</b>\n"
        f"🟢 CALL votes: {confirmation['calls']} | "
        f"🔴 PUT votes: {confirmation['puts']}\n"
        f"⚠️ <i>Reference levels only; not tested profit guarantees. "
        f"No order execution.</i>"
    )


# ============================================================
# DUPLICATE ALERT CONTROL
# ============================================================

def is_duplicate(opportunity):
    key = opportunity["contract_symbol"] + "_" + opportunity["signal"]
    last_sent = STATE["last_sent"].get(key)

    if last_sent is None:
        return False

    return (
        (time.time() - last_sent) / 60
        < SIGNAL_COOLDOWN_MINUTES
    )


def mark_sent(opportunity):
    key = opportunity["contract_symbol"] + "_" + opportunity["signal"]
    STATE["last_sent"][key] = time.time()


# ============================================================
# PREDICTION LOG
# ============================================================

def print_prediction(symbol, prediction):
    if not prediction:
        log(f"[ML] {symbol}: NO PREDICTION")
        return

    log(
        f"[ML] {symbol}: {prediction['signal']} | "
        f"UP={prediction['p_up']*100:.1f}% | "
        f"DOWN={prediction['p_down']*100:.1f}% | "
        f"AUC={prediction['auc']:.3f} | "
        f"rows={prediction['rows']} | "
        f"RSI={prediction['rsi']:.1f} | "
        f"momentum={prediction['momentum']*100:.2f}% | "
        f"{prediction['regime']}"
    )


# ============================================================
# MAIN SCAN
# ============================================================

def run_scan():
    market_ok, market_reason = market_open_now()

    if not market_ok:
        log(f"[MARKET] WAIT: {market_reason}")
        return

    # 1. Fetch the real index. No SPY x 10 substitution.
    spx_df = fetch_real_spx()

    if not data_is_fresh(spx_df):
        log("[STOP] Real SPX index data unavailable or stale")
        return

    # 2. Fetch Alpaca stock/context bars.
    frames = fetch_alpaca_bars(ALL_SYMBOLS)
    report_data_health(frames)

    spy = frames.get("SPY", pd.DataFrame())
    qqq = frames.get("QQQ", pd.DataFrame())

    if not data_is_fresh(spy):
        log("[STOP] SPY confirmation data unavailable or stale")
        return

    if not data_is_fresh(qqq):
        log("[STOP] QQQ confirmation data unavailable or stale")
        return

    # 3. Train SPX model on actual ^GSPC bars.
    spx_features = make_features(
        spx_df,
        market_df=spy,
        qqq_df=qqq,
    )

    spx_model = train_model(spx_features)

    if not spx_model:
        log("[STOP] SPX model failed training/validation")
        return

    models = {
        "SPX": {
            "features": spx_features,
            "model": spx_model,
        }
    }

    # 4. Train context models.
    for symbol in ALL_SYMBOLS:
        df = frames.get(symbol)

        if not data_is_fresh(df):
            log(f"[MODEL] {symbol}: skipped (missing/stale)")
            continue

        features = make_features(
            df,
            market_df=spy,
            qqq_df=qqq,
        )

        model_info = train_model(features)

        if model_info:
            models[symbol] = {
                "features": features,
                "model": model_info,
            }

    # 5. Predict.
    predictions = {}

    for symbol, info in models.items():
        prediction = predict(
            info["model"],
            info["features"],
        )

        if prediction:
            predictions[symbol] = prediction
            print_prediction(symbol, prediction)

    spx_prediction = predictions.get("SPX")

    if not spx_prediction:
        log("[STOP] No SPX prediction")
        return

    # 6. Confirmation excludes SPX itself.
    context_predictions = {
        symbol: prediction
        for symbol, prediction in predictions.items()
        if symbol != "SPX"
    }

    confirmation = calculate_confirmation(context_predictions)

    log(
        f"[CONFIRMATION] {confirmation['bias']} | "
        f"strength={confirmation['strength']*100:.1f}% | "
        f"CALL={confirmation['calls']} | "
        f"PUT={confirmation['puts']}"
    )

    log(
        "[CONFIRMATION DETAILS] "
        + " | ".join(confirmation["details"])
    )

    # 7. Validate model signal.
    valid, reason = validate_spx_signal(
        spx_prediction,
        confirmation,
    )

    if not valid:
        log(f"[SPX] WAIT: {reason}")
        return

    # 8. Find actual SPXW contracts and fresh quotes.
    opportunity, reason = build_opportunity(
        spx_prediction,
        confirmation,
    )

    if not opportunity:
        log(f"[SPXW] WAIT: {reason}")
        return

    if is_duplicate(opportunity):
        log(
            f"[COOLDOWN] Already alerted recently: "
            f"{opportunity['contract_symbol']}"
        )
        return

    log(
        f"[FINAL] {opportunity['signal']} | "
        f"score={opportunity['score']}/100 | "
        f"contract={opportunity['contract_symbol']} | "
        f"entry reference=${opportunity['entry']:.2f}"
    )

    sent = telegram_send(format_alert(opportunity))

    if sent:
        mark_sent(opportunity)
        log("[TELEGRAM] Alert sent")
    else:
        log("[TELEGRAM] Alert failed; cooldown not recorded")


def main():
    section(f"SPX 0DTE AI ADVISOR {VERSION}")

    if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
        log(
            "[FATAL] Set ALPACA_API_KEY and "
            "ALPACA_SECRET_KEY in environment variables"
        )
        return

    log("Primary index: Yahoo Finance ^GSPC (actual index)")
    log("Options: Alpaca SPXW only, if returned by API")
    log(f"Stock data feed: {DATA_FEED.upper()}")
    log(f"Options quote feed: {OPTIONS_FEED.upper()}")
    log("Mode: recommendations only; no order execution")
    log("Note: indicative quotes are not official OPRA quotes")

    telegram_send(
        f"🤖 <b>SPX 0DTE ADVISOR {VERSION}</b>\n"
        f"بدأ التشغيل.\n"
        f"المؤشر: Yahoo ^GSPC الفعلي.\n"
        f"الخيارات: SPXW عبر Alpaca إذا كانت متاحة.\n"
        f"لا يوجد تنفيذ أوامر."
    )

    cycle = 0

    while True:
        cycle += 1
        section(f"SCAN #{cycle}")

        try:
            run_scan()
        except KeyboardInterrupt:
            log("[STOP] Interrupted by user")
            break
        except Exception as exc:
            log(f"[MAIN ERROR] {type(exc).__name__}: {exc}")

        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
