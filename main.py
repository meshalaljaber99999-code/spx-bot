
# ============================================================
# SPX 0DTE AI ADVISOR v18.2
# Recommendation Only — NO automatic order execution
# ============================================================

import os
import time
import logging
import warnings
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
import yfinance as yf

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, roc_auc_score

warnings.filterwarnings("ignore")

# ======================== SETTINGS ==========================

NY = ZoneInfo("America/New_York")
UTC = timezone.utc

API_KEY = (
    os.getenv("ALPACA_API_KEY")
    or os.getenv("APCA_API_KEY_ID")
    or ""
).strip()

API_SECRET = (
    os.getenv("ALPACA_SECRET_KEY")
    or os.getenv("APCA_API_SECRET_KEY")
    or ""
).strip()

# Keep paper trading as the default.
TRADE_URL = os.getenv(
    "ALPACA_TRADE_URL",
    "https://paper-api.alpaca.markets"
).rstrip("/")

DATA_URL = os.getenv(
    "ALPACA_DATA_URL",
    "https://data.alpaca.markets"
).rstrip("/")

DATA_FEED = os.getenv("ALPACA_DATA_FEED", "iex").lower()
OPTIONS_FEED = os.getenv("ALPACA_OPTIONS_FEED", "indicative").lower()

TG_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TG_CHAT = os.getenv("TELEGRAM_CHAT_ID", "").strip()

MIN_AUC = float(os.getenv("MIN_MODEL_AUC", "0.52"))
MIN_CONFIDENCE = float(os.getenv("MIN_MODEL_CONFIDENCE", "0.56"))
MAX_SPREAD_PCT = float(os.getenv("MAX_OPTION_SPREAD_PCT", "0.20"))

SCAN_SECONDS = max(20, int(os.getenv("SCAN_SECONDS", "60")))
LABEL_HORIZON = 3
MAX_BARS = 5000
MAX_CONTRACTS_TO_CHECK = 15

FEATURES = [
    "ret_1", "ret_3", "ret_6",
    "rsi", "ema_spread", "ema_long_spread",
    "atr_pct", "range_pct", "body_ratio",
    "volatility", "momentum_accel",
    "spy_ret_1", "spy_ret_3", "spy_ema_spread", "spy_rsi",
    "qqq_ret_1", "qqq_ret_3", "qqq_ema_spread", "qqq_rsi",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

HTTP = requests.Session()


def log(message, level="info"):
    getattr(logging, level, logging.info)(message)


def utc_now():
    return datetime.now(UTC)


def ny_now():
    return datetime.now(NY)


def alpaca_headers():
    if not API_KEY or not API_SECRET:
        raise RuntimeError(
            "Missing Alpaca keys. Set ALPACA_API_KEY and "
            "ALPACA_SECRET_KEY in Railway Variables."
        )
    return {
        "APCA-API-KEY-ID": API_KEY,
        "APCA-API-SECRET-KEY": API_SECRET,
    }


def normalize_bars(df):
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(
            df["timestamp"], utc=True, errors="coerce"
        )
        df = df.dropna(subset=["timestamp"])
        df = df.set_index("timestamp")
    else:
        df.index = pd.to_datetime(
            df.index, utc=True, errors="coerce"
        )
        df = df.loc[~df.index.isna()]

    df.index = pd.DatetimeIndex(
        pd.to_datetime(df.index, utc=True)
    )
    df.index.name = "timestamp"
    df = df[~df.index.duplicated(keep="last")].sort_index()

    for col in ["open", "high", "low", "close", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    required = [
        col for col in ["open", "high", "low", "close"]
        if col in df.columns
    ]
    if required:
        df = df.dropna(subset=required)

    return df


# ======================== MARKET DATA =======================

def get_spx_bars():
    """SPX index bars from Yahoo Finance, not SPY multiplied by 10."""
    try:
        raw = yf.download(
            "^GSPC",
            period="60d",
            interval="5m",
            auto_adjust=False,
            progress=False,
            threads=False,
        )

        if raw is None or raw.empty:
            raise RuntimeError("Yahoo returned no ^GSPC data")

        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)

        raw = raw.rename(columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        })

        raw = normalize_bars(raw)

        raw = raw[
            ["open", "high", "low", "close"]
        ].tail(MAX_BARS)

        if len(raw) < 300:
            raise RuntimeError(f"Too few SPX bars: {len(raw)}")

        age = (
            utc_now() - raw.index[-1].to_pydatetime()
        ).total_seconds() / 60

        log(
            f"[SPX DATA] ^GSPC bars={len(raw)} "
            f"| last={raw.index[-1]} | age={age:.1f}m"
        )
        return raw

    except Exception as exc:
        log(f"[SPX DATA ERROR] {exc}", "error")
        return pd.DataFrame()


def get_stock_bars(symbol):
    """Fetch 5-minute bars for SPY or QQQ from Alpaca."""
    try:
        end = utc_now()
        start = end - timedelta(days=10)

        params = {
            "timeframe": "5Min",
            "start": start.isoformat().replace("+00:00", "Z"),
            "end": end.isoformat().replace("+00:00", "Z"),
            "limit": 10000,
            "adjustment": "raw",
            "feed": DATA_FEED,
            "sort": "asc",
        }

        response = HTTP.get(
            f"{DATA_URL}/v2/stocks/{symbol}/bars",
            headers=alpaca_headers(),
            params=params,
            timeout=25,
        )

        if response.status_code >= 400:
            raise RuntimeError(
                f"Alpaca HTTP {response.status_code}: "
                f"{response.text[:250]}"
            )

        rows = response.json().get("bars", [])
        if not rows:
            raise RuntimeError(
                f"No {symbol} bars; feed={DATA_FEED}"
            )

        df = pd.DataFrame(rows).rename(columns={
            "t": "timestamp",
            "o": "open",
            "h": "high",
            "l": "low",
            "c": "close",
            "v": "volume",
        })

        df = normalize_bars(df)
        df = df[
            ["open", "high", "low", "close"]
        ].tail(MAX_BARS)

        if len(df) < 100:
            raise RuntimeError(
                f"Too few {symbol} bars: {len(df)}"
            )

        age = (
            utc_now() - df.index[-1].to_pydatetime()
        ).total_seconds() / 60

        log(
            f"[DATA] {symbol}: {len(df)} bars "
            f"| last={df.index[-1]} | age={age:.1f}m"
        )
        return df

    except Exception as exc:
        log(f"[DATA ERROR] {symbol}: {exc}", "error")
        return pd.DataFrame()


# ======================== INDICATORS =======================

def calc_rsi(close, period=14):
    delta = close.diff()

    gain = delta.clip(lower=0).ewm(
        alpha=1 / period,
        min_periods=period,
        adjust=False,
    ).mean()

    loss = (-delta.clip(upper=0)).ewm(
        alpha=1 / period,
        min_periods=period,
        adjust=False,
    ).mean()

    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def build_features(bars, prefix=""):
    df = normalize_bars(bars)

    if df.empty:
        return pd.DataFrame()

    o = df["open"]
    h = df["high"]
    low = df["low"]
    c = df["close"]

    result = pd.DataFrame(index=df.index)

    result[f"{prefix}ret_1"] = c.pct_change(1)
    result[f"{prefix}ret_3"] = c.pct_change(3)
    result[f"{prefix}ret_6"] = c.pct_change(6)

    result[f"{prefix}rsi"] = calc_rsi(c)

    ema9 = c.ewm(span=9, adjust=False).mean()
    ema21 = c.ewm(span=21, adjust=False).mean()
    ema50 = c.ewm(span=50, adjust=False).mean()

    result[f"{prefix}ema_spread"] = (
        (ema9 - ema21) / c.replace(0, np.nan)
    )

    result[f"{prefix}ema_long_spread"] = (
        (ema21 - ema50) / c.replace(0, np.nan)
    )

    previous_close = c.shift(1)

    true_range = pd.concat([
        h - low,
        (h - previous_close).abs(),
        (low - previous_close).abs(),
    ], axis=1).max(axis=1)

    atr = true_range.rolling(14).mean()

    result[f"{prefix}atr_pct"] = (
        atr / c.replace(0, np.nan)
    )

    result[f"{prefix}range_pct"] = (
        (h - low) / c.replace(0, np.nan)
    )

    result[f"{prefix}body_ratio"] = (
        (c - o) / (h - low).replace(0, np.nan)
    )

    result[f"{prefix}volatility"] = (
        c.pct_change().rolling(20).std()
    )

    result[f"{prefix}momentum_accel"] = (
        c.pct_change().diff(3)
    )

    return result.replace([np.inf, -np.inf], np.nan)


def make_features(spx, spy, qqq):
    spx = normalize_bars(spx)
    spy = normalize_bars(spy)
    qqq = normalize_bars(qqq)

    if spx.empty or spy.empty or qqq.empty:
        raise RuntimeError("SPX, SPY or QQQ data is missing")

    spx_features = build_features(spx)
    spy_features = build_features(spy, "spy_")
    qqq_features = build_features(qqq, "qqq_")

    spy_features = spy_features[
        [
            "spy_ret_1", "spy_ret_3",
            "spy_ema_spread", "spy_rsi",
        ]
    ]

    qqq_features = qqq_features[
        [
            "qqq_ret_1", "qqq_ret_3",
            "qqq_ema_spread", "qqq_rsi",
        ]
    ]

    # Normalize timestamps to the same UTC representation.
    for frame in [spx_features, spy_features, qqq_features]:
        frame.index = pd.DatetimeIndex(
            pd.to_datetime(frame.index, utc=True)
        )
        frame.index.name = "timestamp"

    merged = spx_features.join(
        spy_features, how="inner"
    ).join(
        qqq_features, how="inner"
    )

    merged = merged.join(spx[["close"]], how="inner")
    merged = merged.replace([np.inf, -np.inf], np.nan)
    merged = merged.dropna(subset=FEATURES + ["close"])

    if len(merged) < 400:
        raise RuntimeError(
            f"Not enough aligned feature rows: {len(merged)}"
        )

    return merged.sort_index()


# ======================== MACHINE LEARNING ==================

def new_model():
    return make_pipeline(
        SimpleImputer(strategy="median"),
        HistGradientBoostingClassifier(
            max_iter=120,
            learning_rate=0.06,
            max_leaf_nodes=15,
            l2_regularization=1.0,
            random_state=42,
        ),
    )


def train_model(df):
    feature_cols = FEATURES.copy()
    data = df.sort_index().copy()

    future_close = data["close"].shift(-LABEL_HORIZON)
    valid = future_close.notna()

    X = data.loc[valid, feature_cols].replace(
        [np.inf, -np.inf], np.nan
    )

    y = (
        future_close.loc[valid] > data.loc[valid, "close"]
    ).astype(int)

    if len(X) < 500:
        raise RuntimeError(f"Not enough model rows: {len(X)}")

    # Chronological split. Purge the label horizon before test data.
    split = int(len(X) * 0.80)
    train_end = split - LABEL_HORIZON

    if train_end < 200 or len(X) - split < 100:
        raise RuntimeError("Insufficient train/test data")

    X_train = X.iloc[:train_end]
    y_train = y.iloc[:train_end]

    X_test = X.iloc[split:]
    y_test = y.iloc[split:]

    if y_train.nunique() < 2 or y_test.nunique() < 2:
        raise RuntimeError(
            "Train or test labels contain only one class"
        )

    evaluator = new_model()
    evaluator.fit(X_train, y_train)

    probabilities = evaluator.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    accuracy = float(
        accuracy_score(y_test, predictions)
    )
    auc = float(
        roc_auc_score(y_test, probabilities)
    )

    baseline_class = int(y_train.mean() >= 0.5)
    baseline_accuracy = float(
        (y_test == baseline_class).mean()
    )

    log(
        f"[MODEL] rows={len(df)} "
        f"| train={len(X_train)} | test={len(X_test)} "
        f"| accuracy={accuracy:.3f} "
        f"| AUC={auc:.3f} "
        f"| baseline={baseline_accuracy:.3f}"
    )

    # Fit the live model on all available labeled rows.
    live_model = new_model()
    live_model.fit(X, y)

    return live_model, feature_cols, auc, accuracy


def calculate_signal(df, model, feature_cols, auc):
    if auc < MIN_AUC:
        return (
            "WAIT",
            0.0,
            f"AUC {auc:.3f} below threshold {MIN_AUC:.3f}",
        )

    latest = df.iloc[-1]
    live_x = latest[feature_cols].to_frame().T

    probability_up = float(
        model.predict_proba(live_x)[0, 1]
    )
    probability_down = 1.0 - probability_up

    spx_up = (
        latest["ret_1"] > 0
        and latest["ema_spread"] > 0
    )
    spy_up = (
        latest["spy_ret_1"] > 0
        and latest["spy_ema_spread"] > 0
    )
    qqq_up = (
        latest["qqq_ret_1"] > 0
        and latest["qqq_ema_spread"] > 0
    )

    up_votes = sum([bool(spx_up), bool(spy_up), bool(qqq_up)])
    down_votes = 3 - up_votes

    if (
        probability_up >= MIN_CONFIDENCE
        and up_votes >= 2
    ):
        return (
            "CALL",
            probability_up,
            f"P(up)={probability_up:.1%}; votes={up_votes}/3",
        )

    if (
        probability_down >= MIN_CONFIDENCE
        and down_votes >= 2
    ):
        return (
            "PUT",
            probability_down,
            f"P(down)={probability_down:.1%}; votes={down_votes}/3",
        )

    return (
        "WAIT",
        max(probability_up, probability_down),
        f"No aligned setup; P(up)={probability_up:.1%}; "
        f"P(down)={probability_down:.1%}; "
        f"votes up/down={up_votes}/{down_votes}",
    )


# ======================== TELEGRAM ==========================

def send_telegram(message):
    if not TG_TOKEN or not TG_CHAT:
        log("[TELEGRAM] Token or chat ID missing", "warning")
        return False

    try:
        response = HTTP.post(
            f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage",
            json={
                "chat_id": TG_CHAT,
                "text": message,
                "disable_web_page_preview": True,
            },
            timeout=20,
        )

        if response.status_code >= 400:
            log(
                f"[TELEGRAM ERROR] {response.status_code}: "
                f"{response.text[:200]}",
                "error",
            )
            return False

        if not response.json().get("ok", False):
            log("[TELEGRAM ERROR] API returned ok=false", "error")
            return False

        log("[TELEGRAM] Message sent")
        return True

    except Exception as exc:
        log(f"[TELEGRAM ERROR] {exc}", "error")
        return False


# ======================== SPXW OPTIONS ======================

def get_spxw_contracts(direction):
    try:
        params = {
            "underlying_symbols": "SPX",
            "expiration_date": ny_now().date().isoformat(),
            "type": "call" if direction == "CALL" else "put",
            "status": "active",
            "limit": 1000,
        }

        response = HTTP.get(
            f"{TRADE_URL}/v2/options/contracts",
            headers=alpaca_headers(),
            params=params,
            timeout=25,
        )

        if response.status_code >= 400:
            raise RuntimeError(
                f"Contracts HTTP {response.status_code}: "
                f"{response.text[:250]}"
            )

        payload = response.json()
        contracts = payload.get(
            "option_contracts",
            payload.get("contracts", []),
        ) or []

        results = []

        for contract in contracts:
            symbol = contract.get("symbol", "")
            root = contract.get("root_symbol", "")

            if not symbol:
                continue

            if root != "SPXW" and not symbol.startswith("SPXW"):
                continue

            try:
                strike = float(contract["strike_price"])
            except (KeyError, TypeError, ValueError):
                continue

            results.append({
                "symbol": symbol,
                "strike": strike,
                "expiration": contract.get("expiration_date", ""),
            })

        return results

    except Exception as exc:
        log(f"[CONTRACT ERROR] {exc}", "error")
        return []


def get_option_quotes(symbols):
    results = {}
    endpoint = f"{DATA_URL}/v1beta1/options/quotes/latest"

    # Quote requests are batched, not sent once per contract.
    for start in range(0, len(symbols), 100):
        batch = symbols[start:start + 100]

        try:
            response = HTTP.get(
                endpoint,
                headers=alpaca_headers(),
                params={
                    "symbols": ",".join(batch),
                    "feed": OPTIONS_FEED,
                },
                timeout=25,
            )

            if response.status_code >= 400:
                log(
                    f"[QUOTE ERROR] HTTP {response.status_code}: "
                    f"{response.text[:200]}",
                    "warning",
                )
                continue

            payload = response.json()
            quotes = payload.get("quotes", {}) or {}

            if isinstance(quotes, dict):
                results.update(quotes)

        except Exception as exc:
            log(f"[QUOTE ERROR] {exc}", "warning")

    return results


def number_from(quote, *keys):
    for key in keys:
        try:
            value = quote.get(key)
            if value is not None:
                value = float(value)
                if np.isfinite(value):
                    return value
        except (TypeError, ValueError):
            pass

    return None


def choose_contract(direction, spot_price):
    contracts = get_spxw_contracts(direction)

    if not contracts:
        log("[CONTRACT] No SPXW contracts found", "warning")
        return None

    # Only inspect the 15 strikes closest to actual SPX spot.
    contracts.sort(
        key=lambda item: abs(item["strike"] - spot_price)
    )
    candidates = contracts[:MAX_CONTRACTS_TO_CHECK]

    quotes = get_option_quotes(
        [item["symbol"] for item in candidates]
    )

    viable = []

    for contract in candidates:
        quote = quotes.get(contract["symbol"])

        if not isinstance(quote, dict):
            continue

        bid = number_from(quote, "bp", "bid_price", "bid")
        ask = number_from(quote, "ap", "ask_price", "ask")

        if (
            bid is None or ask is None
            or bid <= 0 or ask <= 0 or ask < bid
        ):
            continue

        mid = (bid + ask) / 2.0
        spread_pct = (ask - bid) / mid if mid > 0 else 1.0

        if spread_pct > MAX_SPREAD_PCT:
            continue

        viable.append({
            **contract,
            "bid": bid,
            "ask": ask,
            "mid": mid,
            "spread_pct": spread_pct,
            "distance": abs(contract["strike"] - spot_price),
        })

    if not viable:
        log(
            "[CONTRACT] No nearby contract passed quote/spread checks",
            "warning",
        )
        return None

    # First prefer distance from SPX spot, then narrower spread.
    viable.sort(
        key=lambda item: (
            item["distance"],
            item["spread_pct"],
            item["mid"],
        )
    )

    selected = viable[0]

    log(
        f"[CONTRACT] {selected['symbol']} "
        f"| strike={selected['strike']:.2f} "
        f"| bid={selected['bid']:.2f} "
        f"| ask={selected['ask']:.2f} "
        f"| spread={selected['spread_pct']:.1%} "
        f"| SPX={spot_price:.2f}"
    )

    return selected


# ======================== SCANNER ===========================

def market_is_open():
    now = ny_now()

    if now.weekday() >= 5:
        return False

    minutes = now.hour * 60 + now.minute

    return (
        9 * 60 + 30 <= minutes < 16 * 60
    )


def run_scan():
    log("========== NEW SCAN ==========")

    spx = get_spx_bars()
    spy = get_stock_bars("SPY")
    qqq = get_stock_bars("QQQ")

    if spx.empty or spy.empty or qqq.empty:
        log("[SIGNAL] WAIT | Market data unavailable", "warning")
        return

    try:
        merged = make_features(spx, spy, qqq)

        model, columns, auc, accuracy = train_model(merged)

        direction, confidence, reason = calculate_signal(
            merged, model, columns, auc
        )

        log(
            f"[SIGNAL] {direction} "
            f"| confidence={confidence:.1%} "
            f"| {reason}"
        )

        if direction == "WAIT":
            return

        # This is the ^GSPC index level, not SPY x 10.
        spot_price = float(merged.iloc[-1]["close"])

        contract = choose_contract(direction, spot_price)

        if contract is None:
            send_telegram(
                "⚠️ SPX 0DTE AI ADVISOR v18.2\n"
                f"Signal: {direction}\n"
                f"Confidence: {confidence:.1%}\n"
                f"SPX: {spot_price:,.2f}\n"
                f"Model AUC: {auc:.3f}\n"
                "No contract passed quote and spread filters.\n"
                "Recommendation only — no order was placed."
            )
            return

        message = (
            "📊 SPX 0DTE AI ADVISOR v18.2\n\n"
            f"Signal: {direction}\n"
            f"Model confidence: {confidence:.1%}\n"
            f"SPX index: {spot_price:,.2f}\n"
            f"Contract: {contract['symbol']}\n"
            f"Strike: {contract['strike']:,.2f}\n"
            f"Bid / Ask: {contract['bid']:.2f} / "
            f"{contract['ask']:.2f}\n"
            f"Mid estimate: {contract['mid']:.2f}\n"
            f"Spread: {contract['spread_pct']:.1%}\n"
            f"Model AUC: {auc:.3f}\n"
            f"Test accuracy: {accuracy:.3f}\n"
            f"Reason: {reason}\n\n"
            "⚠️ Recommendation only. No order was placed. "
            "0DTE options can lose value rapidly."
        )

        send_telegram(message)

    except Exception as exc:
        log(
            f"[SCAN ERROR] {type(exc).__name__}: {exc}",
            "error",
        )


def main():
    log("SPX 0DTE AI ADVISOR v18.2 starting")
    log("MODE: RECOMMENDATION ONLY — NO ORDER EXECUTION")
    log(
        f"Data feed={DATA_FEED} | Options feed={OPTIONS_FEED} "
        f"| Min AUC={MIN_AUC:.3f} "
        f"| Min confidence={MIN_CONFIDENCE:.2f}"
    )

    if not API_KEY or not API_SECRET:
        log(
            "Missing Alpaca API credentials in Railway Variables",
            "error",
        )

    if not TG_TOKEN or not TG_CHAT:
        log(
            "Telegram credentials missing; alerts will not be sent",
            "warning",
        )

    while True:
        try:
            if market_is_open():
                run_scan()
            else:
                log(
                    "[MARKET] Closed | New York time "
                    + ny_now().strftime("%Y-%m-%d %H:%M:%S %Z")
                )

        except KeyboardInterrupt:
            log("Stopped by user")
            break

        except Exception as exc:
            log(
                f"[FATAL LOOP ERROR] {type(exc).__name__}: {exc}",
                "error",
            )

        time.sleep(SCAN_SECONDS)


if __name__ == "__main__":
    main()
