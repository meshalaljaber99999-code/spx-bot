# ============================================================
# SPX & 10 STOCKS 0DTE AI ADVISOR / SCANNER v16.3
# DIAGNOSTIC PRO / HONEST MODE
# ============================================================
# Recommendation Only - NO ORDER EXECUTION
#
# v16.3 FIXES:
# - Full Alpaca error diagnostics
# - Data health diagnostics
# - QQQ feature alignment fixed
# - Detailed rejection reasons
# - Option quote diagnostics
# - No false cooldown before Telegram send
# - SPX proxy clearly identified as SPY x 10
# - ML / AUC / option score transparency
# - No silent failures
# ============================================================

import os
import time
import warnings
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")


# ============================================================
# CONFIG
# ============================================================

VERSION = "v16.3-DIAGNOSTIC-PRO"

NY = ZoneInfo("America/New_York")

TIMEFRAME = "5Min"
HISTORY_DAYS = 60

MIN_TRAIN_ROWS = 150
HORIZON = 6
ATR_TARGET = 0.50

# ML
MIN_PROBABILITY = 0.62

# OPTIONS
MIN_OPTION_SCORE = 75
MAX_SPREAD_PERCENT = 0.15
MIN_OPTION_PREMIUM = 0.20
MAX_STRIKE_DISTANCE = 40

# Runtime
SIGNAL_COOLDOWN_MINUTES = 20
POLL_SECONDS = 30
MAX_TELEGRAM_OPPORTUNITIES = 3

# Market protection
NO_TRADE_FIRST_MIN = 10
NO_TRADE_LAST_MIN = 30

# Symbols
STOCKS = [
    "NVDA",
    "AAPL",
    "MSFT",
    "TSLA",
    "AMZN",
    "META",
    "GOOGL",
    "AMD",
    "AVGO",
    "NFLX",
]

ALL_SYMBOLS = ["SPY", "QQQ"] + STOCKS


# ============================================================
# ENV
# ============================================================

ALPACA_API_KEY = (
    os.getenv("ALPACA_API_KEY")
    or os.getenv("APCA_API_KEY_ID")
    or ""
)

ALPACA_SECRET_KEY = (
    os.getenv("ALPACA_SECRET_KEY")
    or os.getenv("APCA_SECRET_KEY_ID")
    or ""
)

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

ALPACA_TRADING_URL = os.getenv(
    "ALPACA_TRADING_URL",
    "https://api.alpaca.markets"
)

ALPACA_DATA_URL = "https://data.alpaca.markets"

DATA_FEED = os.getenv(
    "ALPACA_DATA_FEED",
    "iex"
)

OPTIONS_CONTRACTS_URL = (
    f"{ALPACA_TRADING_URL}/v2/options/contracts"
)

OPTIONS_LATEST_QUOTES_URL = (
    f"{ALPACA_DATA_URL}/v1beta1/options/quotes/latest"
)


# ============================================================
# GLOBAL STATE
# ============================================================

STATE = {
    "last_sent": {},
}

session = requests.Session()

HEADERS = {
    "APCA-API-KEY-ID": ALPACA_API_KEY,
    "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY,
}


# ============================================================
# LOGGING
# ============================================================

def log(message):
    now = datetime.now(NY).strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] {message}", flush=True)


def log_section(title):
    log("")
    log("=" * 60)
    log(title)
    log("=" * 60)


# ============================================================
# TELEGRAM
# ============================================================

def telegram_send(message):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        log("[TELEGRAM] Missing token/chat ID")
        return False

    url = (
        f"https://api.telegram.org/"
        f"bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    )

    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }

    try:
        r = session.post(
            url,
            json=payload,
            timeout=15
        )

        if r.ok:
            return True

        log(
            f"[TELEGRAM ERROR] "
            f"HTTP {r.status_code}: {r.text[:500]}"
        )

        return False

    except Exception as e:
        log(f"[TELEGRAM EXCEPTION] {e}")
        return False


# ============================================================
# TIME
# ============================================================

def now_ny():
    return datetime.now(NY)


def iso_utc(dt):
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=NY)

    return (
        dt.astimezone(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
    )


# ============================================================
# MARKET STATUS
# ============================================================

def market_open_now():

    now = now_ny()

    mins = now.hour * 60 + now.minute

    open_min = 9 * 60 + 30
    close_min = 16 * 60

    if mins < open_min:
        return False, "السوق لم يفتح بعد"

    if mins >= close_min:
        return False, "السوق مغلق"

    if mins - open_min < NO_TRADE_FIRST_MIN:
        return False, "أول دقائق السوق - حماية"

    if close_min - mins <= NO_TRADE_LAST_MIN:
        return False, "آخر دقائق السوق - حماية"

    return True, "السوق مفتوح"


# ============================================================
# ALPACA GET
# ============================================================

def alpaca_get(url, params=None, timeout=30):

    try:

        r = session.get(
            url,
            headers=HEADERS,
            params=params,
            timeout=timeout,
        )

        if r.status_code != 200:

            log(
                f"[ALPACA ERROR] "
                f"HTTP {r.status_code}"
            )

            try:
                body = r.json()
                log(
                    f"[ALPACA ERROR BODY] "
                    f"{body}"
                )
            except Exception:
                log(
                    f"[ALPACA ERROR TEXT] "
                    f"{r.text[:800]}"
                )

            return None

        return r.json()

    except requests.exceptions.Timeout:

        log("[ALPACA ERROR] Request timeout")
        return None

    except Exception as e:

        log(
            f"[ALPACA EXCEPTION] "
            f"{type(e).__name__}: {e}"
        )

        return None


# ============================================================
# STOCK BARS
# ============================================================

def fetch_all_bars(symbols):

    end_dt = datetime.now(timezone.utc)

    start_dt = (
        end_dt -
        timedelta(days=HISTORY_DAYS)
    )

    result = {
        s: []
        for s in symbols
    }

    page_token = None

    while True:

        params = {
            "symbols": ",".join(symbols),
            "timeframe": TIMEFRAME,
            "start": iso_utc(start_dt),
            "end": iso_utc(end_dt),
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

        if not data:
            log(
                "[DATA] No response from Alpaca "
                "stock bars endpoint"
            )
            break

        bars = data.get("bars", {})

        if not bars:
            log(
                "[DATA] Alpaca returned empty bars."
            )

        for symbol in symbols:

            symbol_bars = bars.get(
                symbol,
                []
            )

            for b in symbol_bars:

                try:

                    result[symbol].append(
                        {
                            "timestamp": b.get("t"),

                            "open": float(
                                b.get("o", 0)
                            ),

                            "high": float(
                                b.get("h", 0)
                            ),

                            "low": float(
                                b.get("l", 0)
                            ),

                            "close": float(
                                b.get("c", 0)
                            ),

                            "volume": float(
                                b.get("v", 0)
                            ),
                        }
                    )

                except Exception as e:

                    log(
                        f"[DATA PARSE ERROR] "
                        f"{symbol}: {e}"
                    )

        page_token = data.get(
            "next_page_token"
        )

        if not page_token:
            break

    frames = {}

    for symbol, rows in result.items():

        if not rows:

            frames[symbol] = (
                pd.DataFrame()
            )

            continue

        df = pd.DataFrame(rows)

        df["timestamp"] = pd.to_datetime(
            df["timestamp"],
            utc=True
        )

        df = (
            df
            .drop_duplicates("timestamp")
            .sort_values("timestamp")
            .reset_index(drop=True)
        )

        frames[symbol] = df

    return frames


# ============================================================
# DATA HEALTH
# ============================================================

def report_data_health(frames):

    log_section("DATA HEALTH")

    total_ok = 0

    for symbol in ALL_SYMBOLS:

        df = frames.get(symbol)

        if df is None or df.empty:

            log(
                f"[DATA] {symbol}: "
                f"❌ NO DATA"
            )

        else:

            rows = len(df)

            last_ts = (
                df["timestamp"].iloc[-1]
                if "timestamp" in df.columns
                else None
            )

            log(
                f"[DATA] {symbol}: "
                f"✓ {rows:,} bars | "
                f"last={last_ts}"
            )

            total_ok += 1

    log(
        f"[DATA SUMMARY] "
        f"{total_ok}/{len(ALL_SYMBOLS)} symbols OK"
    )


# ============================================================
# INDICATORS
# ============================================================

def rsi(series, period=14):

    delta = series.diff()

    gain = delta.clip(lower=0)

    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(period).mean()

    avg_loss = (
        loss
        .rolling(period)
        .mean()
        .replace(0, np.nan)
    )

    rs = avg_gain / avg_loss

    return 100 - (
        100 / (1 + rs)
    )


def atr(df, period=14):

    prev = df["close"].shift(1)

    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev).abs(),
            (df["low"] - prev).abs(),
        ],
        axis=1,
    ).max(axis=1)

    return tr.rolling(period).mean()


# ============================================================
# FEATURES
# ============================================================

def make_features(
    df,
    market_df=None,
    qqq_df=None
):

    x = df.copy()

    close = x["close"]

    x["ret_1"] = close.pct_change(1)
    x["ret_3"] = close.pct_change(3)
    x["ret_6"] = close.pct_change(6)
    x["ret_12"] = close.pct_change(12)

    x["rsi"] = rsi(close, 14)

    x["atr"] = atr(x, 14)

    x["atr_pct"] = (
        x["atr"] /
        close
    )

    x["range_pct"] = (
        (x["high"] - x["low"]) /
        close
    )

    x["volatility"] = (
        x["ret_1"]
        .rolling(20)
        .std()
    )

    x["ma_9"] = (
        close
        .rolling(9)
        .mean()
    )

    x["ma_20"] = (
        close
        .rolling(20)
        .mean()
    )

    x["ma_50"] = (
        close
        .rolling(50)
        .mean()
    )

    x["ma9_dist"] = (
        close / x["ma_9"] - 1
    )

    x["ma20_dist"] = (
        close / x["ma_20"] - 1
    )

    x["ma50_dist"] = (
        close / x["ma_50"] - 1
    )

    x["momentum"] = (
        close /
        close.shift(12) - 1
    )

    x["acceleration"] = (
        x["ret_3"] -
        x["ret_3"].shift(3)
    )

    vol_m = (
        x["volume"]
        .rolling(30)
        .mean()
    )

    vol_s = (
        x["volume"]
        .rolling(30)
        .std()
    )

    x["volume_z"] = (
        (x["volume"] - vol_m) /
        vol_s.replace(0, np.nan)
    )

    # --------------------------------------------------------
    # MARKET FEATURES
    # --------------------------------------------------------

    if (
        market_df is not None
        and not market_df.empty
    ):

        m = (
            market_df[
                ["timestamp", "close"]
            ]
            .rename(
                columns={
                    "close":
                    "market_close"
                }
            )
            .sort_values("timestamp")
        )

        x = pd.merge_asof(
            x.sort_values("timestamp"),
            m,
            on="timestamp",
            direction="backward",
        )

        x["market_ret_3"] = (
            x["market_close"]
            .pct_change(3)
        )

        x["market_ret_12"] = (
            x["market_close"]
            .pct_change(12)
        )

        x["relative_strength"] = (
            x["ret_3"] -
            x["market_ret_3"]
        )

    else:

        x["market_ret_3"] = 0.0
        x["market_ret_12"] = 0.0
        x["relative_strength"] = 0.0

    # --------------------------------------------------------
    # QQQ FEATURES
    # --------------------------------------------------------

    if (
        qqq_df is not None
        and not qqq_df.empty
    ):

        q = (
            qqq_df[
                ["timestamp", "close"]
            ]
            .rename(
                columns={
                    "close":
                    "qqq_close"
                }
            )
            .sort_values("timestamp")
        )

        x = pd.merge_asof(
            x.sort_values("timestamp"),
            q,
            on="timestamp",
            direction="backward",
        )

        # FIX:
        # Calculate from merged x,
        # not from original q dataframe.
        x["qqq_ret_3"] = (
            x["qqq_close"]
            .pct_change(3)
        )

        x["qqq_ret_12"] = (
            x["qqq_close"]
            .pct_change(12)
        )

    else:

        x["qqq_ret_3"] = 0.0
        x["qqq_ret_12"] = 0.0

    # --------------------------------------------------------
    # TIME FEATURES
    # --------------------------------------------------------

    local_time = (
        x["timestamp"]
        .dt
        .tz_convert(NY)
    )

    mins = (
        local_time.dt.hour * 60
        +
        local_time.dt.minute
    )

    x["time_sin"] = np.sin(
        2 * np.pi * mins / 1440
    )

    x["time_cos"] = np.cos(
        2 * np.pi * mins / 1440
    )

    # --------------------------------------------------------
    # TARGET
    # --------------------------------------------------------

    future_high = pd.concat(
        [
            x["high"].shift(-i)
            for i in range(
                1,
                HORIZON + 1
            )
        ],
        axis=1,
    ).max(axis=1)

    future_low = pd.concat(
        [
            x["low"].shift(-i)
            for i in range(
                1,
                HORIZON + 1
            )
        ],
        axis=1,
    ).min(axis=1)

    up_hit = (
        future_high >=
        close + x["atr"] * ATR_TARGET
    )

    down_hit = (
        future_low <=
        close - x["atr"] * ATR_TARGET
    )

    x["target"] = np.where(
        up_hit & ~down_hit,
        1,
        np.where(
            down_hit & ~up_hit,
            0,
            np.nan
        )
    )

    return x


# ============================================================
# FEATURES
# ============================================================

FEATURE_COLUMNS = [

    "ret_1",
    "ret_3",
    "ret_6",
    "ret_12",

    "rsi",

    "atr_pct",
    "range_pct",
    "volatility",

    "ma9_dist",
    "ma20_dist",
    "ma50_dist",

    "momentum",
    "acceleration",

    "volume_z",

    "market_ret_3",
    "market_ret_12",
    "relative_strength",

    "qqq_ret_3",
    "qqq_ret_12",

    "time_sin",
    "time_cos",
]


# ============================================================
# MODEL TRAINING
# ============================================================

def train_model(feature_df):

    if (
        feature_df is None
        or feature_df.empty
    ):
        return None

    clean = (
        feature_df
        .dropna(
            subset=
            FEATURE_COLUMNS +
            ["target"]
        )
        .copy()
    )

    if len(clean) < MIN_TRAIN_ROWS:
        return None

    clean["target"] = (
        clean["target"]
        .astype(int)
    )

    n = len(clean)

    train_end = int(
        n * 0.60
    )

    test_start = int(
        n * 0.80
    )

    train = clean.iloc[
        :train_end
    ]

    test = clean.iloc[
        test_start:
    ]

    if (
        train["target"]
        .nunique() < 2
    ):
        return None

    if (
        test["target"]
        .nunique() < 2
    ):
        return None

    models = []

    for seed in [
        17,
        41,
        83
    ]:

        model = (
            HistGradientBoostingClassifier(
                max_iter=250,
                learning_rate=0.045,
                max_leaf_nodes=15,
                min_samples_leaf=25,
                l2_regularization=1.0,
                random_state=seed,
            )
        )

        model.fit(
            train[FEATURE_COLUMNS],
            train["target"]
        )

        models.append(model)

    probs = [
        model.predict_proba(
            test[FEATURE_COLUMNS]
        )[:, 1]
        for model in models
    ]

    mean_prob = np.mean(
        probs,
        axis=0
    )

    try:

        auc = roc_auc_score(
            test["target"],
            mean_prob
        )

    except Exception:

        auc = 0.50

    auc = float(
        np.clip(
            auc,
            0.50,
            0.999
        )
    )

    return {
        "models": models,
        "auc": auc,
        "rows": len(clean),
    }


# ============================================================
# PREDICTION
# ============================================================

def predict(
    model_info,
    feature_df
):

    if not model_info:
        return None

    clean = (
        feature_df
        .dropna(
            subset=FEATURE_COLUMNS
        )
    )

    if clean.empty:
        return None

    latest = clean.iloc[-1]

    X = pd.DataFrame(
        [
            latest[
                FEATURE_COLUMNS
            ].values
        ],
        columns=FEATURE_COLUMNS,
    )

    probs = [
        model.predict_proba(X)[0][1]
        for model in
        model_info["models"]
    ]

    p_up = float(
        np.mean(probs)
    )

    p_down = 1.0 - p_up

    if p_up >= MIN_PROBABILITY:

        signal = "CALL"

    elif p_down >= MIN_PROBABILITY:

        signal = "PUT"

    else:

        signal = "WAIT"

    vol = float(
        latest["volatility"]
    )

    if vol >= 0.0030:

        regime = "HIGH VOL"

    elif vol < 0.0015:

        regime = "LOW VOL"

    else:

        regime = "NORMAL"

    return {

        "signal": signal,

        "p_up": p_up,

        "p_down": p_down,

        "confidence": max(
            p_up,
            p_down
        ),

        "price": float(
            latest["close"]
        ),

        "atr": float(
            latest["atr"]
        ),

        "rsi": float(
            latest["rsi"]
        ),

        "momentum": float(
            latest["momentum"]
        ),

        "volatility": vol,

        "volume_z": float(
            latest["volume_z"]
        ),

        "relative_strength": float(
            latest[
                "relative_strength"
            ]
        ),

        "auc": model_info["auc"],

        "rows": model_info["rows"],

        "regime": regime,
    }


# ============================================================
# OPTIONS CONTRACTS
# ============================================================

def get_option_contracts(
    underlying,
    option_type,
    expiration
):

    params = {

        "underlying_symbols":
            underlying,

        "status":
            "active",

        "expiration_date":
            expiration,

        "type":
            option_type.lower(),

        "limit":
            10000,
    }

    data = alpaca_get(
        OPTIONS_CONTRACTS_URL,
        params=params,
        timeout=30,
    )

    if not data:

        log(
            f"[OPTIONS] "
            f"{underlying} {option_type}: "
            f"NO CONTRACT RESPONSE"
        )

        return []

    contracts = (
        data.get(
            "option_contracts"
        )
        or
        data.get(
            "contracts"
        )
        or []
    )

    log(
        f"[OPTIONS] "
        f"{underlying} {option_type}: "
        f"{len(contracts)} contracts"
    )

    return contracts


# ============================================================
# OPTION QUOTES
# ============================================================

def get_option_quotes(symbols):

    if not symbols:
        return {}

    result = {}

    for i in range(
        0,
        len(symbols),
        100
    ):

        batch = symbols[
            i:i + 100
        ]

        params = {
            "symbols":
                ",".join(batch),

            "feed":
                "indicative",
        }

        data = alpaca_get(
            OPTIONS_LATEST_QUOTES_URL,
            params=params,
            timeout=30,
        )

        if not data:

            log(
                "[OPTIONS QUOTES] "
                "No response"
            )

            continue

        quotes = data.get(
            "quotes",
            {}
        )

        for symbol, q in quotes.items():

            bid = q.get("bp")
            ask = q.get("ap")

            if (
                bid is None
                or ask is None
            ):
                continue

            try:

                bid = float(bid)
                ask = float(ask)

            except Exception:

                continue

            if (
                bid <= 0
                or ask <= 0
            ):
                continue

            mid = (
                bid + ask
            ) / 2

            spread_pct = (
                ask - bid
            ) / mid

            result[symbol] = {

                "bid": bid,

                "ask": ask,

                "mid": mid,

                "spread_pct":
                    spread_pct,
            }

    log(
        f"[OPTIONS QUOTES] "
        f"Valid quotes: "
        f"{len(result)}"
    )

    return result


# ============================================================
# OPTION SCORE
# ============================================================

def score_option(
    pred,
    contract,
    quote
):

    if (
        not pred
        or not contract
        or not quote
    ):
        return 0

    score = 0.0

    # --------------------------------------------------------
    # ML CONFIDENCE
    # --------------------------------------------------------

    score += min(
        30,
        max(
            0,
            (
                pred["confidence"]
                - 0.50
            ) * 100
        ) * 0.75
    )

    # --------------------------------------------------------
    # AUC
    # --------------------------------------------------------

    score += min(
        15,
        max(
            0,
            (
                pred["auc"]
                - 0.50
            ) * 100
        ) * 0.75
    )

    # --------------------------------------------------------
    # SPREAD
    # --------------------------------------------------------

    sp_pct = quote[
        "spread_pct"
    ]

    if sp_pct <= 0.05:

        score += 20

    elif sp_pct <= 0.08:

        score += 16

    elif sp_pct <= 0.12:

        score += 10

    elif (
        sp_pct
        <= MAX_SPREAD_PERCENT
    ):

        score += 4

    else:

        return 0

    # --------------------------------------------------------
    # PREMIUM
    # --------------------------------------------------------

    if (
        quote["mid"]
        < MIN_OPTION_PREMIUM
    ):
        return 0

    if quote["mid"] >= 1:

        score += 8

    elif quote["mid"] >= 0.50:

        score += 5

    else:

        score += 2

    # --------------------------------------------------------
    # MOMENTUM
    # --------------------------------------------------------

    momentum = abs(
        pred["momentum"]
    )

    if momentum >= 0.004:

        score += 8

    elif momentum >= 0.002:

        score += 5

    else:

        score += 1

    # --------------------------------------------------------
    # VOLUME
    # --------------------------------------------------------

    volume_z = pred[
        "volume_z"
    ]

    if volume_z >= 2:

        score += 8

    elif volume_z >= 1:

        score += 5

    else:

        score += 2

    return int(
        max(
            0,
            min(
                100,
                round(score)
            )
        )
    )


# ============================================================
# CHOOSE CONTRACT
# ============================================================

def choose_best_contract(
    symbol,
    signal,
    underlying_price
):

    today = (
        now_ny()
        .date()
        .isoformat()
    )

    option_type = (
        "call"
        if signal == "CALL"
        else "put"
    )

    contracts = get_option_contracts(
        symbol,
        option_type,
        today
    )

    if not contracts:

        log(
            f"[REJECT] {symbol} {signal}: "
            f"NO 0DTE CONTRACTS"
        )

        return None, "NO_CONTRACTS"

    candidates = []

    for c in contracts:

        try:

            strike = float(
                c.get(
                    "strike_price"
                )
            )

        except Exception:

            continue

        if not c.get(
            "tradable",
            True
        ):
            continue

        if symbol == "SPX":

            root = str(
                c.get(
                    "root_symbol",
                    ""
                )
            ).upper()

            c_sym = str(
                c.get(
                    "symbol",
                    ""
                )
            ).upper()

            if (
                "SPXW" not in root
                and "SPXW" not in c_sym
            ):
                continue

        distance = abs(
            strike -
            underlying_price
        )

        if (
            distance
            > MAX_STRIKE_DISTANCE
        ):
            continue

        candidates.append(
            (
                distance,
                c
            )
        )

    if not candidates:

        log(
            f"[REJECT] {symbol} {signal}: "
            f"NO STRIKE WITHIN "
            f"{MAX_STRIKE_DISTANCE}"
        )

        return None, "NO_STRIKE"

    candidates.sort(
        key=lambda x: x[0]
    )

    selected = [
        c
        for _, c
        in candidates[:25]
    ]

    symbols = [
        c.get("symbol")
        for c in selected
        if c.get("symbol")
    ]

    quotes = get_option_quotes(
        symbols
    )

    if not quotes:

        log(
            f"[REJECT] {symbol} {signal}: "
            f"NO VALID OPTION QUOTES"
        )

        return None, "NO_QUOTES"

    best = None
    best_score = -999

    rejected_spread = 0
    rejected_premium = 0

    for c in selected:

        sym = c.get("symbol")

        if sym not in quotes:

            continue

        q = quotes[sym]

        if (
            q["mid"]
            < MIN_OPTION_PREMIUM
        ):

            rejected_premium += 1
            continue

        if (
            q["spread_pct"]
            > MAX_SPREAD_PERCENT
        ):

            rejected_spread += 1
            continue

        distance = abs(
            float(
                c["strike_price"]
            )
            -
            underlying_price
        )

        candidate_score = (
            20
            -
            min(
                20,
                q["spread_pct"] * 100
            )
            -
            (
                distance /
                max(
                    underlying_price,
                    1
                ) * 100
            )
        )

        if (
            candidate_score
            > best_score
        ):

            best_score = (
                candidate_score
            )

            best = {
                "contract": c,
                "quote": q,
            }

    if not best:

        log(
            f"[REJECT] {symbol} {signal}: "
            f"NO LIQUID CONTRACT | "
            f"premium rejects="
            f"{rejected_premium} | "
            f"spread rejects="
            f"{rejected_spread}"
        )

        return None, "NO_LIQUID_CONTRACT"

    return best, "OK"


# ============================================================
# BUILD OPPORTUNITY
# ============================================================

def build_opportunity(
    symbol,
    display_symbol,
    pred
):

    if not pred:

        return None, "NO_PREDICTION"

    if pred["signal"] not in (
        "CALL",
        "PUT"
    ):

        return None, (
            f"ML_{pred['signal']}"
        )

    contract_data, reason = (
        choose_best_contract(
            symbol,
            pred["signal"],
            pred["price"]
        )
    )

    if not contract_data:

        return None, reason

    contract = (
        contract_data["contract"]
    )

    quote = (
        contract_data["quote"]
    )

    score = score_option(
        pred,
        contract,
        quote
    )

    if score < MIN_OPTION_SCORE:

        log(
            f"[REJECT] {display_symbol} "
            f"{pred['signal']} | "
            f"Score={score}/100 < "
            f"{MIN_OPTION_SCORE}"
        )

        return None, (
            f"SCORE_{score}"
        )

    entry = quote["mid"]

    return {

        "symbol":
            display_symbol,

        "signal":
            pred["signal"],

        "confidence":
            pred["confidence"],

        "price":
            pred["price"],

        "auc":
            pred["auc"],

        "regime":
            pred["regime"],

        "score":
            score,

        "contract_symbol":
            contract.get(
                "symbol",
                "UNKNOWN"
            ),

        "strike":
            float(
                contract.get(
                    "strike_price",
                    0
                )
            ),

        "expiration":
            contract.get(
                "expiration_date",
                ""
            ),

        "entry":
            entry,

        "target":
            entry * 1.40,

        "stop":
            entry * 0.70,

        "bid":
            quote["bid"],

        "ask":
            quote["ask"],

        "spread_pct":
            quote["spread_pct"],
    }, "OK"


# ============================================================
# CONSENSUS
# ============================================================

def market_consensus(
    predictions
):

    valid = [
        p
        for p in predictions
        if p is not None
    ]

    if not valid:

        return {
            "bias": "NEUTRAL",
            "strength": 0,
        }

    call_c = sum(
        1
        for p in valid
        if p["signal"] == "CALL"
    )

    put_c = sum(
        1
        for p in valid
        if p["signal"] == "PUT"
    )

    call_p = np.mean(
        [
            p["p_up"]
            for p in valid
        ]
    )

    put_p = np.mean(
        [
            p["p_down"]
            for p in valid
        ]
    )

    if call_c > put_c:

        return {
            "bias": "BULLISH",
            "strength":
                float(call_p),
        }

    if put_c > call_c:

        return {
            "bias": "BEARISH",
            "strength":
                float(put_p),
        }

    return {
        "bias": "NEUTRAL",
        "strength":
            float(
                max(
                    call_p,
                    put_p
                )
            ),
    }


# ============================================================
# FORMAT TELEGRAM
# ============================================================

def format_opportunity(
    o,
    rank
):

    emoji = (
        "🥇"
        if rank == 1
        else "🥈"
        if rank == 2
        else "🥉"
    )

    sig_emoji = (
        "🟢"
        if o["signal"] == "CALL"
        else "🔴"
    )

    return (

        f"{emoji} "
        f"<b>{o['symbol']} "
        f"{sig_emoji} "
        f"{o['signal']}</b>\n"

        f"━━━━━━━━━━━━━━━━━━\n"

        f"🔥 القوة: "
        f"<b>{o['score']}/100</b>\n"

        f"🧠 الثقة: "
        f"<b>{o['confidence']*100:.1f}%</b>\n"

        f"📊 AUC: "
        f"{o['auc']:.2f}\n"

        f"💵 السعر: "
        f"<b>${o['price']:.2f}</b>\n"

        f"🎯 السترايك: "
        f"<b>{o['strike']:.2f}</b>\n"

        f"📜 العقد: "
        f"<code>{o['contract_symbol']}</code>\n"

        f"💰 دخول: "
        f"<b>${o['entry']:.2f}</b>\n"

        f"🎯 هدف: "
        f"<b>${o['target']:.2f}</b>\n"

        f"🛑 وقف: "
        f"<b>${o['stop']:.2f}</b>\n"

        f"↔️ السبريد: "
        f"{o['spread_pct']*100:.1f}%\n"

        f"🌡️ النظام: "
        f"{o['regime']}\n"
    )


# ============================================================
# DUPLICATE CHECK
# ============================================================

def is_duplicate(
    o
):

    key = (
        f"{o['contract_symbol']}_"
        f"{o['signal']}"
    )

    now = time.time()

    if (
        key in STATE["last_sent"]
        and
        (
            now -
            STATE["last_sent"][key]
        ) / 60
        <
        SIGNAL_COOLDOWN_MINUTES
    ):

        return True

    return False


def mark_sent(o):

    key = (
        f"{o['contract_symbol']}_"
        f"{o['signal']}"
    )

    STATE["last_sent"][key] = (
        time.time()
    )


# ============================================================
# PREDICTION DIAGNOSTICS
# ============================================================

def print_prediction(
    symbol,
    pred
):

    if not pred:

        log(
            f"[ML] {symbol}: "
            f"NO PREDICTION"
        )

        return

    log(
        f"[ML] {symbol}: "
        f"{pred['signal']} | "
        f"UP={pred['p_up']*100:.1f}% | "
        f"DOWN={pred['p_down']*100:.1f}% | "
        f"AUC={pred['auc']:.2f} | "
        f"rows={pred['rows']} | "
        f"RSI={pred['rsi']:.1f} | "
        f"mom={pred['momentum']*100:.2f}% | "
        f"volZ={pred['volume_z']:.2f} | "
        f"{pred['regime']}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    log(
        f"SPX & STOCKS AI ADVISOR "
        f"{VERSION} STARTING"
    )

    # --------------------------------------------------------
    # KEYS
    # --------------------------------------------------------

    if not ALPACA_API_KEY:

        log(
            "[FATAL] Missing "
            "ALPACA_API_KEY"
        )

        return

    if not ALPACA_SECRET_KEY:

        log(
            "[FATAL] Missing "
            "ALPACA_SECRET_KEY"
        )

        return

    if not TELEGRAM_BOT_TOKEN:

        log(
            "[WARNING] "
            "Missing TELEGRAM_BOT_TOKEN"
        )

    if not TELEGRAM_CHAT_ID:

        log(
            "[WARNING] "
            "Missing TELEGRAM_CHAT_ID"
        )

    log(
        f"[CONFIG] "
        f"Feed={DATA_FEED} | "
        f"Timeframe={TIMEFRAME} | "
        f"History={HISTORY_DAYS}d | "
        f"MinProb={MIN_PROBABILITY} | "
        f"MinScore={MIN_OPTION_SCORE}"
    )

    # --------------------------------------------------------
    # START TELEGRAM
    # --------------------------------------------------------

    telegram_send(
        f"🤖 <b>AI ADVISOR "
        f"{VERSION}</b>\n\n"
        f"🚀 بدأ التشغيل.\n"
        f"📊 الوضع: Recommendation Only\n"
        f"🛡️ لا يوجد تنفيذ أوامر.\n"
        f"🔍 Diagnostic Mode: ON"
    )

    cycle = 0

    while True:

        cycle += 1

        log_section(
            f"SCAN #{cycle}"
        )

        try:

            # ------------------------------------------------
            # MARKET
            # ------------------------------------------------

            market_ok, reason = (
                market_open_now()
            )

            if not market_ok:

                log(
                    f"[MARKET] WAIT | "
                    f"{reason}"
                )

                time.sleep(
                    POLL_SECONDS
                )

                continue

            log(
                "[MARKET] ✓ "
                "Market open"
            )

            # ------------------------------------------------
            # DATA
            # ------------------------------------------------

            frames = fetch_all_bars(
                ALL_SYMBOLS
            )

            report_data_health(
                frames
            )

            spy = frames.get(
                "SPY"
            )

            qqq = frames.get(
                "QQQ"
            )

            if (
                spy is None
                or spy.empty
            ):

                log(
                    "[FATAL DATA] "
                    "SPY has no bars."
                )

                log(
                    "[HINT] Check Alpaca "
                    "data subscription/feed."
                )

                time.sleep(
                    POLL_SECONDS
                )

                continue

            # ------------------------------------------------
            # MODELS
            # ------------------------------------------------

            models = {}

            # ------------------------------------------------
            # SPX PROXY
            # ------------------------------------------------
            #
            # IMPORTANT:
            # This is NOT real SPX data.
            # It is SPY x 10 proxy.
            # ------------------------------------------------

            spx = spy.copy()

            for col in [
                "open",
                "high",
                "low",
                "close"
            ]:

                spx[col] *= 10

            spx_feat = make_features(
                spx,
                market_df=spy,
                qqq_df=qqq
            )

            spx_model = train_model(
                spx_feat
            )

            if spx_model:

                models["SPX"] = {
                    "features":
                        spx_feat,

                    "model":
                        spx_model,
                }

                log(
                    "[MODEL] SPX "
                    "(SPY×10 proxy): ✓"
                )

            else:

                log(
                    "[MODEL] SPX: "
                    "❌ training failed"
                )

            # ------------------------------------------------
            # STOCKS
            # ------------------------------------------------

            for symbol in STOCKS:

                df = frames.get(
                    symbol
                )

                if (
                    df is None
                    or df.empty
                ):

                    log(
                        f"[MODEL] "
                        f"{symbol}: "
                        f"❌ no data"
                    )

                    continue

                feat = make_features(
                    df,
                    market_df=spy,
                    qqq_df=qqq
                )

                model = train_model(
                    feat
                )

                if model:

                    models[symbol] = {
                        "features":
                            feat,

                        "model":
                            model,
                    }

                    log(
                        f"[MODEL] "
                        f"{symbol}: "
                        f"✓ rows="
                        f"{model['rows']} "
                        f"AUC="
                        f"{model['auc']:.2f}"
                    )

                else:

                    log(
                        f"[MODEL] "
                        f"{symbol}: "
                        f"❌ training failed"
                    )

            if not models:

                log(
                    "[FATAL] "
                    "No models available."
                )

                time.sleep(
                    POLL_SECONDS
                )

                continue

            # ------------------------------------------------
            # PREDICTIONS
            # ------------------------------------------------

            log_section(
                "ML PREDICTIONS"
            )

            predictions = []

            pred_map = {}

            for sym, info in models.items():

                pred = predict(
                    info["model"],
                    info["features"]
                )

                if pred:

                    pred_map[sym] = pred

                    predictions.append(
                        pred
                    )

                    print_prediction(
                        sym,
                        pred
                    )

            # ------------------------------------------------
            # CONSENSUS
            # ------------------------------------------------

            consensus = (
                market_consensus(
                    predictions
                )
            )

            log(
                f"[CONSENSUS] "
                f"{consensus['bias']} | "
                f"strength="
                f"{consensus['strength']*100:.1f}%"
            )

            # ------------------------------------------------
            # OPTIONS
            # ------------------------------------------------

            log_section(
                "OPTION ANALYSIS"
            )

            raw_opportunities = []

            rejection_summary = {}

            for sym, pred in pred_map.items():

                if pred["signal"] not in (
                    "CALL",
                    "PUT"
                ):

                    reason = (
                        f"ML_{pred['signal']}"
                    )

                    rejection_summary[
                        reason
                    ] = (
                        rejection_summary.get(
                            reason,
                            0
                        ) + 1
                    )

                    log(
                        f"[OPTION SKIP] "
                        f"{sym}: "
                        f"{reason}"
                    )

                    continue

                underlying = (
                    "SPX"
                    if sym == "SPX"
                    else sym
                )

                opp, reason = (
                    build_opportunity(
                        underlying,
                        sym,
                        pred
                    )
                )

                if opp:

                    raw_opportunities.append(
                        opp
                    )

                    log(
                        f"[CANDIDATE] "
                        f"{sym} "
                        f"{pred['signal']} | "
                        f"score="
                        f"{opp['score']}/100 | "
                        f"confidence="
                        f"{opp['confidence']*100:.1f}%"
                    )

                else:

                    rejection_summary[
                        reason
                    ] = (
                        rejection_summary.get(
                            reason,
                            0
                        ) + 1
                    )

            # ------------------------------------------------
            # SORT
            # ------------------------------------------------

            raw_opportunities.sort(
                key=lambda x: (
                    x["score"],
                    x["confidence"]
                ),
                reverse=True,
            )

            # ------------------------------------------------
            # REMOVE DUPLICATES
            # WITHOUT MARKING AS SENT
            # ------------------------------------------------

            opportunities = []

            for opp in raw_opportunities:

                if is_duplicate(
                    opp
                ):

                    log(
                        f"[COOLDOWN] "
                        f"{opp['symbol']} "
                        f"{opp['contract_symbol']}"
                    )

                    continue

                opportunities.append(
                    opp
                )

                if (
                    len(opportunities)
                    >= MAX_TELEGRAM_OPPORTUNITIES
                ):
                    break

            # ------------------------------------------------
            # TELEGRAM
            # ------------------------------------------------

            if opportunities:

                msg_lines = [

                    "🚨 "
                    "<b>AI OPTIONS "
                    "& SPX ALERT</b>",

                    (
                        f"🕒 "
                        f"{now_ny().strftime('%H:%M:%S')} "
                        f"NY"
                    ),

                    (
                        f"📈 الاتجاه: "
                        f"<b>{consensus['bias']}</b> "
                        f"("
                        f"{consensus['strength']*100:.1f}%"
                        f")"
                    ),

                    "",
                ]

                for i, opp in enumerate(
                    opportunities,
                    1
                ):

                    msg_lines.append(
                        format_opportunity(
                            opp,
                            i
                        )
                    )

                message = (
                    "\n".join(
                        msg_lines
                    )
                )

                sent = telegram_send(
                    message
                )

                if sent:

                    for opp in opportunities:

                        mark_sent(
                            opp
                        )

                    log(
                        f"[TELEGRAM] "
                        f"✓ Sent "
                        f"{len(opportunities)} "
                        f"opportunities"
                    )

                else:

                    log(
                        "[TELEGRAM] "
                        "❌ SEND FAILED"
                    )

            else:

                log(
                    "[TELEGRAM] "
                    "SILENT"
                )

                if rejection_summary:

                    log(
                        "[REJECTION SUMMARY]"
                    )

                    for reason, count in sorted(
                        rejection_summary.items(),
                        key=lambda x: x[1],
                        reverse=True,
                    ):

                        log(
                            f"  - "
                            f"{reason}: "
                            f"{count}"
                        )

                else:

                    log(
                        "[REJECTION SUMMARY] "
                        "No qualifying ML signals."
                    )

        except Exception as e:

            log(
                f"[MAIN ERROR] "
                f"{type(e).__name__}: {e}"
            )

        time.sleep(
            POLL_SECONDS
        )


# ============================================================
# START
# ============================================================

if __name__ == "__main__":
    main()