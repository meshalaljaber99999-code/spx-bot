# ============================================================
# SPX 0DTE AI ADVISOR v17.0
# SPXW FOCUS / PATIENT / DIAGNOSTIC / HONEST MODE
# ============================================================
#
# Recommendation Only - NO ORDER EXECUTION
#
# v17.0:
# - SPX/SPXW is the PRIMARY instrument
# - Stocks are confirmation/context only
# - No fake SPY x 10 SPX option pricing
# - Explicit 0DTE diagnostics
# - Better SPXW contract filtering
# - Option quote diagnostics
# - Market confirmation from SPY / QQQ / stocks
# - ML ensemble
# - AUC transparency
# - Confidence filter
# - Momentum filter
# - Market alignment filter
# - Liquidity / spread filter
# - Patience mode
# - Telegram only for HIGH QUALITY opportunities
# - No false cooldown
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

VERSION = "v17.0-SPX-0DTE-FOCUS"

NY = ZoneInfo("America/New_York")

TIMEFRAME = "5Min"
HISTORY_DAYS = 60

MIN_TRAIN_ROWS = 150
HORIZON = 6
ATR_TARGET = 0.50

# ------------------------------------------------------------
# ML
# ------------------------------------------------------------

MIN_PROBABILITY = 0.64
MIN_AUC = 0.55

# ------------------------------------------------------------
# SPX SIGNAL QUALITY
# ------------------------------------------------------------

MIN_SPX_SCORE = 78

# Require multiple confirmations
MIN_CONFIRMATIONS = 2

# ------------------------------------------------------------
# OPTIONS
# ------------------------------------------------------------

MAX_SPREAD_PERCENT = 0.12
MIN_OPTION_PREMIUM = 0.50

# Distance from SPX reference
MAX_SPX_STRIKE_DISTANCE = 35

# Avoid extremely cheap / far OTM contracts
MIN_DELTA_PROXY = 0.20
MAX_DELTA_PROXY = 0.80

# ------------------------------------------------------------
# RISK / PATIENCE
# ------------------------------------------------------------

SIGNAL_COOLDOWN_MINUTES = 20
POLL_SECONDS = 30

# No trade during unstable opening/closing periods
NO_TRADE_FIRST_MIN = 10
NO_TRADE_LAST_MIN = 30

# Require fresh market data
MAX_DATA_AGE_MINUTES = 10

# ------------------------------------------------------------
# STOCKS USED AS CONFIRMATION
# ------------------------------------------------------------

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

MARKET_SYMBOLS = [
    "SPY",
    "QQQ",
]

ALL_SYMBOLS = (
    MARKET_SYMBOLS +
    STOCKS
)


# ============================================================
# ENVIRONMENT
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

TELEGRAM_BOT_TOKEN = os.getenv(
    "TELEGRAM_BOT_TOKEN",
    ""
)

TELEGRAM_CHAT_ID = os.getenv(
    "TELEGRAM_CHAT_ID",
    ""
)

ALPACA_TRADING_URL = os.getenv(
    "ALPACA_TRADING_URL",
    "https://api.alpaca.markets"
)

ALPACA_DATA_URL = (
    "https://data.alpaca.markets"
)

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
# GLOBAL
# ============================================================

session = requests.Session()

HEADERS = {
    "APCA-API-KEY-ID":
        ALPACA_API_KEY,

    "APCA-API-SECRET-KEY":
        ALPACA_SECRET_KEY,
}

STATE = {
    "last_sent": {}
}


# ============================================================
# LOGGING
# ============================================================

def log(message):

    now = datetime.now(
        NY
    ).strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    print(
        f"[{now}] {message}",
        flush=True
    )


def section(title):

    log("")
    log("=" * 72)
    log(title)
    log("=" * 72)


# ============================================================
# TELEGRAM
# ============================================================

def telegram_send(message):

    if (
        not TELEGRAM_BOT_TOKEN
        or not TELEGRAM_CHAT_ID
    ):

        log(
            "[TELEGRAM] "
            "Missing token/chat ID"
        )

        return False

    url = (
        "https://api.telegram.org/"
        f"bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    )

    payload = {
        "chat_id":
            TELEGRAM_CHAT_ID,

        "text":
            message,

        "parse_mode":
            "HTML",

        "disable_web_page_preview":
            True,
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
            "[TELEGRAM ERROR] "
            f"HTTP {r.status_code}: "
            f"{r.text[:500]}"
        )

        return False

    except Exception as e:

        log(
            "[TELEGRAM EXCEPTION] "
            f"{e}"
        )

        return False


# ============================================================
# TIME
# ============================================================

def now_ny():

    return datetime.now(NY)


def today_ny():

    return now_ny().date().isoformat()


def iso_utc(dt):

    if dt.tzinfo is None:
        dt = dt.replace(
            tzinfo=NY
        )

    return (
        dt.astimezone(timezone.utc)
        .isoformat()
        .replace(
            "+00:00",
            "Z"
        )
    )


# ============================================================
# MARKET STATUS
# ============================================================

def market_open_now():

    now = now_ny()

    mins = (
        now.hour * 60
        +
        now.minute
    )

    open_min = (
        9 * 60 + 30
    )

    close_min = (
        16 * 60
    )

    if mins < open_min:

        return (
            False,
            "السوق لم يفتح بعد"
        )

    if mins >= close_min:

        return (
            False,
            "السوق مغلق"
        )

    if (
        mins - open_min
        <
        NO_TRADE_FIRST_MIN
    ):

        return (
            False,
            "أول دقائق السوق - حماية"
        )

    if (
        close_min - mins
        <=
        NO_TRADE_LAST_MIN
    ):

        return (
            False,
            "آخر دقائق السوق - حماية"
        )

    return (
        True,
        "السوق مفتوح"
    )


# ============================================================
# ALPACA GET
# ============================================================

def alpaca_get(
    url,
    params=None,
    timeout=30
):

    try:

        r = session.get(
            url,
            headers=HEADERS,
            params=params,
            timeout=timeout
        )

        if r.status_code != 200:

            log(
                "[ALPACA ERROR] "
                f"HTTP {r.status_code}"
            )

            try:

                log(
                    "[ALPACA BODY] "
                    f"{r.json()}"
                )

            except Exception:

                log(
                    "[ALPACA TEXT] "
                    f"{r.text[:1000]}"
                )

            return None

        try:

            return r.json()

        except Exception as e:

            log(
                "[ALPACA JSON ERROR] "
                f"{e}"
            )

            return None

    except requests.exceptions.Timeout:

        log(
            "[ALPACA ERROR] "
            "Request timeout"
        )

        return None

    except Exception as e:

        log(
            "[ALPACA EXCEPTION] "
            f"{type(e).__name__}: {e}"
        )

        return None


# ============================================================
# STOCK BARS
# ============================================================

def fetch_all_bars(symbols):

    end_dt = datetime.now(
        timezone.utc
    )

    start_dt = (
        end_dt -
        timedelta(
            days=HISTORY_DAYS
        )
    )

    result = {
        s: []
        for s in symbols
    }

    page_token = None

    while True:

        params = {

            "symbols":
                ",".join(symbols),

            "timeframe":
                TIMEFRAME,

            "start":
                iso_utc(start_dt),

            "end":
                iso_utc(end_dt),

            "limit":
                10000,

            "feed":
                DATA_FEED,

            "sort":
                "asc",
        }

        if page_token:

            params[
                "page_token"
            ] = page_token

        data = alpaca_get(
            f"{ALPACA_DATA_URL}/v2/stocks/bars",
            params=params,
            timeout=45
        )

        if not data:

            log(
                "[DATA] "
                "No response from bars API"
            )

            break

        bars = data.get(
            "bars",
            {}
        )

        for symbol in symbols:

            for b in bars.get(
                symbol,
                []
            ):

                try:

                    result[symbol].append(
                        {
                            "timestamp":
                                b.get("t"),

                            "open":
                                float(
                                    b.get(
                                        "o",
                                        0
                                    )
                                ),

                            "high":
                                float(
                                    b.get(
                                        "h",
                                        0
                                    )
                                ),

                            "low":
                                float(
                                    b.get(
                                        "l",
                                        0
                                    )
                                ),

                            "close":
                                float(
                                    b.get(
                                        "c",
                                        0
                                    )
                                ),

                            "volume":
                                float(
                                    b.get(
                                        "v",
                                        0
                                    )
                                ),
                        }
                    )

                except Exception as e:

                    log(
                        f"[PARSE ERROR] "
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

        df = pd.DataFrame(
            rows
        )

        df["timestamp"] = (
            pd.to_datetime(
                df["timestamp"],
                utc=True
            )
        )

        df = (
            df
            .drop_duplicates(
                "timestamp"
            )
            .sort_values(
                "timestamp"
            )
            .reset_index(
                drop=True
            )
        )

        frames[symbol] = df

    return frames


# ============================================================
# DATA HEALTH
# ============================================================

def report_data_health(frames):

    section(
        "DATA HEALTH"
    )

    ok = 0

    for symbol in ALL_SYMBOLS:

        df = frames.get(
            symbol
        )

        if (
            df is None
            or df.empty
        ):

            log(
                f"[DATA] {symbol}: "
                "❌ NO DATA"
            )

            continue

        last_ts = (
            df["timestamp"]
            .iloc[-1]
        )

        age_min = (
            datetime.now(
                timezone.utc
            )
            -
            last_ts.to_pydatetime()
        ).total_seconds() / 60

        if age_min <= MAX_DATA_AGE_MINUTES:

            status = "✓ FRESH"

        else:

            status = (
                f"⚠ STALE "
                f"{age_min:.1f}m"
            )

        log(
            f"[DATA] {symbol}: "
            f"{len(df):,} bars | "
            f"last={last_ts} | "
            f"{status}"
        )

        ok += 1

    log(
        f"[DATA SUMMARY] "
        f"{ok}/{len(ALL_SYMBOLS)} "
        "symbols available"
    )


# ============================================================
# INDICATORS
# ============================================================

def rsi(
    series,
    period=14
):

    delta = series.diff()

    gain = (
        delta.clip(
            lower=0
        )
    )

    loss = (
        -delta.clip(
            upper=0
        )
    )

    avg_gain = (
        gain
        .rolling(period)
        .mean()
    )

    avg_loss = (
        loss
        .rolling(period)
        .mean()
        .replace(
            0,
            np.nan
        )
    )

    rs = (
        avg_gain /
        avg_loss
    )

    return (
        100 -
        (
            100 /
            (1 + rs)
        )
    )


def atr(
    df,
    period=14
):

    prev = (
        df["close"]
        .shift(1)
    )

    tr = pd.concat(
        [
            df["high"] -
            df["low"],

            (
                df["high"] -
                prev
            ).abs(),

            (
                df["low"] -
                prev
            ).abs(),
        ],
        axis=1
    ).max(
        axis=1
    )

    return (
        tr
        .rolling(period)
        .mean()
    )


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

    x["ret_1"] = (
        close.pct_change(1)
    )

    x["ret_3"] = (
        close.pct_change(3)
    )

    x["ret_6"] = (
        close.pct_change(6)
    )

    x["ret_12"] = (
        close.pct_change(12)
    )

    x["rsi"] = rsi(
        close
    )

    x["atr"] = atr(
        x
    )

    x["atr_pct"] = (
        x["atr"] /
        close
    )

    x["range_pct"] = (
        (
            x["high"] -
            x["low"]
        ) /
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
        close /
        x["ma_9"] - 1
    )

    x["ma20_dist"] = (
        close /
        x["ma_20"] - 1
    )

    x["ma50_dist"] = (
        close /
        x["ma_50"] - 1
    )

    x["momentum"] = (
        close /
        close.shift(12) - 1
    )

    x["acceleration"] = (
        x["ret_3"] -
        x["ret_3"].shift(3)
    )

    vol_mean = (
        x["volume"]
        .rolling(30)
        .mean()
    )

    vol_std = (
        x["volume"]
        .rolling(30)
        .std()
    )

    x["volume_z"] = (
        (
            x["volume"] -
            vol_mean
        )
        /
        vol_std.replace(
            0,
            np.nan
        )
    )

    # --------------------------------------------------------
    # MARKET
    # --------------------------------------------------------

    if (
        market_df is not None
        and not market_df.empty
    ):

        m = (
            market_df[
                [
                    "timestamp",
                    "close"
                ]
            ]
            .rename(
                columns={
                    "close":
                        "market_close"
                }
            )
            .sort_values(
                "timestamp"
            )
        )

        x = pd.merge_asof(
            x.sort_values(
                "timestamp"
            ),
            m,
            on="timestamp",
            direction="backward"
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

        x[
            "market_ret_3"
        ] = 0.0

        x[
            "market_ret_12"
        ] = 0.0

        x[
            "relative_strength"
        ] = 0.0

    # --------------------------------------------------------
    # QQQ
    # --------------------------------------------------------

    if (
        qqq_df is not None
        and not qqq_df.empty
    ):

        q = (
            qqq_df[
                [
                    "timestamp",
                    "close"
                ]
            ]
            .rename(
                columns={
                    "close":
                        "qqq_close"
                }
            )
            .sort_values(
                "timestamp"
            )
        )

        x = pd.merge_asof(
            x.sort_values(
                "timestamp"
            ),
            q,
            on="timestamp",
            direction="backward"
        )

        x["qqq_ret_3"] = (
            x["qqq_close"]
            .pct_change(3)
        )

        x["qqq_ret_12"] = (
            x["qqq_close"]
            .pct_change(12)
        )

    else:

        x[
            "qqq_ret_3"
        ] = 0.0

        x[
            "qqq_ret_12"
        ] = 0.0

    # --------------------------------------------------------
    # TIME
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

    x["time_sin"] = (
        np.sin(
            2 *
            np.pi *
            mins /
            1440
        )
    )

    x["time_cos"] = (
        np.cos(
            2 *
            np.pi *
            mins /
            1440
        )
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
        axis=1
    ).max(
        axis=1
    )

    future_low = pd.concat(
        [
            x["low"].shift(-i)
            for i in range(
                1,
                HORIZON + 1
            )
        ],
        axis=1
    ).min(
        axis=1
    )

    up_hit = (
        future_high >=
        close +
        x["atr"] *
        ATR_TARGET
    )

    down_hit = (
        future_low <=
        close -
        x["atr"] *
        ATR_TARGET
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
# TRAIN
# ============================================================

def train_model(
    feature_df
):

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

    if (
        len(clean)
        <
        MIN_TRAIN_ROWS
    ):

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
        .nunique()
        <
        2
    ):

        return None

    if (
        test["target"]
        .nunique()
        <
        2
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
            train[
                FEATURE_COLUMNS
            ],
            train["target"]
        )

        models.append(
            model
        )

    probs = [

        model.predict_proba(
            test[
                FEATURE_COLUMNS
            ]
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

        "models":
            models,

        "auc":
            auc,

        "rows":
            len(clean),
    }


# ============================================================
# PREDICT
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
            subset=
            FEATURE_COLUMNS
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
        columns=FEATURE_COLUMNS
    )

    probs = [

        model.predict_proba(
            X
        )[0][1]

        for model in
        model_info["models"]
    ]

    p_up = float(
        np.mean(probs)
    )

    p_down = (
        1.0 -
        p_up
    )

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

        "signal":
            signal,

        "p_up":
            p_up,

        "p_down":
            p_down,

        "confidence":
            max(
                p_up,
                p_down
            ),

        "price":
            float(
                latest["close"]
            ),

        "atr":
            float(
                latest["atr"]
            ),

        "rsi":
            float(
                latest["rsi"]
            ),

        "momentum":
            float(
                latest["momentum"]
            ),

        "volatility":
            vol,

        "volume_z":
            float(
                latest["volume_z"]
            ),

        "relative_strength":
            float(
                latest[
                    "relative_strength"
                ]
            ),

        "auc":
            model_info["auc"],

        "rows":
            model_info["rows"],

        "regime":
            regime,
    }


# ============================================================
# MARKET CONFIRMATION
# ============================================================

def calculate_confirmation(
    pred_map
):

    call_votes = 0
    put_votes = 0

    total = 0

    details = []

    for symbol in (
        ["SPY", "QQQ"] +
        STOCKS
    ):

        p = pred_map.get(
            symbol
        )

        if not p:
            continue

        if p["signal"] == "CALL":

            call_votes += 1
            total += 1

            details.append(
                f"{symbol}:CALL"
            )

        elif p["signal"] == "PUT":

            put_votes += 1
            total += 1

            details.append(
                f"{symbol}:PUT"
            )

    if total == 0:

        return {
            "bias":
                "NEUTRAL",

            "strength":
                0.0,

            "calls":
                0,

            "puts":
                0,

            "total":
                0,

            "details":
                [],
        }

    if call_votes > put_votes:

        bias = "BULLISH"

        strength = (
            call_votes /
            total
        )

    elif put_votes > call_votes:

        bias = "BEARISH"

        strength = (
            put_votes /
            total
        )

    else:

        bias = "NEUTRAL"

        strength = (
            max(
                call_votes,
                put_votes
            )
            /
            total
        )

    return {

        "bias":
            bias,

        "strength":
            float(strength),

        "calls":
            call_votes,

        "puts":
            put_votes,

        "total":
            total,

        "details":
            details,
    }


# ============================================================
# SPX SIGNAL FILTER
# ============================================================

def validate_spx_signal(
    spx_pred,
    confirmation
):

    if not spx_pred:

        return False, (
            "NO_SPX_PREDICTION"
        )

    if spx_pred["signal"] not in (
        "CALL",
        "PUT"
    ):

        return False, "ML_WAIT"

    if (
        spx_pred["auc"]
        < MIN_AUC
    ):

        return False, (
            f"AUC_{spx_pred['auc']:.2f}"
        )

    if (
        spx_pred["confidence"]
        < MIN_PROBABILITY
    ):

        return False, (
            "LOW_CONFIDENCE"
        )

    if (
        abs(
            spx_pred["momentum"]
        )
        < 0.001
    ):

        return False, (
            "WEAK_MOMENTUM"
        )

    desired_bias = (
        "BULLISH"
        if spx_pred["signal"]
        == "CALL"
        else
        "BEARISH"
    )

    if (
        confirmation["bias"]
        != desired_bias
    ):

        return False, (
            "MARKET_NOT_ALIGNED"
        )

    if (
        confirmation["strength"]
        < 0.50
    ):

        return False, (
            "WEAK_CONFIRMATION"
        )

    return True, "OK"


# ============================================================
# OPTION CONTRACTS
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
            option_type,

        "limit":
            10000,
    }

    log(
        f"[OPTIONS REQUEST] "
        f"{underlying} "
        f"{option_type.upper()} "
        f"expiration={expiration}"
    )

    data = alpaca_get(
        OPTIONS_CONTRACTS_URL,
        params=params,
        timeout=30
    )

    if data is None:

        log(
            f"[OPTIONS] "
            f"{underlying}: "
            "❌ API RESPONSE FAILED"
        )

        return []

    contracts = (
        data.get(
            "option_contracts"
        )
        or data.get(
            "contracts"
        )
        or []
    )

    log(
        f"[OPTIONS] "
        f"{underlying} "
        f"{option_type.upper()}: "
        f"{len(contracts)} contracts"
    )

    return contracts


# ============================================================
# SPXW CONTRACT FILTER
# ============================================================

def filter_spxw_contracts(
    contracts
):

    result = []

    for c in contracts:

        symbol = str(
            c.get(
                "symbol",
                ""
            )
        ).upper()

        root = str(
            c.get(
                "root_symbol",
                ""
            )
        ).upper()

        # SPXW must be present
        if (
            "SPXW" not in symbol
            and
            "SPXW" not in root
        ):

            continue

        if not c.get(
            "tradable",
            True
        ):

            continue

        result.append(
            c
        )

    return result


# ============================================================
# OPTION QUOTES
# ============================================================

def get_option_quotes(
    symbols
):

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

            # Try indicative.
            # If the account has no valid
            # options feed, diagnostics
            # will expose the problem.
            "feed":
                "indicative",
        }

        data = alpaca_get(
            OPTIONS_LATEST_QUOTES_URL,
            params=params,
            timeout=30
        )

        if data is None:

            log(
                "[OPTIONS QUOTES] "
                "API FAILED"
            )

            continue

        quotes = data.get(
            "quotes",
            {}
        )

        for symbol, q in (
            quotes.items()
        ):

            try:

                bid = float(
                    q.get(
                        "bp",
                        0
                    )
                )

                ask = float(
                    q.get(
                        "ap",
                        0
                    )
                )

            except Exception:

                continue

            if (
                bid <= 0
                or ask <= 0
                or ask < bid
            ):

                continue

            mid = (
                bid + ask
            ) / 2

            if mid <= 0:

                continue

            spread_pct = (
                ask - bid
            ) / mid

            result[symbol] = {

                "bid":
                    bid,

                "ask":
                    ask,

                "mid":
                    mid,

                "spread_pct":
                    spread_pct,
            }

    log(
        "[OPTIONS QUOTES] "
        f"Valid={len(result)}"
    )

    return result


# ============================================================
# OPTION SCORE
# ============================================================

def score_spx_option(
    pred,
    confirmation,
    quote,
    distance
):

    score = 0.0

    # --------------------------------------------------------
    # ML confidence: 35
    # --------------------------------------------------------

    confidence = (
        pred["confidence"]
    )

    score += min(
        35,
        max(
            0,
            (
                confidence -
                0.50
            ) * 100
        ) * 0.875
    )

    # --------------------------------------------------------
    # AUC: 15
    # --------------------------------------------------------

    score += min(
        15,
        max(
            0,
            (
                pred["auc"] -
                0.50
            ) * 100
        ) * 0.75
    )

    # --------------------------------------------------------
    # Market confirmation: 20
    # --------------------------------------------------------

    score += min(
        20,
        confirmation["strength"]
        * 20
    )

    # --------------------------------------------------------
    # Spread: 15
    # --------------------------------------------------------

    spread = (
        quote["spread_pct"]
    )

    if spread <= 0.04:

        score += 15

    elif spread <= 0.06:

        score += 13

    elif spread <= 0.08:

        score += 10

    elif spread <= 0.10:

        score += 6

    elif spread <= 0.12:

        score += 2

    else:

        return 0

    # --------------------------------------------------------
    # Distance: 10
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Momentum: 5
    # --------------------------------------------------------

    momentum = abs(
        pred["momentum"]
    )

    if momentum >= 0.004:

        score += 5

    elif momentum >= 0.002:

        score += 3

    else:

        score += 1

    return int(
        min(
            100,
            max(
                0,
                round(score)
            )
        )
    )


# ============================================================
# CHOOSE SPXW 0DTE CONTRACT
# ============================================================

def choose_spx_contract(
    signal,
    spx_price,
    pred,
    confirmation
):

    expiration = today_ny()

    option_type = (
        "call"
        if signal == "CALL"
        else
        "put"
    )

    contracts = get_option_contracts(
        "SPX",
        option_type,
        expiration
    )

    if not contracts:

        return None, (
            "NO_0DTE_CONTRACTS"
        )

    contracts = (
        filter_spxw_contracts(
            contracts
        )
    )

    if not contracts:

        log(
            "[SPXW] "
            "API returned contracts, "
            "but no SPXW contracts found."
        )

        return None, (
            "NO_SPXW_CONTRACTS"
        )

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

        distance = abs(
            strike -
            spx_price
        )

        if (
            distance
            >
            MAX_SPX_STRIKE_DISTANCE
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
            "[SPXW] "
            "No strike within "
            f"${MAX_SPX_STRIKE_DISTANCE}"
        )

        return None, (
            "NO_SPXW_STRIKE"
        )

    candidates.sort(
        key=lambda x: x[0]
    )

    # More than one candidate
    # so we do not blindly take
    # the first strike.
    selected = [
        c
        for _, c
        in candidates[:30]
    ]

    symbols = [
        c.get(
            "symbol"
        )
        for c in selected
        if c.get(
            "symbol"
        )
    ]

    quotes = get_option_quotes(
        symbols
    )

    if not quotes:

        return None, (
            "NO_OPTION_QUOTES"
        )

    best = None
    best_score = -1

    rejected_spread = 0
    rejected_premium = 0

    for c in selected:

        symbol = c.get(
            "symbol"
        )

        quote = quotes.get(
            symbol
        )

        if not quote:

            continue

        premium = quote[
            "mid"
        ]

        spread = quote[
            "spread_pct"
        ]

        if (
            premium
            <
            MIN_OPTION_PREMIUM
        ):

            rejected_premium += 1
            continue

        if (
            spread
            >
            MAX_SPREAD_PERCENT
        ):

            rejected_spread += 1
            continue

        try:

            strike = float(
                c.get(
                    "strike_price"
                )
            )

        except Exception:

            continue

        distance = abs(
            strike -
            spx_price
        )

        score = score_spx_option(
            pred,
            confirmation,
            quote,
            distance
        )

        if score > best_score:

            best_score = score

            best = {

                "contract":
                    c,

                "quote":
                    quote,

                "score":
                    score,

                "distance":
                    distance,
            }

    if best is None:

        log(
            "[SPXW REJECT] "
            "No liquid contract | "
            f"premium_rejects="
            f"{rejected_premium} | "
            f"spread_rejects="
            f"{rejected_spread}"
        )

        return None, (
            "NO_LIQUID_SPXW"
        )

    if (
        best["score"]
        <
        MIN_SPX_SCORE
    ):

        log(
            "[SPXW REJECT] "
            f"Score={best['score']}/100 "
            f"< {MIN_SPX_SCORE}"
        )

        return None, (
            f"SCORE_{best['score']}"
        )

    return best, "OK"


# ============================================================
# BUILD SPX OPPORTUNITY
# ============================================================

def build_spx_opportunity(
    pred,
    confirmation
):

    if not pred:

        return None, (
            "NO_SPX_PREDICTION"
        )

    valid, reason = (
        validate_spx_signal(
            pred,
            confirmation
        )
    )

    if not valid:

        return None, reason

    result, reason = (
        choose_spx_contract(
            pred["signal"],
            pred["price"],
            pred,
            confirmation
        )
    )

    if not result:

        return None, reason

    c = result[
        "contract"
    ]

    q = result[
        "quote"
    ]

    entry = q["mid"]

    return {

        "symbol":
            "SPX",

        "signal":
            pred["signal"],

        "confidence":
            pred["confidence"],

        "auc":
            pred["auc"],

        "spx_price":
            pred["price"],

        "score":
            result["score"],

        "contract_symbol":
            c.get(
                "symbol",
                "UNKNOWN"
            ),

        "strike":
            float(
                c.get(
                    "strike_price",
                    0
                )
            ),

        "expiration":
            c.get(
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
            q["bid"],

        "ask":
            q["ask"],

        "spread_pct":
            q["spread_pct"],

        "confirmation":
            confirmation,

        "regime":
            pred["regime"],

        "momentum":
            pred["momentum"],
    }, "OK"


# ============================================================
# DUPLICATE
# ============================================================

def is_duplicate(
    opportunity
):

    key = (
        opportunity[
            "contract_symbol"
        ]
        +
        "_"
        +
        opportunity[
            "signal"
        ]
    )

    last = (
        STATE[
            "last_sent"
        ].get(
            key
        )
    )

    if last is None:

        return False

    elapsed = (
        time.time() -
        last
    ) / 60

    return (
        elapsed
        <
        SIGNAL_COOLDOWN_MINUTES
    )


def mark_sent(
    opportunity
):

    key = (
        opportunity[
            "contract_symbol"
        ]
        +
        "_"
        +
        opportunity[
            "signal"
        ]
    )

    STATE[
        "last_sent"
    ][key] = time.time()


# ============================================================
# TELEGRAM FORMAT
# ============================================================

def format_spx_alert(
    opportunity
):

    signal = (
        opportunity[
            "signal"
        ]
    )

    emoji = (
        "🟢"
        if signal == "CALL"
        else
        "🔴"
    )

    confirmation = (
        opportunity[
            "confirmation"
        ]
    )

    return (

        f"🚨 <b>SPX 0DTE "
        f"{emoji} {signal}</b>\n"

        f"━━━━━━━━━━━━━━━━━━\n"

        f"🔥 القوة: "
        f"<b>{opportunity['score']}/100</b>\n"

        f"🧠 ML: "
        f"<b>{opportunity['confidence']*100:.1f}%</b>\n"

        f"📊 AUC: "
        f"<b>{opportunity['auc']:.2f}</b>\n"

        f"📈 SPX: "
        f"<b>{opportunity['spx_price']:.2f}</b>\n"

        f"🎯 Strike: "
        f"<b>{opportunity['strike']:.2f}</b>\n"

        f"📜 العقد:\n"
        f"<code>"
        f"{opportunity['contract_symbol']}"
        f"</code>\n"

        f"💰 دخول: "
        f"<b>${opportunity['entry']:.2f}</b>\n"

        f"🎯 هدف +40%: "
        f"<b>${opportunity['target']:.2f}</b>\n"

        f"🛑 وقف -30%: "
        f"<b>${opportunity['stop']:.2f}</b>\n"

        f"↔️ السبريد: "
        f"{opportunity['spread_pct']*100:.1f}%\n"

        f"🌡️ النظام: "
        f"{opportunity['regime']}\n"

        f"📊 تأكيد السوق: "
        f"<b>{confirmation['bias']}</b>\n"

        f"🟢 CALL: "
        f"{confirmation['calls']} | "
        f"🔴 PUT: "
        f"{confirmation['puts']}\n"

        f"📈 قوة التأكيد: "
        f"{confirmation['strength']*100:.1f}%\n"

        f"⚠️ <i>Recommendation only — "
        f"لا يوجد تنفيذ أوامر</i>"
    )


# ============================================================
# PRINT PREDICTION
# ============================================================

def print_prediction(
    symbol,
    pred
):

    if not pred:

        log(
            f"[ML] {symbol}: "
            "NO PREDICTION"
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

    section(
        f"SPX 0DTE AI ADVISOR "
        f"{VERSION}"
    )

    log(
        "🚀 Starting..."
    )

    log(
        "🎯 PRIMARY = SPX/SPXW 0DTE"
    )

    log(
        "🧠 STOCKS = MARKET CONFIRMATION"
    )

    log(
        "🛡️ Recommendation Only"
    )

    # --------------------------------------------------------
    # KEYS
    # --------------------------------------------------------

    if not ALPACA_API_KEY:

        log(
            "[FATAL] "
            "Missing ALPACA_API_KEY"
        )

        return

    if not ALPACA_SECRET_KEY:

        log(
            "[FATAL] "
            "Missing ALPACA_SECRET_KEY"
        )

        return

    log(
        f"[CONFIG] "
        f"Feed={DATA_FEED} | "
        f"Timeframe={TIMEFRAME} | "
        f"History={HISTORY_DAYS}d"
    )

    log(
        f"[CONFIG] "
        f"MinProb={MIN_PROBABILITY} | "
        f"MinAUC={MIN_AUC} | "
        f"MinScore={MIN_SPX_SCORE}"
    )

    log(
        f"[CONFIG] "
        f"MaxSpread="
        f"{MAX_SPREAD_PERCENT*100:.1f}%"
    )

    # --------------------------------------------------------
    # START MESSAGE
    # --------------------------------------------------------

    telegram_send(
        "🤖 <b>SPX 0DTE AI ADVISOR "
        f"{VERSION}</b>\n\n"
        "🚀 بدأ التشغيل.\n"
        "🎯 التركيز: SPXW 0DTE\n"
        "🧠 الأسهم: تأكيد للسوق\n"
        "🛡️ لا يوجد تنفيذ أوامر\n"
        "🔎 Patient Mode: ON"
    )

    cycle = 0

    while True:

        cycle += 1

        section(
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
                "[MARKET] ✓ OPEN"
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
                    "SPY unavailable"
                )

                time.sleep(
                    POLL_SECONDS
                )

                continue

            # ------------------------------------------------
            # TRAIN MODELS
            # ------------------------------------------------

            models = {}

            # ------------------------------------------------
            # SPX MODEL
            # ------------------------------------------------
            #
            # IMPORTANT:
            # We DO NOT call SPY×10 "real SPX".
            #
            # Until a real SPX data source is available,
            # SPY is used only as an INDEX-MARKET PROXY.
            #
            # This model is therefore labeled honestly.
            # ------------------------------------------------

            spx_proxy = spy.copy()

            for col in [
                "open",
                "high",
                "low",
                "close"
            ]:

                spx_proxy[
                    col
                ] *= 10

            spx_features = make_features(
                spx_proxy,
                market_df=spy,
                qqq_df=qqq
            )

            spx_model = train_model(
                spx_features
            )

            if spx_model:

                models["SPX"] = {

                    "features":
                        spx_features,

                    "model":
                        spx_model,
                }

                log(
                    "[MODEL] SPX "
                    "proxy model: ✓ "
                    f"AUC={spx_model['auc']:.2f}"
                )

            else:

                log(
                    "[MODEL] SPX "
                    "proxy: ❌ failed"
                )

            # ------------------------------------------------
            # STOCK MODELS
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
                        "NO DATA"
                    )

                    continue

                features = make_features(
                    df,
                    market_df=spy,
                    qqq_df=qqq
                )

                model = train_model(
                    features
                )

                if model:

                    models[symbol] = {

                        "features":
                            features,

                        "model":
                            model,
                    }

                    log(
                        f"[MODEL] "
                        f"{symbol}: "
                        f"AUC={model['auc']:.2f} "
                        f"rows={model['rows']}"
                    )

            if "SPX" not in models:

                log(
                    "[FATAL] "
                    "SPX model unavailable"
                )

                time.sleep(
                    POLL_SECONDS
                )

                continue

            # ------------------------------------------------
            # PREDICTIONS
            # ------------------------------------------------

            section(
                "ML PREDICTIONS"
            )

            pred_map = {}

            for symbol, info in (
                models.items()
            ):

                pred = predict(
                    info["model"],
                    info["features"]
                )

                if pred:

                    pred_map[
                        symbol
                    ] = pred

                    print_prediction(
                        symbol,
                        pred
                    )

            spx_pred = pred_map.get(
                "SPX"
            )

            if not spx_pred:

                log(
                    "[SPX] "
                    "NO PREDICTION"
                )

                time.sleep(
                    POLL_SECONDS
                )

                continue

            # ------------------------------------------------
            # MARKET CONFIRMATION
            # ------------------------------------------------

            section(
                "MARKET CONFIRMATION"
            )

            confirmation = (
                calculate_confirmation(
                    pred_map
                )
            )

            log(
                f"[CONFIRMATION] "
                f"{confirmation['bias']} | "
                f"strength="
                f"{confirmation['strength']*100:.1f}% | "
                f"CALL="
                f"{confirmation['calls']} | "
                f"PUT="
                f"{confirmation['puts']}"
            )

            log(
                "[CONFIRMATION DETAILS] "
                +
                " | ".join(
                    confirmation[
                        "details"
                    ]
                )
            )

            # ------------------------------------------------
            # SPX VALIDATION
            # ------------------------------------------------

            section(
                "SPX SIGNAL VALIDATION"
            )

            valid, reason = (
                validate_spx_signal(
                    spx_pred,
                    confirmation
                )
            )

            if not valid:

                log(
                    f"[SPX] "
                    f"❌ WAIT | "
                    f"reason={reason}"
                )

                log(
                    "⏳ Patient Mode: "
                    "لا توجد فرصة مؤكدة."
                )

                time.sleep(
                    POLL_SECONDS
                )

                continue

            log(
                f"[SPX] "
                f"✓ {spx_pred['signal']} "
                f"passed signal filters"
            )

            # ------------------------------------------------
            # OPTION ANALYSIS
            # ------------------------------------------------

            section(
                "SPXW 0DTE OPTION ANALYSIS"
            )

            opportunity, reason = (
                build_spx_opportunity(
                    spx_pred,
                    confirmation
                )
            )

            if not opportunity:

                log(
                    f"[SPXW] "
                    f"❌ WAIT | "
                    f"reason={reason}"
                )

                log(
                    "⏳ No high-quality "
                    "SPX 0DTE contract."
                )

                time.sleep(
                    POLL_SECONDS
                )

                continue

            # ------------------------------------------------
            # DUPLICATE
            # ------------------------------------------------

            if is_duplicate(
                opportunity
            ):

                log(
                    "[COOLDOWN] "
                    f"{opportunity['contract_symbol']} "
                    "already sent recently"
                )

                time.sleep(
                    POLL_SECONDS
                )

                continue

            # ------------------------------------------------
            # FINAL
            # ------------------------------------------------

            log(
                "🔥 "
                "HIGH QUALITY SPX 0DTE "
                "OPPORTUNITY FOUND"
            )

            log(
                f"[FINAL] "
                f"{opportunity['signal']} | "
                f"score="
                f"{opportunity['score']}/100 | "
                f"confidence="
                f"{opportunity['confidence']*100:.1f}% | "
                f"strike="
                f"{opportunity['strike']} | "
                f"entry="
                f"${opportunity['entry']:.2f}"
            )

            # ------------------------------------------------
            # TELEGRAM
            # ------------------------------------------------

            message = (
                format_spx_alert(
                    opportunity
                )
            )

            sent = telegram_send(
                message
            )

            if sent:

                mark_sent(
                    opportunity
                )

                log(
                    "[TELEGRAM] "
                    "✓ SPX ALERT SENT"
                )

            else:

                log(
                    "[TELEGRAM] "
                    "❌ SEND FAILED"
                )

        except Exception as e:

            log(
                "[MAIN ERROR] "
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