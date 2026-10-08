# ============================================================
# SPX & 10 STOCKS 0DTE AI ADVISOR / SCANNER v16.0
# ============================================================
# Recommendation Only - NO ORDER EXECUTION
# ============================================================

import os
import time
import warnings
from datetime import datetime, timedelta, timezone, time as dt_time
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

VERSION = "v16.0-AUTH-CACHED-FIRSTHIT"

NY = ZoneInfo("America/New_York")

SPY_SYMBOL = "SPY"
QQQ_SYMBOL = "QQQ"

TIMEFRAME = "5Min"

HISTORY_DAYS = int(os.getenv("HISTORY_DAYS", "60"))
MIN_TRAIN_ROWS = int(os.getenv("MIN_TRAIN_ROWS", "150"))

HORIZON = int(os.getenv("HORIZON_BARS", "6"))
ATR_TARGET = float(os.getenv("ATR_TARGET", "0.50"))

MIN_PROBABILITY = float(
    os.getenv("MIN_PROBABILITY", "0.62")
)

MIN_OPTION_SCORE = float(
    os.getenv("MIN_OPTION_SCORE", "75")
)

MAX_SPREAD_PERCENT = float(
    os.getenv("MAX_SPREAD_PERCENT", "0.15")
)

MIN_OPTION_PREMIUM = float(
    os.getenv("MIN_OPTION_PREMIUM", "0.20")
)

MAX_SPX_STRIKE_DISTANCE = float(
    os.getenv("MAX_SPX_STRIKE_DISTANCE", "40")
)

MAX_STOCK_STRIKE_DISTANCE_PCT = float(
    os.getenv("MAX_STOCK_STRIKE_DISTANCE_PCT", "0.04")
)

SIGNAL_COOLDOWN_MINUTES = int(
    os.getenv("SIGNAL_COOLDOWN_MINUTES", "20")
)

POLL_SECONDS = int(
    os.getenv("POLL_SECONDS", "30")
)

FRESH_BAR_REFRESH_SECONDS = int(
    os.getenv("FRESH_BAR_REFRESH_SECONDS", "60")
)

MODEL_REFRESH_SECONDS = int(
    os.getenv("MODEL_REFRESH_SECONDS", "3600")
)

MAX_TELEGRAM_OPPORTUNITIES = int(
    os.getenv("MAX_TELEGRAM_OPPORTUNITIES", "3")
)

NO_TRADE_FIRST_MINUTES = int(
    os.getenv("NO_TRADE_FIRST_MINUTES", "10")
)

NO_TRADE_LAST_MINUTES = int(
    os.getenv("NO_TRADE_LAST_MINUTES", "30")
)

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

ALL_SYMBOLS = [
    SPY_SYMBOL,
    QQQ_SYMBOL,
] + STOCKS

# ============================================================
# ENV
# ============================================================

ALPACA_API_KEY = (
    os.getenv("ALPACA_API_KEY", "").strip()
    or os.getenv("APCA_API_KEY_ID", "").strip()
)

ALPACA_SECRET_KEY = (
    os.getenv("ALPACA_SECRET_KEY", "").strip()
    or os.getenv("APCA_API_SECRET_KEY", "").strip()
)

TELEGRAM_BOT_TOKEN = os.getenv(
    "TELEGRAM_BOT_TOKEN", ""
).strip()

TELEGRAM_CHAT_ID = os.getenv(
    "TELEGRAM_CHAT_ID", ""
).strip()

ALPACA_TRADING_URL = os.getenv(
    "ALPACA_TRADING_URL",
    "https://api.alpaca.markets"
).strip().rstrip("/")

ALPACA_DATA_URL = (
    "https://data.alpaca.markets"
)

DATA_FEED = os.getenv(
    "ALPACA_DATA_FEED",
    "iex"
).strip()

OPTIONS_CONTRACTS_URL = (
    f"{ALPACA_TRADING_URL}/v2/options/contracts"
)

OPTIONS_LATEST_QUOTES_URL = (
    f"{ALPACA_DATA_URL}/v1beta1/options/quotes/latest"
)

# ============================================================
# GLOBAL STATE
# ============================================================

session = requests.Session()

BARS_CACHE = {}
MODEL_CACHE = {}

LAST_TRAIN_TIME = 0.0
LAST_BARS_TIME = 0.0

LAST_SIGNAL = {}

STARTUP_SENT = False

# ============================================================
# LOG
# ============================================================

def log(message):
    now = datetime.now(NY).strftime(
        "%Y-%m-%d %H:%M:%S"
    )
    print(
        f"[{now}] {message}",
        flush=True
    )

# ============================================================
# ALPACA HEADERS
# ============================================================

def alpaca_headers():

    return {
        "APCA-API-KEY-ID":
            ALPACA_API_KEY.strip(),

        "APCA-API-SECRET-KEY":
            ALPACA_SECRET_KEY.strip(),

        "Accept":
            "application/json",
    }

# ============================================================
# ALPACA AUTH TEST
# ============================================================

def test_alpaca_auth():

    if not ALPACA_API_KEY:
        log("❌ ALPACA_API_KEY missing")
        return False

    if not ALPACA_SECRET_KEY:
        log("❌ ALPACA_SECRET_KEY missing")
        return False

    url = (
        f"{ALPACA_TRADING_URL}/v2/account"
    )

    log(
        "🔐 Testing Alpaca authentication..."
    )

    log(
        f"URL: {ALPACA_TRADING_URL}"
    )

    log(
        f"API key length: "
        f"{len(ALPACA_API_KEY)}"
    )

    log(
        f"Secret length: "
        f"{len(ALPACA_SECRET_KEY)}"
    )

    try:

        response = session.get(
            url,
            headers=alpaca_headers(),
            timeout=15
        )

    except Exception as e:

        log(
            f"❌ Alpaca connection error: {e}"
        )

        return False

    if response.status_code == 200:

        log(
            "✅ ALPACA AUTHENTICATION OK"
        )

        return True

    log(
        f"❌ ALPACA AUTH FAILED "
        f"HTTP {response.status_code}"
    )

    log(
        response.text[:500]
    )

    if response.status_code == 401:

        log(
            "⚠️ 401 = غالبًا المفتاح والسر "
            "لا يطابقان بيئة Live/Paper."
        )

        log(
            "إذا كانت المفاتيح Paper استخدم:"
        )

        log(
            "https://paper-api.alpaca.markets"
        )

        log(
            "إذا كانت Live استخدم:"
        )

        log(
            "https://api.alpaca.markets"
        )

    return False

# ============================================================
# ALPACA GET
# ============================================================

def alpaca_get(
    url,
    params=None,
    timeout=30
):

    try:

        response = session.get(
            url,
            headers=alpaca_headers(),
            params=params,
            timeout=timeout
        )

    except Exception as e:

        log(
            f"❌ Alpaca request error: {e}"
        )

        return None

    if response.status_code != 200:

        log(
            f"❌ Alpaca HTTP "
            f"{response.status_code}"
        )

        log(
            f"URL: {url}"
        )

        log(
            response.text[:500]
        )

        return None

    try:

        return response.json()

    except Exception as e:

        log(
            f"❌ Alpaca JSON error: {e}"
        )

        return None

# ============================================================
# TELEGRAM
# ============================================================

def telegram_send(message):

    if (
        not TELEGRAM_BOT_TOKEN
        or not TELEGRAM_CHAT_ID
    ):

        log(
            "⚠️ Telegram credentials missing"
        )

        return False

    url = (
        "https://api.telegram.org/bot"
        f"{TELEGRAM_BOT_TOKEN}/sendMessage"
    )

    payload = {
        "chat_id":
            TELEGRAM_CHAT_ID,

        "text":
            message,

        "disable_web_page_preview":
            True
    }

    try:

        response = session.post(
            url,
            json=payload,
            timeout=15
        )

        if response.status_code == 200:
            return True

        log(
            f"❌ Telegram HTTP "
            f"{response.status_code}: "
            f"{response.text[:300]}"
        )

    except Exception as e:

        log(
            f"❌ Telegram error: {e}"
        )

    return False

# ============================================================
# TIME
# ============================================================

def now_ny():
    return datetime.now(NY)


def today_date_ny():
    return now_ny().date().isoformat()


def regular_market_open():

    now = now_ny()

    if now.weekday() >= 5:
        return False

    t = now.time()

    return (
        dt_time(9, 30)
        <= t
        <= dt_time(16, 0)
    )


def trading_window_allowed():

    now = now_ny()

    if not regular_market_open():
        return False

    minutes_from_open = (
        now.hour * 60
        + now.minute
        - 570
    )

    minutes_to_close = (
        960
        - (
            now.hour * 60
            + now.minute
        )
    )

    if (
        minutes_from_open
        < NO_TRADE_FIRST_MINUTES
    ):
        return False

    if (
        minutes_to_close
        <= NO_TRADE_LAST_MINUTES
    ):
        return False

    return True

# ============================================================
# DATA
# ============================================================

def fetch_all_bars(
    symbols,
    start=None,
    end=None
):

    if start is None:

        start = (
            datetime.now(timezone.utc)
            - timedelta(
                days=HISTORY_DAYS
            )
        ).isoformat()

    if end is None:

        end = (
            datetime.now(timezone.utc)
        ).isoformat()

    result = {}

    for symbol in symbols:

        rows = []
        page_token = None

        while True:

            params = {
                "symbols":
                    symbol,

                "timeframe":
                    TIMEFRAME,

                "start":
                    start,

                "end":
                    end,

                "limit":
                    10000,

                "feed":
                    DATA_FEED,

                "adjustment":
                    "raw",

                "sort":
                    "asc"
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
                break

            bars = data.get(
                "bars",
                {}
            )

            rows.extend(
                bars.get(
                    symbol,
                    []
                )
            )

            page_token = data.get(
                "next_page_token"
            )

            if not page_token:
                break

            if len(rows) > 250000:
                break

        if not rows:

            log(
                f"⚠️ {symbol}: "
                "no bars"
            )

            result[symbol] = (
                pd.DataFrame()
            )

            continue

        df = pd.DataFrame(rows)

        df["timestamp"] = (
            pd.to_datetime(
                df["t"],
                utc=True
            ).dt.tz_convert(NY)
        )

        df = df.rename(
            columns={
                "o": "open",
                "h": "high",
                "l": "low",
                "c": "close",
                "v": "volume"
            }
        )

        columns = [
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume"
        ]

        df = df[columns]

        for col in columns[1:]:

            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            )

        df = (
            df
            .dropna()
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

        result[symbol] = df

        log(
            f"📊 {symbol}: "
            f"{len(df):,} bars"
        )

    return result

# ============================================================
# HISTORICAL CACHE
# ============================================================

def refresh_training_data(
    force=False
):

    global LAST_TRAIN_TIME
    global BARS_CACHE

    if (
        not force
        and BARS_CACHE
        and (
            time.time()
            - LAST_TRAIN_TIME
            < MODEL_REFRESH_SECONDS
        )
    ):

        return BARS_CACHE

    log(
        "📚 Downloading historical "
        f"{HISTORY_DAYS}-day data..."
    )

    data = fetch_all_bars(
        ALL_SYMBOLS
    )

    if data:

        BARS_CACHE = data

        LAST_TRAIN_TIME = (
            time.time()
        )

        log(
            "✅ Historical data cached"
        )

    return BARS_CACHE

# ============================================================
# RECENT DATA
# ============================================================

def refresh_recent_data(
    force=False
):

    global LAST_BARS_TIME
    global BARS_CACHE

    if (
        not force
        and BARS_CACHE
        and (
            time.time()
            - LAST_BARS_TIME
            < FRESH_BAR_REFRESH_SECONDS
        )
    ):

        return BARS_CACHE

    end = datetime.now(
        timezone.utc
    )

    start = (
        end
        - timedelta(days=3)
    )

    log(
        "🔄 Updating recent bars..."
    )

    recent = fetch_all_bars(
        ALL_SYMBOLS,
        start=start.isoformat(),
        end=end.isoformat()
    )

    if not recent:
        return BARS_CACHE

    for symbol, new_df in recent.items():

        if new_df.empty:
            continue

        old_df = BARS_CACHE.get(
            symbol,
            pd.DataFrame()
        )

        if old_df.empty:

            merged = new_df

        else:

            merged = pd.concat(
                [
                    old_df,
                    new_df
                ],
                ignore_index=True
            )

        merged = (
            merged
            .drop_duplicates(
                "timestamp"
            )
            .sort_values(
                "timestamp"
            )
            .tail(5000)
            .reset_index(
                drop=True
            )
        )

        BARS_CACHE[symbol] = merged

    LAST_BARS_TIME = time.time()

    return BARS_CACHE

# ============================================================
# INDICATORS
# ============================================================

def rsi(
    series,
    period=14
):

    delta = series.diff()

    gain = delta.clip(
        lower=0
    )

    loss = -delta.clip(
        upper=0
    )

    avg_gain = gain.rolling(
        period
    ).mean()

    avg_loss = loss.rolling(
        period
    ).mean()

    rs = (
        avg_gain
        / avg_loss.replace(
            0,
            np.nan
        )
    )

    return (
        100
        - (
            100
            / (1 + rs)
        )
    )

# ============================================================
# FEATURES
# ============================================================

FEATURE_COLUMNS = [
    "ret1",
    "ret3",
    "ret6",
    "ret12",
    "rsi",
    "atr_pct",
    "range_pct",
    "volatility",
    "ma9_dist",
    "ma20_dist",
    "ma50_dist",
    "momentum12",
    "acceleration",
    "volume_z",
    "spy_ret3",
    "spy_ret6",
    "qqq_ret3",
    "qqq_ret6",
    "relative_spy",
    "relative_qqq",
    "time_sin",
    "time_cos"
]

# ============================================================
# FEATURE ENGINEERING
# ============================================================

def add_features(
    df,
    spy_df=None,
    qqq_df=None
):

    d = df.copy()

    close = d["close"]
    high = d["high"]
    low = d["low"]

    d["ret1"] = (
        close.pct_change(1)
    )

    d["ret3"] = (
        close.pct_change(3)
    )

    d["ret6"] = (
        close.pct_change(6)
    )

    d["ret12"] = (
        close.pct_change(12)
    )

    d["rsi"] = rsi(
        close,
        14
    )

    previous = close.shift(1)

    tr = pd.concat(
        [
            high - low,
            (
                high - previous
            ).abs(),
            (
                low - previous
            ).abs()
        ],
        axis=1
    ).max(axis=1)

    atr = tr.rolling(
        14
    ).mean()

    d["atr_pct"] = (
        atr / close
    )

    d["range_pct"] = (
        (high - low)
        / close
    )

    d["volatility"] = (
        d["ret1"]
        .rolling(20)
        .std()
    )

    d["ma9_dist"] = (
        close
        / close.rolling(9)
        - 1
    )

    d["ma20_dist"] = (
        close
        / close.rolling(20)
        - 1
    )

    d["ma50_dist"] = (
        close
        / close.rolling(50)
        - 1
    )

    d["momentum12"] = (
        close
        / close.shift(12)
        - 1
    )

    d["acceleration"] = (
        d["ret3"]
        - d["ret3"].shift(3)
    )

    volume_mean = (
        d["volume"]
        .rolling(30)
        .mean()
    )

    volume_std = (
        d["volume"]
        .rolling(30)
        .std()
    )

    d["volume_z"] = (
        (
            d["volume"]
            - volume_mean
        )
        / volume_std.replace(
            0,
            np.nan
        )
    )

    minutes = (
        d["timestamp"].dt.hour
        * 60
        + d["timestamp"].dt.minute
    )

    session_minutes = (
        minutes - 570
    )

    d["time_sin"] = np.sin(
        2 * np.pi
        * session_minutes
        / 390
    )

    d["time_cos"] = np.cos(
        2 * np.pi
        * session_minutes
        / 390
    )

    if (
        spy_df is not None
        and not spy_df.empty
    ):

        spy = spy_df[
            ["timestamp", "close"]
        ].copy()

        spy["spy_ret3"] = (
            spy["close"]
            .pct_change(3)
        )

        spy["spy_ret6"] = (
            spy["close"]
            .pct_change(6)
        )

        d = pd.merge_asof(
            d.sort_values(
                "timestamp"
            ),
            spy[
                [
                    "timestamp",
                    "spy_ret3",
                    "spy_ret6"
                ]
            ].sort_values(
                "timestamp"
            ),
            on="timestamp",
            direction="backward"
        )

    else:

        d["spy_ret3"] = 0
        d["spy_ret6"] = 0

    if (
        qqq_df is not None
        and not qqq_df.empty
    ):

        qqq = qqq_df[
            ["timestamp", "close"]
        ].copy()

        qqq["qqq_ret3"] = (
            qqq["close"]
            .pct_change(3)
        )

        qqq["qqq_ret6"] = (
            qqq["close"]
            .pct_change(6)
        )

        d = pd.merge_asof(
            d.sort_values(
                "timestamp"
            ),
            qqq[
                [
                    "timestamp",
                    "qqq_ret3",
                    "qqq_ret6"
                ]
            ].sort_values(
                "timestamp"
            ),
            on="timestamp",
            direction="backward"
        )

    else:

        d["qqq_ret3"] = 0
        d["qqq_ret6"] = 0

    d["relative_spy"] = (
        d["ret3"]
        - d["spy_ret3"]
    )

    d["relative_qqq"] = (
        d["ret3"]
        - d["qqq_ret3"]
    )

    return d

# ============================================================
# FIRST-HIT TARGET
# ============================================================

def make_first_hit_target(df):

    close = (
        df["close"]
        .to_numpy()
    )

    high = (
        df["high"]
        .to_numpy()
    )

    low = (
        df["low"]
        .to_numpy()
    )

    previous = (
        df["close"]
        .shift(1)
    )

    tr = pd.concat(
        [
            df["high"] - df["low"],
            (
                df["high"]
                - previous
            ).abs(),
            (
                df["low"]
                - previous
            ).abs()
        ],
        axis=1
    ).max(axis=1)

    atr = (
        tr
        .rolling(14)
        .mean()
        .to_numpy()
    )

    target = np.full(
        len(df),
        np.nan
    )

    for i in range(
        len(df) - HORIZON
    ):

        if (
            not np.isfinite(
                atr[i]
            )
            or atr[i] <= 0
        ):
            continue

        up_level = (
            close[i]
            + atr[i]
            * ATR_TARGET
        )

        down_level = (
            close[i]
            - atr[i]
            * ATR_TARGET
        )

        result = None

        for step in range(
            1,
            HORIZON + 1
        ):

            k = i + step

            up_hit = (
                high[k]
                >= up_level
            )

            down_hit = (
                low[k]
                <= down_level
            )

            if up_hit and down_hit:

                result = None
                break

            if up_hit:

                result = 1.0
                break

            if down_hit:

                result = 0.0
                break

        if result is not None:
            target[i] = result

    return pd.Series(
        target,
        index=df.index,
        name="target"
    )

# ============================================================
# TRAINING FRAME
# ============================================================

def prepare_training_frame(
    symbol,
    data
):

    if symbol not in data:
        return None

    df = data[symbol]

    if df.empty:
        return None

    # SPX model is actually based on SPY proxy.
    if symbol == "SPX":
        return None

    feat = add_features(
        df,
        spy_df=data.get(
            SPY_SYMBOL
        ),
        qqq_df=data.get(
            QQQ_SYMBOL
        )
    )

    feat["target"] = (
        make_first_hit_target(
            feat
        )
    )

    feat = feat.dropna(
        subset=
            FEATURE_COLUMNS
            + ["target"]
    )

    if len(feat) < MIN_TRAIN_ROWS:
        return None

    values = feat[
        FEATURE_COLUMNS
    ].astype(float)

    mask = np.isfinite(
        values
    ).all(axis=1)

    feat = feat.loc[
        mask
    ].copy()

    feat["target"] = (
        feat["target"]
        .astype(int)
    )

    if (
        len(feat)
        < MIN_TRAIN_ROWS
    ):
        return None

    if (
        feat["target"]
        .nunique()
        < 2
    ):
        return None

    return feat

# ============================================================
# TRAIN MODEL
# ============================================================

def train_model(
    symbol,
    data
):

    frame = prepare_training_frame(
        symbol,
        data
    )

    if frame is None:

        log(
            f"⚠️ {symbol}: "
            "not enough usable data"
        )

        return None

    X = frame[
        FEATURE_COLUMNS
    ]

    y = frame["target"]

    n = len(frame)

    train_end = int(
        n * 0.60
    )

    valid_end = int(
        n * 0.80
    )

    if (
        train_end < 50
        or valid_end <= train_end
    ):
        return None

    X_train = X.iloc[
        :train_end
    ]

    y_train = y.iloc[
        :train_end
    ]

    X_test = X.iloc[
        valid_end:
    ]

    y_test = y.iloc[
        valid_end:
    ]

    if (
        y_train.nunique()
        < 2
        or y_test.nunique()
        < 2
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
                learning_rate=0.055,
                max_iter=180,
                max_leaf_nodes=15,
                min_samples_leaf=25,
                l2_regularization=0.5,
                random_state=seed
            )
        )

        model.fit(
            X_train,
            y_train
        )

        models.append(
            model
        )

    test_predictions = []

    for model in models:

        p = (
            model
            .predict_proba(
                X_test
            )[:, 1]
        )

        test_predictions.append(
            p
        )

    average_probability = (
        np.mean(
            test_predictions,
            axis=0
        )
    )

    try:

        auc = float(
            roc_auc_score(
                y_test,
                average_probability
            )
        )

    except Exception:

        auc = float("nan")

    if np.isfinite(auc):

        log(
            f"🤖 {symbol} trained | "
            f"rows={n} | "
            f"AUC={auc:.3f}"
        )

    else:

        log(
            f"🤖 {symbol} trained | "
            f"rows={n} | AUC=N/A"
        )

    return {
        "models":
            models,

        "auc":
            auc,

        "rows":
            n,

        "last_timestamp":
            frame[
                "timestamp"
            ].iloc[-1]
    }

# ============================================================
# TRAIN ALL
# ============================================================

def train_all_models(data):

    models = {}

    # SPX = SPY proxy
    if (
        SPY_SYMBOL in data
        and not data[
            SPY_SYMBOL
        ].empty
    ):

        model = train_model(
            SPY_SYMBOL,
            data
        )

        if model:
            models[
                "SPX"
            ] = model

    for symbol in STOCKS:

        model = train_model(
            symbol,
            data
        )

        if model:
            models[
                symbol
            ] = model

    log(
        f"🧠 Models ready: "
        f"{len(models)}/11"
    )

    return models

# ============================================================
# LATEST FEATURES
# ============================================================

def latest_feature_row(
    symbol,
    data
):

    actual_symbol = (
        SPY_SYMBOL
        if symbol == "SPX"
        else symbol
    )

    if actual_symbol not in data:
        return None

    df = data[
        actual_symbol
    ]

    if len(df) < 60:
        return None

    feat = add_features(
        df,
        spy_df=data.get(
            SPY_SYMBOL
        ),
        qqq_df=data.get(
            QQQ_SYMBOL
        )
    )

    feat = feat.dropna(
        subset=
            FEATURE_COLUMNS
    )

    if feat.empty:
        return None

    row = feat.iloc[-1]

    values = (
        row[
            FEATURE_COLUMNS
        ]
        .astype(float)
    )

    if not np.isfinite(
        values
    ).all():

        return None

    return row

# ============================================================
# PREDICTION
# ============================================================

def predict(
    symbol,
    data,
    model_info
):

    row = latest_feature_row(
        symbol,
        data
    )

    if (
        row is None
        or model_info is None
    ):
        return None

    X = pd.DataFrame(
        [
            row[
                FEATURE_COLUMNS
            ].astype(float).values
        ],
        columns=FEATURE_COLUMNS
    )

    probabilities = []

    for model in model_info[
        "models"
    ]:

        try:

            probabilities.append(
                float(
                    model
                    .predict_proba(X)[
                        0, 1
                    ]
                )
            )

        except Exception:
            pass

    if not probabilities:
        return None

    p_up = float(
        np.mean(
            probabilities
        )
    )

    p_down = (
        1.0 - p_up
    )

    if (
        p_up
        >= MIN_PROBABILITY
    ):

        direction = "CALL"
        confidence = p_up

    elif (
        p_down
        >= MIN_PROBABILITY
    ):

        direction = "PUT"
        confidence = p_down

    else:

        direction = "WAIT"
        confidence = max(
            p_up,
            p_down
        )

    volatility = float(
        row["volatility"]
    )

    if volatility >= 0.003:

        regime = "HIGH_VOL"

    elif volatility < 0.0015:

        regime = "LOW_VOL"

    else:

        regime = "NORMAL"

    return {
        "symbol":
            symbol,

        "direction":
            direction,

        "p_up":
            p_up,

        "p_down":
            p_down,

        "confidence":
            confidence,

        "auc":
            model_info["auc"],

        "regime":
            regime,

        "row":
            row
    }

# ============================================================
# OPTION CONTRACTS
# ============================================================

def get_option_contracts(
    underlying,
    option_type
):

    params = {

        "underlying_symbols":
            underlying,

        "status":
            "active",

        "expiration_date":
            today_date_ny(),

        "type":
            option_type.lower(),

        "limit":
            10000
    }

    return alpaca_get(
        OPTIONS_CONTRACTS_URL,
        params=params,
        timeout=30
    )


def parse_contracts(data):

    if not data:
        return []

    contracts = (
        data.get(
            "option_contracts"
        )
    )

    if contracts is None:

        contracts = (
            data.get(
                "contracts"
            )
        )

    return contracts or []

# ============================================================
# OPTION QUOTES
# ============================================================

def get_option_quotes(
    symbols
):

    if not symbols:
        return {}

    result = {}

    for start in range(
        0,
        len(symbols),
        100
    ):

        batch = symbols[
            start:start + 100
        ]

        params = {
            "symbols":
                ",".join(batch),

            "feed":
                "indicative"
        }

        data = alpaca_get(
            OPTIONS_LATEST_QUOTES_URL,
            params=params,
            timeout=30
        )

        if not data:
            continue

        quotes = data.get(
            "quotes",
            {}
        )

        if not isinstance(
            quotes,
            dict
        ):
            continue

        for symbol, quote in (
            quotes.items()
        ):

            try:

                bid = float(
                    quote.get(
                        "bp",
                        quote.get(
                            "bid_price",
                            0
                        )
                    )
                    or 0
                )

                ask = float(
                    quote.get(
                        "ap",
                        quote.get(
                            "ask_price",
                            0
                        )
                    )
                    or 0
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

            spread = (
                ask - bid
            ) / mid

            result[
                symbol
            ] = {

                "bid":
                    bid,

                "ask":
                    ask,

                "mid":
                    mid,

                "spread":
                    spread
            }

    return result

# ============================================================
# HELPERS
# ============================================================

def clamp(
    value,
    low,
    high
):

    return max(
        low,
        min(
            high,
            value
        )
    )

# ============================================================
# STRIKE DISTANCE
# ============================================================

def strike_distance_score(
    underlying,
    strike,
    spot
):

    if spot <= 0:
        return 0

    distance = abs(
        strike - spot
    )

    if underlying == "SPX":

        max_distance = (
            MAX_SPX_STRIKE_DISTANCE
        )

    else:

        max_distance = (
            spot
            * MAX_STOCK_STRIKE_DISTANCE_PCT
        )

    if max_distance <= 0:
        return 0

    ratio = (
        distance
        / max_distance
    )

    return (
        clamp(
            1 - ratio,
            0,
            1
        )
        * 100
    )

# ============================================================
# OPTION SCORE
# ============================================================

def score_option(
    underlying,
    prediction,
    quote,
    strike,
    spot
):

    row = prediction[
        "row"
    ]

    direction = (
        prediction[
            "direction"
        ]
    )

    confidence = float(
        prediction[
            "confidence"
        ]
    )

    auc = prediction[
        "auc"
    ]

    # --------------------------------------------------------
    # 1. AI confidence - 30
    # --------------------------------------------------------

    confidence_score = (
        clamp(
            (
                confidence
                - 0.50
            ) / 0.25,
            0,
            1
        )
        * 30
    )

    # --------------------------------------------------------
    # 2. Historical AUC - 20
    # --------------------------------------------------------

    if np.isfinite(auc):

        auc_score = (
            clamp(
                (
                    auc
                    - 0.50
                ) / 0.20,
                0,
                1
            )
            * 20
        )

    else:

        auc_score = 0

    # --------------------------------------------------------
    # 3. Momentum - 10
    # --------------------------------------------------------

    momentum = float(
        row.get(
            "momentum12",
            0
        )
    )

    aligned_momentum = (
        momentum
        if direction == "CALL"
        else -momentum
    )

    momentum_score = (
        clamp(
            0.5
            + aligned_momentum
            / 0.02,
            0,
            1
        )
        * 10
    )

    # --------------------------------------------------------
    # 4. RSI - 8
    # --------------------------------------------------------

    rsi_value = float(
        row.get(
            "rsi",
            50
        )
    )

    if direction == "CALL":

        rsi_alignment = (
            clamp(
                (
                    rsi_value
                    - 45
                ) / 15,
                0,
                1
            )
        )

    else:

        rsi_alignment = (
            clamp(
                (
                    55
                    - rsi_value
                ) / 15,
                0,
                1
            )
        )

    rsi_score = (
        rsi_alignment
        * 8
    )

    # --------------------------------------------------------
    # 5. Volume - 7
    # --------------------------------------------------------

    volume_z = float(
        row.get(
            "volume_z",
            0
        )
    )

    volume_score = (
        clamp(
            (
                volume_z + 1
            ) / 3,
            0,
            1
        )
        * 7
    )

    # --------------------------------------------------------
    # 6. Relative strength - 7
    # --------------------------------------------------------

    rel_spy = float(
        row.get(
            "relative_spy",
            0
        )
    )

    rel_qqq = float(
        row.get(
            "relative_qqq",
            0
        )
    )

    relative = (
        rel_spy
        + rel_qqq
    ) / 2

    if direction == "PUT":
        relative = -relative

    relative_score = (
        clamp(
            0.5
            + relative
            / 0.015,
            0,
            1
        )
        * 7
    )

    # --------------------------------------------------------
    # 7. Spread - 8
    # --------------------------------------------------------

    spread = float(
        quote[
            "spread"
        ]
    )

    if spread > MAX_SPREAD_PERCENT:
        return 0

    spread_score = (
        clamp(
            1
            - spread
            / MAX_SPREAD_PERCENT,
            0,
            1
        )
        * 8
    )

    # --------------------------------------------------------
    # 8. Strike - 10
    # --------------------------------------------------------

    distance_score = (
        strike_distance_score(
            underlying,
            strike,
            spot
        )
        / 100
        * 10
    )

    total = (
        confidence_score
        + auc_score
        + momentum_score
        + rsi_score
        + volume_score
        + relative_score
        + spread_score
        + distance_score
    )

    return round(
        clamp(
            total,
            0,
            100
        ),
        1
    )

# ============================================================
# CHOOSE BEST OPTION
# ============================================================

def choose_best_contract(
    underlying,
    prediction,
    spot
):

    direction = (
        prediction[
            "direction"
        ]
    )

    if direction not in (
        "CALL",
        "PUT"
    ):
        return None

    option_type = (
        "call"
        if direction == "CALL"
        else "put"
    )

    data = get_option_contracts(
        underlying,
        option_type
    )

    contracts = parse_contracts(
        data
    )

    if not contracts:

        log(
            f"⚠️ {underlying}: "
            "no 0DTE contracts"
        )

        return None

    candidates = []

    for contract in contracts:

        try:

            symbol = str(
                contract.get(
                    "symbol",
                    ""
                )
            )

            if not symbol:
                continue

            strike = float(
                contract[
                    "strike_price"
                ]
            )

            if not contract.get(
                "tradable",
                True
            ):
                continue

            # ------------------------------------------------
            # SPX must be SPXW
            # ------------------------------------------------

            if underlying == "SPX":

                root = str(
                    contract.get(
                        "root_symbol",
                        ""
                    )
                ).upper()

                if (
                    root != "SPXW"
                    and "SPXW"
                    not in symbol.upper()
                ):
                    continue

                max_distance = (
                    MAX_SPX_STRIKE_DISTANCE
                )

            else:

                max_distance = (
                    spot
                    * MAX_STOCK_STRIKE_DISTANCE_PCT
                )

            distance = abs(
                strike - spot
            )

            if (
                distance
                > max_distance
            ):
                continue

            candidates.append(
                (
                    distance,
                    symbol,
                    strike,
                    contract
                )
            )

        except Exception:

            continue

    if not candidates:
        return None

    candidates.sort(
        key=lambda x: x[0]
    )

    # Near ATM only.
    candidates = candidates[
        :80
    ]

    quote_map = get_option_quotes(
        [
            item[1]
            for item in candidates
        ]
    )

    best = None

    for (
        distance,
        symbol,
        strike,
        contract
    ) in candidates:

        quote = quote_map.get(
            symbol
        )

        if not quote:
            continue

        if (
            quote["mid"]
            < MIN_OPTION_PREMIUM
        ):
            continue

        if (
            quote["spread"]
            > MAX_SPREAD_PERCENT
        ):
            continue

        score = score_option(
            underlying,
            prediction,
            quote,
            strike,
            spot
        )

        if (
            score
            < MIN_OPTION_SCORE
        ):
            continue

        candidate = {

            "underlying":
                underlying,

            "contract":
                symbol,

            "strike":
                strike,

            "option_type":
                option_type.upper(),

            "bid":
                quote["bid"],

            "ask":
                quote["ask"],

            "mid":
                quote["mid"],

            "spread":
                quote["spread"],

            "score":
                score,

            "contract_data":
                contract
        }

        if (
            best is None
            or candidate[
                "score"
            ]
            > best[
                "score"
            ]
        ):

            best = candidate

    return best

# ============================================================
# SPX PROXY
# ============================================================

def get_spx_proxy(
    data
):

    spy = data.get(
        SPY_SYMBOL
    )

    if (
        spy is None
        or spy.empty
    ):
        return None

    try:

        return (
            float(
                spy[
                    "close"
                ].iloc[-1]
            )
            * 10
        )

    except Exception:

        return None

# ============================================================
# BUILD OPPORTUNITY
# ============================================================

def build_opportunity(
    underlying,
    prediction,
    option
):

    if not option:
        return None

    # Use ASK as conservative entry.
    entry = option[
        "ask"
    ]

    # +40% target
    target = (
        entry * 1.40
    )

    # -30% stop
    stop = (
        entry * 0.70
    )

    return {

        "underlying":
            underlying,

        "direction":
            prediction[
                "direction"
            ],

        "confidence":
            prediction[
                "confidence"
            ],

        "p_up":
            prediction[
                "p_up"
            ],

        "p_down":
            prediction[
                "p_down"
            ],

        "auc":
            prediction[
                "auc"
            ],

        "regime":
            prediction[
                "regime"
            ],

        "contract":
            option[
                "contract"
            ],

        "strike":
            option[
                "strike"
            ],

        "bid":
            option[
                "bid"
            ],

        "ask":
            option[
                "ask"
            ],

        "mid":
            option[
                "mid"
            ],

        "spread":
            option[
                "spread"
            ],

        "score":
            option[
                "score"
            ],

        "entry":
            entry,

        "target":
            target,

        "stop":
            stop,

        "timestamp":
            now_ny()
    }

# ============================================================
# SIGNAL COOLDOWN
# ============================================================

def signal_allowed(
    opportunity
):

    key = (
        opportunity[
            "underlying"
        ],
        opportunity[
            "direction"
        ],
        opportunity[
            "contract"
        ]
    )

    previous = (
        LAST_SIGNAL.get(
            key
        )
    )

    if previous is None:
        return True

    elapsed = (
        time.time()
        - previous
    ) / 60

    return (
        elapsed
        >= SIGNAL_COOLDOWN_MINUTES
    )


def mark_signal_sent(
    opportunity
):

    key = (
        opportunity[
            "underlying"
        ],
        opportunity[
            "direction"
        ],
        opportunity[
            "contract"
        ]
    )

    LAST_SIGNAL[
        key
    ] = time.time()

# ============================================================
# TELEGRAM MESSAGE
# ============================================================

def format_opportunity(
    opportunity
):

    auc = opportunity[
        "auc"
    ]

    if np.isfinite(auc):

        auc_text = (
            f"{auc * 100:.1f}%"
        )

    else:

        auc_text = "N/A"

    direction = (
        opportunity[
            "direction"
        ]
    )

    emoji = (
        "🟢"
        if direction == "CALL"
        else "🔴"
    )

    return (

        f"🚨 <b>AI OPTIONS "
        f"OPPORTUNITY</b>\n"

        f"━━━━━━━━━━━━━━━━━━\n"

        f"🎯 الأصل: "
        f"<b>{opportunity['underlying']}</b>\n"

        f"📈 الاتجاه: "
        f"{emoji} <b>{direction}</b>\n"

        f"🔥 القوة: "
        f"<b>{opportunity['score']}/100</b>\n"

        f"🧠 AI Confidence: "
        f"<b>{opportunity['confidence'] * 100:.1f}%</b>\n"

        f"📊 AUC: "
        f"<b>{auc_text}</b>\n"

        f"🌡 النظام: "
        f"{opportunity['regime']}\n"

        f"━━━━━━━━━━━━━━━━━━\n"

        f"📜 العقد:\n"
        f"<code>{opportunity['contract']}</code>\n"

        f"🎯 Strike: "
        f"<b>{opportunity['strike']:.2f}</b>\n"

        f"💵 Bid: "
        f"${opportunity['bid']:.2f}\n"

        f"💵 Ask: "
        f"${opportunity['ask']:.2f}\n"

        f"↔️ Spread: "
        f"{opportunity['spread'] * 100:.1f}%\n"

        f"━━━━━━━━━━━━━━━━━━\n"

        f"🟢 دخول: "
        f"<b>${opportunity['entry']:.2f}</b>\n"

        f"🎯 هدف +40%: "
        f"<b>${opportunity['target']:.2f}</b>\n"

        f"🛑 وقف -30%: "
        f"<b>${opportunity['stop']:.2f}</b>\n"

        f"━━━━━━━━━━━━━━━━━━\n"

        f"🕒 "
        f"{opportunity['timestamp'].strftime('%H:%M:%S')} NY\n"

        f"🤖 {VERSION}\n\n"

        f"⚠️ <i>توصية آلية وليست "
        f"ضمانًا للربح.</i>\n"

        f"🚫 لا يوجد تنفيذ تلقائي."
    )

# ============================================================
# STARTUP
# ============================================================

def send_startup():

    global STARTUP_SENT

    if STARTUP_SENT:
        return

    message = (

        f"🤖 <b>SPX & 10 STOCKS "
        f"AI ADVISOR</b>\n\n"

        f"Version: "
        f"<b>{VERSION}</b>\n"

        f"Mode: "
        f"<b>Recommendation Only</b>\n\n"

        f"📌 SPX + "
        f"{len(STOCKS)} أسهم\n"

        f"🧠 3-model ML ensemble\n"

        f"📚 History: "
        f"{HISTORY_DAYS} days\n"

        f"⏱ Timeframe: "
        f"{TIMEFRAME}\n"

        f"🔥 Minimum score: "
        f"{MIN_OPTION_SCORE}/100\n\n"

        f"✅ Alpaca authentication OK\n"

        f"🚫 WAIT لن يتم إرساله\n"

        f"🚫 لا يوجد شراء/بيع تلقائي"
    )

    if telegram_send(
        message
    ):

        STARTUP_SENT = True

# ============================================================
# MARKET CONSENSUS
# ============================================================

def market_consensus(
    predictions
):

    if not predictions:

        return {
            "bias":
                "NEUTRAL",

            "strength":
                0
        }

    calls = [
        p for p in predictions
        if p["direction"] == "CALL"
    ]

    puts = [
        p for p in predictions
        if p["direction"] == "PUT"
    ]

    if len(calls) > len(puts):

        strength = np.mean(
            [
                p["confidence"]
                for p in calls
            ]
        )

        return {
            "bias":
                "BULLISH",

            "strength":
                float(strength)
        }

    if len(puts) > len(calls):

        strength = np.mean(
            [
                p["confidence"]
                for p in puts
            ]
        )

        return {
            "bias":
                "BEARISH",

            "strength":
                float(strength)
        }

    return {
        "bias":
            "NEUTRAL",

        "strength":
            0
    }

# ============================================================
# SCAN
# ============================================================

def scan_market(
    data,
    models
):

    opportunities = []

    predictions = []

    # --------------------------------------------------------
    # Generate predictions
    # --------------------------------------------------------

    for symbol in (
        ["SPX"]
        + STOCKS
    ):

        model = models.get(
            symbol
        )

        if not model:
            continue

        prediction = predict(
            symbol,
            data,
            model
        )

        if not prediction:
            continue

        prediction[
            "scan_symbol"
        ] = symbol

        predictions.append(
            prediction
        )

    consensus = (
        market_consensus(
            predictions
        )
    )

    # --------------------------------------------------------
    # Scan options
    # --------------------------------------------------------

    for prediction in predictions:

        if (
            prediction[
                "direction"
            ]
            not in (
                "CALL",
                "PUT"
            )
        ):
            continue

        symbol = prediction[
            "scan_symbol"
        ]

        if symbol == "SPX":

            underlying = "SPX"

            spot = get_spx_proxy(
                data
            )

        else:

            underlying = symbol

            df = data.get(
                symbol
            )

            if (
                df is None
                or df.empty
            ):
                continue

            spot = float(
                df[
                    "close"
                ].iloc[-1]
            )

        if spot is None:
            continue

        # ----------------------------------------------------
        # Extra quality filter
        # ----------------------------------------------------

        if (
            prediction[
                "confidence"
            ]
            < MIN_PROBABILITY
        ):
            continue

        option = (
            choose_best_contract(
                underlying,
                prediction,
                spot
            )
        )

        if not option:
            continue

        opportunity = (
            build_opportunity(
                underlying,
                prediction,
                option
            )
        )

        if not opportunity:
            continue

        if not signal_allowed(
            opportunity
        ):
            continue

        opportunity[
            "consensus_bias"
        ] = consensus[
            "bias"
        ]

        opportunity[
            "consensus_strength"
        ] = consensus[
            "strength"
        ]

        opportunities.append(
            opportunity
        )

    # --------------------------------------------------------
    # Rank
    # --------------------------------------------------------

    opportunities.sort(
        key=lambda x: (
            x["score"],
            x["confidence"],
            (
                x["auc"]
                if np.isfinite(
                    x["auc"]
                )
                else 0
            )
        ),
        reverse=True
    )

    return opportunities[
        :MAX_TELEGRAM_OPPORTUNITIES
    ]

# ============================================================
# MAIN
# ============================================================

def main():

    log(
        "=" * 70
    )

    log(
        f"🚀 SPX & 10 STOCKS "
        f"AI ADVISOR {VERSION}"
    )

    log(
        "=" * 70
    )

    # --------------------------------------------------------
    # Credentials
    # --------------------------------------------------------

    if (
        not ALPACA_API_KEY
        or not ALPACA_SECRET_KEY
    ):

        log(
            "❌ Alpaca credentials missing"
        )

        return

    # --------------------------------------------------------
    # AUTHENTICATION FIRST
    # --------------------------------------------------------

    if not test_alpaca_auth():

        log(
            "🛑 Bot stopped because "
            "Alpaca authentication failed."
        )

        return

    if (
        not TELEGRAM_BOT_TOKEN
        or not TELEGRAM_CHAT_ID
    ):

        log(
            "⚠️ Telegram credentials "
            "missing."
        )

    # --------------------------------------------------------
    # Initial data
    # --------------------------------------------------------

    data = (
        refresh_training_data(
            force=True
        )
    )

    if not data:

        log(
            "❌ Historical data unavailable"
        )

        return

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    models = (
        train_all_models(
            data
        )
    )

    if not models:

        log(
            "❌ No models trained"
        )

        return

    # --------------------------------------------------------
    # Startup
    # --------------------------------------------------------

    send_startup()

    log(
        "✅ Scanner running"
    )

    log(
        "🚫 Recommendation only"
    )

    log(
        "🚫 No automatic orders"
    )

    # --------------------------------------------------------
    # Main loop
    # --------------------------------------------------------

    while True:

        try:

            if not regular_market_open():

                time.sleep(
                    POLL_SECONDS
                )

                continue

            # ------------------------------------------------
            # Retrain periodically
            # ------------------------------------------------

            if (
                time.time()
                - LAST_TRAIN_TIME
                >= MODEL_REFRESH_SECONDS
            ):

                new_data = (
                    refresh_training_data(
                        force=True
                    )
                )

                new_models = (
                    train_all_models(
                        new_data
                    )
                )

                if new_models:

                    models = (
                        new_models
                    )

            # ------------------------------------------------
            # Update recent bars
            # ------------------------------------------------

            data = (
                refresh_recent_data(
                    force=False
                )
            )

            if not trading_window_allowed():

                time.sleep(
                    POLL_SECONDS
                )

                continue

            # ------------------------------------------------
            # Scan
            # ------------------------------------------------

            opportunities = (
                scan_market(
                    data,
                    models
                )
            )

            if opportunities:

                for opportunity in opportunities:

                    message = (
                        format_opportunity(
                            opportunity
                        )
                    )

                    if telegram_send(
                        message
                    ):

                        mark_signal_sent(
                            opportunity
                        )

                        log(
                            "🚨 SENT | "
                            f"{opportunity['underlying']} "
                            f"{opportunity['direction']} | "
                            f"score="
                            f"{opportunity['score']}"
                        )

            else:

                log(
                    "🔎 Scan complete | "
                    "No qualifying opportunity."
                )

            time.sleep(
                POLL_SECONDS
            )

        except KeyboardInterrupt:

            log(
                "🛑 Stopped by user"
            )

            break

        except Exception as e:

            log(
                f"❌ MAIN ERROR | "
                f"{type(e).__name__}: "
                f"{e}"
            )

            time.sleep(30)

# ============================================================
# START
# ============================================================

if __name__ == "__main__":
    main()