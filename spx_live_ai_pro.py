# ============================================================
# SPX 0DTE AI ADVISOR v14.6
# ============================================================
# Recommendation Only
#
# SPY 5m -> SPX Proxy -> ML Ensemble
# -> SPXW 0DTE Option Chain
# -> CALL / PUT / WAIT
# -> Entry / Target / Stop
# -> Telegram
#
# NO ORDER EXECUTION
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
from sklearn.calibration import CalibratedClassifierCV

warnings.filterwarnings("ignore")


# ============================================================
# CONFIG
# ============================================================

VERSION = "v14.6"

SPY_SYMBOL = "SPY"
TIMEFRAME = "5Min"

HISTORY_DAYS = 60
MIN_TRAIN_ROWS = 150

HORIZON = 6
ATR_TARGET = 0.50

MIN_PROBABILITY = 0.57

SIGNAL_COOLDOWN_MINUTES = 20
WAIT_MESSAGE_MINUTES = 15

POLL_SECONDS = 30

NO_TRADE_FIRST_MIN = 10
NO_TRADE_LAST_MIN = 30

OPTION_EXPIRY_DAYS = 0

# عقد الأوبشن:
OPTION_MAX_SPREAD_PERCENT = 15.0

# لا نختار عقود بعيدة جدًا عن السعر
MAX_STRIKE_DISTANCE = 40

# الحد الأدنى التقريبي لسعر العقد
MIN_OPTION_PREMIUM = 0.20


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
    or os.getenv("APCA_API_SECRET_KEY")
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


STOCK_DATA_URL = (
    "https://data.alpaca.markets/v2/stocks/bars"
)

OPTIONS_CONTRACTS_URL = (
    "https://paper-api.alpaca.markets/v2/options/contracts"
)

OPTIONS_LATEST_QUOTES_URL = (
    "https://data.alpaca.markets/v1beta1/options/quotes/latest"
)

NY = ZoneInfo(
    "America/New_York"
)


# ============================================================
# STATE
# ============================================================

STATE = {
    "models": [],
    "auc": None,
    "trained": False,

    "last_signal": None,
    "last_signal_time": None,

    "last_wait_time": None,

    "last_training_time": None,

    "last_training_rows": 0,
}


# ============================================================
# LOG
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


# ============================================================
# TELEGRAM
# ============================================================

def telegram_send(message):

    if (
        not TELEGRAM_BOT_TOKEN
        or not TELEGRAM_CHAT_ID
    ):
        log(
            "[TELEGRAM] credentials missing"
        )
        return False

    url = (
        "https://api.telegram.org/bot"
        f"{TELEGRAM_BOT_TOKEN}/sendMessage"
    )

    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "disable_web_page_preview": True,
    }

    try:

        r = requests.post(
            url,
            json=payload,
            timeout=15
        )

        if r.ok:
            return True

        log(
            f"[TELEGRAM ERROR] "
            f"{r.status_code}: "
            f"{r.text[:300]}"
        )

        return False

    except Exception as e:

        log(
            f"[TELEGRAM ERROR] {e}"
        )

        return False


# ============================================================
# TIME
# ============================================================

def now_ny():

    return datetime.now(
        NY
    )


def iso_utc(dt):

    if dt.tzinfo is None:

        dt = dt.replace(
            tzinfo=NY
        )

    return (
        dt.astimezone(
            timezone.utc
        )
        .isoformat()
        .replace(
            "+00:00",
            "Z"
        )
    )


# ============================================================
# ALPACA HEADERS
# ============================================================

def alpaca_headers():

    return {
        "APCA-API-KEY-ID":
            ALPACA_API_KEY,

        "APCA-API-SECRET-KEY":
            ALPACA_SECRET_KEY,
    }


# ============================================================
# DOWNLOAD STOCK BARS
# ============================================================

def download_bars(
    symbol=SPY_SYMBOL,
    timeframe=TIMEFRAME,
    days=60,
    limit=10000
):

    if (
        not ALPACA_API_KEY
        or not ALPACA_SECRET_KEY
    ):

        log(
            "[DATA ERROR] "
            "Alpaca keys missing"
        )

        return pd.DataFrame()

    end_dt = now_ny()

    start_dt = (
        end_dt
        - timedelta(days=days)
    )

    params = {

        "symbols":
            symbol,

        "timeframe":
            timeframe,

        "start":
            iso_utc(start_dt),

        "end":
            iso_utc(end_dt),

        "limit":
            min(limit, 10000),

        "feed":
            "iex",

        "adjustment":
            "raw",
    }

    rows = []

    page_token = None

    try:

        for _ in range(20):

            p = dict(
                params
            )

            if page_token:

                p[
                    "page_token"
                ] = page_token

            r = requests.get(
                STOCK_DATA_URL,
                headers=alpaca_headers(),
                params=p,
                timeout=30
            )

            if not r.ok:

                log(
                    f"[DATA ERROR] "
                    f"{r.status_code}: "
                    f"{r.text[:500]}"
                )

                break

            data = r.json()

            bars = data.get(
                "bars",
                {}
            )

            symbol_rows = bars.get(
                symbol,
                []
            )

            rows.extend(
                symbol_rows
            )

            page_token = data.get(
                "next_page_token"
            )

            if not page_token:

                break

        if not rows:

            return pd.DataFrame()

        df = pd.DataFrame(
            rows
        )

        df = df.rename(
            columns={
                "t": "timestamp",
                "o": "open",
                "h": "high",
                "l": "low",
                "c": "close",
                "v": "volume",
                "vw": "vwap",
                "n": "trade_count",
            }
        )

        df["timestamp"] = (
            pd.to_datetime(
                df["timestamp"],
                utc=True
            )
        )

        numeric = [
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]

        for col in numeric:

            if col in df.columns:

                df[col] = pd.to_numeric(
                    df[col],
                    errors="coerce"
                )

        df = df.dropna(
            subset=numeric
        )

        df = df.sort_values(
            "timestamp"
        )

        df = df.drop_duplicates(
            "timestamp"
        )

        df = df.reset_index(
            drop=True
        )

        log(
            f"[DATA] {symbol}: "
            f"{len(df)} bars"
        )

        return df

    except Exception as e:

        log(
            f"[DATA ERROR] {e}"
        )

        return pd.DataFrame()


# ============================================================
# LOCAL DATA
# ============================================================

LOCAL_FILE = (
    "spx_advisor_data.csv"
)


def load_local():

    if not os.path.exists(
        LOCAL_FILE
    ):
        return pd.DataFrame()

    try:

        df = pd.read_csv(
            LOCAL_FILE
        )

        df["timestamp"] = (
            pd.to_datetime(
                df["timestamp"],
                utc=True
            )
        )

        return df

    except Exception:

        return pd.DataFrame()


def save_local(df):

    try:

        df.to_csv(
            LOCAL_FILE,
            index=False
        )

    except Exception as e:

        log(
            f"[SAVE ERROR] {e}"
        )


# ============================================================
# UPDATE DATA
# ============================================================

def update_data(
    initial=False
):

    days = (
        HISTORY_DAYS
        if initial
        else 7
    )

    df = download_bars(
        SPY_SYMBOL,
        TIMEFRAME,
        days=days
    )

    if df.empty:

        return pd.DataFrame()

    # --------------------------------------------------------
    # SPY -> SPX proxy
    # --------------------------------------------------------

    df["spx_open"] = (
        df["open"] * 10
    )

    df["spx_high"] = (
        df["high"] * 10
    )

    df["spx_low"] = (
        df["low"] * 10
    )

    df["spx_close"] = (
        df["close"] * 10
    )

    df["spx_volume"] = (
        df["volume"]
    )

    old = load_local()

    if not old.empty:

        df = pd.concat(
            [old, df],
            ignore_index=True
        )

    df["timestamp"] = (
        pd.to_datetime(
            df["timestamp"],
            utc=True
        )
    )

    df = df.sort_values(
        "timestamp"
    )

    df = df.drop_duplicates(
        "timestamp",
        keep="last"
    )

    cutoff = (
        pd.Timestamp.now(
            tz="UTC"
        )
        - pd.Timedelta(
            days=90
        )
    )

    df = df[
        df["timestamp"]
        >= cutoff
    ]

    df = df.reset_index(
        drop=True
    )

    save_local(df)

    return df


# ============================================================
# REGULAR SESSION
# ============================================================

def regular_session(df):

    if df.empty:

        return df

    x = df.copy()

    ny = (
        x["timestamp"]
        .dt.tz_convert(NY)
    )

    minutes = (
        ny.dt.hour * 60
        + ny.dt.minute
    )

    start = (
        9 * 60 + 30
    )

    end = (
        16 * 60
    )

    x = x[
        (minutes >= start)
        & (minutes <= end)
    ]

    return x


# ============================================================
# RSI
# ============================================================

def rsi(series, period=14):

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
# ATR
# ============================================================

def atr(df, period=14):

    high = df[
        "spx_high"
    ]

    low = df[
        "spx_low"
    ]

    close = df[
        "spx_close"
    ]

    prev = close.shift(1)

    tr = pd.concat(
        [
            high - low,
            (high - prev).abs(),
            (low - prev).abs(),
        ],
        axis=1
    ).max(
        axis=1
    )

    return tr.rolling(
        period
    ).mean()


# ============================================================
# FEATURES
# ============================================================

FEATURES = [

    "ret_1",
    "ret_3",
    "ret_6",
    "ret_12",

    "rsi",

    "atr_pct",

    "vol_12",
    "vol_24",

    "range_pct",

    "close_vs_ma20",
    "close_vs_ma50",

    "volume_z",

    "momentum_3",
    "momentum_6",

    "hour_sin",
    "hour_cos",
]


# ============================================================
# PREPARE
# ============================================================

def prepare(df):

    if df.empty:

        return pd.DataFrame()

    x = regular_session(
        df.copy()
    )

    if len(x) < 100:

        return pd.DataFrame()

    close = x[
        "spx_close"
    ]

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
        x["atr"]
        / close
    )

    x["vol_12"] = (
        x["ret_1"]
        .rolling(12)
        .std()
    )

    x["vol_24"] = (
        x["ret_1"]
        .rolling(24)
        .std()
    )

    x["range_pct"] = (
        x["spx_high"]
        - x["spx_low"]
    ) / close

    ma20 = (
        close.rolling(20)
        .mean()
    )

    ma50 = (
        close.rolling(50)
        .mean()
    )

    x["close_vs_ma20"] = (
        close / ma20
    ) - 1

    x["close_vs_ma50"] = (
        close / ma50
    ) - 1

    vm = (
        x["spx_volume"]
        .rolling(30)
        .mean()
    )

    vs = (
        x["spx_volume"]
        .rolling(30)
        .std()
    )

    x["volume_z"] = (
        x["spx_volume"]
        - vm
    ) / vs.replace(
        0,
        np.nan
    )

    x["momentum_3"] = (
        close
        / close.shift(3)
    ) - 1

    x["momentum_6"] = (
        close
        / close.shift(6)
    ) - 1

    ny = (
        x["timestamp"]
        .dt.tz_convert(NY)
    )

    mins = (
        ny.dt.hour * 60
        + ny.dt.minute
        - 570
    )

    angle = (
        2
        * np.pi
        * mins
        / 390
    )

    x["hour_sin"] = np.sin(
        angle
    )

    x["hour_cos"] = np.cos(
        angle
    )

    # ========================================================
    # TARGET
    # ========================================================

    target = []

    for i in range(
        len(x)
    ):

        if (
            i + HORIZON
            >= len(x)
        ):

            target.append(
                np.nan
            )

            continue

        entry = float(
            x["spx_close"]
            .iloc[i]
        )

        current_atr = float(
            x["atr"].iloc[i]
        )

        if (
            not np.isfinite(
                current_atr
            )
            or current_atr <= 0
        ):

            target.append(
                np.nan
            )

            continue

        upper = (
            entry
            + ATR_TARGET
            * current_atr
        )

        lower = (
            entry
            - ATR_TARGET
            * current_atr
        )

        result = np.nan

        for j in range(
            i + 1,
            min(
                i + HORIZON + 1,
                len(x)
            )
        ):

            hi = float(
                x["spx_high"]
                .iloc[j]
            )

            lo = float(
                x["spx_low"]
                .iloc[j]
            )

            up = (
                hi >= upper
            )

            down = (
                lo <= lower
            )

            # Ambiguous candle
            if up and down:

                result = np.nan
                break

            if up:

                result = 1
                break

            if down:

                result = 0
                break

        target.append(
            result
        )

    x["target"] = target

    return x


# ============================================================
# TRAIN
# ============================================================

def train(df):

    work = df.dropna(
        subset=FEATURES + ["target"]
    ).copy()

    work = work.replace(
        [np.inf, -np.inf],
        np.nan
    )

    work = work.dropna(
        subset=FEATURES + ["target"]
    )

    if len(work) < MIN_TRAIN_ROWS:

        log(
            f"[TRAIN] "
            f"{len(work)}/{MIN_TRAIN_ROWS}"
        )

        return False

    n = len(work)

    train_end = int(
        n * 0.60
    )

    test_start = int(
        n * 0.80
    )

    train_df = work.iloc[
        :train_end
    ]

    test_df = work.iloc[
        test_start:
    ]

    X_train = train_df[
        FEATURES
    ]

    y_train = train_df[
        "target"
    ].astype(int)

    X_test = test_df[
        FEATURES
    ]

    y_test = test_df[
        "target"
    ].astype(int)

    if (
        y_train.nunique() < 2
        or y_test.nunique() < 2
    ):

        log(
            "[TRAIN] "
            "not enough target classes"
        )

        return False

    models = []

    for seed in [
        11,
        29,
        71,
    ]:

        try:

            base = (
                HistGradientBoostingClassifier(
                    learning_rate=0.045,
                    max_iter=250,
                    max_leaf_nodes=15,
                    min_samples_leaf=15,
                    l2_regularization=1.0,
                    random_state=seed,
                )
            )

            model = (
                CalibratedClassifierCV(
                    base,
                    method="sigmoid",
                    cv=3,
                )
            )

            model.fit(
                X_train,
                y_train
            )

            models.append(
                model
            )

        except Exception as e:

            log(
                f"[TRAIN MODEL ERROR] {e}"
            )

    if not models:

        return False

    predictions = []

    for model in models:

        predictions.append(
            model.predict_proba(
                X_test
            )[:, 1]
        )

    probability = np.mean(
        predictions,
        axis=0
    )

    try:

        auc_score = (
            roc_auc_score(
                y_test,
                probability
            )
        )

    except Exception:

        auc_score = np.nan

    STATE["models"] = models

    STATE["auc"] = (
        float(auc_score)
        if np.isfinite(
            auc_score
        )
        else None
    )

    STATE["trained"] = True

    STATE[
        "last_training_rows"
    ] = len(work)

    STATE[
        "last_training_time"
    ] = now_ny()

    log(
        f"[TRAIN] DONE | "
        f"rows={len(work)} | "
        f"AUC={auc_score:.3f}"
    )

    telegram_send(
        "🧠 SPX AI Advisor\n"
        "━━━━━━━━━━━━━━━━━━\n"
        "✅ تم تدريب النموذج\n\n"
        f"📊 بيانات التدريب: {len(work)}\n"
        f"🧪 Test AUC: {auc_score:.3f}\n\n"
        "جاهز لتحليل SPX."
    )

    return True


# ============================================================
# REGIME
# ============================================================

def regime(df):

    if df.empty:

        return "UNKNOWN"

    x = df.dropna(
        subset=["vol_24"]
    )

    if x.empty:

        return "UNKNOWN"

    value = float(
        x["vol_24"].iloc[-1]
    )

    if value >= 0.018:

        return "HIGH_VOL"

    if value <= 0.008:

        return "LOW_VOL"

    return "NORMAL"


# ============================================================
# ML SIGNAL
# ============================================================

def ml_signal(df):

    if not STATE["trained"]:

        return {
            "signal": "WAIT",
            "probability": 0.50,
            "reason":
                "النموذج غير جاهز",
        }

    x = df.dropna(
        subset=FEATURES
    )

    if x.empty:

        return {
            "signal": "WAIT",
            "probability": 0.50,
            "reason":
                "لا توجد بيانات صالحة",
        }

    latest = x.iloc[
        [-1]
    ][FEATURES]

    probs = []

    for model in STATE[
        "models"
    ]:

        try:

            p = model.predict_proba(
                latest
            )[0][1]

            probs.append(
                float(p)
            )

        except Exception:
            pass

    if not probs:

        return {
            "signal": "WAIT",
            "probability": 0.50,
            "reason":
                "فشل النموذج",
        }

    p_call = float(
        np.mean(probs)
    )

    p_put = (
        1 - p_call
    )

    market_regime = regime(
        df
    )

    threshold = (
        0.60
        if market_regime
        == "HIGH_VOL"
        else MIN_PROBABILITY
    )

    if p_call >= threshold:

        signal = "CALL"

        confidence = p_call

        reason = (
            "النموذج يميل للصعود"
        )

    elif p_put >= threshold:

        signal = "PUT"

        confidence = p_put

        reason = (
            "النموذج يميل للهبوط"
        )

    else:

        signal = "WAIT"

        confidence = max(
            p_call,
            p_put
        )

        reason = (
            "الاحتمالية غير كافية"
        )

    return {

        "signal":
            signal,

        "p_call":
            p_call,

        "p_put":
            p_put,

        "probability":
            p_call,

        "confidence":
            confidence,

        "regime":
            market_regime,

        "reason":
            reason,
    }


# ============================================================
# MARKET TIME
# ============================================================

def market_open_now():

    now = now_ny()

    mins = (
        now.hour * 60
        + now.minute
    )

    open_min = (
        9 * 60 + 30
    )

    close_min = (
        16 * 60
    )

    if mins < open_min:

        return False, (
            "السوق لم يفتح بعد"
        )

    if mins >= close_min:

        return False, (
            "السوق مغلق"
        )

    if (
        mins
        - open_min
        < NO_TRADE_FIRST_MIN
    ):

        return False, (
            "أول دقائق السوق"
        )

    if (
        close_min
        - mins
        <= NO_TRADE_LAST_MIN
    ):

        return False, (
            "آخر دقائق السوق"
        )

    return True, (
        "السوق مفتوح"
    )


# ============================================================
# OPTION CHAIN
# ============================================================

def get_spxw_contracts(
    signal,
    underlying_price
):

    expiry = (
        now_ny()
        .date()
        .isoformat()
    )

    contract_type = (
        "call"
        if signal == "CALL"
        else "put"
    )

    params = {

        "underlying_symbols":
            "SPX",

        "status":
            "active",

        "expiration_date":
            expiry,

        "type":
            contract_type,

        "limit":
            100,

        "root_symbol":
            "SPX",
    }

    try:

        r = requests.get(
            OPTIONS_CONTRACTS_URL,
            headers=alpaca_headers(),
            params=params,
            timeout=20
        )

        if not r.ok:

            log(
                f"[OPTIONS] "
                f"HTTP {r.status_code}: "
                f"{r.text[:300]}"
            )

            return []

        data = r.json()

        contracts = data.get(
            "option_contracts",
            data.get(
                "contracts",
                []
            )
        )

        if not contracts:

            return []

        selected = []

        for c in contracts:

            symbol = (
                c.get("symbol")
                or c.get(
                    "tradingsymbol"
                )
            )

            strike = c.get(
                "strike_price"
            )

            if (
                symbol is None
                or strike is None
            ):
                continue

            try:

                strike = float(
                    strike
                )

            except Exception:

                continue

            distance = abs(
                strike
                - underlying_price
            )

            if (
                distance
                <= MAX_STRIKE_DISTANCE
            ):

                selected.append(
                    {
                        "symbol":
                            symbol,

                        "strike":
                            strike,

                        "type":
                            contract_type,

                        "expiration":
                            expiry,
                    }
                )

        return selected

    except Exception as e:

        log(
            f"[OPTIONS ERROR] {e}"
        )

        return []


# ============================================================
# OPTION QUOTE
# ============================================================

def get_option_quote(
    symbols
):

    if not symbols:

        return {}

    params = {
        "symbols":
            ",".join(symbols)
    }

    try:

        r = requests.get(
            OPTIONS_LATEST_QUOTES_URL,
            headers=alpaca_headers(),
            params=params,
            timeout=20
        )

        if not r.ok:

            log(
                f"[OPTION QUOTE] "
                f"HTTP {r.status_code}"
            )

            return {}

        data = r.json()

        quotes = data.get(
            "quotes",
            {}
        )

        result = {}

        for symbol, q in quotes.items():

            bid = q.get(
                "bp",
                q.get(
                    "bid_price",
                    0
                )
            )

            ask = q.get(
                "ap",
                q.get(
                    "ask_price",
                    0
                )
            )

            try:

                bid = float(
                    bid or 0
                )

                ask = float(
                    ask or 0
                )

            except Exception:

                continue

            if bid > 0 and ask > 0:

                mid = (
                    bid + ask
                ) / 2

                spread = (
                    (ask - bid)
                    / mid
                    * 100
                )

                result[symbol] = {

                    "bid": bid,

                    "ask": ask,

                    "mid": mid,

                    "spread_pct":
                        spread,
                }

        return result

    except Exception as e:

        log(
            f"[OPTION QUOTE ERROR] {e}"
        )

        return {}


# ============================================================
# SELECT OPTION
# ============================================================

def select_option(
    signal,
    spx_price
):

    contracts = (
        get_spxw_contracts(
            signal,
            spx_price
        )
    )

    if not contracts:

        return None

    symbols = [
        c["symbol"]
        for c in contracts
    ]

    quotes = (
        get_option_quote(
            symbols
        )
    )

    candidates = []

    for c in contracts:

        q = quotes.get(
            c["symbol"]
        )

        if not q:

            continue

        if (
            q["mid"]
            < MIN_OPTION_PREMIUM
        ):

            continue

        if (
            q["spread_pct"]
            > OPTION_MAX_SPREAD_PERCENT
        ):

            continue

        distance = abs(
            c["strike"]
            - spx_price
        )

        score = (
            distance
            + q["spread_pct"]
            * 0.15
        )

        item = dict(c)

        item.update(q)

        item["score"] = score

        candidates.append(
            item
        )

    if not candidates:

        return None

    candidates.sort(
        key=lambda x:
            x["score"]
    )

    return candidates[0]


# ============================================================
# OPTION LEVELS
# ============================================================

def option_levels(
    option,
    signal
):

    if not option:

        return None

    entry = float(
        option["mid"]
    )

    if signal == "CALL":

        target = (
            entry * 1.35
        )

        stop = (
            entry * 0.70
        )

    else:

        target = (
            entry * 1.35
        )

        stop = (
            entry * 0.70
        )

    return {

        "entry":
            entry,

        "target":
            target,

        "stop":
            stop,
    }


# ============================================================
# SIGNAL COOLDOWN
# ============================================================

def can_send(signal):

    if signal not in [
        "CALL",
        "PUT"
    ]:

        return False

    if (
        STATE["last_signal"]
        is None
    ):

        return True

    if signal != (
        STATE["last_signal"]
    ):

        return True

    if (
        STATE["last_signal_time"]
        is None
    ):

        return True

    elapsed = (
        now_ny()
        - STATE[
            "last_signal_time"
        ]
    ).total_seconds() / 60

    return (
        elapsed
        >= SIGNAL_COOLDOWN_MINUTES
    )


# ============================================================
# WAIT TELEGRAM
# ============================================================

def send_wait(
    ml,
    df,
    reason=None,
    force=False
):

    now = now_ny()

    if not force:

        last = (
            STATE["last_wait_time"]
        )

        if last:

            elapsed = (
                now - last
            ).total_seconds() / 60

            if (
                elapsed
                < WAIT_MESSAGE_MINUTES
            ):

                return

    spx = float(
        df["spx_close"].iloc[-1]
    )

    atr_value = float(
        df["atr"].iloc[-1]
    )

    p_call = ml.get(
        "p_call",
        0.5
    )

    p_put = ml.get(
        "p_put",
        0.5
    )

    if p_call > p_put:

        expected = (
            "🟢 صعود / CALL"
        )

    elif p_put > p_call:

        expected = (
            "🔴 هبوط / PUT"
        )

    else:

        expected = (
            "⚪ محايد"
        )

    suggested_strike = (
        round(spx / 5)
        * 5
    )

    target_up = (
        spx
        + ATR_TARGET
        * atr_value
    )

    target_down = (
        spx
        - ATR_TARGET
        * atr_value
    )

    auc = STATE["auc"]

    auc_text = (
        f"{auc:.3f}"
        if auc is not None
        else "N/A"
    )

    message = (
        "⚪ SPX AI — WAIT\n"
        "━━━━━━━━━━━━━━━━━━\n\n"

        f"📍 SPX Proxy: {spx:.2f}\n"

        f"📈 احتمال CALL: "
        f"{p_call:.1%}\n"

        f"📉 احتمال PUT: "
        f"{p_put:.1%}\n\n"

        f"🔮 التوقع المفضل: "
        f"{expected}\n"

        f"🎯 Strike المفضل: "
        f"{suggested_strike}\n\n"

        f"🌡 حالة السوق: "
        f"{ml.get('regime', 'N/A')}\n"

        f"🧪 Test AUC: "
        f"{auc_text}\n\n"

        f"🎯 هدف صعود SPX: "
        f"{target_up:.2f}\n"

        f"🎯 هدف هبوط SPX: "
        f"{target_down:.2f}\n\n"

        "💵 عقد 0DTE:\n"
        "غير متاح حاليًا — "
        "لم يتم العثور على Quote صالح.\n\n"

        f"🧠 السبب: "
        f"{reason or ml.get('reason')}\n\n"

        "⏳ القرار الحالي: WAIT\n"
        "⚠️ توصية فقط — لا يوجد تنفيذ أوامر\n"
        "⚠️ SPX هنا Proxy من SPY × 10"
    )

    telegram_send(
        message
    )

    STATE[
        "last_wait_time"
    ] = now


# ============================================================
# SEND CALL / PUT
# ============================================================

def send_recommendation(
    ml,
    df
):

    signal = ml["signal"]

    spx = float(
        df["spx_close"].iloc[-1]
    )

    auc = STATE["auc"]

    auc_text = (
        f"{auc:.3f}"
        if auc is not None
        else "N/A"
    )

    option = select_option(
        signal,
        spx
    )

    levels = (
        option_levels(
            option,
            signal
        )
        if option
        else None
    )

    if signal == "CALL":

        emoji = "🟢"

    else:

        emoji = "🔴"

    strike = (
        option["strike"]
        if option
        else round(spx / 5) * 5
    )

    if option and levels:

        option_text = (
            f"🎫 العقد: "
            f"{option['symbol']}\n"
            f"🎯 Strike: "
            f"{strike:.0f}\n"
            f"💵 Entry: "
            f"${levels['entry']:.2f}\n"
            f"🎯 Target: "
            f"${levels['target']:.2f}\n"
            f"🛑 Stop: "
            f"${levels['stop']:.2f}\n"
            f"📊 Bid/Ask: "
            f"${option['bid']:.2f} / "
            f"${option['ask']:.2f}\n"
            f"↔️ Spread: "
            f"{option['spread_pct']:.1f}%"
        )

    else:

        option_text = (
            "🎫 عقد 0DTE: "
            "غير متاح حاليًا\n"
            "لم يتم العثور على Quote "
            "صالح من Alpaca."
        )

    message = (
        "🚨 SPX 0DTE AI RECOMMENDATION\n"
        "━━━━━━━━━━━━━━━━━━\n\n"

        f"{emoji} {signal}\n\n"

        f"📍 SPX Proxy: "
        f"{spx:.2f}\n"

        f"📈 Confidence: "
        f"{ml['confidence']:.1%}\n"

        f"🧪 Test AUC: "
        f"{auc_text}\n"

        f"🌡 Regime: "
        f"{ml['regime']}\n\n"

        f"{option_text}\n\n"

        f"🧠 السبب: "
        f"{ml['reason']}\n\n"

        "⚠️ توصية فقط — "
        "لا يوجد تنفيذ أوامر\n"

        "⚠️ SPX Proxy محسوب من SPY × 10"
    )

    telegram_send(
        message
    )

    STATE[
        "last_signal"
    ] = signal

    STATE[
        "last_signal_time"
    ] = now_ny()

    log(
        f"[SIGNAL] "
        f"{signal} | "
        f"confidence="
        f"{ml['confidence']:.1%} | "
        f"SPX={spx:.2f}"
    )


# ============================================================
# STARTUP
# ============================================================

def startup():

    log(
        f"SPX {VERSION} "
        "— بدء التشغيل"
    )

    telegram_send(
        f"🤖 SPX AI Advisor {VERSION}\n\n"
        "🚀 بدأ التشغيل\n"
        "📚 جاري تحميل التاريخ "
        "وتدريب النموذج..."
    )

    for attempt in range(
        1,
        11
    ):

        log(
            f"[STARTUP] "
            f"محاولة {attempt}/10"
        )

        raw = update_data(
            initial=True
        )

        if raw.empty:

            send_wait(
                {
                    "p_call": 0.5,
                    "p_put": 0.5,
                    "regime":
                        "UNKNOWN",
                },
                pd.DataFrame(
                    {
                        "spx_close": [],
                        "atr": [],
                    }
                ),
                "تعذر جلب البيانات",
                force=(attempt == 1)
            ) if False else None

            time.sleep(10)

            continue

        prepared = prepare(
            raw
        )

        if prepared.empty:

            log(
                "[STARTUP] "
                "البيانات غير كافية بعد التنظيف"
            )

            time.sleep(10)

            continue

        usable = prepared.dropna(
            subset=FEATURES + ["target"]
        )

        log(
            f"[STARTUP] "
            f"raw={len(raw)} | "
            f"prepared={len(prepared)} | "
            f"usable={len(usable)}"
        )

        if (
            len(usable)
            < MIN_TRAIN_ROWS
        ):

            time.sleep(10)

            continue

        if train(
            prepared
        ):

            log(
                "[STARTUP] "
                "MODEL READY"
            )

            return True

    telegram_send(
        "❌ SPX AI Advisor\n\n"
        "فشل تدريب النموذج.\n"
        "تحقق من بيانات Alpaca."
    )

    return False


# ============================================================
# LIVE LOOP
# ============================================================

def live_loop():

    last_retrain = now_ny()

    while True:

        try:

            raw = update_data(
                initial=False
            )

            if raw.empty:

                log(
                    "[ANALYSIS] "
                    "WAIT | no data"
                )

                time.sleep(
                    POLL_SECONDS
                )

                continue

            prepared = prepare(
                raw
            )

            if prepared.empty:

                time.sleep(
                    POLL_SECONDS
                )

                continue

            # ------------------------------------------------
            # RETRAIN
            # ------------------------------------------------

            if (
                now_ny()
                - last_retrain
            ).total_seconds()
            >= 3600:

                log(
                    "[TRAIN] "
                    "إعادة تدريب..."
                )

                if train(
                    prepared
                ):

                    last_retrain = (
                        now_ny()
                    )

            # ------------------------------------------------
            # MARKET
            # ------------------------------------------------

            market_ok, reason = (
                market_open_now()
            )

            if not market_ok:

                log(
                    f"[ANALYSIS] "
                    f"WAIT | {reason}"
                )

                time.sleep(
                    POLL_SECONDS
                )

                continue

            # ------------------------------------------------
            # ML
            # ------------------------------------------------

            ml = ml_signal(
                prepared
            )

            log(
                f"[ANALYSIS] "
                f"{ml['signal']} | "
                f"CALL="
                f"{ml.get('p_call', 0):.1%} | "
                f"PUT="
                f"{ml.get('p_put', 0):.1%} | "
                f"regime="
                f"{ml.get('regime')}"
            )

            # ------------------------------------------------
            # SIGNAL
            # ------------------------------------------------

            if ml["signal"] in [
                "CALL",
                "PUT"
            ]:

                if can_send(
                    ml["signal"]
                ):

                    send_recommendation(
                        ml,
                        prepared
                    )

                else:

                    log(
                        "[FILTER] "
                        "duplicate signal"
                    )

            else:

                send_wait(
                    ml,
                    prepared
                )

        except KeyboardInterrupt:

            log(
                "إيقاف البوت"
            )

            break

        except Exception as e:

            log(
                f"[MAIN ERROR] {e}"
            )

        time.sleep(
            POLL_SECONDS
        )


# ============================================================
# MAIN
# ============================================================

def main():

    if not ALPACA_API_KEY:

        log(
            "❌ ALPACA_API_KEY غير موجود"
        )

        return

    if not ALPACA_SECRET_KEY:

        log(
            "❌ ALPACA_SECRET_KEY غير موجود"
        )

        return

    if not startup():

        return

    live_loop()


if __name__ == "__main__":

    main()