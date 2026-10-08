# ============================================================
# SPX 0DTE ADVISOR v14.4
# HONEST ML SIGNAL ENGINE — RECOMMENDATION ONLY
# ============================================================
#
# الوظيفة:
#   SPX -> Features -> ML Ensemble -> Probability
#       -> Market Filters -> CALL / PUT / WAIT
#       -> 0DTE Contract Suggestion -> Telegram
#
# لا يوجد تنفيذ أوامر.
# لا يوجد شراء/بيع حقيقي.
#
# ملاحظة:
#   يستخدم SPY كمؤشر بديل لحركة SPX عندما لا تتوفر بيانات SPX
#   مباشرة من مصدر البيانات، مع إبقاء ذلك واضحاً في التوصية.
# ============================================================

import os
import time
import warnings
from math import sqrt, floor, ceil
from datetime import datetime
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

NY = ZoneInfo("America/New_York")

TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

APCA_KEY = os.getenv("APCA_API_KEY_ID", "")
APCA_SECRET = os.getenv("APCA_API_SECRET_KEY", "")

APCA_DATA_URL = "https://data.alpaca.markets"

POLL_SECONDS = 30

MARKET_OPEN_MIN = 9 * 60 + 30
MARKET_CLOSE_MIN = 16 * 60

# لا تداول/توصيات في أول وآخر دقائق
NO_TRADE_FIRST_MIN = 10
NO_TRADE_LAST_MIN = 30

MAX_BAR_AGE_MIN = 15

LOCAL_DB_CSV = "local_market_db_v14_4.csv"

# ============================================================
# ML
# ============================================================

MIN_TRAIN_ROWS = 150
HORIZON = 6

# خفضنا العتبة من 0.55 قليلاً لكن ما زالت فلترة حقيقية
MIN_PROBABILITY = 0.57

# لا نسمح بتجميل AUC
MIN_ACCEPTABLE_AUC = 0.50

RETRAIN_EVERY_MIN = 180

# ============================================================
# 0DTE
# ============================================================

STRIKE_STEP = 5.0
STRIKE_OFFSET_STEPS = 1

STOP_LOSS_PCT = 0.40
TAKE_PROFIT_PCT = 0.50

MAX_HOLD_MINUTES = 30

# ============================================================
# TELEGRAM / SIGNAL CONTROL
# ============================================================

DIAG_THROTTLE_SEC = 60

# لا تكرر نفس الاتجاه باستمرار
SIGNAL_COOLDOWN_MIN = 20

# لا نرسل WAIT كل 30 ثانية
WAIT_REPORT_MIN = 15

# ============================================================
# STATE
# ============================================================

STATE = {
    "models": {},
    "auc": 0.0,
    "last_train": None,
    "date": None,
    "daily_pnl": 0.0,
    "signals_today": 0,
    "last_signal": None,
    "last_signal_time": None,
    "last_wait_time": None,
    "last_spot": None,
    "data_source": "UNKNOWN",
    "train_rows": 0,
}

_DIAG_LAST = {}


# ============================================================
# BASIC
# ============================================================

def now_ny():
    return datetime.now(NY)


def say(msg):
    print(
        f"[{now_ny().strftime('%Y-%m-%d %H:%M:%S')}] {msg}",
        flush=True
    )


def why(msg):
    now = time.time()

    if now - _DIAG_LAST.get(msg, 0) >= DIAG_THROTTLE_SEC:
        _DIAG_LAST[msg] = now
        say(f"[WHY-WAIT] {msg}")


def minutes_of_day(dt=None):
    dt = dt or now_ny()
    return dt.hour * 60 + dt.minute


def market_time_ok():
    n = now_ny()

    return (
        n.weekday() < 5
        and MARKET_OPEN_MIN <= minutes_of_day(n) < MARKET_CLOSE_MIN
    )


def session_allowed():
    m = minutes_of_day() - MARKET_OPEN_MIN

    return (
        NO_TRADE_FIRST_MIN
        <= m
        <= (MARKET_CLOSE_MIN - MARKET_OPEN_MIN) - NO_TRADE_LAST_MIN
    )


def reset_daily_state():

    today = now_ny().date()

    if STATE["date"] != today:

        STATE["date"] = today
        STATE["daily_pnl"] = 0.0
        STATE["signals_today"] = 0
        STATE["last_signal"] = None
        STATE["last_signal_time"] = None
        STATE["last_wait_time"] = None


# ============================================================
# TELEGRAM
# ============================================================

def send_telegram(message):

    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:

        say(
            "Telegram غير مفعّل: "
            "TELEGRAM_BOT_TOKEN أو TELEGRAM_CHAT_ID غير موجود"
        )

        return False

    try:

        url = (
            f"https://api.telegram.org/"
            f"bot{TELEGRAM_TOKEN}/sendMessage"
        )

        response = requests.post(
            url,
            json={
                "chat_id": TELEGRAM_CHAT_ID,
                "text": message
            },
            timeout=10
        )

        if response.status_code != 200:

            say(
                f"Telegram HTTP {response.status_code}: "
                f"{response.text[:300]}"
            )

            return False

        return True

    except Exception as e:

        say(f"Telegram error: {e}")

        return False


# ============================================================
# ALPACA DATA
# ============================================================

def get_alpaca_headers():

    return {
        "APCA-API-KEY-ID": APCA_KEY,
        "APCA-API-SECRET-KEY": APCA_SECRET,
        "accept": "application/json",
    }


def download_bars(symbol, timeframe="5Min", limit=1000):

    if not APCA_KEY or not APCA_SECRET:

        why("Alpaca API keys غير موجودة")

        return None

    url = f"{APCA_DATA_URL}/v2/stocks/bars"

    params = {
        "symbols": symbol,
        "timeframe": timeframe,
        "limit": limit,
        "feed": "iex",
    }

    try:

        response = requests.get(
            url,
            headers=get_alpaca_headers(),
            params=params,
            timeout=15
        )

        if response.status_code != 200:

            why(
                f"Alpaca bars HTTP "
                f"{response.status_code} لـ {symbol}"
            )

            return None

        payload = response.json()

        bars = payload.get("bars", {}).get(symbol, [])

        if not bars:

            return None

        df = pd.DataFrame(bars)

        if df.empty:
            return None

        df["timestamp"] = (
            pd.to_datetime(df["t"], utc=True)
            .dt.tz_convert(NY)
        )

        out = pd.DataFrame({

            "timestamp": df["timestamp"],

            "open": pd.to_numeric(
                df["o"],
                errors="coerce"
            ),

            "high": pd.to_numeric(
                df["h"],
                errors="coerce"
            ),

            "low": pd.to_numeric(
                df["l"],
                errors="coerce"
            ),

            "close": pd.to_numeric(
                df["c"],
                errors="coerce"
            ),

            "volume": pd.to_numeric(
                df["v"],
                errors="coerce"
            ),

        })

        out = (
            out
            .dropna(subset=["close"])
            .sort_values("timestamp")
            .drop_duplicates("timestamp")
        )

        return out

    except Exception as e:

        why(f"Alpaca exception {symbol}: {e}")

        return None


# ============================================================
# DATABASE UPDATE
# ============================================================

def update_local_database():

    spy = download_bars(
        "SPY",
        timeframe="5Min",
        limit=1000
    )

    if spy is None or spy.empty:

        why("لم يتم الحصول على بيانات SPY")

        return False

    df = pd.DataFrame()

    df["timestamp"] = spy["timestamp"]

    # Proxy فقط وليس SPX حقيقي
    df["spx_open"] = spy["open"] * 10.0
    df["spx_high"] = spy["high"] * 10.0
    df["spx_low"] = spy["low"] * 10.0
    df["spx_close"] = spy["close"] * 10.0

    df["spy_close"] = spy["close"]
    df["spy_volume"] = spy["volume"]

    # VIX غير متوفر هنا بشكل مباشر
    # نستخدم proxy مبني على volatility بدلاً من قيمة ثابتة
    returns = spy["close"].pct_change()

    realized = (
        returns
        .rolling(12)
        .std()
        * sqrt(252 * 78)
        * 100
    )

    df["vix"] = realized.fillna(18.0)

    df["timestamp_str"] = df["timestamp"].astype(str)

    if os.path.exists(LOCAL_DB_CSV):

        try:
            old = pd.read_csv(LOCAL_DB_CSV)

            combined = pd.concat(
                [old, df],
                ignore_index=True
            )

            combined = (
                combined
                .drop_duplicates(
                    subset=["timestamp_str"],
                    keep="last"
                )
                .sort_values("timestamp_str")
                .tail(40000)
            )

            combined.to_csv(
                LOCAL_DB_CSV,
                index=False
            )

        except Exception:

            df.to_csv(
                LOCAL_DB_CSV,
                index=False
            )

    else:

        df.to_csv(
            LOCAL_DB_CSV,
            index=False
        )

    STATE["data_source"] = "SPY_PROXY"

    return True


# ============================================================
# GET DATA
# ============================================================

def get_data():

    update_local_database()

    if not os.path.exists(LOCAL_DB_CSV):
        return None

    try:

        df = pd.read_csv(LOCAL_DB_CSV)

        if df.empty:
            return None

        df["timestamp"] = (
            pd.to_datetime(
                df["timestamp"],
                utc=True
            )
            .dt.tz_convert(NY)
        )

        t = (
            df["timestamp"].dt.hour * 60
            + df["timestamp"].dt.minute
        )

        df = df[
            (t >= MARKET_OPEN_MIN)
            &
            (t < MARKET_CLOSE_MIN)
        ]

        return (
            df
            .dropna()
            .sort_values("timestamp")
            .reset_index(drop=True)
        )

    except Exception as e:

        why(f"Database read error: {e}")

        return None


# ============================================================
# INDICATORS
# ============================================================

def rsi(series, period=14):

    d = series.diff()

    gain = (
        d.clip(lower=0)
        .ewm(
            alpha=1 / period,
            adjust=False
        )
        .mean()
    )

    loss = (
        -d.clip(upper=0)
        .ewm(
            alpha=1 / period,
            adjust=False
        )
        .mean()
    )

    rs = gain / loss.replace(0, np.nan)

    return 100 - (100 / (1 + rs))


def atr(
    df,
    high_col,
    low_col,
    close_col,
    period=14
):

    previous_close = df[close_col].shift(1)

    tr = pd.concat(
        [
            df[high_col] - df[low_col],
            (
                df[high_col]
                - previous_close
            ).abs(),
            (
                df[low_col]
                - previous_close
            ).abs(),
        ],
        axis=1
    ).max(axis=1)

    return tr.ewm(
        alpha=1 / period,
        adjust=False
    ).mean()


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def prepare(raw):

    df = raw.copy()

    df["date"] = df["timestamp"].dt.date

    c = df["spx_close"]

    for n in (1, 3, 6, 12):

        df[f"spx_ret_{n}"] = c.pct_change(n)

        df[f"spy_ret_{n}"] = (
            df["spy_close"].pct_change(n)
        )

    df["rsi"] = rsi(c)

    df["spx_atr"] = atr(
        df,
        "spx_high",
        "spx_low",
        "spx_close"
    )

    df["atr_pct"] = (
        df["spx_atr"] / c
    )

    df["realized_vol"] = (
        c.pct_change()
        .rolling(12)
        .std()
        * sqrt(252 * 78)
        * 100
    )

    df["vol_spread"] = (
        df["vix"]
        - df["realized_vol"]
    )

    df["vix_chg_5"] = (
        df["vix"]
        .diff(5)
        .fillna(0)
    )

    df["vix_acceleration"] = (
        df["vix_chg_5"]
        .diff()
        .fillna(0)
    )

    # --------------------------------------------------------
    # PREVIOUS DAY LEVELS
    # --------------------------------------------------------

    daily_high = (
        df.groupby("date")["spx_high"]
        .max()
    )

    daily_low = (
        df.groupby("date")["spx_low"]
        .min()
    )

    previous_high = daily_high.shift(1)
    previous_low = daily_low.shift(1)

    df["prev_day_high"] = (
        df["date"]
        .map(previous_high)
        .fillna(df["spx_high"])
    )

    df["prev_day_low"] = (
        df["date"]
        .map(previous_low)
        .fillna(df["spx_low"])
    )

    df["dist_prev_high"] = (
        c - df["prev_day_high"]
    ) / c

    df["dist_prev_low"] = (
        c - df["prev_day_low"]
    ) / c

    # --------------------------------------------------------
    # OPENING RANGE
    # --------------------------------------------------------

    df["mins_open"] = (
        df["timestamp"].dt.hour * 60
        + df["timestamp"].dt.minute
        - MARKET_OPEN_MIN
    )

    first_30 = (
        (df["mins_open"] >= 0)
        &
        (df["mins_open"] < 30)
    )

    or_high = (
        df[first_30]
        .groupby("date")["spx_high"]
        .max()
    )

    or_low = (
        df[first_30]
        .groupby("date")["spx_low"]
        .min()
    )

    df["orh"] = df["date"].map(or_high)
    df["orl"] = df["date"].map(or_low)

    df.loc[
        df["mins_open"] < 30,
        ["orh", "orl"]
    ] = np.nan

    df["dist_from_orh"] = np.where(
        df["orh"].notna(),
        (c - df["orh"]) / c,
        0.0
    )

    df["dist_from_orl"] = np.where(
        df["orl"].notna(),
        (c - df["orl"]) / c,
        0.0
    )

    # --------------------------------------------------------
    # VOLUME
    # --------------------------------------------------------

    volume_mean = (
        df["spy_volume"]
        .rolling(
            30,
            min_periods=5
        )
        .mean()
    )

    volume_std = (
        df["spy_volume"]
        .rolling(
            30,
            min_periods=5
        )
        .std()
        .replace(0, 1)
    )

    df["volume_zscore"] = (
        df["spy_volume"]
        - volume_mean
    ) / volume_std

    # --------------------------------------------------------
    # TARGET
    # --------------------------------------------------------

    targets = []

    highs = df["spx_high"].values
    lows = df["spx_low"].values
    closes = c.values
    atrs = df["spx_atr"].values
    dates = df["date"].values

    for i in range(len(df)):

        if (
            i + HORIZON >= len(df)
            or dates[i] != dates[i + HORIZON]
        ):

            targets.append(np.nan)
            continue

        entry = closes[i]
        atr_value = atrs[i]

        if (
            not np.isfinite(atr_value)
            or atr_value <= 0
        ):

            targets.append(np.nan)
            continue

        upper = entry + 0.5 * atr_value
        lower = entry - 0.5 * atr_value

        hit_up = False
        hit_down = False

        for h in range(1, HORIZON + 1):

            idx = i + h

            up = highs[idx] >= upper
            down = lows[idx] <= lower

            # إذا ضرب الاثنين بنفس الشمعة:
            # نتجاهل الحالة بدلاً من افتراض الاتجاه
            if up and down:
                hit_up = False
                hit_down = False
                break

            if up:
                hit_up = True
                break

            if down:
                hit_down = True
                break

        if hit_up:
            targets.append(1.0)

        elif hit_down:
            targets.append(0.0)

        else:
            targets.append(np.nan)

    df["target"] = targets

    return df


# ============================================================
# FEATURES
# ============================================================

TREND_FEATURES = [
    "spx_ret_3",
    "spx_ret_6",
    "rsi",
    "atr_pct",
    "dist_from_orh",
    "dist_from_orl",
    "dist_prev_high",
]

MOMENTUM_FEATURES = [
    "spx_ret_1",
    "spy_ret_3",
    "spy_ret_6",
    "volume_zscore",
    "realized_vol",
]

VOLATILITY_FEATURES = [
    "vix",
    "vix_chg_5",
    "vix_acceleration",
    "vol_spread",
    "mins_open",
]


ALL_FEATURES = list(
    dict.fromkeys(
        TREND_FEATURES
        + MOMENTUM_FEATURES
        + VOLATILITY_FEATURES
    )
)


# ============================================================
# TRAINING
# ============================================================

def train_ensemble(df):

    needed = ALL_FEATURES + ["target"]

    data = (
        df
        .dropna(subset=needed)
        .reset_index(drop=True)
    )

    STATE["train_rows"] = len(data)

    if len(data) < MIN_TRAIN_ROWS:

        why(
            f"بيانات التدريب غير كافية: "
            f"{len(data)}/{MIN_TRAIN_ROWS}"
        )

        return {}, 0.0

    # --------------------------------------------------------
    # TIME-ORDERED SPLIT
    # --------------------------------------------------------

    n = len(data)

    train_end = int(n * 0.60)
    validation_end = int(n * 0.80)

    train_df = data.iloc[:train_end]
    validation_df = data.iloc[
        train_end:validation_end
    ]
    test_df = data.iloc[validation_end:]

    models = {}
    aucs = []

    subsets = {
        "trend": TREND_FEATURES,
        "momentum": MOMENTUM_FEATURES,
        "volatility": VOLATILITY_FEATURES,
    }

    for name, features in subsets.items():

        try:

            if train_df["target"].nunique() < 2:
                continue

            model = HistGradientBoostingClassifier(
                max_depth=3,
                learning_rate=0.03,
                max_iter=150,
                min_samples_leaf=20,
                l2_regularization=1.0,
                random_state=42
            )

            model.fit(
                train_df[features],
                train_df["target"]
            )

            # ------------------------------------------------
            # VALIDATION
            # ------------------------------------------------

            if (
                len(validation_df) > 20
                and validation_df["target"].nunique() > 1
            ):

                val_prob = model.predict_proba(
                    validation_df[features]
                )[:, 1]

                val_auc = roc_auc_score(
                    validation_df["target"],
                    val_prob
                )

            # ------------------------------------------------
            # OUT-OF-SAMPLE TEST
            # ------------------------------------------------

            if (
                len(test_df) > 20
                and test_df["target"].nunique() > 1
            ):

                test_prob = model.predict_proba(
                    test_df[features]
                )[:, 1]

                test_auc = roc_auc_score(
                    test_df["target"],
                    test_prob
                )

                aucs.append(test_auc)

                say(
                    f"MODEL {name}: "
                    f"TEST AUC={test_auc:.3f}"
                )

            models[name] = model

        except Exception as e:

            say(
                f"Model {name} training error: {e}"
            )

    if not models:
        return {}, 0.0

    real_auc = (
        float(np.mean(aucs))
        if aucs
        else 0.0
    )

    return models, real_auc


# ============================================================
# MARKET REGIME
# ============================================================

def detect_market_regime(df):

    if df is None or len(df) < 30:
        return "UNKNOWN"

    last = df.iloc[-1]

    atr_pct = float(last["atr_pct"])
    rv = float(last["realized_vol"])

    if atr_pct > 0.004 or rv > 35:
        return "HIGH_VOL"

    if atr_pct < 0.0015 and rv < 12:
        return "LOW_VOL"

    return "NORMAL"


# ============================================================
# ENSEMBLE SIGNAL
# ============================================================

def calculate_ensemble_signal(
    df,
    regime
):

    if df is None or len(df) < 50:

        return {
            "direction": "WAIT",
            "probability": 0.50,
            "reason": "بيانات غير كافية"
        }

    last = df.iloc[-1]

    age = (
        now_ny()
        - last["timestamp"]
    ).total_seconds() / 60

    if age > MAX_BAR_AGE_MIN:

        return {
            "direction": "WAIT",
            "probability": 0.50,
            "reason": (
                f"آخر شمعة قديمة "
                f"{age:.1f} دقيقة"
            )
        }

    if not STATE["models"]:

        return {
            "direction": "WAIT",
            "probability": 0.50,
            "reason": "النموذج لم يكتمل تدريبه"
        }

    probabilities = []

    try:

        for name, features in [
            ("trend", TREND_FEATURES),
            ("momentum", MOMENTUM_FEATURES),
            ("volatility", VOLATILITY_FEATURES),
        ]:

            model = STATE["models"].get(name)

            if model is None:
                continue

            p = model.predict_proba(
                pd.DataFrame(
                    [last[features]]
                )
            )[0][1]

            probabilities.append(float(p))

    except Exception as e:

        return {
            "direction": "WAIT",
            "probability": 0.50,
            "reason": f"Model error: {e}"
        }

    if len(probabilities) < 2:

        return {
            "direction": "WAIT",
            "probability": 0.50,
            "reason": "عدد النماذج الفعالة غير كافٍ"
        }

    # --------------------------------------------------------
    # Ensemble
    # --------------------------------------------------------

    p_up = float(
        np.mean(probabilities)
    )

    p_down = 1.0 - p_up

    # --------------------------------------------------------
    # REGIME FILTER
    # --------------------------------------------------------

    if regime == "LOW_VOL":

        # لا نمنع تماماً، لكن نحتاج ثقة أعلى
        threshold = 0.60

    elif regime == "HIGH_VOL":

        # في التقلب العالي نحتاج تأكيد أكبر
        threshold = 0.60

    else:

        threshold = MIN_PROBABILITY

    if p_up >= threshold:

        return {
            "direction": "CALL",
            "probability": p_up,
            "reason": (
                f"Ensemble bullish "
                f"{p_up*100:.1f}%"
            )
        }

    if p_down >= threshold:

        return {
            "direction": "PUT",
            "probability": p_down,
            "reason": (
                f"Ensemble bearish "
                f"{p_down*100:.1f}%"
            )
        }

    return {
        "direction": "WAIT",
        "probability": max(
            p_up,
            p_down
        ),
        "reason": (
            f"الثقة غير كافية "
            f"(UP={p_up*100:.1f}% "
            f"DOWN={p_down*100:.1f}%)"
        )
    }


# ============================================================
# STRIKE
# ============================================================

def suggested_strike(
    spot,
    direction
):

    if direction == "CALL":

        return (
            ceil(spot / STRIKE_STEP)
            * STRIKE_STEP
            + STRIKE_STEP
            * (STRIKE_OFFSET_STEPS - 1)
        )

    return (
        floor(spot / STRIKE_STEP)
        * STRIKE_STEP
        - STRIKE_STEP
        * (STRIKE_OFFSET_STEPS - 1)
    )


# ============================================================
# RECOMMENDATION
# ============================================================

def build_recommendation(
    df,
    regime
):

    if not market_time_ok():

        return {
            "status": "WAIT",
            "reason": "السوق مغلق"
        }

    if not session_allowed():

        return {
            "status": "WAIT",
            "reason": "خارج نافذة التوصيات"
        }

    signal = calculate_ensemble_signal(
        df,
        regime
    )

    if signal["direction"] == "WAIT":

        return {
            "status": "WAIT",
            "probability": signal["probability"],
            "reason": signal["reason"]
        }

    last = df.iloc[-1]

    spot = float(
        last["spx_close"]
    )

    atr_value = float(
        last["spx_atr"]
    )

    direction = signal["direction"]

    strike = suggested_strike(
        spot,
        direction
    )

    if direction == "CALL":

        target = (
            spot
            + 0.5 * atr_value
        )

        stop = (
            spot
            - 0.5 * atr_value
        )

    else:

        target = (
            spot
            - 0.5 * atr_value
        )

        stop = (
            spot
            + 0.5 * atr_value
        )

    return {

        "status": direction,

        "probability":
            signal["probability"],

        "reason":
            signal["reason"],

        "spx":
            spot,

        "strike":
            strike,

        "target":
            target,

        "stop":
            stop,

        "data_source":
            STATE["data_source"],

        "time":
            now_ny(),

    }


# ============================================================
# SIGNAL DUPLICATE FILTER
# ============================================================

def should_send_signal(rec):

    direction = rec["status"]

    now = now_ny()

    last_direction = STATE[
        "last_signal"
    ]

    last_time = STATE[
        "last_signal_time"
    ]

    if (
        last_direction == direction
        and last_time is not None
    ):

        elapsed = (
            now - last_time
        ).total_seconds() / 60

        if elapsed < SIGNAL_COOLDOWN_MIN:

            return False

    return True


# ============================================================
# SEND RECOMMENDATION
# ============================================================

def send_recommendation(rec):

    direction = rec["status"]

    emoji = (
        "🟢" if direction == "CALL"
        else "🔴"
    )

    msg = f"""
{emoji} SPX 0DTE — {direction}

📊 الثقة: {rec["probability"]*100:.1f}%

SPX: {rec["spx"]:.2f}

🎯 العقد المقترح:
{direction} {rec["strike"]:.0f} 0DTE

🎯 هدف SPX:
{rec["target"]:.2f}

🛑 وقف SPX:
{rec["stop"]:.2f}

📈 Regime:
{STATE.get("regime", "NORMAL")}

🧠 السبب:
{rec["reason"]}

📡 البيانات:
{rec["data_source"]}

⚠️ توصية فقط — لا يوجد تنفيذ آلي.
""".strip()

    say(
        msg.replace(
            "\n",
            " | "
        )
    )

    ok = send_telegram(msg)

    if ok:

        STATE["last_signal"] = direction
        STATE["last_signal_time"] = now_ny()
        STATE["signals_today"] += 1

    return ok


# ============================================================
# WAIT REPORT
# ============================================================

def maybe_send_wait(reason):

    now = now_ny()

    last_wait = STATE[
        "last_wait_time"
    ]

    if last_wait is not None:

        elapsed = (
            now - last_wait
        ).total_seconds() / 60

        if elapsed < WAIT_REPORT_MIN:
            return

    msg = (
        "⚪ SPX 0DTE — WAIT\n\n"
        f"السبب: {reason}\n\n"
        "لا توجد إشارة CALL/PUT قوية "
        "حاليًا.\n"
        "النظام مستمر بالمراقبة."
    )

    send_telegram(msg)

    STATE[
        "last_wait_time"
    ] = now


# ============================================================
# STARTUP
# ============================================================

def startup_test():

    say(
        "SPX v14.4 — "
        "بدء التشغيل"
    )

    if not APCA_KEY or not APCA_SECRET:

        say(
            "⚠️ مفاتيح Alpaca غير موجودة"
        )

    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:

        say(
            "⚠️ Telegram غير مهيأ"
        )

    else:

        send_telegram(
            "✅ SPX v14.4 اشتغل بنجاح\n"
            "🧠 ML Recommendation Engine\n"
            "📡 المراقبة بدأت\n"
            "⚠️ توصيات فقط — بدون تنفيذ أوامر"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    startup_test()

    reset_daily_state()

    # --------------------------------------------------------
    # WAIT FOR DATA
    # --------------------------------------------------------

    while True:

        df_raw = get_data()

        if (
            df_raw is not None
            and len(df_raw) >= 100
        ):

            break

        why(
            "بانتظار بيانات كافية "
            "لبناء النموذج..."
        )

        time.sleep(30)

    # --------------------------------------------------------
    # FIRST TRAIN
    # --------------------------------------------------------

    say(
        f"بيانات أولية: "
        f"{len(df_raw)} شمعة"
    )

    df_prep = prepare(
        df_raw
    )

    models, auc = train_ensemble(
        df_prep
    )

    if models:

        STATE.update(
            models=models,
            auc=auc,
            last_train=now_ny()
        )

        send_telegram(
            "📚 تم تدريب النموذج\n\n"
            f"Training rows: "
            f"{STATE['train_rows']}\n"
            f"Out-of-sample AUC: "
            f"{auc:.3f}\n\n"
            "ملاحظة: AUC يعرض "
            "النتيجة الحقيقية فقط."
        )

    else:

        send_telegram(
            "⚠️ لم يكتمل تدريب ML بعد.\n"
            "النظام مستمر في جمع البيانات."
        )

    # --------------------------------------------------------
    # MAIN LOOP
    # --------------------------------------------------------

    while True:

        try:

            reset_daily_state()

            # خارج السوق
            if not market_time_ok():

                time.sleep(30)

                continue

            # ------------------------------------------------
            # GET DATA
            # ------------------------------------------------

            df_raw = get_data()

            if (
                df_raw is None
                or len(df_raw) < 50
            ):

                why(
                    "بيانات غير كافية "
                    "لإنتاج إشارة"
                )

                time.sleep(
                    POLL_SECONDS
                )

                continue

            # ------------------------------------------------
            # PREPARE
            # ------------------------------------------------

            df_prep = prepare(
                df_raw
            )

            regime = detect_market_regime(
                df_prep
            )

            STATE["regime"] = regime

            # ------------------------------------------------
            # RETRAIN
            # ------------------------------------------------

            if STATE["last_train"] is None:

                retrain = True

            else:

                retrain = (
                    (
                        now_ny()
                        - STATE["last_train"]
                    ).total_seconds()
                    >
                    RETRAIN_EVERY_MIN * 60
                )

            if retrain:

                say(
                    "🔄 إعادة تدريب ML..."
                )

                models, auc = train_ensemble(
                    df_prep
                )

                if models:

                    STATE.update(
                        models=models,
                        auc=auc,
                        last_train=now_ny()
                    )

                    say(
                        f"Training complete | "
                        f"AUC={auc:.3f}"
                    )

            # ------------------------------------------------
            # LIVE BAR
            # ------------------------------------------------

            last = df_prep.iloc[-1]

            STATE["last_spot"] = float(
                last["spx_close"]
            )

            # ------------------------------------------------
            # RECOMMENDATION
            # ------------------------------------------------

            rec = build_recommendation(
                df_prep,
                regime
            )

            # ------------------------------------------------
            # SIGNAL
            # ------------------------------------------------

            if rec["status"] in (
                "CALL",
                "PUT"
            ):

                if should_send_signal(rec):

                    send_recommendation(
                        rec
                    )

                else:

                    why(
                        "تم منع تكرار نفس "
                        "التوصية خلال فترة التهدئة"
                    )

            else:

                maybe_send_wait(
                    rec.get(
                        "reason",
                        "لا توجد إشارة"
                    )
                )

            time.sleep(
                POLL_SECONDS
            )

        except KeyboardInterrupt:

            say(
                "تم إيقاف النظام يدويًا."
            )

            break

        except Exception as e:

            say(
                f"MAIN LOOP ERROR: {e}"
            )

            time.sleep(
                POLL_SECONDS
            )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()