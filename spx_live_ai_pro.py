# ============================================================
# SPX 0DTE AI ADVISOR v14.5
# ============================================================
# Recommendation Only
#
# SPY 5m -> SPX Proxy -> Features -> ML Ensemble
# -> Historical Training -> CALL / PUT / WAIT
# -> Telegram
#
# لا يوجد تنفيذ أوامر.
# ============================================================

import os
import sys
import time
import json
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

BOT_VERSION = "v14.5"

SPY_SYMBOL = "SPY"
TIMEFRAME = "5Min"

HISTORY_DAYS = 60

# لا ننزل بهذا الرقم فقط لإجبار البوت على إصدار إشارات
MIN_TRAIN_ROWS = 150

HORIZON = 6

ATR_TARGET = 0.50

MIN_PROBABILITY = 0.57

HIGH_VOL_THRESHOLD = 0.018
LOW_VOL_THRESHOLD = 0.008

SIGNAL_COOLDOWN_MINUTES = 20
WAIT_MESSAGE_MINUTES = 15

POLL_SECONDS = 30

NO_TRADE_FIRST_MIN = 10
NO_TRADE_LAST_MIN = 30

# عدد المحاولات عند بداية التشغيل
STARTUP_RETRIES = 10

# إذا كان أقل من هذا، لا ندعي أن النموذج ممتاز
MIN_REASONABLE_AUC = 0.50


# ============================================================
# ENV
# ============================================================

ALPACA_API_KEY = os.getenv("ALPACA_API_KEY", "")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "")

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

DATA_URL = "https://data.alpaca.markets/v2/stocks/bars"

NY = ZoneInfo("America/New_York")


# ============================================================
# STATE
# ============================================================

STATE = {
    "models": [],
    "feature_columns": [],
    "auc": None,
    "last_signal": None,
    "last_signal_time": None,
    "last_wait_time": None,
    "trained": False,
    "last_training_rows": 0,
    "last_data_update": None,
}


# ============================================================
# LOG
# ============================================================

def log(message):
    now = datetime.now(NY).strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] {message}", flush=True)


# ============================================================
# TELEGRAM
# ============================================================

def telegram_send(message):

    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        log("[TELEGRAM] مفاتيح Telegram غير موجودة")
        return False

    url = (
        f"https://api.telegram.org/bot"
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

        log(f"[TELEGRAM ERROR] {r.status_code} {r.text[:300]}")
        return False

    except Exception as e:
        log(f"[TELEGRAM ERROR] {e}")
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
# ALPACA HISTORICAL DATA
# ============================================================

def download_bars(
    symbol=SPY_SYMBOL,
    timeframe=TIMEFRAME,
    days=None,
    limit=10000
):

    if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
        log("[DATA ERROR] ALPACA_API_KEY / ALPACA_SECRET_KEY غير موجودة")
        return pd.DataFrame()

    end_dt = now_ny()

    params = {
        "symbols": symbol,
        "timeframe": timeframe,
        "limit": min(limit, 10000),
        "feed": "iex",
        "adjustment": "raw",
    }

    if days is not None:
        start_dt = end_dt - timedelta(days=days)

        params["start"] = iso_utc(start_dt)
        params["end"] = iso_utc(end_dt)

    headers = {
        "APCA-API-KEY-ID": ALPACA_API_KEY,
        "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY,
    }

    all_rows = []

    page_token = None

    max_pages = 20

    try:

        for page in range(max_pages):

            request_params = dict(params)

            if page_token:
                request_params["page_token"] = page_token

            r = requests.get(
                DATA_URL,
                headers=headers,
                params=request_params,
                timeout=30
            )

            if not r.ok:

                log(
                    f"[DATA ERROR] HTTP {r.status_code}: "
                    f"{r.text[:500]}"
                )

                break

            data = r.json()

            bars = data.get("bars", {})

            symbol_rows = bars.get(symbol, [])

            if symbol_rows:
                all_rows.extend(symbol_rows)

            page_token = data.get("next_page_token")

            if not page_token:
                break

            log(
                f"[DATA] صفحة تاريخية إضافية: "
                f"{page + 1}"
            )

        if not all_rows:

            log("[DATA] لم تصل أي شموع")

            return pd.DataFrame()

        df = pd.DataFrame(all_rows)

        if "t" not in df.columns:
            log("[DATA ERROR] تنسيق البيانات غير متوقع")
            return pd.DataFrame()

        df = df.rename(
            columns={
                "t": "timestamp",
                "o": "open",
                "h": "high",
                "l": "low",
                "c": "close",
                "v": "volume",
                "n": "trade_count",
                "vw": "vwap",
            }
        )

        df["timestamp"] = pd.to_datetime(
            df["timestamp"],
            utc=True
        )

        df = df.sort_values("timestamp")

        df = df.drop_duplicates(
            subset=["timestamp"]
        )

        numeric_cols = [
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]

        for col in numeric_cols:

            if col in df.columns:
                df[col] = pd.to_numeric(
                    df[col],
                    errors="coerce"
                )

        df = df.dropna(
            subset=[
                "open",
                "high",
                "low",
                "close",
                "volume",
            ]
        )

        log(
            f"[DATA] {symbol}: "
            f"{len(df)} شمعة 5m تم تحميلها"
        )

        return df.reset_index(drop=True)

    except Exception as e:

        log(f"[DATA ERROR] {e}")

        return pd.DataFrame()


# ============================================================
# LOCAL DATA
# ============================================================

LOCAL_FILE = "spx_advisor_data.csv"


def save_local_data(df):

    try:
        df.to_csv(
            LOCAL_FILE,
            index=False
        )

        STATE["last_data_update"] = now_ny()

    except Exception as e:
        log(f"[SAVE ERROR] {e}")


def load_local_data():

    if not os.path.exists(LOCAL_FILE):
        return pd.DataFrame()

    try:

        df = pd.read_csv(
            LOCAL_FILE
        )

        if "timestamp" in df.columns:

            df["timestamp"] = pd.to_datetime(
                df["timestamp"],
                utc=True
            )

        return df

    except Exception as e:

        log(f"[LOAD ERROR] {e}")

        return pd.DataFrame()


# ============================================================
# UPDATE DATABASE
# ============================================================

def update_local_database(initial=False):

    if initial:

        log(
            f"[DATA] جلب تاريخ {HISTORY_DAYS} يوم "
            f"للتدريب الأول..."
        )

        df = download_bars(
            symbol=SPY_SYMBOL,
            timeframe=TIMEFRAME,
            days=HISTORY_DAYS,
            limit=10000
        )

    else:

        df = download_bars(
            symbol=SPY_SYMBOL,
            timeframe=TIMEFRAME,
            days=7,
            limit=5000
        )

    if df.empty:
        return pd.DataFrame()

    # ========================================================
    # SPY -> SPX PROXY
    # ========================================================

    df["spx_open"] = df["open"] * 10.0
    df["spx_high"] = df["high"] * 10.0
    df["spx_low"] = df["low"] * 10.0
    df["spx_close"] = df["close"] * 10.0

    df["spx_volume"] = df["volume"]

    # ========================================================
    # MERGE WITH LOCAL
    # ========================================================

    old = load_local_data()

    if not old.empty:

        combined = pd.concat(
            [old, df],
            ignore_index=True
        )

    else:

        combined = df

    combined["timestamp"] = pd.to_datetime(
        combined["timestamp"],
        utc=True
    )

    combined = combined.sort_values(
        "timestamp"
    )

    combined = combined.drop_duplicates(
        subset=["timestamp"],
        keep="last"
    )

    # Keep roughly 90 days locally
    cutoff = (
        pd.Timestamp.now(tz="UTC")
        - pd.Timedelta(days=90)
    )

    combined = combined[
        combined["timestamp"] >= cutoff
    ]

    combined = combined.reset_index(drop=True)

    save_local_data(combined)

    log(
        f"[DATA] إجمالي البيانات المحلية: "
        f"{len(combined)} شمعة"
    )

    return combined


# ============================================================
# SESSION FILTER
# ============================================================

def regular_session(df):

    if df.empty:
        return df

    x = df.copy()

    x["ny_time"] = (
        x["timestamp"]
        .dt.tz_convert(NY)
    )

    x["time_only"] = (
        x["ny_time"].dt.hour * 60
        + x["ny_time"].dt.minute
    )

    start_min = 9 * 60 + 30
    end_min = 16 * 60

    x = x[
        (x["time_only"] >= start_min)
        & (x["time_only"] <= end_min)
    ]

    return x.drop(
        columns=["time_only"],
        errors="ignore"
    )


# ============================================================
# RSI
# ============================================================

def calculate_rsi(series, period=14):

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

    rs = avg_gain / avg_loss.replace(
        0,
        np.nan
    )

    rsi = 100 - (
        100 / (1 + rs)
    )

    return rsi


# ============================================================
# ATR
# ============================================================

def calculate_atr(df, period=14):

    high = df["spx_high"]
    low = df["spx_low"]
    close = df["spx_close"]

    prev_close = close.shift(1)

    tr1 = high - low

    tr2 = (
        high - prev_close
    ).abs()

    tr3 = (
        low - prev_close
    ).abs()

    tr = pd.concat(
        [tr1, tr2, tr3],
        axis=1
    ).max(axis=1)

    return tr.rolling(
        period
    ).mean()


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


def prepare_features(df):

    if df.empty:
        return pd.DataFrame()

    x = regular_session(
        df.copy()
    )

    if len(x) < 100:
        return pd.DataFrame()

    close = x["spx_close"]

    # --------------------------------------------------------
    # Returns
    # --------------------------------------------------------

    x["ret_1"] = close.pct_change(1)

    x["ret_3"] = close.pct_change(3)

    x["ret_6"] = close.pct_change(6)

    x["ret_12"] = close.pct_change(12)

    # --------------------------------------------------------
    # RSI
    # --------------------------------------------------------

    x["rsi"] = calculate_rsi(
        close,
        14
    )

    # --------------------------------------------------------
    # ATR
    # --------------------------------------------------------

    x["atr"] = calculate_atr(
        x,
        14
    )

    x["atr_pct"] = (
        x["atr"] / close
    )

    # --------------------------------------------------------
    # Realized volatility
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Candle range
    # --------------------------------------------------------

    x["range_pct"] = (
        x["spx_high"]
        - x["spx_low"]
    ) / close

    # --------------------------------------------------------
    # Moving averages
    # --------------------------------------------------------

    ma20 = close.rolling(20).mean()

    ma50 = close.rolling(50).mean()

    x["close_vs_ma20"] = (
        close / ma20
    ) - 1

    x["close_vs_ma50"] = (
        close / ma50
    ) - 1

    # --------------------------------------------------------
    # Volume Z
    # --------------------------------------------------------

    volume_mean = (
        x["spx_volume"]
        .rolling(30)
        .mean()
    )

    volume_std = (
        x["spx_volume"]
        .rolling(30)
        .std()
    )

    x["volume_z"] = (
        x["spx_volume"]
        - volume_mean
    ) / volume_std.replace(
        0,
        np.nan
    )

    # --------------------------------------------------------
    # Momentum
    # --------------------------------------------------------

    x["momentum_3"] = (
        close
        - close.shift(3)
    ) / close.shift(3)

    x["momentum_6"] = (
        close
        - close.shift(6)
    ) / close.shift(6)

    # --------------------------------------------------------
    # Time
    # --------------------------------------------------------

    ny_time = (
        x["timestamp"]
        .dt.tz_convert(NY)
    )

    minutes = (
        ny_time.dt.hour * 60
        + ny_time.dt.minute
    )

    minutes_from_open = (
        minutes - (9 * 60 + 30)
    )

    angle = (
        2
        * np.pi
        * minutes_from_open
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
    #
    # 1 = السعر يضرب +0.5 ATR قبل -0.5 ATR
    # 0 = السعر يضرب -0.5 ATR قبل +0.5 ATR
    #
    # إذا ضرب الاثنين داخل نفس الشمعة:
    # نستبعدها حتى لا نخلق انحيازًا مصطنعًا.
    # ========================================================

    targets = []

    for i in range(len(x)):

        if i + HORIZON >= len(x):

            targets.append(np.nan)

            continue

        entry = x["spx_close"].iloc[i]

        atr = x["atr"].iloc[i]

        if (
            not np.isfinite(entry)
            or not np.isfinite(atr)
            or atr <= 0
        ):

            targets.append(np.nan)

            continue

        upper = entry + (
            ATR_TARGET * atr
        )

        lower = entry - (
            ATR_TARGET * atr
        )

        result = np.nan

        for j in range(
            i + 1,
            min(
                i + HORIZON + 1,
                len(x)
            )
        ):

            hi = x["spx_high"].iloc[j]
            lo = x["spx_low"].iloc[j]

            hit_up = hi >= upper
            hit_down = lo <= lower

            if hit_up and hit_down:

                result = np.nan
                break

            if hit_up:

                result = 1
                break

            if hit_down:

                result = 0
                break

        targets.append(result)

    x["target"] = targets

    return x


# ============================================================
# TRAIN ENSEMBLE
# ============================================================

def train_ensemble(df):

    if df.empty:

        log(
            "[TRAIN] لا توجد بيانات"
        )

        return False

    required = (
        FEATURE_COLUMNS
        + ["target"]
    )

    work = df.dropna(
        subset=required
    ).copy()

    work = work.replace(
        [np.inf, -np.inf],
        np.nan
    )

    work = work.dropna(
        subset=required
    )

    if len(work) < MIN_TRAIN_ROWS:

        log(
            "[TRAIN] بيانات غير كافية: "
            f"{len(work)}/{MIN_TRAIN_ROWS}"
        )

        return False

    # ========================================================
    # TIME ORDERED SPLIT
    # ========================================================

    n = len(work)

    train_end = int(
        n * 0.60
    )

    val_end = int(
        n * 0.80
    )

    train = work.iloc[
        :train_end
    ]

    val = work.iloc[
        train_end:val_end
    ]

    test = work.iloc[
        val_end:
    ]

    X_train = train[
        FEATURE_COLUMNS
    ]

    y_train = train[
        "target"
    ].astype(int)

    X_val = val[
        FEATURE_COLUMNS
    ]

    y_val = val[
        "target"
    ].astype(int)

    X_test = test[
        FEATURE_COLUMNS
    ]

    y_test = test[
        "target"
    ].astype(int)

    # ========================================================
    # CHECK BOTH CLASSES
    # ========================================================

    if (
        y_train.nunique() < 2
        or y_test.nunique() < 2
    ):

        log(
            "[TRAIN] لا توجد فئتان كافيتان "
            "CALL/PUT في البيانات"
        )

        return False

    models = []

    seeds = [
        11,
        29,
        71,
    ]

    for seed in seeds:

        base = HistGradientBoostingClassifier(
            learning_rate=0.045,
            max_iter=250,
            max_leaf_nodes=15,
            min_samples_leaf=15,
            l2_regularization=1.0,
            random_state=seed
        )

        try:

            model = CalibratedClassifierCV(
                base,
                method="sigmoid",
                cv=3
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
                f"[TRAIN] فشل نموذج: {e}"
            )

    if not models:

        log(
            "[TRAIN] فشل تدريب جميع النماذج"
        )

        return False

    # ========================================================
    # TEST AUC
    # ========================================================

    probabilities = []

    for model in models:

        p = model.predict_proba(
            X_test
        )[:, 1]

        probabilities.append(p)

    mean_probability = np.mean(
        probabilities,
        axis=0
    )

    try:

        auc = roc_auc_score(
            y_test,
            mean_probability
        )

    except Exception:

        auc = np.nan

    # ========================================================
    # SAVE STATE
    # ========================================================

    STATE["models"] = models

    STATE["feature_columns"] = (
        FEATURE_COLUMNS.copy()
    )

    STATE["auc"] = (
        float(auc)
        if np.isfinite(auc)
        else None
    )

    STATE["trained"] = True

    STATE["last_training_rows"] = len(work)

    log(
        f"[TRAIN] اكتمل التدريب | "
        f"usable={len(work)} | "
        f"test={len(test)} | "
        f"AUC={auc:.3f}"
    )

    telegram_send(
        "🧠 SPX AI Advisor\n\n"
        "✅ اكتمل التدريب\n"
        f"📊 بيانات التدريب: {len(work)}\n"
        f"🧪 Test AUC: {auc:.3f}\n\n"
        "البوت الآن جاهز لتحليل السوق."
    )

    return True


# ============================================================
# MARKET REGIME
# ============================================================

def detect_market_regime(df):

    if df.empty:
        return "UNKNOWN"

    x = df.dropna(
        subset=["vol_24"]
    )

    if x.empty:
        return "UNKNOWN"

    vol = float(
        x["vol_24"].iloc[-1]
    )

    if vol >= HIGH_VOL_THRESHOLD:

        return "HIGH_VOL"

    if vol <= LOW_VOL_THRESHOLD:

        return "LOW_VOL"

    return "NORMAL"


# ============================================================
# SIGNAL
# ============================================================

def calculate_signal(df):

    if not STATE["trained"]:

        return {
            "signal": "WAIT",
            "probability": 0.50,
            "reason": "النموذج لم يكتمل تدريبه",
        }

    x = df.dropna(
        subset=FEATURE_COLUMNS
    ).copy()

    if x.empty:

        return {
            "signal": "WAIT",
            "probability": 0.50,
            "reason": "لا توجد شمعة صالحة للتحليل",
        }

    latest = x.iloc[
        [-1]
    ][FEATURE_COLUMNS]

    probabilities = []

    for model in STATE["models"]:

        try:

            p = model.predict_proba(
                latest
            )[0][1]

            probabilities.append(
                float(p)
            )

        except Exception:
            pass

    if not probabilities:

        return {
            "signal": "WAIT",
            "probability": 0.50,
            "reason": "فشل حساب احتمالية النماذج",
        }

    probability = float(
        np.mean(probabilities)
    )

    regime = detect_market_regime(
        df
    )

    threshold = MIN_PROBABILITY

    # في التقلب العالي نكون أكثر تحفظًا
    if regime == "HIGH_VOL":

        threshold = 0.60

    # ========================================================
    # DIRECTION
    # ========================================================

    if probability >= threshold:

        signal = "CALL"

        confidence = probability

        reason = (
            f"النموذج يميل للصعود "
            f"({probability:.1%})"
        )

    elif probability <= (
        1 - threshold
    ):

        signal = "PUT"

        confidence = 1 - probability

        reason = (
            f"النموذج يميل للهبوط "
            f"({1-probability:.1%})"
        )

    else:

        signal = "WAIT"

        confidence = max(
            probability,
            1 - probability
        )

        reason = (
            f"الاحتمالية غير كافية "
            f"(CALL={probability:.1%})"
        )

    return {
        "signal": signal,
        "probability": probability,
        "confidence": confidence,
        "regime": regime,
        "reason": reason,
    }


# ============================================================
# MARKET TIME FILTER
# ============================================================

def trading_window_status():

    now = now_ny()

    minutes = (
        now.hour * 60
        + now.minute
    )

    market_open = (
        9 * 60 + 30
    )

    market_close = (
        16 * 60
    )

    if minutes < market_open:

        return False, "السوق لم يفتح بعد"

    if minutes >= market_close:

        return False, "السوق مغلق"

    from_open = (
        minutes - market_open
    )

    to_close = (
        market_close - minutes
    )

    if from_open < NO_TRADE_FIRST_MIN:

        return (
            False,
            "أول دقائق السوق"
        )

    if to_close <= NO_TRADE_LAST_MIN:

        return (
            False,
            "آخر دقائق السوق"
        )

    return True, "داخل نافذة التداول"


# ============================================================
# SPX PRICE
# ============================================================

def current_spx(df):

    if df.empty:
        return np.nan

    return float(
        df["spx_close"].iloc[-1]
    )


# ============================================================
# RECOMMENDATION
# ============================================================

def send_signal(
    signal_data,
    df
):

    signal = signal_data["signal"]

    if signal == "WAIT":

        return False

    spx = current_spx(df)

    if not np.isfinite(spx):

        return False

    atr = float(
        df["atr"].iloc[-1]
    )

    if not np.isfinite(atr) or atr <= 0:

        return False

    confidence = (
        signal_data["confidence"]
    )

    # ========================================================
    # SUGGESTED LEVELS
    # ========================================================

    if signal == "CALL":

        target = spx + (
            ATR_TARGET * atr
        )

        stop = spx - (
            0.35 * atr
        )

        direction = "🟢 CALL"

    else:

        target = spx - (
            ATR_TARGET * atr
        )

        stop = spx + (
            0.35 * atr
        )

        direction = "🔴 PUT"

    # تقريب strike إلى 5 نقاط
    suggested_strike = (
        round(spx / 5)
        * 5
    )

    auc = STATE["auc"]

    auc_text = (
        f"{auc:.3f}"
        if auc is not None
        else "N/A"
    )

    message = (
        "🚨 SPX 0DTE AI RECOMMENDATION\n"
        "━━━━━━━━━━━━━━━━━━\n"
        f"{direction}\n\n"
        f"📍 SPX Proxy: {spx:.2f}\n"
        f"🎯 Suggested Strike: {suggested_strike}\n"
        f"📈 Confidence: {confidence:.1%}\n"
        f"🧪 Test AUC: {auc_text}\n"
        f"🌡 Regime: {signal_data['regime']}\n\n"
        f"🎯 Target: {target:.2f}\n"
        f"🛑 Stop: {stop:.2f}\n\n"
        f"🧠 السبب: {signal_data['reason']}\n\n"
        "⚠️ توصية فقط — لا يوجد تنفيذ أوامر\n"
        "⚠️ SPX محسوب تقريبيًا من SPY × 10"
    )

    telegram_send(message)

    STATE["last_signal"] = signal

    STATE["last_signal_time"] = now_ny()

    log(
        f"[SIGNAL] {signal} | "
        f"confidence={confidence:.1%} | "
        f"SPX={spx:.2f}"
    )

    return True


# ============================================================
# WAIT MESSAGE
# ============================================================

def maybe_send_wait(
    reason,
    force=False
):

    now = now_ny()

    last = STATE["last_wait_time"]

    if not force and last is not None:

        elapsed = (
            now - last
        ).total_seconds() / 60

        if elapsed < WAIT_MESSAGE_MINUTES:

            return

    message = (
        "⚪ SPX AI — WAIT\n"
        "━━━━━━━━━━━━━━━━━━\n"
        f"السبب: {reason}\n\n"
        "البوت يراقب السوق ولا يصدر "
        "إشارة CALL/PUT إلا عند توفر "
        "احتمالية كافية."
    )

    telegram_send(message)

    STATE["last_wait_time"] = now

    log(
        f"[WHY-WAIT] {reason}"
    )


# ============================================================
# DUPLICATE SIGNAL FILTER
# ============================================================

def should_send_signal(signal):

    if signal not in [
        "CALL",
        "PUT"
    ]:

        return False

    last_signal = (
        STATE["last_signal"]
    )

    last_time = (
        STATE["last_signal_time"]
    )

    if (
        last_signal is None
        or last_time is None
    ):

        return True

    if signal != last_signal:

        return True

    elapsed = (
        now_ny() - last_time
    ).total_seconds() / 60

    return (
        elapsed
        >= SIGNAL_COOLDOWN_MINUTES
    )


# ============================================================
# INITIAL DATA + TRAINING
# ============================================================

def initialize_model():

    log(
        f"SPX {BOT_VERSION} "
        "— بدء التشغيل"
    )

    telegram_send(
        f"🤖 SPX AI Advisor {BOT_VERSION}\n\n"
        "بدأ التشغيل.\n"
        "📚 جاري جلب البيانات التاريخية "
        "وتدريب النموذج..."
    )

    for attempt in range(
        1,
        STARTUP_RETRIES + 1
    ):

        log(
            f"[STARTUP] محاولة {attempt}/"
            f"{STARTUP_RETRIES}"
        )

        df_raw = update_local_database(
            initial=True
        )

        if df_raw.empty:

            maybe_send_wait(
                "تعذر جلب بيانات SPY من Alpaca",
                force=(attempt == 1)
            )

            time.sleep(10)

            continue

        df_prepared = prepare_features(
            df_raw
        )

        if df_prepared.empty:

            maybe_send_wait(
                "البيانات وصلت لكن لا توجد "
                "شموع كافية بعد تنظيفها",
                force=(attempt == 1)
            )

            time.sleep(10)

            continue

        usable = df_prepared.dropna(
            subset=FEATURE_COLUMNS + ["target"]
        )

        log(
            f"[STARTUP] raw={len(df_raw)} | "
            f"prepared={len(df_prepared)} | "
            f"usable={len(usable)}"
        )

        if len(usable) < MIN_TRAIN_ROWS:

            maybe_send_wait(
                f"بيانات التدريب غير كافية: "
                f"{len(usable)}/{MIN_TRAIN_ROWS}",
                force=(attempt == 1)
            )

            time.sleep(10)

            continue

        if train_ensemble(
            df_prepared
        ):

            log(
                "[STARTUP] النموذج جاهز ✅"
            )

            return True

        time.sleep(10)

    log(
        "[STARTUP] فشل بناء النموذج "
        "بعد عدة محاولات"
    )

    telegram_send(
        "❌ SPX AI Advisor\n\n"
        "تعذر بناء النموذج.\n"
        "تحقق من مفاتيح Alpaca وبيانات السوق."
    )

    return False


# ============================================================
# LIVE LOOP
# ============================================================

def run_live():

    last_retrain = now_ny()

    retrain_every_minutes = 60

    while True:

        try:

            # ------------------------------------------------
            # تحديث البيانات
            # ------------------------------------------------

            raw = update_local_database(
                initial=False
            )

            if raw.empty:

                maybe_send_wait(
                    "لا توجد بيانات حديثة من Alpaca"
                )

                time.sleep(
                    POLL_SECONDS
                )

                continue

            prepared = prepare_features(
                raw
            )

            if prepared.empty:

                maybe_send_wait(
                    "لا توجد بيانات صالحة للتحليل"
                )

                time.sleep(
                    POLL_SECONDS
                )

                continue

            # ------------------------------------------------
            # إعادة التدريب كل ساعة
            # ------------------------------------------------

            elapsed_retrain = (
                now_ny()
                - last_retrain
            ).total_seconds() / 60

            if (
                elapsed_retrain
                >= retrain_every_minutes
            ):

                log(
                    "[TRAIN] إعادة تدريب دورية..."
                )

                if train_ensemble(
                    prepared
                ):

                    last_retrain = now_ny()

            # ------------------------------------------------
            # وقت السوق
            # ------------------------------------------------

            market_ok, reason = (
                trading_window_status()
            )

            if not market_ok:

                maybe_send_wait(
                    reason
                )

                time.sleep(
                    POLL_SECONDS
                )

                continue

            # ------------------------------------------------
            # SIGNAL
            # ------------------------------------------------

            signal_data = calculate_signal(
                prepared
            )

            signal = signal_data[
                "signal"
            ]

            log(
                f"[ANALYSIS] "
                f"{signal} | "
                f"prob="
                f"{signal_data.get('probability', 0):.1%} | "
                f"regime="
                f"{signal_data.get('regime', 'N/A')}"
            )

            if signal in [
                "CALL",
                "PUT"
            ]:

                if should_send_signal(
                    signal
                ):

                    send_signal(
                        signal_data,
                        prepared
                    )

                else:

                    log(
                        "[FILTER] "
                        "تم منع تكرار نفس الإشارة"
                    )

            else:

                maybe_send_wait(
                    signal_data["reason"]
                )

        except KeyboardInterrupt:

            log(
                "إيقاف البوت..."
            )

            break

        except Exception as e:

            log(
                f"[MAIN ERROR] {e}"
            )

            maybe_send_wait(
                f"خطأ مؤقت: {str(e)[:150]}"
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

    initialize_ok = (
        initialize_model()
    )

    if not initialize_ok:

        log(
            "البوت توقف لأن النموذج "
            "لم يكتمل تدريبه."
        )

        return

    run_live()


if __name__ == "__main__":

    main()