# ============================================================
# SPX 0DTE AI ADVISOR v19.4 — صَفاء التحليل
# HONEST DATA + REAL DELTA FILTER + JOURNAL FIXES
# FRESH FINAL QUOTE RECHECK + STRICT SPXW 0DTE EXPIRY
# SPOT/OPTION TIMESTAMP ALIGNMENT
# SPXW 0DTE recommendations only — NO ORDER EXECUTION
# ============================================================

import os
import time
import logging
import warnings
from datetime import datetime, timedelta, timezone, date
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

# ======================== CONFIG =============================

VERSION = "19.4"

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

TRADE_URL = os.getenv(
    "ALPACA_TRADE_URL",
    "https://paper-api.alpaca.markets"
).rstrip("/")

DATA_URL = os.getenv(
    "ALPACA_DATA_URL",
    "https://data.alpaca.markets"
).rstrip("/")

DATA_FEED = os.getenv("ALPACA_DATA_FEED", "iex").lower()
OPTIONS_FEED = os.getenv(
    "ALPACA_OPTIONS_FEED",
    "indicative"
).lower()

TG_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TG_CHAT = os.getenv("TELEGRAM_CHAT_ID", "").strip()

MIN_AUC = float(os.getenv("MIN_MODEL_AUC", "0.55"))
MIN_CONFIDENCE = float(
    os.getenv("MIN_MODEL_CONFIDENCE", "0.62")
)

MAX_SPREAD_PCT = float(
    os.getenv("MAX_OPTION_SPREAD_PCT", "0.12")
)

MIN_OPTION_PREMIUM = float(
    os.getenv("MIN_OPTION_PREMIUM", "0.50")
)

MAX_OPTION_PREMIUM = float(
    os.getenv("MAX_OPTION_PREMIUM", "50.00")
)

MIN_TARGET_DISTANCE = float(
    os.getenv("MIN_TARGET_DISTANCE_POINTS", "1.00")
)

MAX_TARGET_DISTANCE = float(
    os.getenv("MAX_TARGET_DISTANCE_POINTS", "15.00")
)

MIN_DIRECTION_VOTES = int(
    os.getenv("MIN_DIRECTION_VOTES", "2")
)

MAX_SPX_BAR_AGE_MINUTES = float(
    os.getenv("MAX_SPX_BAR_AGE_MINUTES", "12")
)

MAX_STOCK_BAR_AGE_MINUTES = float(
    os.getenv("MAX_STOCK_BAR_AGE_MINUTES", "12")
)

MAX_COMMON_BAR_AGE_MINUTES = float(
    os.getenv("MAX_COMMON_BAR_AGE_MINUTES", "12")
)

MAX_SOURCE_SYNC_GAP_MINUTES = float(
    os.getenv("MAX_SOURCE_SYNC_GAP_MINUTES", "8")
)

MAX_SPOT_AGE_SECONDS = float(
    os.getenv("MAX_SPOT_AGE_SECONDS", "180")
)

MAX_OPTION_QUOTE_AGE_SECONDS = float(
    os.getenv("MAX_OPTION_QUOTE_AGE_SECONDS", "180")
)

MIN_ABS_DELTA = float(os.getenv("MIN_ABS_DELTA", "0.25"))
MAX_ABS_DELTA = float(os.getenv("MAX_ABS_DELTA", "0.65"))
TARGET_ABS_DELTA = float(
    os.getenv("TARGET_ABS_DELTA", "0.45")
)

MAX_DELTA_QUOTE_AGE_SECONDS = float(
    os.getenv("MAX_DELTA_QUOTE_AGE_SECONDS", "90")
)

MAX_SPOT_OPTION_GAP_SECONDS = float(
    os.getenv("MAX_SPOT_OPTION_GAP_SECONDS", "180")
)

MAX_SIGNAL_AGE_BEFORE_SEND_SECONDS = float(
    os.getenv("MAX_SIGNAL_AGE_BEFORE_SEND_SECONDS", "180")
)

SCAN_SECONDS = max(
    20,
    int(os.getenv("SCAN_SECONDS", "60"))
)

ALERT_COOLDOWN_MINUTES = max(
    1,
    int(os.getenv("ALERT_COOLDOWN_MINUTES", "20"))
)

LABEL_HORIZON = 3
MAX_BARS = 5000
MAX_CONTRACTS_TO_CHECK = 100

SIGNAL_LOG_PATH = os.getenv(
    "SIGNAL_LOG_PATH",
    "spxw_signal_journal.csv"
)

FEATURES = [
    "ret_1", "ret_3", "ret_6", "rsi",
    "ema_spread", "ema_long_spread",
    "atr_pct", "range_pct", "body_ratio",
    "volatility", "momentum_accel",
    "resistance_dist", "support_dist",
    "spy_ret_1", "spy_ret_3", "spy_ema_spread",
    "spy_rsi", "spy_vwap_dist",
    "qqq_ret_1", "qqq_ret_3", "qqq_ema_spread",
    "qqq_rsi", "qqq_vwap_dist",
]

OUTCOME_MINUTES = (1, 3, 5, 10)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

HTTP = requests.Session()
HTTP.headers.update({"User-Agent": f"SafaaAnalysis/{VERSION}"})

LAST_ALERTS = {}


# ======================== LOGGING / TIME =====================

def log(message, level="info"):
    getattr(logging, level, logging.info)(message)


def utc_now():
    return datetime.now(UTC)


def ny_now():
    return datetime.now(NY)


def alpaca_headers():
    if not API_KEY or not API_SECRET:
        raise RuntimeError(
            "مفاتيح Alpaca غير موجودة في متغيرات البيئة."
        )

    return {
        "APCA-API-KEY-ID": API_KEY,
        "APCA-API-SECRET-KEY": API_SECRET,
    }


def parse_timestamp(value):
    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    try:
        ts = pd.Timestamp(value)

        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")

        return ts.to_pydatetime()

    except Exception:
        return None


def age_seconds(ts):
    if ts is None:
        return None

    return (
        utc_now() - ts.astimezone(UTC)
    ).total_seconds()


def timestamp_gap_seconds(ts1, ts2):
    if ts1 is None or ts2 is None:
        return None

    return abs(
        (
            ts1.astimezone(UTC)
            - ts2.astimezone(UTC)
        ).total_seconds()
    )


def valid_timestamp_age(ts, max_age):
    age = age_seconds(ts)

    return (
        age is not None
        and -30 <= age <= max_age
    )


# ======================== BAR NORMALIZATION ==================

def normalize_bars(df):
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(
            df["timestamp"],
            utc=True,
            errors="coerce"
        )
        df = df.dropna(
            subset=["timestamp"]
        ).set_index("timestamp")
    else:
        df.index = pd.to_datetime(
            df.index,
            utc=True,
            errors="coerce"
        )
        df = df.loc[~df.index.isna()]

    df.index = pd.DatetimeIndex(
        pd.to_datetime(df.index, utc=True)
    )

    df.index.name = "timestamp"
    df = df[
        ~df.index.duplicated(keep="last")
    ].sort_index()

    for col in [
        "open", "high", "low", "close", "volume"
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            )

    required = [
        c for c in ["open", "high", "low", "close"]
        if c in df.columns
    ]

    if required:
        df = df.dropna(subset=required)

    return df


def validate_bar_freshness(df, source, max_age_minutes):
    if df is None or df.empty:
        raise RuntimeError(f"{source}: لا توجد شموع")

    last_ts = df.index[-1].to_pydatetime()

    age_min = (
        utc_now() - last_ts
    ).total_seconds() / 60

    log(
        f"[حداثة البيانات] {source}: "
        f"آخر شمعة={last_ts.isoformat()} | "
        f"العمر={age_min:.1f} دقيقة"
    )

    if age_min < -2:
        raise RuntimeError(
            f"{source}: توقيت آخر شمعة في المستقبل"
        )

    if age_min > max_age_minutes:
        raise RuntimeError(
            f"{source}: بيانات قديمة ({age_min:.1f} دقيقة)"
        )

    return age_min


# ======================== MARKET DATA ========================

def get_spx_bars():
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
            raise RuntimeError(
                "لم تصل بيانات SPX من Yahoo Finance"
            )

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

        keep = [
            c for c in [
                "open", "high", "low", "close", "volume"
            ]
            if c in raw.columns
        ]

        raw = raw[keep].tail(MAX_BARS)

        if len(raw) < 300:
            raise RuntimeError(
                f"عدد شموع SPX غير كافٍ: {len(raw)}"
            )

        validate_bar_freshness(
            raw,
            "SPX 5m",
            MAX_SPX_BAR_AGE_MINUTES
        )

        return raw

    except Exception as exc:
        log(f"[خطأ بيانات SPX] {exc}", "error")
        return pd.DataFrame()


def get_spx_spot():
    """
    Yahoo 1-minute close is a reference price, not a guaranteed
    live exchange tick.
    """
    try:
        raw = yf.download(
            "^GSPC",
            period="1d",
            interval="1m",
            auto_adjust=False,
            progress=False,
            threads=False,
        )

        if raw is None or raw.empty:
            raise RuntimeError("لم تصل بيانات SPX الدقيقة")

        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)

        raw = raw.rename(columns={"Close": "close"})
        raw = normalize_bars(raw)

        if raw.empty or "close" not in raw.columns:
            raise RuntimeError("بيانات SPX الدقيقة غير صالحة")

        ts = raw.index[-1].to_pydatetime()
        spot = float(raw["close"].iloc[-1])
        age = age_seconds(ts)

        log(
            f"[سعر SPX المرجعي] {spot:,.2f} | "
            f"التوقيت={ts.isoformat()} | العمر={age:.0f} ثانية"
        )

        if not valid_timestamp_age(ts, MAX_SPOT_AGE_SECONDS):
            raise RuntimeError(
                f"سعر SPX المرجعي قديم/غير صالح؛ العمر={age}"
            )

        return spot, ts

    except Exception as exc:
        log(f"[خطأ سعر SPX المرجعي] {exc}", "error")
        return None, None


def get_stock_bars(symbol):
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
                f"{response.text[:200]}"
            )

        rows = response.json().get("bars", [])

        if not rows:
            raise RuntimeError(
                f"لا توجد بيانات {symbol}; feed={DATA_FEED}"
            )

        df = pd.DataFrame(rows).rename(columns={
            "t": "timestamp",
            "o": "open",
            "h": "high",
            "l": "low",
            "c": "close",
            "v": "volume",
        })

        df = normalize_bars(df).tail(MAX_BARS)

        if len(df) < 100:
            raise RuntimeError(
                f"عدد شموع {symbol} غير كافٍ: {len(df)}"
            )

        validate_bar_freshness(
            df,
            f"{symbol} 5m",
            MAX_STOCK_BAR_AGE_MINUTES
        )

        return df

    except Exception as exc:
        log(f"[خطأ بيانات {symbol}] {exc}", "error")
        return pd.DataFrame()


# ======================== FEATURES ===========================

def calc_rsi(close, period=14):
    delta = close.diff()

    gain = delta.clip(lower=0).ewm(
        alpha=1 / period,
        min_periods=period,
        adjust=False
    ).mean()

    loss = (-delta.clip(upper=0)).ewm(
        alpha=1 / period,
        min_periods=period,
        adjust=False
    ).mean()

    rs = gain / loss.replace(0, np.nan)

    return 100 - (100 / (1 + rs))


def build_features(bars, prefix=""):
    df = normalize_bars(bars)

    if df.empty:
        return pd.DataFrame()

    o = df["open"]
    h = df["high"]
    lo = df["low"]
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

    prev = c.shift(1)

    tr = pd.concat([
        h - lo,
        (h - prev).abs(),
        (lo - prev).abs(),
    ], axis=1).max(axis=1)

    result[f"{prefix}atr_pct"] = (
        tr.rolling(14).mean() / c.replace(0, np.nan)
    )

    result[f"{prefix}range_pct"] = (
        (h - lo) / c.replace(0, np.nan)
    )

    result[f"{prefix}body_ratio"] = (
        (c - o) / (h - lo).replace(0, np.nan)
    )

    result[f"{prefix}volatility"] = (
        c.pct_change().rolling(20).std()
    )

    result[f"{prefix}momentum_accel"] = (
        c.pct_change().diff(3)
    )

    prior_high = h.shift(1).rolling(20).max()
    prior_low = lo.shift(1).rolling(20).min()

    if prefix == "":
        result["resistance_dist"] = (
            (prior_high - c) / c.replace(0, np.nan)
        )

        result["support_dist"] = (
            (c - prior_low) / c.replace(0, np.nan)
        )

    if prefix in ("spy_", "qqq_"):
        if (
            "volume" in df.columns
            and df["volume"].fillna(0).sum() > 0
        ):
            typical = (h + lo + c) / 3.0
            vol = df["volume"].fillna(0).clip(lower=0)

            vwap_num = (typical * vol).rolling(20).sum()
            vwap_den = vol.rolling(20).sum().replace(0, np.nan)
            vwap = vwap_num / vwap_den

            result[f"{prefix}vwap_dist"] = (
                (c - vwap) / c.replace(0, np.nan)
            )
        else:
            result[f"{prefix}vwap_dist"] = np.nan

    return result.replace([np.inf, -np.inf], np.nan)


def make_features(spx, spy, qqq):
    spx = normalize_bars(spx)
    spy = normalize_bars(spy)
    qqq = normalize_bars(qqq)

    if spx.empty or spy.empty or qqq.empty:
        raise RuntimeError(
            "بيانات SPX أو SPY أو QQQ غير متوفرة"
        )

    newest = [spx.index[-1], spy.index[-1], qqq.index[-1]]

    sync_gap = (max(newest) - min(newest)).total_seconds() / 60

    if sync_gap > MAX_SOURCE_SYNC_GAP_MINUTES:
        raise RuntimeError(
            f"مصادر البيانات غير متزامنة؛ الفرق {sync_gap:.1f} دقيقة"
        )

    spxf = build_features(spx)

    spyf = build_features(spy, "spy_")[[
        "spy_ret_1",
        "spy_ret_3",
        "spy_ema_spread",
        "spy_rsi",
        "spy_vwap_dist",
    ]]

    qqf = build_features(qqq, "qqq_")[[
        "qqq_ret_1",
        "qqq_ret_3",
        "qqq_ema_spread",
        "qqq_rsi",
        "qqq_vwap_dist",
    ]]

    merged = (
        spxf.join(spyf, how="inner")
        .join(qqf, how="inner")
        .join(spx[["close"]], how="inner")
    )

    merged = (
        merged.replace([np.inf, -np.inf], np.nan)
        .dropna(subset=FEATURES + ["close"])
        .sort_index()
    )

    if len(merged) < 400:
        raise RuntimeError(
            f"عدد صفوف البيانات المتطابقة غير كافٍ: {len(merged)}"
        )

    common_age = (
        utc_now() - merged.index[-1].to_pydatetime()
    ).total_seconds() / 60

    if (
        common_age > MAX_COMMON_BAR_AGE_MINUTES
        or common_age < -2
    ):
        raise RuntimeError(
            f"آخر شمعة مشتركة قديمة/غير صالحة: {common_age:.1f} دقيقة"
        )

    log(
        f"[تزامن البيانات] الفرق={sync_gap:.1f} دقيقة | "
        f"عمر آخر شمعة مشتركة={common_age:.1f} دقيقة"
    )

    return merged


# ======================== MACHINE LEARNING ===================

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
    data = df.sort_index().copy()

    future_close = data["close"].shift(-LABEL_HORIZON)
    valid = future_close.notna()

    X = data.loc[valid, FEATURES].replace(
        [np.inf, -np.inf], np.nan
    )

    y = (
        future_close.loc[valid] > data.loc[valid, "close"]
    ).astype(int)

    if len(X) < 500:
        raise RuntimeError(f"بيانات التدريب غير كافية: {len(X)}")

    split = int(len(X) * 0.80)
    train_end = split - LABEL_HORIZON

    if train_end < 200 or len(X) - split < 100:
        raise RuntimeError(
            "بيانات التدريب أو الاختبار غير كافية"
        )

    X_train = X.iloc[:train_end]
    y_train = y.iloc[:train_end]
    X_test = X.iloc[split:]
    y_test = y.iloc[split:]

    if y_train.nunique() < 2 or y_test.nunique() < 2:
        raise RuntimeError(
            "بيانات التدريب أو الاختبار تحتوي على فئة واحدة"
        )

    evaluator = new_model()
    evaluator.fit(X_train, y_train)

    probabilities = evaluator.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    accuracy = float(accuracy_score(y_test, predictions))
    auc = float(roc_auc_score(y_test, probabilities))

    baseline = int(y_train.mean() >= 0.5)
    baseline_accuracy = float((y_test == baseline).mean())

    log(
        f"[النموذج] صفوف={len(X)} | اختبار={len(X_test)} | "
        f"دقة الاتجاه={accuracy:.3f} | AUC={auc:.3f} | "
        f"خط الأساس={baseline_accuracy:.3f}"
    )

    live_model = new_model()
    live_model.fit(X, y)

    return live_model, FEATURES.copy(), auc, accuracy


def directional_votes(latest):
    rsi = float(latest["rsi"])

    spx_up = (
        latest["ret_1"] > 0
        and latest["ema_spread"] > 0
        and 45 <= rsi <= 72
    )

    spx_down = (
        latest["ret_1"] < 0
        and latest["ema_spread"] < 0
        and 28 <= rsi <= 55
    )

    spy_rsi = float(latest["spy_rsi"])
    qqq_rsi = float(latest["qqq_rsi"])

    spy_up = (
        latest["spy_ret_1"] > 0
        and latest["spy_ema_spread"] > 0
        and latest["spy_vwap_dist"] > 0
        and 45 <= spy_rsi <= 72
    )

    spy_down = (
        latest["spy_ret_1"] < 0
        and latest["spy_ema_spread"] < 0
        and latest["spy_vwap_dist"] < 0
        and 28 <= spy_rsi <= 55
    )

    qqq_up = (
        latest["qqq_ret_1"] > 0
        and latest["qqq_ema_spread"] > 0
        and latest["qqq_vwap_dist"] > 0
        and 45 <= qqq_rsi <= 72
    )

    qqq_down = (
        latest["qqq_ret_1"] < 0
        and latest["qqq_ema_spread"] < 0
        and latest["qqq_vwap_dist"] < 0
        and 28 <= qqq_rsi <= 55
    )

    up = sum(map(bool, [spx_up, spy_up, qqq_up]))
    down = sum(map(bool, [spx_down, spy_down, qqq_down]))

    return up, down


def calculate_signal(df, model, feature_cols, auc):
    if auc < MIN_AUC:
        return (
            "WAIT",
            0.0,
            f"AUC={auc:.3f} أقل من الحد {MIN_AUC:.3f}",
        )

    latest = df.iloc[-1]
    x = latest[feature_cols].to_frame().T

    p_up = float(model.predict_proba(x)[0, 1])
    p_down = 1.0 - p_up

    up_votes, down_votes = directional_votes(latest)

    resistance_dist = float(latest["resistance_dist"])
    support_dist = float(latest["support_dist"])

    if (
        p_up >= MIN_CONFIDENCE
        and up_votes >= MIN_DIRECTION_VOTES
        and down_votes == 0
    ):
        if 0 <= resistance_dist < 0.0015:
            return "WAIT", p_up, "السعر قريب جدًا من مقاومة حديثة"

        return (
            "CALL",
            p_up,
            f"ترجيح صعود={p_up:.1%}؛ تأكيدات صاعدة={up_votes}/3",
        )

    if (
        p_down >= MIN_CONFIDENCE
        and down_votes >= MIN_DIRECTION_VOTES
        and up_votes == 0
    ):
        if 0 <= support_dist < 0.0015:
            return "WAIT", p_down, "السعر قريب جدًا من دعم حديث"

        return (
            "PUT",
            p_down,
            f"ترجيح هبوط={p_down:.1%}؛ تأكيدات هابطة={down_votes}/3",
        )

    return (
        "WAIT",
        max(p_up, p_down),
        f"التأكيد غير كافٍ؛ صعود={p_up:.1%}؛ "
        f"هبوط={p_down:.1%}؛ الأصوات={up_votes}/{down_votes}",
    )


# ======================== TELEGRAM ===========================

def send_telegram(message):
    if not TG_TOKEN or not TG_CHAT:
        log(
            "[تيليجرام] رمز البوت أو معرف المحادثة غير موجود",
            "warning"
        )
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
                f"[خطأ تيليجرام] HTTP {response.status_code}: "
                f"{response.text[:200]}",
                "error"
            )
            return False

        if not response.json().get("ok", False):
            log(
                "[خطأ تيليجرام] لم تؤكد الواجهة إرسال الرسالة",
                "error"
            )
            return False

        log("[تيليجرام] تم إرسال الرسالة")
        return True

    except Exception as exc:
        log(f"[خطأ تيليجرام] {exc}", "error")
        return False


# ======================== OPTIONS CONTRACTS ==================

def get_spxw_contracts(direction):
    try:
        today = ny_now().date().isoformat()

        params = {
            "underlying_symbols": "SPX",
            "expiration_date": today,
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
                f"جلب العقود HTTP {response.status_code}: "
                f"{response.text[:200]}"
            )

        payload = response.json()
        contracts = (
            payload.get("option_contracts")
            or payload.get("contracts")
            or []
        )

        results = []

        for item in contracts:
            symbol = item.get("symbol", "")
            root = item.get("root_symbol", "")

            if not symbol:
                continue

            if root != "SPXW" and not symbol.startswith("SPXW"):
                continue

            try:
                strike = float(item["strike_price"])
            except (KeyError, TypeError, ValueError):
                continue

            expiry = item.get("expiration_date", "")

            # FIX: reject missing expiry and anything not expiring today.
            if not expiry or str(expiry)[:10] != today:
                continue

            results.append({
                "symbol": symbol,
                "strike": strike,
                "expiration": expiry,
            })

        return results

    except Exception as exc:
        log(f"[خطأ العقود] {exc}", "error")
        return []


def get_option_quotes(symbols):
    results = {}
    endpoint = f"{DATA_URL}/v1beta1/options/quotes/latest"

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
                    f"[خطأ أسعار الخيارات] HTTP {response.status_code}: "
                    f"{response.text[:200]}",
                    "warning"
                )
                continue

            quotes = response.json().get("quotes", {}) or {}

            if isinstance(quotes, dict):
                results.update(quotes)

        except Exception as exc:
            log(f"[خطأ أسعار الخيارات] {exc}", "warning")

    return results


def get_option_snapshots(symbols):
    results = {}
    endpoint = f"{DATA_URL}/v1beta1/options/snapshots"

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
                    f"[خطأ لقطات الخيارات/Delta] HTTP "
                    f"{response.status_code}: {response.text[:200]}",
                    "warning"
                )
                continue

            payload = response.json().get("snapshots", {}) or {}

            if isinstance(payload, dict):
                results.update(payload)

        except Exception as exc:
            log(f"[خطأ لقطات الخيارات/Delta] {exc}", "warning")

    return results


# ======================== QUOTE UTILITIES ====================

def quote_timestamp(quote):
    if not isinstance(quote, dict):
        return None

    for key in ("t", "timestamp", "time"):
        ts = parse_timestamp(quote.get(key))

        if ts is not None:
            return ts

    return None


def number_from(quote, *keys):
    if not isinstance(quote, dict):
        return None

    for key in keys:
        try:
            value = quote.get(key)

            if value is None:
                continue

            value = float(value)

            if np.isfinite(value):
                return value

        except (TypeError, ValueError):
            pass

    return None


def normalize_quote(quote):
    if not isinstance(quote, dict):
        return None

    ts = quote_timestamp(quote)
    age = age_seconds(ts) if ts else None

    if (
        age is None
        or age < -30
        or age > MAX_OPTION_QUOTE_AGE_SECONDS
    ):
        return None

    bid = number_from(quote, "bp", "bid_price", "bid")
    ask = number_from(quote, "ap", "ask_price", "ask")

    if (
        bid is None
        or ask is None
        or bid <= 0
        or ask <= 0
        or ask < bid
    ):
        return None

    mid = (bid + ask) / 2

    if mid <= 0:
        return None

    spread = (ask - bid) / mid

    return {
        "bid": bid,
        "ask": ask,
        "mid": mid,
        "spread_pct": spread,
        "quote_ts": ts,
        "quote_age_seconds": age,
    }


def validate_real_delta(direction, snapshot):
    """
    Delta must be provider-supplied, finite, correctly signed,
    within the permitted magnitude, and associated with a fresh
    latestQuote timestamp.

    IMPORTANT:
    latestQuote timestamp is NOT proof that Greeks were calculated
    at that timestamp. No separate Greeks calculation timestamp is
    confirmed by this implementation.
    """
    if not isinstance(snapshot, dict):
        return None, None

    greeks = snapshot.get("greeks") or {}

    try:
        delta = float(greeks.get("delta"))
    except (TypeError, ValueError):
        return None, None

    if not np.isfinite(delta) or delta == 0 or abs(delta) > 1:
        return None, None

    if direction == "CALL" and delta <= 0:
        return None, None

    if direction == "PUT" and delta >= 0:
        return None, None

    if not MIN_ABS_DELTA <= abs(delta) <= MAX_ABS_DELTA:
        return None, None

    latest_quote = (
        snapshot.get("latestQuote")
        or snapshot.get("latest_quote")
        or {}
    )

    ts = quote_timestamp(latest_quote)
    age = age_seconds(ts) if ts else None

    if (
        age is None
        or age < -30
        or age > MAX_DELTA_QUOTE_AGE_SECONDS
    ):
        return None, None

    return delta, ts


# ======================== CONTRACT SELECTION =================

def choose_contract(direction, spot_price, spot_ts):
    spot_age = age_seconds(spot_ts)

    if (
        spot_ts is None
        or spot_age is None
        or spot_age < -30
        or spot_age > MAX_SPOT_AGE_SECONDS
    ):
        log(
            "[العقود] WAIT — سعر SPX المرجعي قديم أو غير صالح",
            "warning"
        )
        return None

    contracts = get_spxw_contracts(direction)

    if not contracts:
        log(
            "[العقود] لم يتم العثور على عقود SPXW لليوم",
            "warning"
        )
        return None

    if direction == "CALL":
        contracts = [
            c for c in contracts
            if MIN_TARGET_DISTANCE
            <= c["strike"] - spot_price
            <= MAX_TARGET_DISTANCE
        ]
    else:
        contracts = [
            c for c in contracts
            if MIN_TARGET_DISTANCE
            <= spot_price - c["strike"]
            <= MAX_TARGET_DISTANCE
        ]

    if not contracts:
        log(
            f"[العقود] لا يوجد سعر تنفيذ ضمن "
            f"{MIN_TARGET_DISTANCE:.1f}–"
            f"{MAX_TARGET_DISTANCE:.1f} نقطة",
            "info"
        )
        return None

    contracts.sort(
        key=lambda c: abs(c["strike"] - spot_price)
    )

    candidates = contracts[:MAX_CONTRACTS_TO_CHECK]
    symbols = [c["symbol"] for c in candidates]

    quotes = get_option_quotes(symbols)
    snapshots = get_option_snapshots(symbols)

    viable = []

    for contract in candidates:
        symbol = contract["symbol"]
        q = normalize_quote(quotes.get(symbol))

        if q is None:
            log(
                f"[رفض العقد] {symbol}: سعر قديم أو غير صالح",
                "debug"
            )
            continue

        if q["spread_pct"] > MAX_SPREAD_PCT:
            continue

        if not MIN_OPTION_PREMIUM <= q["ask"] <= MAX_OPTION_PREMIUM:
            continue

        delta, delta_ts = validate_real_delta(
            direction,
            snapshots.get(symbol)
        )

        if delta is None or delta_ts is None:
            continue

        quote_delta_gap = timestamp_gap_seconds(
            q["quote_ts"], delta_ts
        )

        if (
            quote_delta_gap is None
            or quote_delta_gap > MAX_SPOT_OPTION_GAP_SECONDS
        ):
            continue

        spot_quote_gap = timestamp_gap_seconds(
            spot_ts, q["quote_ts"]
        )

        spot_delta_gap = timestamp_gap_seconds(
            spot_ts, delta_ts
        )

        if (
            spot_quote_gap is None
            or spot_delta_gap is None
            or spot_quote_gap > MAX_SPOT_OPTION_GAP_SECONDS
            or spot_delta_gap > MAX_SPOT_OPTION_GAP_SECONDS
        ):
            log(
                f"[رفض العقد] {symbol}: اختلاف توقيت "
                f"SPX/الخيار/Delta؛ السعر={spot_quote_gap} ثانية، "
                f"Delta={spot_delta_gap} ثانية",
                "info"
            )
            continue

        if not valid_timestamp_age(
            spot_ts, MAX_SPOT_AGE_SECONDS
        ):
            log(
                "[العقود] WAIT — انتهت صلاحية سعر SPX أثناء فحص العقود",
                "warning"
            )
            return None

        if not valid_timestamp_age(
            q["quote_ts"], MAX_OPTION_QUOTE_AGE_SECONDS
        ):
            continue

        if not valid_timestamp_age(
            delta_ts, MAX_DELTA_QUOTE_AGE_SECONDS
        ):
            continue

        viable.append({
            **contract,
            **q,
            "delta": delta,
            "delta_quote_ts": delta_ts,
            "delta_quote_age_seconds": age_seconds(delta_ts),
            "distance": abs(contract["strike"] - spot_price),
            "delta_sync_gap_seconds": quote_delta_gap,
            "spot_quote_gap_seconds": spot_quote_gap,
            "spot_delta_gap_seconds": spot_delta_gap,
        })

    if not viable:
        log(
            "[العقود] WAIT — لا يوجد عقد يجتاز فلاتر "
            "السعر والسبريد وDelta والتوقيت",
            "warning"
        )
        return None

    viable.sort(key=lambda c: (
        abs(abs(c["delta"]) - TARGET_ABS_DELTA),
        c["spread_pct"],
        c["distance"],
        c["ask"],
    ))

    selected = viable[0]

    log(
        f"[العقد] {selected['symbol']} | "
        f"strike={selected['strike']:.2f} | "
        f"SPX={spot_price:.2f} | "
        f"Delta={selected['delta']:+.3f} | "
        f"distance={selected['distance']:.2f} | "
        f"bid={selected['bid']:.2f} ask={selected['ask']:.2f} | "
        f"spread={selected['spread_pct']:.1%} | "
        f"quote age={selected['quote_age_seconds']:.0f}s | "
        f"Delta quote age="
        f"{selected['delta_quote_age_seconds']:.0f}s | "
        f"SPX/quote gap="
        f"{selected['spot_quote_gap_seconds']:.0f}s"
    )

    return selected


# FIX 1: fetch a fresh quote and snapshot immediately before Telegram.
def final_contract_check(
    contract,
    spot_ts,
    spot_price,
    direction
):
    """
    Fresh network recheck immediately before Telegram.
    latestQuote timestamp is a proxy only; it is not a confirmed
    Greeks calculation timestamp.
    """
    if not valid_timestamp_age(
        spot_ts,
        MAX_SPOT_AGE_SECONDS
    ):
        return False, "سعر SPX المرجعي أصبح قديمًا"

    symbol = contract.get("symbol")

    if not symbol:
        return False, "رمز العقد مفقود"

    # Fetch new data instead of trusting the quote selected earlier.
    quotes = get_option_quotes([symbol])
    snapshots = get_option_snapshots([symbol])

    q = normalize_quote(quotes.get(symbol))

    if q is None:
        return False, "تعذر الحصول على عرض سعر حديث للعقد"

    if q["spread_pct"] > MAX_SPREAD_PCT:
        return False, "السبريد تجاوز الحد المسموح"

    if not MIN_OPTION_PREMIUM <= q["ask"] <= MAX_OPTION_PREMIUM:
        return False, "سعر العقد خارج النطاق المسموح"

    delta, delta_ts = validate_real_delta(
        direction,
        snapshots.get(symbol)
    )

    if delta is None or delta_ts is None:
        return False, "Delta غير متاحة أو قديمة أو غير صالحة"

    quote_delta_gap = timestamp_gap_seconds(
        q["quote_ts"], delta_ts
    )

    spot_quote_gap = timestamp_gap_seconds(
        spot_ts, q["quote_ts"]
    )

    spot_delta_gap = timestamp_gap_seconds(
        spot_ts, delta_ts
    )

    if any(x is None for x in (
        quote_delta_gap,
        spot_quote_gap,
        spot_delta_gap
    )):
        return False, "تعذر التحقق من تزامن التوقيت"

    if max(
        quote_delta_gap,
        spot_quote_gap,
        spot_delta_gap
    ) > MAX_SPOT_OPTION_GAP_SECONDS:
        return False, "توقيت SPX والعقد وDelta غير متزامن"

    if not valid_timestamp_age(
        q["quote_ts"],
        MAX_OPTION_QUOTE_AGE_SECONDS
    ):
        return False, "عرض سعر العقد أصبح قديمًا"

    if not valid_timestamp_age(
        delta_ts,
        MAX_DELTA_QUOTE_AGE_SECONDS
    ):
        return False, "توقيت عرض Delta أصبح قديمًا"

    # Refresh every quote-dependent field with the new data.
    contract.update(q)
    contract["delta"] = delta
    contract["delta_quote_ts"] = delta_ts
    contract["delta_quote_age_seconds"] = age_seconds(delta_ts)
    contract["delta_sync_gap_seconds"] = quote_delta_gap
    contract["spot_quote_gap_seconds"] = spot_quote_gap
    contract["spot_delta_gap_seconds"] = spot_delta_gap
    contract["distance"] = abs(contract["strike"] - spot_price)

    return True, "OK"


# ======================== OUTCOME JOURNAL ====================

def _journal_columns():
    cols = [
        "signal_time_utc",
        "direction",
        "symbol",
        "spot",
        "spot_quote_time_utc",
        "strike",
        "entry_bid",
        "entry_ask",
        "entry_mid",
        "entry_delta",
        "delta_quote_time_utc",
        "delta_timestamp_basis",
        "confidence",
        "auc",
        "quote_time_utc",
    ]

    for minutes in OUTCOME_MINUTES:
        cols += [
            f"bid_after_{minutes}m",
            f"observed_at_{minutes}m_utc",
            f"source_quote_time_{minutes}m_utc",
            f"actual_observation_delay_seconds_{minutes}m",
            f"quote_to_poll_delay_seconds_{minutes}m",
            f"return_vs_entry_ask_{minutes}m",
        ]

    return cols


def read_journal():
    if not os.path.exists(SIGNAL_LOG_PATH):
        return pd.DataFrame(columns=_journal_columns())

    try:
        df = pd.read_csv(SIGNAL_LOG_PATH)
    except Exception as exc:
        raise RuntimeError(
            f"تعذرت قراءة سجل CSV الحالي: {exc}. "
            "لم يتم استبدال الملف تلقائيًا."
        )

    if df.empty:
        for col in _journal_columns():
            if col not in df.columns:
                df[col] = pd.Series(dtype="object")
        return df

    for col in _journal_columns():
        if col not in df.columns:
            df[col] = np.nan

    return df


def write_journal(df):
    expected = _journal_columns()
    extra = [c for c in df.columns if c not in expected]
    ordered = expected + extra

    for col in ordered:
        if col not in df.columns:
            df[col] = np.nan

    df[ordered].to_csv(SIGNAL_LOG_PATH, index=False)


def append_signal_journal(row):
    try:
        df = read_journal()
        new_row = pd.DataFrame([row])

        for col in df.columns:
            if col not in new_row.columns:
                new_row[col] = np.nan

        for col in new_row.columns:
            if col not in df.columns:
                df[col] = np.nan

        combined = pd.concat([df, new_row], ignore_index=True)
        write_journal(combined)

        log(
            f"[سجل التوصيات] تمت إضافة "
            f"{row.get('symbol')} إلى {SIGNAL_LOG_PATH}"
        )

    except Exception as exc:
        log(f"[سجل التوصيات] تعذر حفظ السجل: {exc}", "warning")


def update_outcome_journal():
    """
    Each fresh option quote may update at most one outcome target
    per signal per scan. This is an observed quote, not an exact
    historical replay, executable fill, or guaranteed exit price.
    """
    try:
        if not os.path.exists(SIGNAL_LOG_PATH):
            return

        df = read_journal()

        if df.empty or "signal_time_utc" not in df.columns:
            return

        now = utc_now()
        pending = {}

        for idx, row in df.iterrows():
            try:
                symbol = row.get("symbol")

                if not isinstance(symbol, str) or not symbol:
                    continue

                start = parse_timestamp(row.get("signal_time_utc"))

                if start is None:
                    continue

                elapsed = (now - start).total_seconds()
                due_targets = []

                for minutes in OUTCOME_MINUTES:
                    bid_col = f"bid_after_{minutes}m"

                    if not pd.isna(row.get(bid_col)):
                        continue

                    if elapsed >= minutes * 60:
                        due_targets.append(minutes)

                if not due_targets:
                    continue

                used_quote_times = set()

                for minutes in OUTCOME_MINUTES:
                    used_col = f"source_quote_time_{minutes}m_utc"
                    used_ts = parse_timestamp(row.get(used_col))

                    if used_ts is not None:
                        used_quote_times.add(used_ts.isoformat())

                pending.setdefault(symbol, []).append({
                    "idx": idx,
                    "start": start,
                    "minutes": due_targets[0],
                    "entry_ask": row.get("entry_ask"),
                    "used_quote_times": used_quote_times,
                })

            except Exception as exc:
                log(
                    f"[سجل النتائج] تعذر تجهيز صف {idx}: {exc}",
                    "debug"
                )

        if not pending:
            return

        quotes = get_option_quotes(list(pending.keys()))
        changed = False

        for symbol, entries in pending.items():
            q = normalize_quote(quotes.get(symbol))

            if q is None:
                continue

            q_ts = q["quote_ts"]

            for item in entries:
                idx = item["idx"]
                minutes = item["minutes"]
                start = item["start"]

                target_time = start + timedelta(minutes=minutes)

                if q_ts < target_time:
                    continue

                if q_ts.isoformat() in item["used_quote_times"]:
                    continue

                bid_col = f"bid_after_{minutes}m"

                if not pd.isna(df.at[idx, bid_col]):
                    continue

                df.at[idx, bid_col] = q["bid"]

                df.at[
                    idx, f"observed_at_{minutes}m_utc"
                ] = now.isoformat()

                df.at[
                    idx, f"source_quote_time_{minutes}m_utc"
                ] = q_ts.isoformat()

                quote_delay = (q_ts - target_time).total_seconds()

                df.at[
                    idx,
                    f"actual_observation_delay_seconds_{minutes}m"
                ] = max(0.0, quote_delay)

                poll_delay = (now - q_ts).total_seconds()

                df.at[
                    idx,
                    f"quote_to_poll_delay_seconds_{minutes}m"
                ] = poll_delay

                try:
                    entry = float(item["entry_ask"])

                    df.at[
                        idx,
                        f"return_vs_entry_ask_{minutes}m"
                    ] = (
                        (q["bid"] - entry) / entry
                        if entry > 0
                        else np.nan
                    )

                except (TypeError, ValueError):
                    df.at[
                        idx,
                        f"return_vs_entry_ask_{minutes}m"
                    ] = np.nan

                changed = True
                break

        if changed:
            write_journal(df)
            log(
                "[سجل النتائج] تم تحديث هدف واحد كحد أقصى "
                "لكل إشارة باستخدام توقيت عرض السعر"
            )

    except Exception as exc:
        log(f"[سجل النتائج] تعذر تحديث النتائج: {exc}", "warning")


# ======================== DUPLICATE SUPPRESSION ==============

def is_duplicate(direction, symbol):
    key = f"{direction}:{symbol}"
    last = LAST_ALERTS.get(key)

    if last and (
        utc_now() - last
    ).total_seconds() < ALERT_COOLDOWN_MINUTES * 60:
        return True

    return False


def mark_alert_sent(direction, symbol):
    LAST_ALERTS[f"{direction}:{symbol}"] = utc_now()


# ======================== TELEGRAM MESSAGE ===================

def build_telegram_message(
    direction,
    confidence,
    spot,
    spot_ts,
    contract,
    signal_time,
    auc,
):
    direction_ar = (
        "شراء كول" if direction == "CALL"
        else "شراء بوت"
    )

    arrow = "🟢" if direction == "CALL" else "🔴"
    target_direction = "صعودًا" if direction == "CALL" else "هبوطًا"
    distance = abs(contract["strike"] - spot)
    delta_age = contract["delta_quote_age_seconds"]

    return (
        f"🦅 صَفاء التحليل — v{VERSION}\n"
        "مستشار توصيات خيارات SPXW 0DTE\n"
        "━━━━━━━━━━━━━━━━━━\n\n"
        f"{arrow} نوع الإشارة: {direction_ar}\n"
        f"💹 مستوى SPX المرجعي: {spot:,.2f}\n"
        f"🕒 توقيت مرجع SPX: "
        f"{spot_ts.astimezone(NY).strftime('%H:%M:%S')} نيويورك\n"
        f"🎯 سعر التنفيذ: {contract['strike']:,.2f}\n"
        f"📏 المسافة إلى سعر التنفيذ: {distance:.2f} نقطة "
        f"({target_direction})\n"
        f"🧮 Delta من بيانات المزود: {contract['delta']:+.3f}\n"
        f"📊 درجة النموذج الاتجاهية: {confidence:.1%}\n"
        f"📈 AUC في الاختبار الزمني: {auc:.3f}\n\n"
        "📑 تفاصيل عقد SPXW\n"
        f"🔹 رمز العقد: {contract['symbol']}\n"
        f"🟢 Ask: {contract['ask']:.2f}\n"
        f"🔴 Bid: {contract['bid']:.2f}\n"
        f"⚖️ متوسط العرض والطلب: {contract['mid']:.2f}\n"
        f"📉 السبريد: {contract['spread_pct']:.1%}\n"
        f"⏱️ عمر عرض العقد: "
        f"{contract['quote_age_seconds']:.0f} ثانية\n"
        f"⏱️ عمر توقيت عرض Delta: {delta_age:.0f} ثانية\n"
        f"🔄 فرق توقيت SPX والعقد: "
        f"{contract['spot_quote_gap_seconds']:.0f} ثانية\n"
        f"🕒 وقت الإشارة: "
        f"{signal_time.astimezone(NY).strftime('%H:%M:%S')} نيويورك\n\n"
        "━━━━━━━━━━━━━━━━━━\n"
        "⚠️ درجة النموذج ليست احتمال ربح العقد.\n"
        "⚠️ AUC لا يضمن نجاح التوصية.\n"
        "⚠️ توقيت عرض السعر لا يثبت توقيت حساب Greeks.\n"
        "⚠️ سعر Yahoo مرجعي وليس بالضرورة سعر SPX اللحظي.\n"
        "⚠️ الأسعار المعروضة ليست ضمانًا لسعر التنفيذ.\n\n"
        "صَفاء التحليل • توصيات فقط • لا تنفيذ تلقائي"
    )


# ======================== MARKET HOLIDAYS ====================

def nth_weekday(year, month, weekday, n):
    first = date(year, month, 1)
    offset = (weekday - first.weekday()) % 7
    day = 1 + offset + (n - 1) * 7
    return date(year, month, day)


def last_weekday(year, month, weekday):
    if month == 12:
        current = date(year + 1, 1, 1) - timedelta(days=1)
    else:
        current = date(year, month + 1, 1) - timedelta(days=1)

    while current.weekday() != weekday:
        current -= timedelta(days=1)

    return current


def easter_sunday(year):
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451

    month = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1

    return date(year, month, day)


def observed_fixed_holiday(day):
    if day.weekday() == 5:
        return day - timedelta(days=1)

    if day.weekday() == 6:
        return day + timedelta(days=1)

    return day


def market_holidays(year):
    holidays = set()

    # FIX 2: New Year's Day observance, including prior Dec 31
    # when January 1 falls on Saturday.
    new_year = date(year, 1, 1)

    if new_year.weekday() <= 4:
        holidays.add(new_year)
    elif new_year.weekday() == 6:
        holidays.add(new_year + timedelta(days=1))

    if new_year.weekday() == 5:
        holidays.add(new_year - timedelta(days=1))

    holidays.add(nth_weekday(year, 1, 0, 3))   # MLK Day
    holidays.add(nth_weekday(year, 2, 0, 3))   # Presidents Day
    holidays.add(easter_sunday(year) - timedelta(days=2))
    holidays.add(last_weekday(year, 5, 0))     # Memorial Day

    if year >= 2022:
        holidays.add(
            observed_fixed_holiday(date(year, 6, 19))
        )

    holidays.add(observed_fixed_holiday(date(year, 7, 4)))
    holidays.add(nth_weekday(year, 9, 0, 1))   # Labor Day
    holidays.add(nth_weekday(year, 11, 3, 4))  # Thanksgiving
    holidays.add(observed_fixed_holiday(date(year, 12, 25)))

    return holidays


def is_market_holiday(day):
    for year in (day.year - 1, day.year, day.year + 1):
        if day in market_holidays(year):
            return True

    return False


def market_close_time(day):
    if day.weekday() >= 5 or is_market_holiday(day):
        return None

    thanksgiving = nth_weekday(day.year, 11, 3, 4)

    if day == thanksgiving + timedelta(days=1):
        return (13, 0)

    if day.month == 7 and day.day == 3:
        return (13, 0)

    if day.month == 12 and day.day == 24:
        return (13, 0)

    return (16, 0)


def market_is_open():
    now = ny_now()
    day = now.date()

    if day.weekday() >= 5:
        return False

    if is_market_holiday(day):
        return False

    close_time = market_close_time(day)

    if close_time is None:
        return False

    current_minutes = now.hour * 60 + now.minute
    open_minutes = 9 * 60 + 30
    close_minutes = close_time[0] * 60 + close_time[1]

    return open_minutes <= current_minutes < close_minutes


# ======================== SCANNER ============================

def run_scan():
    log(
        f"========== بدء فحص جديد | صَفاء التحليل v{VERSION} =========="
    )

    update_outcome_journal()

    if not API_KEY or not API_SECRET:
        log(
            "[الإشارة] WAIT — مفاتيح Alpaca غير موجودة",
            "error"
        )
        return

    spx = get_spx_bars()
    spy = get_stock_bars("SPY")
    qqq = get_stock_bars("QQQ")

    if spx.empty or spy.empty or qqq.empty:
        log(
            "[الإشارة] WAIT — بيانات السوق غير مكتملة أو قديمة",
            "warning"
        )
        return

    spot_price, spot_ts = get_spx_spot()

    if spot_price is None or spot_ts is None:
        log(
            "[الإشارة] WAIT — لا يوجد سعر SPX مرجعي حديث",
            "warning"
        )
        return

    try:
        merged = make_features(spx, spy, qqq)
        model, columns, auc, accuracy = train_model(merged)

        direction, confidence, reason = calculate_signal(
            merged, model, columns, auc
        )

        log(
            f"[الإشارة] {direction} | "
            f"درجة الاتجاه={confidence:.1%} | "
            f"AUC={auc:.3f} | دقة الاختبار={accuracy:.3f} | "
            f"السبب={reason}"
        )

        if direction == "WAIT":
            return

        if not valid_timestamp_age(spot_ts, MAX_SPOT_AGE_SECONDS):
            log(
                "[الإشارة] WAIT — سعر SPX المرجعي أصبح قديمًا",
                "warning"
            )
            return

        contract = choose_contract(direction, spot_price, spot_ts)

        if contract is None:
            return

        if is_duplicate(direction, contract["symbol"]):
            log(
                f"[منع التكرار] تجاهل إشارة متكررة: "
                f"{direction} {contract['symbol']}"
            )
            return

        # FIX 3: pass spot price and direction to final recheck.
        valid, reason_check = final_contract_check(
            contract,
            spot_ts,
            spot_price,
            direction
        )

        if not valid:
            log(
                f"[الإشارة] WAIT — فشل الفحص النهائي: {reason_check}",
                "warning"
            )
            return

        spot_age = age_seconds(spot_ts)

        if (
            spot_age is None
            or spot_age < -30
            or spot_age > MAX_SIGNAL_AGE_BEFORE_SEND_SECONDS
        ):
            log(
                "[الإشارة] WAIT — تجاوز سعر SPX حد العمر "
                "المسموح قبل الإرسال",
                "warning"
            )
            return

        # Check quote freshness again after the final network calls.
        if not valid_timestamp_age(
            contract.get("quote_ts"),
            MAX_OPTION_QUOTE_AGE_SECONDS
        ):
            log(
                "[الإشارة] WAIT — عرض العقد أصبح قديمًا قبل الإرسال",
                "warning"
            )
            return

        signal_time = utc_now()

        message = build_telegram_message(
            direction,
            confidence,
            spot_price,
            spot_ts,
            contract,
            signal_time,
            auc
        )

        if send_telegram(message):
            mark_alert_sent(direction, contract["symbol"])

            append_signal_journal({
                "signal_time_utc": signal_time.isoformat(),
                "direction": direction,
                "symbol": contract["symbol"],
                "spot": spot_price,
                "spot_quote_time_utc": spot_ts.isoformat(),
                "strike": contract["strike"],
                "entry_bid": contract["bid"],
                "entry_ask": contract["ask"],
                "entry_mid": contract["mid"],
                "entry_delta": contract["delta"],
                "delta_quote_time_utc": (
                    contract["delta_quote_ts"].isoformat()
                ),
                "delta_timestamp_basis": (
                    "latestQuote timestamp proxy; "
                    "not a confirmed Greeks calculation timestamp"
                ),
                "confidence": confidence,
                "auc": auc,
                "quote_time_utc": contract["quote_ts"].isoformat(),
            })

    except Exception as exc:
        log(
            f"[خطأ الفحص] {type(exc).__name__}: {exc}",
            "error"
        )


# ======================== MAIN ===============================

def main():
    log(
        f"بدء تشغيل صَفاء التحليل — "
        f"SPXW 0DTE AI ADVISOR v{VERSION}"
    )

    log("الوضع: توصيات فقط — لا يوجد تنفيذ تلقائي للصفقات")

    log(
        f"بيانات الأسهم={DATA_FEED} | "
        f"أسعار الخيارات={OPTIONS_FEED} | "
        f"AUC الأدنى={MIN_AUC:.3f} | "
        f"درجة الاتجاه الأدنى={MIN_CONFIDENCE:.2f}"
    )

    log(
        f"عمر شمعة SPX الأقصى={MAX_SPX_BAR_AGE_MINUTES:.1f} دقيقة | "
        f"عمر سعر العقد الأقصى={MAX_OPTION_QUOTE_AGE_SECONDS:.0f} ثانية | "
        f"مسافة سعر التنفيذ={MIN_TARGET_DISTANCE:.1f}-"
        f"{MAX_TARGET_DISTANCE:.1f} نقطة"
    )

    log(
        f"Real Delta required: abs Delta "
        f"{MIN_ABS_DELTA:.2f}-{MAX_ABS_DELTA:.2f}; "
        f"target={TARGET_ABS_DELTA:.2f}; "
        f"Delta quote age max={MAX_DELTA_QUOTE_AGE_SECONDS:.0f}s"
    )

    log(
        f"SPX/option timestamp gap max="
        f"{MAX_SPOT_OPTION_GAP_SECONDS:.0f}s"
    )

    log(
        f"Scan interval={SCAN_SECONDS}s | "
        f"Alert cooldown={ALERT_COOLDOWN_MINUTES}m"
    )

    if not API_KEY or not API_SECRET:
        log("مفاتيح Alpaca غير موجودة في متغيرات البيئة", "error")

    if not TG_TOKEN or not TG_CHAT:
        log("إعدادات تيليجرام غير مكتملة", "warning")

    while True:
        try:
            if market_is_open():
                run_scan()
            else:
                log(
                    "[السوق] مغلق | توقيت نيويورك: "
                    + ny_now().strftime("%Y-%m-%d %H:%M:%S %Z")
                )

        except KeyboardInterrupt:
            log("تم إيقاف صَفاء التحليل")
            break

        except Exception as exc:
            log(
                f"[خطأ رئيسي] {type(exc).__name__}: {exc}",
                "error"
            )

        time.sleep(SCAN_SECONDS)


if __name__ == "__main__":
    main()