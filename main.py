# ============================================================
# SPX 0DTE AI ADVISOR v19.9 — صَفاء التحليل
# STRICT BAR COMPLETENESS + PRE-SEND RE-VALIDATION FIX
# 15M NY SESSION ALIGNMENT + ANTI-LOOKAHEAD ENFORCEMENT
# OBSERVED QUOTES JOURNAL — NOT ACTUAL EXECUTION P&L
# SPXW SAME-DAY OPTIONS — RECOMMENDATIONS ONLY
# NO AUTOMATIC ORDER EXECUTION
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

VERSION = "19.9"
BOT_NAME = "صَفاء التحليل"

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
    "ALPACA_OPTIONS_FEED", "indicative"
).lower()

TG_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TG_CHAT = os.getenv("TELEGRAM_CHAT_ID", "").strip()

MIN_AUC = float(os.getenv("MIN_MODEL_AUC", "0.55"))
MIN_CONFIDENCE = float(
    os.getenv("MIN_MODEL_CONFIDENCE", "0.62")
)

BASE_MAX_SPREAD_PCT = float(
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
TARGET_ABS_DELTA = float(os.getenv("TARGET_ABS_DELTA", "0.45"))

MAX_DELTA_QUOTE_AGE_SECONDS = float(
    os.getenv("MAX_DELTA_QUOTE_AGE_SECONDS", "90")
)

MAX_SPOT_OPTION_GAP_SECONDS = float(
    os.getenv("MAX_SPOT_OPTION_GAP_SECONDS", "180")
)

MAX_SIGNAL_AGE_BEFORE_SEND_SECONDS = float(
    os.getenv("MAX_SIGNAL_AGE_BEFORE_SEND_SECONDS", "120")
)

SCAN_SECONDS = max(20, int(os.getenv("SCAN_SECONDS", "60")))
ALERT_COOLDOWN_MINUTES = max(
    1, int(os.getenv("ALERT_COOLDOWN_MINUTES", "20"))
)

LABEL_HORIZON = 3
MAX_BARS = 5000
MAX_CONTRACTS_TO_CHECK = 100

SIGNAL_LOG_PATH = os.getenv(
    "SIGNAL_LOG_PATH", "spxw_signal_journal.csv"
)

OUTCOME_MINUTES = (1, 3, 5, 10)

REQUIRE_INDEPENDENT_GREEKS_TIMESTAMP = (
    os.getenv("REQUIRE_INDEPENDENT_GREEKS_TIMESTAMP", "false").lower()
    in ("1", "true", "yes")
)

FEATURES = [
    "ret_1", "ret_3", "ret_6",
    "rsi", "ema_spread", "ema_long_spread",
    "atr_pct", "range_pct", "body_ratio",
    "volatility", "momentum_accel", "momentum_velocity",
    "resistance_dist", "support_dist",
    "macd_line_pct", "macd_signal_pct", "macd_hist_pct",
    "adx", "plus_di", "minus_di",
    "volume_ratio", "session_vwap_dist",
    "bullish_engulfing", "bearish_engulfing",
    "lower_wick_ratio", "upper_wick_ratio",
    "bullish_rejection", "bearish_rejection",
    "breakout_up", "breakout_down",
    "fakeout_up", "fakeout_down",
    "retest_up", "retest_down",
    "opening_range_position",
    "opening_range_break_up", "opening_range_break_down",
    "spy_ret_1", "spy_ret_3", "spy_ema_spread",
    "spy_rsi", "spy_vwap_dist",
    "spy_macd_hist_pct", "spy_adx", "spy_volume_ratio",
    "qqq_ret_1", "qqq_ret_3", "qqq_ema_spread",
    "qqq_rsi", "qqq_vwap_dist",
    "qqq_macd_hist_pct", "qqq_adx", "qqq_volume_ratio",
    "m15_ret_1", "m15_ema_spread", "m15_rsi",
    "m15_macd_hist_pct", "m15_adx",
]

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
        raise RuntimeError("مفاتيح Alpaca غير موجودة في متغيرات البيئة.")
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
    return (utc_now() - ts.astimezone(UTC)).total_seconds()


def timestamp_gap_seconds(ts1, ts2):
    if ts1 is None or ts2 is None:
        return None
    return abs((ts1.astimezone(UTC) - ts2.astimezone(UTC)).total_seconds())


def valid_timestamp_age(ts, max_age):
    age = age_seconds(ts)
    return age is not None and -30 <= age <= max_age


# ======================== BAR NORMALIZATION ==================

def normalize_bars(df):
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df = df.dropna(subset=["timestamp"]).set_index("timestamp")
    else:
        df.index = pd.DatetimeIndex(pd.to_datetime(df.index, utc=True))
        df.index.name = "timestamp"

    df = df[~df.index.duplicated(keep="last")].sort_index()

    for col in ["open", "high", "low", "close", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    required = [c for c in ["open", "high", "low", "close"] if c in df.columns]
    if required:
        df = df.dropna(subset=required)
    return df


def remove_incomplete_5m_bars(df, source):
    """
    يضمن عدم قبول أي شمعة لم ينقضِ وقت إغلاقها الفعلي تماماً (منع Lookahead).
    """
    df = normalize_bars(df)
    if df.empty:
        return df

    now = pd.Timestamp(utc_now())
    end_times = df.index + pd.Timedelta(minutes=5)
    completed = end_times <= now
    removed = int((~completed).sum())

    if removed:
        log(f"[الشموع غير المكتملة] {source}: استبعاد {removed} شمعة لم يحن وقت اكتمالها")

    return df.loc[completed].copy()


def validate_bar_freshness(df, source, max_age_minutes):
    if df is None or df.empty:
        raise RuntimeError(f"{source}: لا توجد شموع مكتملة")

    last_ts = df.index[-1].to_pydatetime()
    age_min = (utc_now() - last_ts).total_seconds() / 60

    log(f"[حداثة البيانات] {source}: آخر بداية شمعة={last_ts.isoformat()} | العمر={age_min:.1f} دقيقة")

    if age_min < -2:
        raise RuntimeError(f"{source}: توقيت آخر شمعة في المستقبل")
    if age_min > max_age_minutes:
        raise RuntimeError(f"{source}: بيانات قديمة ({age_min:.1f} دقيقة)")

    return age_min


# ======================== MARKET DATA ========================

def get_alpaca_stock_bars(symbol):
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
        response = HTTP.get(f"{DATA_URL}/v2/stocks/{symbol}/bars", headers=alpaca_headers(), params=params, timeout=25)
        if response.status_code >= 400:
            raise RuntimeError(f"Alpaca HTTP {response.status_code}: {response.text[:200]}")

        rows = response.json().get("bars", [])
        if not rows:
            raise RuntimeError(f"لا توجد بيانات {symbol} من Alpaca")

        df = pd.DataFrame(rows).rename(columns={"t": "timestamp", "o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"})
        df = remove_incomplete_5m_bars(df, f"{symbol} Alpaca")
        df = df.tail(MAX_BARS)

        if len(df) < 100:
            raise RuntimeError(f"شموع {symbol} غير كافية: {len(df)}")

        validate_bar_freshness(df, f"{symbol} 5m Alpaca", MAX_SPX_BAR_AGE_MINUTES)
        return df
    except Exception as exc:
        log(f"[تحذير Alpaca {symbol}] {exc} — محاولة التحويل إلى Yahoo Finance", "warning")
        return pd.DataFrame()


def get_spx_bars():
    df = get_alpaca_stock_bars("SPX")
    if not df.empty:
        log("[مصدر SPX] تم الجلب بنجاح من Alpaca Direct Feed.")
        return df

    try:
        raw = yf.download("^GSPC", period="60d", interval="5m", auto_adjust=False, progress=False, threads=False)
        if raw is None or raw.empty:
            raise RuntimeError("لم تصل بيانات SPX من Yahoo Finance")
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)

        raw = raw.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"})
        raw = remove_incomplete_5m_bars(raw, "SPX Yahoo 5m")
        raw = raw.tail(MAX_BARS)

        if len(raw) < 300:
            raise RuntimeError(f"شموع SPX غير كافية: {len(raw)}")

        validate_bar_freshness(raw, "SPX 5m Yahoo", MAX_SPX_BAR_AGE_MINUTES)
        log("[مصدر SPX] تم الاعتماد على Yahoo Finance (Fallback).")
        return raw
    except Exception as exc:
        log(f"[خطأ بيانات SPX النهائي] {exc}", "error")
        return pd.DataFrame()


def get_spx_spot():
    try:
        response = HTTP.get(f"{DATA_URL}/v2/stocks/SPX/trades/latest", headers=alpaca_headers(), timeout=10)
        if response.status_code < 400:
            trade = response.json().get("trade", {})
            price = number_from(trade, "p", "price")
            ts = parse_timestamp(trade.get("t"))
            if price and ts and valid_timestamp_age(ts, MAX_SPOT_AGE_SECONDS):
                return float(price), ts
    except Exception:
        pass

    try:
        raw = yf.download("^GSPC", period="1d", interval="1m", auto_adjust=False, progress=False, threads=False)
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
        if not valid_timestamp_age(ts, MAX_SPOT_AGE_SECONDS):
            raise RuntimeError("سعر SPX المرجعي قديم")
        return spot, ts
    except Exception as exc:
        log(f"[خطأ سعر SPX المرجعي] {exc}", "error")
        return None, None


def get_stock_bars(symbol):
    df = get_alpaca_stock_bars(symbol)
    if not df.empty:
        return df
    try:
        raw = yf.download(symbol, period="10d", interval="5m", auto_adjust=False, progress=False, threads=False)
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        raw = raw.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"})
        return remove_incomplete_5m_bars(raw, f"{symbol} Yahoo")
    except Exception as exc:
        log(f"[خطأ {symbol}] {exc}", "error")
        return pd.DataFrame()


# ======================== TECHNICAL INDICATORS ===============

def calc_rsi(close, period=14):
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def calculate_macd(close):
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    line = ema12 - ema26
    signal = line.ewm(span=9, adjust=False).mean()
    hist = line - signal
    scale = close.replace(0, np.nan)
    return line / scale, signal / scale, hist / scale


def calculate_adx(high, low, close, period=14):
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=close.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=close.index)
    atr = tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    plus_smoothed = plus_dm.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    minus_smoothed = minus_dm.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    plus_di = 100 * plus_smoothed / atr.replace(0, np.nan)
    minus_di = 100 * minus_smoothed / atr.replace(0, np.nan)
    denom = (plus_di + minus_di).replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / denom
    adx = dx.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    return adx, plus_di, minus_di


def add_session_vwap(df):
    result = pd.Series(np.nan, index=df.index, dtype=float)
    if "volume" not in df.columns:
        return result
    volume = pd.to_numeric(df["volume"], errors="coerce").fillna(0).clip(lower=0)
    if volume.sum() <= 0:
        return result
    local_index = df.index.tz_convert(NY)
    dates = pd.Series(local_index.date, index=df.index)
    minutes = pd.Series(local_index.hour * 60 + local_index.minute, index=df.index)
    typical = (df["high"] + df["low"] + df["close"]) / 3

    for session_date in dates.unique():
        mask = ((dates == session_date) & (minutes >= 570) & (minutes < 960))
        if not mask.any():
            continue
        vol = volume.loc[mask]
        price = typical.loc[mask]
        cumulative_volume = vol.cumsum()
        cumulative_value = (price * vol).cumsum()
        result.loc[mask] = cumulative_value / cumulative_volume.replace(0, np.nan)
    return result


def add_opening_range(df):
    high_result = pd.Series(np.nan, index=df.index, dtype=float)
    low_result = pd.Series(np.nan, index=df.index, dtype=float)
    local_index = df.index.tz_convert(NY)
    dates = pd.Series(local_index.date, index=df.index)
    minutes = pd.Series(local_index.hour * 60 + local_index.minute, index=df.index)

    for session_date in dates.unique():
        session_mask = ((dates == session_date) & (minutes >= 570) & (minutes < 960))
        session = df.loc[session_mask]
        if session.empty:
            continue
        session_minutes = minutes.loc[session.index]
        opening = session[(session_minutes >= 570) & (session_minutes < 585)]
        if len(opening) < 3:
            continue
        opening_high = float(opening["high"].max())
        opening_low = float(opening["low"].min())
        after_opening = session_minutes >= 585
        valid_idx = session.index[after_opening]
        high_result.loc[valid_idx] = opening_high
        low_result.loc[valid_idx] = opening_low

    return high_result, low_result


# ======================== FEATURE ENGINEERING ===============

def build_features(bars, prefix=""):
    df = normalize_bars(bars)
    if df.empty:
        return pd.DataFrame()

    o, h, lo, c = df["open"], df["high"], df["low"], df["close"]
    result = pd.DataFrame(index=df.index)

    result[f"{prefix}ret_1"] = c.pct_change(1)
    result[f"{prefix}ret_3"] = c.pct_change(3)
    result[f"{prefix}ret_6"] = c.pct_change(6)
    result[f"{prefix}rsi"] = calc_rsi(c)

    ema9 = c.ewm(span=9, adjust=False).mean()
    ema21 = c.ewm(span=21, adjust=False).mean()
    ema50 = c.ewm(span=50, adjust=False).mean()

    result[f"{prefix}ema_spread"] = (ema9 - ema21) / c.replace(0, np.nan)
    result[f"{prefix}ema_long_spread"] = (ema21 - ema50) / c.replace(0, np.nan)

    prev = c.shift(1)
    tr = pd.concat([h - lo, (h - prev).abs(), (lo - prev).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()

    result[f"{prefix}atr_pct"] = atr / c.replace(0, np.nan)
    candle_range = (h - lo).replace(0, np.nan)
    result[f"{prefix}range_pct"] = (h - lo) / c.replace(0, np.nan)
    result[f"{prefix}body_ratio"] = (c - o) / candle_range
    result[f"{prefix}volatility"] = c.pct_change().rolling(20).std()
    result[f"{prefix}momentum_accel"] = c.pct_change().diff(3)
    result[f"{prefix}momentum_velocity"] = (c - c.shift(3)) / (candle_range.rolling(3).mean().replace(0, np.nan))

    macd_line, macd_signal, macd_hist = calculate_macd(c)
    result[f"{prefix}macd_line_pct"] = macd_line
    result[f"{prefix}macd_signal_pct"] = macd_signal
    result[f"{prefix}macd_hist_pct"] = macd_hist

    adx, plus_di, minus_di = calculate_adx(h, lo, c)
    result[f"{prefix}adx"] = adx
    result[f"{prefix}plus_di"] = plus_di
    result[f"{prefix}minus_di"] = minus_di

    if "volume" in df.columns and df["volume"].fillna(0).sum() > 0:
        volume = pd.to_numeric(df["volume"], errors="coerce").fillna(0).clip(lower=0)
        average = volume.shift(1).rolling(20).mean()
        result[f"{prefix}volume_ratio"] = volume / average.replace(0, np.nan)
    else:
        result[f"{prefix}volume_ratio"] = np.nan

    session_vwap = add_session_vwap(df)
    result[f"{prefix}session_vwap_dist"] = (c - session_vwap) / c.replace(0, np.nan)

    if prefix in ("spy_", "qqq_"):
        result[f"{prefix}vwap_dist"] = result[f"{prefix}session_vwap_dist"]

    upper_wick = h - pd.concat([o, c], axis=1).max(axis=1)
    lower_wick = pd.concat([o, c], axis=1).min(axis=1) - lo
    result[f"{prefix}upper_wick_ratio"] = upper_wick / candle_range
    result[f"{prefix}lower_wick_ratio"] = lower_wick / candle_range

    prev_open = o.shift(1)
    prev_close = c.shift(1)

    result[f"{prefix}bullish_engulfing"] = ((prev_close < prev_open) & (c > o) & (o <= prev_close) & (c >= prev_open)).astype(float)
    result[f"{prefix}bearish_engulfing"] = ((prev_close > prev_open) & (c < o) & (o >= prev_close) & (c <= prev_open)).astype(float)
    result[f"{prefix}bullish_rejection"] = ((lower_wick > 2 * (c - o).abs()) & (lower_wick > upper_wick) & (c > lo + candle_range.fillna(0) * 0.60)).astype(float)
    result[f"{prefix}bearish_rejection"] = ((upper_wick > 2 * (c - o).abs()) & (upper_wick > lower_wick) & (c < h - candle_range.fillna(0) * 0.60)).astype(float)

    prior_high = h.shift(1).rolling(20).max()
    prior_low = lo.shift(1).rolling(20).min()

    if prefix == "":
        result["resistance_dist"] = (prior_high - c) / c.replace(0, np.nan)
        result["support_dist"] = (c - prior_low) / c.replace(0, np.nan)
        result["breakout_up"] = (c > prior_high).astype(float)
        result["breakout_down"] = (c < prior_low).astype(float)

        prior_prior_high = h.shift(2).rolling(20).max()
        prior_prior_low = lo.shift(2).rolling(20).min()

        result["fakeout_up"] = ((c.shift(1) > prior_prior_high) & (c <= prior_high)).astype(float)
        result["fakeout_down"] = ((c.shift(1) < prior_prior_low) & (c >= prior_low)).astype(float)
        result["retest_up"] = ((c.shift(1) > prior_high.shift(1)) & (lo <= prior_high * 1.001) & (c > prior_high)).astype(float)
        result["retest_down"] = ((c.shift(1) < prior_low.shift(1)) & (h >= prior_low * 0.999) & (c < prior_low)).astype(float)

        or_high, or_low = add_opening_range(df)
        result["opening_range_position"] = (c - or_low) / (or_high - or_low).replace(0, np.nan)
        result["opening_range_break_up"] = ((c > or_high) & or_high.notna()).astype(float)
        result["opening_range_break_down"] = ((c < or_low) & or_low.notna()).astype(float)

    return result.replace([np.inf, -np.inf], np.nan)


# ======================== 15-MINUTE ALIGNMENT FIX ===========

def build_15m_features(spx):
    """
    تجميع شموع 15 دقيقة المكتملة بدقة تامة وفق توقيت نيويورك الرسمي،
    مع التأكد من اكتمال الشموع الثلاث الفرعية (5 دقائق لكل منها) ومنع أي تسريب مستقبلي.
    """
    df = normalize_bars(spx)
    if df.empty:
        return pd.DataFrame()

    local = df.copy()
    local.index = local.index.tz_convert(NY)

    bars15 = local.resample("15min", closed="left", label="right").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    })

    counts = local["close"].resample("15min", closed="left", label="right").count()
    bars15 = bars15.loc[counts == 3].dropna(subset=["open", "high", "low", "close"])
    if bars15.empty:
        return pd.DataFrame()

    now_ny = pd.Timestamp(ny_now())
    bars15 = bars15.loc[bars15.index <= now_ny]
    if bars15.empty:
        return pd.DataFrame()

    local_index = bars15.index
    minutes = local_index.hour * 60 + local_index.minute
    bars15 = bars15.loc[(minutes >= 585) & (minutes <= 960)]
    if bars15.empty:
        return pd.DataFrame()

    bars15.index = bars15.index.tz_convert(UTC)
    f15 = build_features(bars15)

    return f15[[
        "ret_1", "ema_spread", "rsi", "macd_hist_pct", "adx",
    ]].rename(columns={
        "ret_1": "m15_ret_1",
        "ema_spread": "m15_ema_spread",
        "rsi": "m15_rsi",
        "macd_hist_pct": "m15_macd_hist_pct",
        "adx": "m15_adx",
    })


# ======================== FEATURE MERGE ======================

def make_features(spx, spy, qqq):
    spx = normalize_bars(spx)
    spy = normalize_bars(spy)
    qqq = normalize_bars(qqq)

    if spx.empty or spy.empty or qqq.empty:
        raise RuntimeError("بيانات SPX أو SPY أو QQQ غير متوفرة")

    newest = [spx.index[-1], spy.index[-1], qqq.index[-1]]
    sync_gap = (max(newest) - min(newest)).total_seconds() / 60
    if sync_gap > MAX_SOURCE_SYNC_GAP_MINUTES:
        raise RuntimeError(f"مصادر البيانات غير متزامنة؛ الفرق {sync_gap:.1f} دقيقة")

    spxf = build_features(spx)
    spy_all = build_features(spy, "spy_")
    qqq_all = build_features(qqq, "qqq_")

    spy_cols = ["spy_ret_1", "spy_ret_3", "spy_ema_spread", "spy_rsi", "spy_vwap_dist", "spy_macd_hist_pct", "spy_adx", "spy_volume_ratio"]
    qqq_cols = ["qqq_ret_1", "qqq_ret_3", "qqq_ema_spread", "qqq_rsi", "qqq_vwap_dist", "qqq_macd_hist_pct", "qqq_adx", "qqq_volume_ratio"]

    m15 = build_15m_features(spx)

    merged = (
        spxf.join(spy_all[spy_cols], how="inner")
        .join(qqq_all[qqq_cols], how="inner")
        .join(m15, how="left")
        .join(spx[["close"]], how="inner")
    )

    merged = merged.replace([np.inf, -np.inf], np.nan).dropna(subset=["close"]).sort_index()
    required_history = ["ret_1", "rsi", "ema_spread", "atr_pct", "macd_hist_pct"]
    merged = merged.dropna(subset=required_history)

    if len(merged) < 400:
        raise RuntimeError(f"صفوف البيانات المتطابقة غير كافية: {len(merged)}")

    return merged


# ======================== MACHINE LEARNING ===================

def new_model():
    return make_pipeline(
        SimpleImputer(strategy="median"),
        HistGradientBoostingClassifier(
            max_iter=100,
            learning_rate=0.05,
            max_leaf_nodes=12,
            l2_regularization=2.0,
            random_state=42,
        ),
    )


def train_model(df):
    data = df.sort_index().copy()
    future_close = data["close"].shift(-LABEL_HORIZON)
    valid = future_close.notna()

    X = data.loc[valid, FEATURES].replace([np.inf, -np.inf], np.nan)
    y = (future_close.loc[valid] > data.loc[valid, "close"]).astype(int)

    if len(X) < 500:
        raise RuntimeError(f"بيانات التدريب غير كافية: {len(X)}")

    split = int(len(X) * 0.80)
    train_end = split - LABEL_HORIZON
    if train_end < 200 or len(X) - split < 100:
        raise RuntimeError("بيانات التدريب أو الاختبار غير كافية")

    X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
    X_test, y_test = X.iloc[split:], y.iloc[split:]

    if y_train.nunique() < 2 or y_test.nunique() < 2:
        raise RuntimeError("بيانات التدريب أو الاختبار تحتوي على فئة واحدة")

    evaluator = new_model()
    evaluator.fit(X_train, y_train)

    probabilities = evaluator.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    accuracy = float(accuracy_score(y_test, predictions))
    auc = float(roc_auc_score(y_test, probabilities))
    baseline_accuracy = float((y_test == int(y_train.mean() >= 0.5)).mean())

    log(f"[اختبار اتجاه SPX] صفوف={len(X)} | اختبار={len(X_test)} | دقة={accuracy:.3f} | AUC={auc:.3f}")

    live_model = new_model()
    live_model.fit(X, y)
    return live_model, FEATURES.copy(), auc, accuracy, baseline_accuracy


# ======================== DIRECTIONAL CONFIRMATION ===========

def _num(row, key, default=np.nan):
    try:
        value = float(row.get(key, default))
        return value if np.isfinite(value) else default
    except (TypeError, ValueError):
        return default


def directional_votes(latest):
    rsi = _num(latest, "rsi")
    macd = _num(latest, "macd_hist_pct")
    velocity = _num(latest, "momentum_velocity", 0.0)

    spx_up = (_num(latest, "ret_1") > 0 and _num(latest, "ema_spread") > 0 and macd > 0 and 45 <= rsi <= 75 and velocity > -0.2)
    spx_down = (_num(latest, "ret_1") < 0 and _num(latest, "ema_spread") < 0 and macd < 0 and 25 <= rsi <= 55 and velocity < 0.2)

    spy_up = (_num(latest, "spy_ret_1") > 0 and _num(latest, "spy_ema_spread") > 0 and _num(latest, "spy_macd_hist_pct") > 0 and 45 <= _num(latest, "spy_rsi") <= 75)
    spy_down = (_num(latest, "spy_ret_1") < 0 and _num(latest, "spy_ema_spread") < 0 and _num(latest, "spy_macd_hist_pct") < 0 and 25 <= _num(latest, "spy_rsi") <= 55)

    qqq_up = (_num(latest, "qqq_ret_1") > 0 and _num(latest, "qqq_ema_spread") > 0 and _num(latest, "qqq_macd_hist_pct") > 0 and 45 <= _num(latest, "qqq_rsi") <= 75)
    qqq_down = (_num(latest, "qqq_ret_1") < 0 and _num(latest, "qqq_ema_spread") < 0 and _num(latest, "qqq_macd_hist_pct") < 0 and 25 <= _num(latest, "qqq_rsi") <= 55)

    m15_up = (_num(latest, "m15_ret_1") > 0 and _num(latest, "m15_ema_spread") > 0 and _num(latest, "m15_macd_hist_pct") > 0)
    m15_down = (_num(latest, "m15_ret_1") < 0 and _num(latest, "m15_ema_spread") < 0 and _num(latest, "m15_macd_hist_pct") < 0)

    up = sum(map(bool, [spx_up, spy_up, qqq_up, m15_up]))
    down = sum(map(bool, [spx_down, spy_down, qqq_down, m15_down]))
    return up, down


def calculate_signal(df, model, feature_cols, auc):
    if not np.isfinite(auc) or auc < MIN_AUC:
        return "WAIT", 0.0, f"AUC={auc:.3f} أقل من الحد"

    latest = df.iloc[-1]
    x = latest[feature_cols].to_frame().T

    p_up = float(model.predict_proba(x)[0, 1])
    p_down = 1.0 - p_up

    up_votes, down_votes = directional_votes(latest)
    adx = _num(latest, "adx")
    velocity = abs(_num(latest, "momentum_velocity", 0.0))

    if np.isfinite(adx) and adx < 14:
        return "WAIT", max(p_up, p_down), f"ADX={adx:.1f}: سوق عرضي"
    if np.isfinite(velocity) and velocity < 0.15:
        return "WAIT", max(p_up, p_down), f"سرعة الزخم منخفضة ({velocity:.2f})"

    if p_up >= MIN_CONFIDENCE and up_votes >= MIN_DIRECTION_VOTES:
        if down_votes > 0:
            return "WAIT", p_up, "تعارض مع الهبوط"
        return "CALL", p_up, f"صعودي={p_up:.1%}"

    if p_down >= MIN_CONFIDENCE and down_votes >= MIN_DIRECTION_VOTES:
        if up_votes > 0:
            return "WAIT", p_down, "تعارض مع الصعود"
        return "PUT", p_down, f"هبوطي={p_down:.1%}"

    return "WAIT", max(p_up, p_down), "ثقة غير كافية"


# ======================== TELEGRAM ===========================

def send_telegram(message):
    if not TG_TOKEN or not TG_CHAT:
        return False
    try:
        response = HTTP.post(
            f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage",
            json={"chat_id": TG_CHAT, "text": message, "disable_web_page_preview": True},
            timeout=20,
        )
        return response.status_code < 400 and response.json().get("ok", False)
    except Exception:
        return False


# ======================== OPTIONS CONTRACTS =================-

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
        response = HTTP.get(f"{TRADE_URL}/v2/options/contracts", headers=alpaca_headers(), params=params, timeout=25)
        if response.status_code >= 400:
            return []
        payload = response.json()
        contracts = payload.get("option_contracts") or payload.get("contracts") or []
        results = []
        for item in contracts:
            symbol = item.get("symbol", "")
            root = item.get("root_symbol", "")
            if not symbol or (root != "SPXW" and not symbol.startswith("SPXW")):
                continue
            try:
                strike = float(item["strike_price"])
            except (KeyError, TypeError, ValueError):
                continue
            if str(item.get("expiration_date", ""))[:10] != today:
                continue
            results.append({"symbol": symbol, "strike": strike, "expiration": today})
        return results
    except Exception:
        return []


def get_option_quotes(symbols):
    results = {}
    endpoint = f"{DATA_URL}/v1beta1/options/quotes/latest"
    for start in range(0, len(symbols), 100):
        batch = symbols[start:start + 100]
        try:
            response = HTTP.get(endpoint, headers=alpaca_headers(), params={"symbols": ",".join(batch), "feed": OPTIONS_FEED}, timeout=25)
            if response.status_code < 400:
                quotes = response.json().get("quotes", {}) or {}
                if isinstance(quotes, dict):
                    results.update(quotes)
        except Exception:
            pass
    return results


def get_option_snapshots(symbols):
    results = {}
    endpoint = f"{DATA_URL}/v1beta1/options/snapshots"
    for start in range(0, len(symbols), 100):
        batch = symbols[start:start + 100]
        try:
            response = HTTP.get(endpoint, headers=alpaca_headers(), params={"symbols": ",".join(batch), "feed": OPTIONS_FEED}, timeout=25)
            if response.status_code < 400:
                payload = response.json().get("snapshots", {}) or {}
                if isinstance(payload, dict):
                    results.update(payload)
        except Exception:
            pass
    return results


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
            value = float(quote.get(key))
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
    if age is None or age < -30 or age > MAX_OPTION_QUOTE_AGE_SECONDS:
        return None

    bid = number_from(quote, "bp", "bid_price", "bid")
    ask = number_from(quote, "ap", "ask_price", "ask")
    if bid is None or ask is None or bid <= 0 or ask <= 0 or ask < bid:
        return None
    mid = (bid + ask) / 2
    if mid <= 0:
        return None
    return {
        "bid": bid, "ask": ask, "mid": mid,
        "spread_pct": (ask - bid) / mid,
        "quote_ts": ts, "quote_age_seconds": age,
    }


def find_greeks_timestamp(snapshot, greeks):
    for container in (greeks, snapshot):
        if not isinstance(container, dict):
            continue
        for key in ("greeks_timestamp", "greeks_time", "calculated_at", "calculation_timestamp"):
            ts = parse_timestamp(container.get(key))
            if ts is not None:
                return ts
    return None


def validate_real_delta(direction, snapshot):
    if not isinstance(snapshot, dict):
        return None, None, "لا توجد لقطة"
    greeks = snapshot.get("greeks") or {}
    try:
        delta = float(greeks.get("delta"))
    except (TypeError, ValueError):
        return None, None, "Delta غير متاحة"
    if not np.isfinite(delta) or delta == 0 or abs(delta) > 1:
        return None, None, "Delta غير صالحة"
    if direction == "CALL" and delta <= 0:
        return None, None, "Delta لا تتوافق مع CALL"
    if direction == "PUT" and delta >= 0:
        return None, None, "Delta لا تتوافق مع PUT"
    if not MIN_ABS_DELTA <= abs(delta) <= MAX_ABS_DELTA:
        return None, None, "Delta خارج النطاق"
    
    greeks_ts = find_greeks_timestamp(snapshot, greeks)
    if greeks_ts is not None:
        if age_seconds(greeks_ts) > MAX_DELTA_QUOTE_AGE_SECONDS:
            return None, None, "وقت Greeks قديم"
        return delta, greeks_ts, "وقت Greeks مستقل"
    
    if REQUIRE_INDEPENDENT_GREEKS_TIMESTAMP:
        return None, None, "المصدر لا يوفر توقيت Greeks مستقل"
    
    latest_quote = snapshot.get("latestQuote") or snapshot.get("latest_quote") or {}
    proxy_ts = quote_timestamp(latest_quote)
    if age_seconds(proxy_ts) > MAX_DELTA_QUOTE_AGE_SECONDS:
        return None, None, "توقيت Delta المرجعي قديم"
    return delta, proxy_ts, "توقيت مرجعي تقريبي"


def get_dynamic_max_spread():
    now = ny_now()
    current_minutes = now.hour * 60 + now.minute
    close_minutes = 16 * 60
    if (close_minutes - current_minutes) <= 30:
        return BASE_MAX_SPREAD_PCT * 0.75
    return BASE_MAX_SPREAD_PCT


def evaluate_contract(contract, direction, spot_price, spot_ts, quote, snapshot):
    q = normalize_quote(quote)
    if q is None:
        return None
    if q["spread_pct"] > get_dynamic_max_spread():
        return None
    if not MIN_OPTION_PREMIUM <= q["ask"] <= MAX_OPTION_PREMIUM:
        return None

    delta, delta_ts, delta_basis = validate_real_delta(direction, snapshot)
    if delta is None or delta_ts is None:
        return None

    spot_quote_gap = timestamp_gap_seconds(spot_ts, q["quote_ts"])
    if spot_quote_gap is None or spot_quote_gap > MAX_SPOT_OPTION_GAP_SECONDS:
        return None

    return {
        **contract, **q,
        "delta": delta,
        "delta_ts": delta_ts,
        "delta_timestamp_basis": delta_basis,
        "delta_quote_age_seconds": age_seconds(delta_ts),
        "distance": abs(contract["strike"] - spot_price),
        "spot_quote_gap_seconds": spot_quote_gap,
    }


def choose_contract(direction, spot_price, spot_ts):
    if not valid_timestamp_age(spot_ts, MAX_SPOT_AGE_SECONDS):
        return None
    contracts = get_spxw_contracts(direction)
    if not contracts:
        return None

    if direction == "CALL":
        contracts = [c for c in contracts if MIN_TARGET_DISTANCE <= c["strike"] - spot_price <= MAX_TARGET_DISTANCE]
    else:
        contracts = [c for c in contracts if MIN_TARGET_DISTANCE <= spot_price - c["strike"] <= MAX_TARGET_DISTANCE]

    if not contracts:
        return None

    contracts.sort(key=lambda c: abs(c["strike"] - spot_price))
    candidates = contracts[:MAX_CONTRACTS_TO_CHECK]
    symbols = [c["symbol"] for c in candidates]

    quotes = get_option_quotes(symbols)
    snapshots = get_option_snapshots(symbols)
    viable = []

    for contract in candidates:
        selected = evaluate_contract(
            contract, direction, spot_price, spot_ts,
            quotes.get(contract["symbol"]), snapshots.get(contract["symbol"])
        )
        if selected:
            viable.append(selected)

    if not viable:
        return None

    viable.sort(key=lambda c: (abs(abs(c["delta"]) - TARGET_ABS_DELTA), c["spread_pct"], c["distance"]))
    return viable[0]


# ======================== OUTCOME JOURNAL ====================

def journal_columns():
    cols = [
        "signal_time_utc", "direction", "symbol", "spot", "spot_quote_time_utc",
        "strike", "entry_bid_observed", "entry_ask_observed", "entry_mid_observed",
        "entry_delta", "delta_timestamp_basis", "confidence", "auc_direction_only",
        "quote_time_utc", "journal_type",
    ]
    for minutes in OUTCOME_MINUTES:
        cols += [
            f"bid_observed_after_{minutes}m", f"ask_observed_after_{minutes}m",
            f"observation_time_{minutes}m_utc", f"source_quote_time_{minutes}m_utc",
            f"observation_delay_seconds_{minutes}m", f"observed_bid_change_vs_entry_ask_{minutes}m",
        ]
    return cols


def read_journal():
    if not os.path.exists(SIGNAL_LOG_PATH):
        return pd.DataFrame(columns=journal_columns())
    df = pd.read_csv(SIGNAL_LOG_PATH)
    for col in journal_columns():
        if col not in df.columns:
            df[col] = np.nan
    return df


def write_journal(df):
    for col in journal_columns():
        if col not in df.columns:
            df[col] = np.nan
    df[journal_columns()].to_csv(SIGNAL_LOG_PATH, index=False)


def append_signal_journal(row):
    try:
        df = read_journal()
        write_journal(pd.concat([df, pd.DataFrame([row])], ignore_index=True))
    except Exception:
        pass


def update_outcome_journal():
    try:
        if not os.path.exists(SIGNAL_LOG_PATH):
            return
        df = read_journal()
        if df.empty:
            return

        now = utc_now()
        pending = {}
        for idx, row in df.iterrows():
            symbol = row.get("symbol")
            start = parse_timestamp(row.get("signal_time_utc"))
            if not isinstance(symbol, str) or not symbol or start is None:
                continue
            elapsed = (now - start).total_seconds()
            for minutes in OUTCOME_MINUTES:
                if pd.isna(row.get(f"bid_observed_after_{minutes}m")) and elapsed >= minutes * 60:
                    pending.setdefault(symbol, []).append({"idx": idx, "start": start, "minutes": minutes, "entry_ask": row.get("entry_ask_observed")})

        if not pending:
            return

        quotes = get_option_quotes(list(pending.keys()))
        changed = False
        for symbol, entries in pending.items():
            q = normalize_quote(quotes.get(symbol))
            if q is None:
                continue
            for item in entries:
                idx, minutes, target = item["idx"], item["minutes"], item["start"] + timedelta(minutes=minutes)
                if q["quote_ts"] < target:
                    continue
                df.at[idx, f"bid_observed_after_{minutes}m"] = q["bid"]
                df.at[idx, f"ask_observed_after_{minutes}m"] = q["ask"]
                df.at[idx, f"observation_time_{minutes}m_utc"] = now.isoformat()
                df.at[idx, f"source_quote_time_{minutes}m_utc"] = q["quote_ts"].isoformat()
                df.at[idx, f"observation_delay_seconds_{minutes}m"] = max(0.0, (q["quote_ts"] - target).total_seconds())
                try:
                    ea = float(item["entry_ask"])
                    df.at[idx, f"observed_bid_change_vs_entry_ask_{minutes}m"] = (q["bid"] - ea) / ea if ea > 0 else np.nan
                except Exception:
                    pass
                changed = True
        if changed:
            write_journal(df)
    except Exception:
        pass


def is_duplicate(direction, symbol):
    key = f"{direction}:{symbol}"
    last = LAST_ALERTS.get(key)
    return bool(last and (utc_now() - last).total_seconds() < ALERT_COOLDOWN_MINUTES * 60)


def mark_alert_sent(direction, symbol):
    LAST_ALERTS[f"{direction}:{symbol}"] = utc_now()


def run_scan():
    log(f"========== بدء فحص | {BOT_NAME} v{VERSION} ==========")
    update_outcome_journal()

    if not API_KEY or not API_SECRET:
        return

    spx = get_spx_bars()
    spy = get_stock_bars("SPY")
    qqq = get_stock_bars("QQQ")

    if spx.empty or spy.empty or qqq.empty:
        return

    # ربط حداثة الإشارة بالتوقيت الحقيقي لآخر شمعة مكتملة
    last_bar_ts = spx.index[-1].to_pydatetime()
    bar_age_sec = (utc_now() - last_bar_ts).total_seconds()
    if bar_age_sec > MAX_SIGNAL_AGE_BEFORE_SEND_SECONDS + 60:
        log("[الإشارة] WAIT — الشمعة الأخيرة قديمة جداً لتوليد إشارة موثوقة", "warning")
        return

    spot_price, spot_ts = get_spx_spot()
    if spot_price is None or spot_ts is None:
        return

    try:
        merged = make_features(spx, spy, qqq)
        model, columns, auc, accuracy, baseline = train_model(merged)
        direction, confidence, reason = calculate_signal(merged, model, columns, auc)

        log(f"[الإشارة] {direction} | الثقة={confidence:.1%} | AUC={auc:.3f} | السبب={reason}")
        if direction == "WAIT":
            return

        contract = choose_contract(direction, spot_price, spot_ts)
        if contract is None or is_duplicate(direction, contract["symbol"]):
            return

        # ==========================================
        # إعادة التحقق الفوري (Pre-Send Re-Validation)
        # ==========================================
        fresh_spot, fresh_spot_ts = get_spx_spot()
        if fresh_spot is None or not valid_timestamp_age(fresh_spot_ts, MAX_SPOT_AGE_SECONDS):
            log("[الإشارة] WAIT — فشل التحقق الفوري لـ SPX قبل الإرسال", "warning")
            return

        fresh_quotes = get_option_quotes([contract["symbol"]])
        fresh_snapshots = get_option_snapshots([contract["symbol"]])
        rechecked_contract = evaluate_contract(
            contract, direction, fresh_spot, fresh_spot_ts,
            fresh_quotes.get(contract["symbol"]), fresh_snapshots.get(contract["symbol"])
        )

        if rechecked_contract is None:
            log("[الإشارة] WAIT — العقد لم يعد يجتاز الشروط الصارمة عند إعادة التحقق الفوري", "warning")
            return

        contract.update(rechecked_contract)
        signal_time = utc_now()

        # فحص الحداثة النهائي قبل الإرسال الفعلي
        if (signal_time - last_bar_ts).total_seconds() > MAX_SIGNAL_AGE_BEFORE_SEND_SECONDS + 120:
            log("[الإشارة] WAIT — تجاوز وقت الإرسال الحد الأقصى للحداثة", "warning")
            return

        message = (
            f"🦅 صَفاء التحليل — v{VERSION}\n"
            f"🟢 اتجاه: {direction} | ثقة: {confidence:.1%}\n"
            f"💹 SPX (المُحقق): {fresh_spot:,.2f} | التنفيذ: {contract['strike']:,.2f}\n"
            f"🧮 Delta: {contract['delta']:+.3f} | السبريد: {contract['spread_pct']:.1%}\n"
            f"📌 السبب: {reason}"
        )

        if send_telegram(message):
            mark_alert_sent(direction, contract["symbol"])
            append_signal_journal({
                "signal_time_utc": signal_time.isoformat(),
                "direction": direction,
                "symbol": contract["symbol"],
                "spot": fresh_spot,
                "spot_quote_time_utc": fresh_spot_ts.isoformat(),
                "strike": contract["strike"],
                "entry_bid_observed": contract["bid"],
                "entry_ask_observed": contract["ask"],
                "entry_mid_observed": contract["mid"],
                "entry_delta": contract["delta"],
                "delta_timestamp_basis": contract["delta_timestamp_basis"],
                "confidence": confidence,
                "auc_direction_only": auc,
                "quote_time_utc": contract["quote_ts"].isoformat(),
                "journal_type": "OBSERVED_QUOTES_ONLY_NOT_ACTUAL_EXECUTION",
            })
    except Exception as exc:
        log(f"[خطأ الفحص] {exc}", "error")


def market_is_open():
    now = ny_now()
    if now.weekday() >= 5:
        return False
    minutes = now.hour * 60 + now.minute
    return (9 * 60 + 30) <= minutes < 16 * 60


def main():
    log(f"بدء تشغيل {BOT_NAME} — v{VERSION}")
    while True:
        try:
            if market_is_open():
                run_scan()
            else:
                log("[السوق] مغلق حالياً")
        except KeyboardInterrupt:
            break
        except Exception as exc:
            log(f"[خطأ رئيسي] {exc}", "error")
        time.sleep(SCAN_SECONDS)


if __name__ == "__main__":
    main()
