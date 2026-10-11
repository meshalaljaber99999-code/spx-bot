# ============================================================
# SPX 0DTE AI ADVISOR v19.0 — صَفاء التحليل
# SPXW 0DTE recommendations only — NO ORDER EXECUTION
# Freshness checks, target-side contract filter, technical confirmation,
# duplicate suppression, Arabic Telegram layout, and outcome journaling.
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

# ======================== CONFIG =============================
NY = ZoneInfo("America/New_York")
UTC = timezone.utc

API_KEY = (os.getenv("ALPACA_API_KEY") or os.getenv("APCA_API_KEY_ID") or "").strip()
API_SECRET = (os.getenv("ALPACA_SECRET_KEY") or os.getenv("APCA_API_SECRET_KEY") or "").strip()

TRADE_URL = os.getenv("ALPACA_TRADE_URL", "https://paper-api.alpaca.markets").rstrip("/")
DATA_URL = os.getenv("ALPACA_DATA_URL", "https://data.alpaca.markets").rstrip("/")
DATA_FEED = os.getenv("ALPACA_DATA_FEED", "iex").lower()
OPTIONS_FEED = os.getenv("ALPACA_OPTIONS_FEED", "indicative").lower()

TG_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TG_CHAT = os.getenv("TELEGRAM_CHAT_ID", "").strip()

# More selective defaults. These are filters, not profit guarantees.
MIN_AUC = float(os.getenv("MIN_MODEL_AUC", "0.55"))
MIN_CONFIDENCE = float(os.getenv("MIN_MODEL_CONFIDENCE", "0.62"))
MAX_SPREAD_PCT = float(os.getenv("MAX_OPTION_SPREAD_PCT", "0.12"))
MIN_OPTION_PREMIUM = float(os.getenv("MIN_OPTION_PREMIUM", "0.50"))
MAX_OPTION_PREMIUM = float(os.getenv("MAX_OPTION_PREMIUM", "50.00"))
MIN_TARGET_DISTANCE = float(os.getenv("MIN_TARGET_DISTANCE_POINTS", "1.00"))
MAX_TARGET_DISTANCE = float(os.getenv("MAX_TARGET_DISTANCE_POINTS", "15.00"))
MIN_DIRECTION_VOTES = int(os.getenv("MIN_DIRECTION_VOTES", "2"))

MAX_SPX_BAR_AGE_MINUTES = float(os.getenv("MAX_SPX_BAR_AGE_MINUTES", "12"))
MAX_STOCK_BAR_AGE_MINUTES = float(os.getenv("MAX_STOCK_BAR_AGE_MINUTES", "12"))
MAX_COMMON_BAR_AGE_MINUTES = float(os.getenv("MAX_COMMON_BAR_AGE_MINUTES", "12"))
MAX_SOURCE_SYNC_GAP_MINUTES = float(os.getenv("MAX_SOURCE_SYNC_GAP_MINUTES", "8"))
MAX_SPOT_AGE_SECONDS = float(os.getenv("MAX_SPOT_AGE_SECONDS", "180"))
MAX_OPTION_QUOTE_AGE_SECONDS = float(os.getenv("MAX_OPTION_QUOTE_AGE_SECONDS", "180"))

SCAN_SECONDS = max(20, int(os.getenv("SCAN_SECONDS", "60")))
ALERT_COOLDOWN_MINUTES = max(1, int(os.getenv("ALERT_COOLDOWN_MINUTES", "20")))
LABEL_HORIZON = 3  # 3 five-minute bars: directional target, NOT option-profit target
MAX_BARS = 5000
MAX_CONTRACTS_TO_CHECK = 100
SIGNAL_LOG_PATH = os.getenv("SIGNAL_LOG_PATH", "spxw_signal_journal.csv")

FEATURES = [
    "ret_1", "ret_3", "ret_6", "rsi", "ema_spread", "ema_long_spread",
    "atr_pct", "range_pct", "body_ratio", "volatility", "momentum_accel",
    "resistance_dist", "support_dist",
    "spy_ret_1", "spy_ret_3", "spy_ema_spread", "spy_rsi", "spy_vwap_dist",
    "qqq_ret_1", "qqq_ret_3", "qqq_ema_spread", "qqq_rsi", "qqq_vwap_dist",
]
OUTCOME_MINUTES = (1, 3, 5, 10)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
HTTP = requests.Session()
LAST_ALERTS = {}  # in-process duplicate suppression; journal persists when disk persists


def log(message, level="info"):
    getattr(logging, level, logging.info)(message)


def utc_now():
    return datetime.now(UTC)


def ny_now():
    return datetime.now(NY)


def alpaca_headers():
    if not API_KEY or not API_SECRET:
        raise RuntimeError("مفاتيح Alpaca غير موجودة في متغيرات البيئة.")
    return {"APCA-API-KEY-ID": API_KEY, "APCA-API-SECRET-KEY": API_SECRET}


def parse_timestamp(value):
    if value is None or value == "":
        return None
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


def normalize_bars(df):
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df = df.dropna(subset=["timestamp"]).set_index("timestamp")
    else:
        df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
        df = df.loc[~df.index.isna()]
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


def validate_bar_freshness(df, source, max_age_minutes):
    if df is None or df.empty:
        raise RuntimeError(f"{source}: لا توجد شموع")
    last_ts = df.index[-1].to_pydatetime()
    age_min = (utc_now() - last_ts).total_seconds() / 60
    log(f"[حداثة البيانات] {source}: آخر شمعة={last_ts.isoformat()} | العمر={age_min:.1f} دقيقة")
    if age_min < -2:
        raise RuntimeError(f"{source}: توقيت آخر شمعة في المستقبل بشكل غير متوقع")
    if age_min > max_age_minutes:
        raise RuntimeError(f"{source}: بيانات قديمة ({age_min:.1f} دقيقة؛ الحد {max_age_minutes:.1f})")
    return age_min


# ======================== MARKET DATA ========================
def get_spx_bars():
    """5-minute SPX candles from Yahoo Finance; fail closed when stale."""
    try:
        raw = yf.download("^GSPC", period="60d", interval="5m", auto_adjust=False,
                          progress=False, threads=False)
        if raw is None or raw.empty:
            raise RuntimeError("لم تصل بيانات SPX من Yahoo Finance")
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        raw = raw.rename(columns={"Open": "open", "High": "high", "Low": "low",
                                  "Close": "close", "Volume": "volume"})
        raw = normalize_bars(raw)
        keep = [c for c in ["open", "high", "low", "close", "volume"] if c in raw.columns]
        raw = raw[keep].tail(MAX_BARS)
        if len(raw) < 300:
            raise RuntimeError(f"عدد شموع SPX غير كافٍ: {len(raw)}")
        validate_bar_freshness(raw, "SPX 5m", MAX_SPX_BAR_AGE_MINUTES)
        return raw
    except Exception as exc:
        log(f"[خطأ بيانات SPX] {exc}", "error")
        return pd.DataFrame()


def get_spx_spot():
    """Fetch a separate 1-minute SPX reference close and reject stale values.
    This is the latest Yahoo minute-bar close, not a guaranteed exchange tick."""
    try:
        raw = yf.download("^GSPC", period="1d", interval="1m", auto_adjust=False,
                          progress=False, threads=False)
        if raw is None or raw.empty:
            raise RuntimeError("لم تصل بيانات SPX الدقيقة")
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        raw = raw.rename(columns={"Close": "close"})
        raw = normalize_bars(raw)
        if raw.empty or "close" not in raw:
            raise RuntimeError("بيانات SPX الدقيقة غير صالحة")
        ts = raw.index[-1].to_pydatetime()
        age = age_seconds(ts)
        log(f"[سعر SPX المرجعي] {raw['close'].iloc[-1]:,.2f} | توقيت={ts.isoformat()} | العمر={age:.0f} ثانية")
        if age is None or age < -30 or age > MAX_SPOT_AGE_SECONDS:
            raise RuntimeError(f"سعر SPX المرجعي قديم/غير صالح (العمر={age})")
        return float(raw["close"].iloc[-1]), ts
    except Exception as exc:
        log(f"[خطأ سعر SPX المرجعي] {exc}", "error")
        return None, None


def get_stock_bars(symbol):
    try:
        end = utc_now()
        start = end - timedelta(days=10)
        params = {"timeframe": "5Min", "start": start.isoformat().replace("+00:00", "Z"),
                  "end": end.isoformat().replace("+00:00", "Z"), "limit": 10000,
                  "adjustment": "raw", "feed": DATA_FEED, "sort": "asc"}
        response = HTTP.get(f"{DATA_URL}/v2/stocks/{symbol}/bars", headers=alpaca_headers(),
                            params=params, timeout=25)
        if response.status_code >= 400:
            raise RuntimeError(f"Alpaca HTTP {response.status_code}: {response.text[:200]}")
        rows = response.json().get("bars", [])
        if not rows:
            raise RuntimeError(f"لا توجد بيانات {symbol}؛ feed={DATA_FEED}")
        df = pd.DataFrame(rows).rename(columns={"t": "timestamp", "o": "open", "h": "high",
                                                "l": "low", "c": "close", "v": "volume"})
        df = normalize_bars(df).tail(MAX_BARS)
        if len(df) < 100:
            raise RuntimeError(f"عدد شموع {symbol} غير كافٍ: {len(df)}")
        validate_bar_freshness(df, f"{symbol} 5m", MAX_STOCK_BAR_AGE_MINUTES)
        return df
    except Exception as exc:
        log(f"[خطأ بيانات {symbol}] {exc}", "error")
        return pd.DataFrame()


# ======================== TECHNICAL FEATURES =================
def calc_rsi(close, period=14):
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


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
    result[f"{prefix}atr_pct"] = tr.rolling(14).mean() / c.replace(0, np.nan)
    result[f"{prefix}range_pct"] = (h - lo) / c.replace(0, np.nan)
    result[f"{prefix}body_ratio"] = (c - o) / (h - lo).replace(0, np.nan)
    result[f"{prefix}volatility"] = c.pct_change().rolling(20).std()
    result[f"{prefix}momentum_accel"] = c.pct_change().diff(3)

    # Prior 20-bar support/resistance levels (exclude current candle).
    prior_high = h.shift(1).rolling(20).max()
    prior_low = lo.shift(1).rolling(20).min()
    if prefix == "":
        result["resistance_dist"] = (prior_high - c) / c.replace(0, np.nan)
        result["support_dist"] = (c - prior_low) / c.replace(0, np.nan)

    # VWAP distance for SPY/QQQ when volume is supplied by the feed.
    if prefix in ("spy_", "qqq_"):
        if "volume" in df.columns and df["volume"].fillna(0).sum() > 0:
            typical = (h + lo + c) / 3.0
            vol = df["volume"].fillna(0).clip(lower=0)
            vwap_num = (typical * vol).rolling(20).sum()
            vwap_den = vol.rolling(20).sum().replace(0, np.nan)
            vwap = vwap_num / vwap_den
            result[f"{prefix}vwap_dist"] = (c - vwap) / c.replace(0, np.nan)
        else:
            result[f"{prefix}vwap_dist"] = np.nan
    return result.replace([np.inf, -np.inf], np.nan)


def make_features(spx, spy, qqq):
    spx, spy, qqq = normalize_bars(spx), normalize_bars(spy), normalize_bars(qqq)
    if spx.empty or spy.empty or qqq.empty:
        raise RuntimeError("بيانات SPX أو SPY أو QQQ غير متوفرة")
    newest = [spx.index[-1], spy.index[-1], qqq.index[-1]]
    sync_gap = (max(newest) - min(newest)).total_seconds() / 60
    if sync_gap > MAX_SOURCE_SYNC_GAP_MINUTES:
        raise RuntimeError(f"مصادر البيانات غير متزامنة؛ الفرق {sync_gap:.1f} دقيقة")

    spxf = build_features(spx)
    spyf = build_features(spy, "spy_")[["spy_ret_1", "spy_ret_3", "spy_ema_spread", "spy_rsi", "spy_vwap_dist"]]
    qqf = build_features(qqq, "qqq_")[["qqq_ret_1", "qqq_ret_3", "qqq_ema_spread", "qqq_rsi", "qqq_vwap_dist"]]
    merged = spxf.join(spyf, how="inner").join(qqf, how="inner").join(spx[["close"]], how="inner")
    merged = merged.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURES + ["close"]).sort_index()
    if len(merged) < 400:
        raise RuntimeError(f"عدد صفوف البيانات المتطابقة غير كافٍ: {len(merged)}")
    common_age = (utc_now() - merged.index[-1].to_pydatetime()).total_seconds() / 60
    if common_age > MAX_COMMON_BAR_AGE_MINUTES or common_age < -2:
        raise RuntimeError(f"آخر شمعة مشتركة قديمة/غير صالحة: {common_age:.1f} دقيقة")
    log(f"[تزامن البيانات] الفرق بين المصادر={sync_gap:.1f} دقيقة | عمر آخر شمعة مشتركة={common_age:.1f} دقيقة")
    return merged


# ======================== MACHINE LEARNING ==================
def new_model():
    return make_pipeline(SimpleImputer(strategy="median"), HistGradientBoostingClassifier(
        max_iter=120, learning_rate=0.06, max_leaf_nodes=15,
        l2_regularization=1.0, random_state=42))


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
    baseline = int(y_train.mean() >= 0.5)
    baseline_accuracy = float((y_test == baseline).mean())
    log(f"[النموذج] صفوف={len(X)} | اختبار={len(X_test)} | دقة الاتجاه={accuracy:.3f} | AUC={auc:.3f} | خط الأساس={baseline_accuracy:.3f}")
    live_model = new_model()
    live_model.fit(X, y)
    return live_model, FEATURES.copy(), auc, accuracy


def directional_votes(latest):
    # Neutral/conflicting signals remain neutral; they are not counted as bearish.
    rsi = float(latest["rsi"])
    spx_up = latest["ret_1"] > 0 and latest["ema_spread"] > 0 and 45 <= rsi <= 72
    spx_down = latest["ret_1"] < 0 and latest["ema_spread"] < 0 and 28 <= rsi <= 55

    spy_rsi, qqq_rsi = float(latest["spy_rsi"]), float(latest["qqq_rsi"])
    spy_up = latest["spy_ret_1"] > 0 and latest["spy_ema_spread"] > 0 and latest["spy_vwap_dist"] > 0 and 45 <= spy_rsi <= 72
    spy_down = latest["spy_ret_1"] < 0 and latest["spy_ema_spread"] < 0 and latest["spy_vwap_dist"] < 0 and 28 <= spy_rsi <= 55
    qqq_up = latest["qqq_ret_1"] > 0 and latest["qqq_ema_spread"] > 0 and latest["qqq_vwap_dist"] > 0 and 45 <= qqq_rsi <= 72
    qqq_down = latest["qqq_ret_1"] < 0 and latest["qqq_ema_spread"] < 0 and latest["qqq_vwap_dist"] < 0 and 28 <= qqq_rsi <= 55
    up = sum(map(bool, [spx_up, spy_up, qqq_up]))
    down = sum(map(bool, [spx_down, spy_down, qqq_down]))
    return up, down


def calculate_signal(df, model, feature_cols, auc):
    if auc < MIN_AUC:
        return "WAIT", 0.0, f"AUC={auc:.3f} أقل من الحد {MIN_AUC:.3f}"
    latest = df.iloc[-1]
    x = latest[feature_cols].to_frame().T
    p_up = float(model.predict_proba(x)[0, 1])
    p_down = 1.0 - p_up
    up_votes, down_votes = directional_votes(latest)

    # Avoid buying directly into nearby resistance/support based on prior 20 bars.
    resistance_dist = float(latest["resistance_dist"])
    support_dist = float(latest["support_dist"])
    if p_up >= MIN_CONFIDENCE and up_votes >= MIN_DIRECTION_VOTES and down_votes == 0:
        if 0 <= resistance_dist < 0.0015:
            return "WAIT", p_up, "السعر قريب جدًا من مقاومة حديثة"
        return "CALL", p_up, f"ترجيح صعود={p_up:.1%}؛ تأكيدات صاعدة={up_votes}/3"
    if p_down >= MIN_CONFIDENCE and down_votes >= MIN_DIRECTION_VOTES and up_votes == 0:
        if 0 <= support_dist < 0.0015:
            return "WAIT", p_down, "السعر قريب جدًا من دعم حديث"
        return "PUT", p_down, f"ترجيح هبوط={p_down:.1%}؛ تأكيدات هابطة={down_votes}/3"
    return "WAIT", max(p_up, p_down), f"التأكيد غير كافٍ؛ صعود={p_up:.1%}؛ هبوط={p_down:.1%}؛ الأصوات={up_votes}/{down_votes}"


# ======================== TELEGRAM ===========================
def send_telegram(message):
    if not TG_TOKEN or not TG_CHAT:
        log("[تيليجرام] رمز البوت أو معرف المحادثة غير موجود", "warning")
        return False
    try:
        response = HTTP.post(f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage",
                             json={"chat_id": TG_CHAT, "text": message, "disable_web_page_preview": True}, timeout=20)
        if response.status_code >= 400:
            log(f"[خطأ تيليجرام] HTTP {response.status_code}: {response.text[:200]}", "error")
            return False
        if not response.json().get("ok", False):
            log("[خطأ تيليجرام] لم تؤكد واجهة تيليجرام إرسال الرسالة", "error")
            return False
        log("[تيليجرام] تم إرسال الرسالة")
        return True
    except Exception as exc:
        log(f"[خطأ تيليجرام] {exc}", "error")
        return False


# ======================== SPXW CONTRACTS =====================
def get_spxw_contracts(direction):
    try:
        params = {"underlying_symbols": "SPX", "expiration_date": ny_now().date().isoformat(),
                  "type": "call" if direction == "CALL" else "put", "status": "active", "limit": 1000}
        response = HTTP.get(f"{TRADE_URL}/v2/options/contracts", headers=alpaca_headers(), params=params, timeout=25)
        if response.status_code >= 400:
            raise RuntimeError(f"جلب العقود HTTP {response.status_code}: {response.text[:200]}")
        payload = response.json()
        contracts = payload.get("option_contracts", payload.get("contracts", [])) or []
        results = []
        for item in contracts:
            symbol, root = item.get("symbol", ""), item.get("root_symbol", "")
            if not symbol or (root != "SPXW" and not symbol.startswith("SPXW")):
                continue
            try:
                strike = float(item["strike_price"])
            except (KeyError, TypeError, ValueError):
                continue
            results.append({"symbol": symbol, "strike": strike, "expiration": item.get("expiration_date", "")})
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
            response = HTTP.get(endpoint, headers=alpaca_headers(),
                                params={"symbols": ",".join(batch), "feed": OPTIONS_FEED}, timeout=25)
            if response.status_code >= 400:
                log(f"[خطأ أسعار الخيارات] HTTP {response.status_code}: {response.text[:200]}", "warning")
                continue
            quotes = response.json().get("quotes", {}) or {}
            if isinstance(quotes, dict):
                results.update(quotes)
        except Exception as exc:
            log(f"[خطأ أسعار الخيارات] {exc}", "warning")
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


def quote_timestamp(quote):
    for key in ("t", "timestamp", "time"):
        ts = parse_timestamp(quote.get(key))
        if ts is not None:
            return ts
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
    spread = (ask - bid) / mid if mid > 0 else 1.0
    return {"bid": bid, "ask": ask, "mid": mid, "spread_pct": spread, "quote_ts": ts, "quote_age_seconds": age}


def choose_contract(direction, spot_price):
    contracts = get_spxw_contracts(direction)
    if not contracts:
        log("[العقود] لم يتم العثور على عقود SPXW لليوم", "warning")
        return None

    # Filter to a target that is still ahead in the intended direction.
    if direction == "CALL":
        contracts = [c for c in contracts if MIN_TARGET_DISTANCE <= c["strike"] - spot_price <= MAX_TARGET_DISTANCE]
    else:
        contracts = [c for c in contracts if MIN_TARGET_DISTANCE <= spot_price - c["strike"] <= MAX_TARGET_DISTANCE]
    if not contracts:
        log(f"[العقود] لا يوجد سعر تنفيذ أمام المؤشر ضمن {MIN_TARGET_DISTANCE:.1f}–{MAX_TARGET_DISTANCE:.1f} نقطة", "info")
        return None

    contracts.sort(key=lambda c: abs(c["strike"] - spot_price))
    candidates = contracts[:MAX_CONTRACTS_TO_CHECK]
    quotes = get_option_quotes([c["symbol"] for c in candidates])
    viable = []
    for contract in candidates:
        q = normalize_quote(quotes.get(contract["symbol"]))
        if q is None:
            continue
        if q["spread_pct"] > MAX_SPREAD_PCT:
            continue
        if q["ask"] < MIN_OPTION_PREMIUM or q["ask"] > MAX_OPTION_PREMIUM:
            continue
        viable.append({**contract, **q, "distance": abs(contract["strike"] - spot_price)})
    if not viable:
        log("[العقود] لم يجتز أي عقد شروط حداثة السعر/السبريد/القسط", "warning")
        return None
    viable.sort(key=lambda c: (c["distance"], c["spread_pct"], c["ask"]))
    selected = viable[0]
    log(f"[العقد] {selected['symbol']} | strike={selected['strike']:.2f} | SPX={spot_price:.2f} | مسافة الهدف={selected['distance']:.2f} | bid={selected['bid']:.2f} ask={selected['ask']:.2f} | spread={selected['spread_pct']:.1%} | quote age={selected['quote_age_seconds']:.0f}s")
    return selected


# ======================== OUTCOME JOURNAL ====================
def _journal_columns():
    cols = ["signal_time_utc", "direction", "symbol", "spot", "strike", "entry_bid", "entry_ask",
            "entry_mid", "confidence", "auc", "quote_time_utc"]
    for m in OUTCOME_MINUTES:
        cols += [f"bid_after_{m}m", f"observed_at_{m}m_utc", f"return_vs_entry_ask_{m}m"]
    return cols


def append_signal_journal(row):
    try:
        exists = os.path.exists(SIGNAL_LOG_PATH)
        pd.DataFrame([row], columns=_journal_columns()).to_csv(
            SIGNAL_LOG_PATH, mode="a", header=not exists, index=False
        )
        log(f"[سجل التوصيات] تمت إضافة {row.get('symbol')} إلى {SIGNAL_LOG_PATH}")
    except Exception as exc:
        log(f"[سجل التوصيات] تعذر حفظ السجل: {exc}", "warning")


def update_outcome_journal():
    """Record observed option bid at 1/3/5/10m after alerts. Entry uses ask;
    this is a quote-based paper observation, not a guaranteed executable fill."""
    try:
        if not os.path.exists(SIGNAL_LOG_PATH):
            return
        df = pd.read_csv(SIGNAL_LOG_PATH)
        if df.empty or "signal_time_utc" not in df:
            return
        now = utc_now()
        pending = {}
        for idx, row in df.iterrows():
            try:
                start = parse_timestamp(row["signal_time_utc"])
                if start is None:
                    continue
                elapsed = (now - start).total_seconds() / 60
                for m in OUTCOME_MINUTES:
                    if elapsed >= m and pd.isna(row.get(f"bid_after_{m}m")):
                        pending.setdefault(row["symbol"], []).append((idx, m, row["entry_ask"]))
            except Exception:
                continue
        if not pending:
            return
        quotes = get_option_quotes(list(pending.keys()))
        changed = False
        for symbol, entries in pending.items():
            q = normalize_quote(quotes.get(symbol))
            if q is None:
                continue
            for idx, m, entry_ask in entries:
                df.at[idx, f"bid_after_{m}m"] = q["bid"]
                df.at[idx, f"observed_at_{m}m_utc"] = now.isoformat()
                try:
                    entry = float(entry_ask)
                    df.at[idx, f"return_vs_entry_ask_{m}m"] = (q["bid"] - entry) / entry if entry > 0 else np.nan
                except (TypeError, ValueError):
                    df.at[idx, f"return_vs_entry_ask_{m}m"] = np.nan
                changed = True
        if changed:
            df.to_csv(SIGNAL_LOG_PATH, index=False)
            log("[سجل النتائج] تم تحديث أسعار متابعة التوصيات")
    except Exception as exc:
        log(f"[سجل النتائج] تعذر تحديث النتائج: {exc}", "warning")


def is_duplicate(direction, symbol):
    key = f"{direction}:{symbol}"
    last = LAST_ALERTS.get(key)
    if last and (utc_now() - last).total_seconds() < ALERT_COOLDOWN_MINUTES * 60:
        return True
    return False


def mark_alert_sent(direction, symbol):
    LAST_ALERTS[f"{direction}:{symbol}"] = utc_now()


# ======================== SIGNAL MESSAGE ====================
def build_telegram_message(direction, confidence, spot, contract, auc, accuracy, signal_time):
    direction_ar = "شراء كول" if direction == "CALL" else "شراء بوت"
    arrow = "🟢" if direction == "CALL" else "🔴"
    target_direction = "صعودًا" if direction == "CALL" else "هبوطًا"
    distance = abs(contract["strike"] - spot)
    # Accuracy/AUC are omitted from the trading card: they measure directional test performance,
    # not a calibrated probability that this option trade will profit.
    return (
        "🦅 صَفاء التحليل\n"
        "ذكاء اصطناعي لتحليل عقود سباكس\n"
        "━━━━━━━━━━━━━━━━━━\n\n"
        f"{arrow} نوع الإشارة: {direction_ar}\n"
        f"💹 مستوى مؤشر SPX: {spot:,.2f}\n"
        f"🎯 سعر التنفيذ: {contract['strike']:,.2f}\n"
        f"📏 المسافة إلى الهدف: {distance:.2f} نقطة ({target_direction})\n"
        f"📊 ثقة النموذج الاتجاهية: {confidence:.1%}\n\n"
        "📑 تفاصيل عقد SPXW\n"
        f"🔹 رمز العقد: {contract['symbol']}\n"
        f"🟢 سعر الطلب (Ask): {contract['ask']:.2f}\n"
        f"🔴 سعر العرض (Bid): {contract['bid']:.2f}\n"
        f"⚖️ متوسط السعر: {contract['mid']:.2f}\n"
        f"📉 فارق العرض والطلب: {contract['spread_pct']:.1%}\n"
        f"🕒 وقت الإشارة: {signal_time.astimezone(NY).strftime('%H:%M:%S')} بتوقيت نيويورك\n\n"
        "━━━━━━━━━━━━━━━━━━\n"
        "صَفاء التحليل • توصيات SPXW 0DTE"
    )


# ======================== MAIN SCANNER =======================
def market_is_open():
    now = ny_now()
    if now.weekday() >= 5:
        return False
    minutes = now.hour * 60 + now.minute
    return 9 * 60 + 30 <= minutes < 16 * 60


def run_scan():
    log("========== بدء فحص جديد | صَفاء التحليل ==========")
    update_outcome_journal()
    spx = get_spx_bars()
    spy = get_stock_bars("SPY")
    qqq = get_stock_bars("QQQ")
    if spx.empty or spy.empty or qqq.empty:
        log("[الإشارة] انتظار — بيانات السوق غير مكتملة أو قديمة", "warning")
        return

    # Current reference price is separately fetched from Yahoo's latest 1m bar.
    spot_price, spot_ts = get_spx_spot()
    if spot_price is None:
        log("[الإشارة] انتظار — لا يوجد سعر SPX حديث يمكن التحقق منه", "warning")
        return

    try:
        merged = make_features(spx, spy, qqq)
        model, columns, auc, accuracy = train_model(merged)
        direction, confidence, reason = calculate_signal(merged, model, columns, auc)
        log(f"[الإشارة] {direction} | ثقة الاتجاه={confidence:.1%} | AUC={auc:.3f} | السبب={reason}")
        if direction == "WAIT":
            return

        contract = choose_contract(direction, spot_price)
        if contract is None:
            # Keep Telegram quiet when no high-quality contract is available.
            return

        if is_duplicate(direction, contract["symbol"]):
            log(f"[منع التكرار] تم تجاهل إشارة متكررة: {direction} {contract['symbol']}")
            return

        signal_time = utc_now()
        message = build_telegram_message(direction, confidence, spot_price, contract, auc, accuracy, signal_time)
        if send_telegram(message):
            mark_alert_sent(direction, contract["symbol"])
            append_signal_journal({
                "signal_time_utc": signal_time.isoformat(), "direction": direction,
                "symbol": contract["symbol"], "spot": spot_price, "strike": contract["strike"],
                "entry_bid": contract["bid"], "entry_ask": contract["ask"], "entry_mid": contract["mid"],
                "confidence": confidence, "auc": auc, "quote_time_utc": contract["quote_ts"].isoformat(),
            })
    except Exception as exc:
        log(f"[خطأ الفحص] {type(exc).__name__}: {exc}", "error")


def main():
    log("بدء تشغيل صَفاء التحليل — SPXW 0DTE AI ADVISOR v19.0")
    log("الوضع: توصيات فقط — لا يوجد تنفيذ تلقائي للصفقات")
    log(f"بيانات الأسهم={DATA_FEED} | أسعار الخيارات={OPTIONS_FEED} | AUC الأدنى={MIN_AUC:.3f} | ثقة الاتجاه الأدنى={MIN_CONFIDENCE:.2f}")
    log(f"عمر شمعة SPX الأقصى={MAX_SPX_BAR_AGE_MINUTES:.1f} دقيقة | عمر سعر العقد الأقصى={MAX_OPTION_QUOTE_AGE_SECONDS:.0f} ثانية | مسافة الهدف={MIN_TARGET_DISTANCE:.1f}-{MAX_TARGET_DISTANCE:.1f} نقطة")
    if not API_KEY or not API_SECRET:
        log("مفاتيح Alpaca غير موجودة في متغيرات البيئة", "error")
    if not TG_TOKEN or not TG_CHAT:
        log("إعدادات تيليجرام غير مكتملة", "warning")
    while True:
        try:
            if market_is_open():
                run_scan()
            else:
                log("[السوق] مغلق | توقيت نيويورك: " + ny_now().strftime("%Y-%m-%d %H:%M:%S %Z"))
        except KeyboardInterrupt:
            log("تم إيقاف صَفاء التحليل")
            break
        except Exception as exc:
            log(f"[خطأ رئيسي] {type(exc).__name__}: {exc}", "error")
        time.sleep(SCAN_SECONDS)


if __name__ == "__main__":
    main()
