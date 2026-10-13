# ============================================================
# صَفاء التحليل — SPY 0DTE AI ADVISOR v21.6
# PART 1/6
# Imports • Configuration • NY Session • Calendar • Market Data
#
# MODE: RECOMMENDATIONS ONLY — SPY FULL DEDICATION & OPTIONS SCANNER
# ============================================================


from __future__ import annotations

import os
import sys
import json
import time
import math
import logging
import warnings
import uuid
import hashlib
import csv

from pathlib import Path
from datetime import (
    datetime,
    date,
    time as dt_time,
    timedelta,
    timezone,
)
from typing import Any, Optional
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

warnings.filterwarnings("ignore", category=FutureWarning)

# ============================================================
# 1. VERSION AND LOGGING
# ============================================================

BOT_NAME = "صَفاء التحليل"
BOT_VERSION = "21.6"
JOURNAL_VERSION = "21.6"

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)

log = logging.getLogger("safa_analysis")

# ============================================================
# 2. ENVIRONMENT CONFIGURATION HELPERS
# ============================================================

def _s(name: str, default: str = "") -> str:
    value = os.getenv(name)
    return default if value is None else value.strip()


def _i(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    try:
        return int(value.strip())
    except (TypeError, ValueError):
        return default


def _f(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    try:
        result = float(value.strip())
        if not math.isfinite(result):
            raise ValueError("non-finite value")
        return result
    except (TypeError, ValueError):
        return default


def _b(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return value.strip().lower() in {
        "1", "true", "yes", "y", "on"
    }


# ============================================================
# 3. API CONFIGURATION
# ============================================================

APCA_API_KEY_ID = _s("APCA_API_KEY_ID")
APCA_API_SECRET_KEY = _s("APCA_API_SECRET_KEY")

ALPACA_DATA_URL = _s(
    "ALPACA_DATA_URL",
    "https://data.alpaca.markets",
).rstrip("/")

ALPACA_TRADING_URL = _s(
    "ALPACA_TRADING_URL",
    "https://paper-api.alpaca.markets",
).rstrip("/")

ALPACA_DATA_FEED = _s("ALPACA_DATA_FEED", "iex").lower()

REQUEST_TIMEOUT_SECONDS = _i("REQUEST_TIMEOUT_SECONDS", 15)
HTTP_RETRIES = _i("HTTP_RETRIES", 3)
HTTP_RETRY_SLEEP_SECONDS = _f("HTTP_RETRY_SLEEP_SECONDS", 1.5)

NO_ORDER_EXECUTION = True
RECOMMENDATION_ONLY = True

# ============================================================
# 4. MARKET AND MODEL SETTINGS
# ============================================================

TIMEZONE_NY = ZoneInfo("America/New_York")
TIMEZONE_UTC = timezone.utc

TIMEFRAME = _s("TIMEFRAME", "5Min")
HISTORY_DAYS = _i("HISTORY_DAYS", 60)
MIN_TRAIN_ROWS = _i("MIN_TRAIN_ROWS", 300)
HORIZON = _i("HORIZON", 6)

MIN_PROBABILITY = _f("MIN_PROBABILITY", 0.64)
MIN_AUC = _f("MIN_AUC", 0.55)
MIN_SPY_SCORE = _f("MIN_SPY_SCORE", 78.0)
MIN_CONFIRMATIONS = _i("MIN_CONFIRMATIONS", 2)

MIN_HOLDOUT_SIGNAL_N = _i("MIN_HOLDOUT_SIGNAL_N", 30)
MIN_HOLDOUT_SIGNAL_ACC = _f("MIN_HOLDOUT_SIGNAL_ACC", 0.55)
MAX_HOLDOUT_BRIER = _f("MAX_HOLDOUT_BRIER", 0.25)

ATR_TARGET = _f("ATR_TARGET", 0.50)

MAX_SPREAD_PERCENT = _f("MAX_SPREAD_PERCENT", 0.12)
MIN_OPTION_PREMIUM = _f("MIN_OPTION_PREMIUM", 0.50)
MAX_OPTION_PREMIUM = _f("MAX_OPTION_PREMIUM", 50.00)
MAX_STRIKE_DISTANCE = _f("MAX_STRIKE_DISTANCE", 5.0)

POLL_SECONDS = _i("POLL_SECONDS", 30)
SIGNAL_COOLDOWN_MINUTES = _i("SIGNAL_COOLDOWN_MINUTES", 20)

PRIMARY_UNDERLYING = "SPY"

# ============================================================
# 5. RISK AND ACCOUNT SETTINGS
# ============================================================

ACCOUNT_EQUITY = _f("ACCOUNT_EQUITY", 0.0)
RISK_PCT = _f("RISK_PCT", 0.50)
MAX_DAILY_LOSS = _f("MAX_DAILY_LOSS", 200.0)
MAX_TRADES_PER_DAY = _i("MAX_TRADES_PER_DAY", 5)

# ============================================================
# 6. TELEGRAM CONFIGURATION
# ============================================================

TELEGRAM_BOT_TOKEN = _s("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = _s("TELEGRAM_CHAT_ID")
TELEGRAM_ENABLED = bool(TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID)

# ============================================================
# 7. JOURNAL FILES
# ============================================================

DATA_DIR = _s("DATA_DIR", ".")
SIGNAL_JOURNAL_PATH = os.path.join(DATA_DIR, "safa_signals_v216.csv")
os.makedirs(DATA_DIR, exist_ok=True)

# ============================================================
# 8. SESSION TIME HELPERS
# ============================================================

def now_utc() -> datetime:
    return datetime.now(TIMEZONE_UTC)

def now_ny() -> datetime:
    return datetime.now(TIMEZONE_NY)

def to_utc(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    try:
        ts = pd.Timestamp(value)
        if pd.isna(ts):
            return None
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        return ts.to_pydatetime()
    except Exception:
        return None

# ============================================================
# 9. HTTP SESSION AND ALPACA HEADERS
# ============================================================

HTTP = requests.Session()
HTTP.headers.update({
    "Accept": "application/json",
    "User-Agent": "SafaaAnalysis/21.6",
})

def alpaca_headers() -> dict[str, str]:
    key_id = str(APCA_API_KEY_ID or "").strip()
    secret_key = str(APCA_API_SECRET_KEY or "").strip()
    if not key_id or not secret_key:
        raise RuntimeError("Alpaca credentials are empty.")
    return {
        "APCA-API-KEY-ID": key_id,
        "APCA-API-SECRET-KEY": secret_key,
        "Accept": "application/json",
    }

def alpaca_get(url: str, params: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    headers = alpaca_headers()
    last_error: Optional[Exception] = None
    for attempt in range(1, HTTP_RETRIES + 1):
        try:
            response = HTTP.get(url, headers=headers, params=params, timeout=REQUEST_TIMEOUT_SECONDS)
            if response.status_code in (401, 403):
                raise RuntimeError(f"Alpaca auth error {response.status_code}")
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict):
                raise RuntimeError("Unexpected JSON response.")
            return payload
        except Exception as exc:
            last_error = exc
            if attempt < HTTP_RETRIES:
                time.sleep(HTTP_RETRY_SLEEP_SECONDS * attempt)
    raise RuntimeError(f"Alpaca GET failed: {last_error}")

# ============================================================
# 10. MARKET DATA & PREDICTION PIPELINE
# ============================================================

REQUIRED_BAR_COLUMNS = ["open", "high", "low", "close", "volume"]

def normalize_bars(raw_bars: list[dict[str, Any]]) -> pd.DataFrame:
    if not raw_bars:
        return pd.DataFrame(columns=REQUIRED_BAR_COLUMNS)
    frame = pd.DataFrame(raw_bars)
    if "t" not in frame.columns:
        raise ValueError("No timestamp field 't'.")
    rename_map = {"t": "timestamp", "o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"}
    frame = frame.rename(columns=rename_map)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    frame = frame.dropna(subset=["timestamp"])
    for column in REQUIRED_BAR_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=REQUIRED_BAR_COLUMNS)
    return frame.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last").set_index("timestamp")

def fetch_stock_bars(symbol: str = "SPY", timeframe: str = TIMEFRAME, history_days: int = HISTORY_DAYS) -> pd.DataFrame:
    end_time = now_utc()
    start_time = end_time - timedelta(days=history_days)
    url = f"{ALPACA_DATA_URL}/v2/stocks/{symbol}/bars"
    params = {
        "timeframe": timeframe,
        "start": start_time.isoformat().replace("+00:00", "Z"),
        "end": end_time.isoformat().replace("+00:00", "Z"),
        "limit": 10000,
        "adjustment": "raw",
        "feed": ALPACA_DATA_FEED,
        "sort": "asc",
    }
    payload = alpaca_get(url, params=params)
    bars = payload.get("bars") or []
    return normalize_bars(bars)

# Feature Engineering
FEATURE_COLUMNS = [
    "ret_1", "ret_3", "ret_6", "ret_12",
    "ema_spread", "ema_trend", "rsi_14", "atr_pct",
    "range_pct", "body_pct", "close_location", "volume_ratio",
]

def _safe_divide(num: pd.Series, den: pd.Series) -> pd.Series:
    den = den.replace(0, np.nan)
    return (num / den).replace([np.inf, -np.inf], np.nan)

def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False, min_periods=span).mean()

def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)
    avg_gain = gains.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = losses.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    rs = _safe_divide(avg_gain, avg_loss)
    res = 100.0 - (100.0 / (1.0 + rs))
    return res.clip(0.0, 100.0)

def _true_range(frame: pd.DataFrame) -> pd.Series:
    pc = frame["close"].shift(1)
    comp = pd.concat([frame["high"] - frame["low"], (frame["high"] - pc).abs(), (frame["low"] - pc).abs()], axis=1)
    return comp.max(axis=1)

def _atr(frame: pd.DataFrame, period: int = 14) -> pd.Series:
    return _true_range(frame).ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()

def build_features(bars: pd.DataFrame) -> pd.DataFrame:
    frame = bars.copy().sort_index()
    close = frame["close"]
    frame["ret_1"] = close.pct_change(1)
    frame["ret_3"] = close.pct_change(3)
    frame["ret_6"] = close.pct_change(6)
    frame["ret_12"] = close.pct_change(12)
    ema9, ema20, ema50 = _ema(close, 9), _ema(close, 20), _ema(close, 50)
    frame["ema_spread"] = _safe_divide(ema9 - ema20, close)
    frame["ema_trend"] = _safe_divide(ema20 - ema50, close)
    frame["rsi_14"] = _rsi(close, 14) / 100.0
    frame["atr_pct"] = _safe_divide(_atr(frame, 14), close)
    cr = (frame["high"] - frame["low"]).clip(lower=0)
    body = close - frame["open"]
    frame["range_pct"] = _safe_divide(cr, close)
    frame["body_pct"] = _safe_divide(body, close)
    frame["close_location"] = _safe_divide(close - frame["low"], cr).where(cr > 0, 0.5)
    v_mean = frame["volume"].rolling(20, min_periods=20).mean()
    frame["volume_ratio"] = _safe_divide(frame["volume"], v_mean)
    return frame.replace([np.inf, -np.inf], np.nan)

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

def train_and_predict(bars: pd.DataFrame) -> dict[str, Any]:
    features = build_features(bars)
    close = bars["close"]
    fc = close.shift(-HORIZON)
    features["target"] = (fc > close).astype("int8")
    dataset = features[FEATURE_COLUMNS + ["target"]].dropna()
    
    if len(dataset) < MIN_TRAIN_ROWS:
        return {"ok": False, "reason": "INSUFFICIENT_ROWS"}
        
    X, y = dataset[FEATURE_COLUMNS], dataset["target"]
    rf = Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42))])
    rf.fit(X, y)
    
    latest = features.iloc[[-1]][FEATURE_COLUMNS]
    prob_up = float(rf.predict_proba(latest)[0][1])
    signal = "CALL" if prob_up >= 0.5 else "PUT"
    
    return {
        "ok": True,
        "signal": signal,
        "probability": max(prob_up, 1.0 - prob_up) * 100,
        "current_price": float(close.iloc[-1])
    }

# ============================================================
# OPTIONS SCANNER & TELEGRAM DISPATCHER
# ============================================================

OPTIONS_FEED = _s("OPTIONS_FEED", "indicative").lower()

def fetch_best_0dte_option(underlying_price: float, direction: str) -> Optional[dict[str, Any]]:
    today_str = now_ny().date().isoformat()
    url = f"{ALPACA_TRADING_URL}/v2/options/contracts"
    params = {
        "underlying_symbols": "SPY",
        "status": "active",
        "expiration_date": today_str,
        "type": "call" if direction == "CALL" else "put",
        "limit": 100
    }
    try:
        payload = alpaca_get(url, params=params)
        contracts = payload.get("option_contracts", payload.get("contracts", []))
        if not contracts:
            return None
            
        # اختيار أقرب سترايك للسعر الحالي (ATM)
        best_contract = None
        min_diff = 999.0
        for c in contracts:
            strike = float(c.get("strike_price", 0))
            diff = abs(strike - underlying_price)
            if diff < min_diff:
                min_diff = diff
                best_contract = c
                
        if not best_contract or min_diff > MAX_STRIKE_DISTANCE:
            return None
            
        sym = best_contract["symbol"]
        q_url = f"{ALPACA_DATA_URL}/v1beta1/options/quotes/latest"
        q_payload = alpaca_get(q_url, params={"symbols": sym, "feed": OPTIONS_FEED})
        quote = q_payload.get("quotes", {}).get(sym, {})
        
        bid = float(quote.get("bp", 0))
        ask = float(quote.get("ap", 0))
        mid = (bid + ask) / 2.0 if ask >= bid else 0.0
        
        return {
            "symbol": sym,
            "strike": best_contract.get("strike_price"),
            "type": direction,
            "expiration": today_str,
            "bid": bid,
            "ask": ask,
            "mid": mid
        }
    except Exception as e:
        log.error(f"Error fetching option contract: {e}")
        return None

def send_telegram(message: str) -> bool:
    if not TELEGRAM_ENABLED or not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return False
    try:
        res = requests.post(f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage", json={"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}, timeout=10)
        return res.status_code == 200
    except Exception:
        return False

def send_market_outlook_telegram(spy_price, direction, target_spy, points_move):
    dir_text = "صعود بنقاط مستهدفة 🟢" if direction == "up" else "هبوط بنقاط مستهدفة 🔴"
    message = (
        "✨ **تحليل حركة السوق (SPY)**\n\n"
        f"▫️ **سعر SPY الحالي:** `{spy_price:.2f}`\n"
        f"▫️ **الوجهة المتوقعة:** {dir_text}\n"
        f"▫️ **الهدف السعري:** `{target_spy:.2f}`\n"
        f"▫️ **مدى الحركة:** `{points_move}`\n\n"
        "⚡ *جاهز للجلسة بناءً على استراتيجيات SPY والـ ATR*\n\n"
        "<i>✨ تصميم أبو سحاب</i>"
    )
    send_telegram(message)

def send_option_recommendation_telegram(pred: dict, opt: dict):
    dir_emoji = "🟢" if pred["signal"] == "CALL" else "🔴"
    message = (
        "🎯 **توصية خيارات جديدة (SPY 0DTE)**\n\n"
        f"▫️ **الرمز الأساسي:** `SPY`\n"
        f"▫️ **نوع العقد:** `{pred['signal']}` {dir_emoji}\n"
        f"▫️ **سعر السترايك (Strike):** `{opt.get('strike', 'N/A')}`\n"
        f"▫️ **تاريخ الانتهاء:** `اليوم (0DTE)`\n"
        f"▫️ **السعر التقريبي (Mid):** `${opt.get('mid', 0):.2f}` (Bid: {opt.get('bid', 0)} / Ask: {opt.get('ask', 0)})\n"
        f"▫️ **نسبة ثقة المودل:** `{pred['probability']:.1f}%`\n\n"
        "⚡ *تنبيه: التوصية لأغراض الدراسة والمتابعة — بدون تنفيذ آلي*\n\n"
        "<i>✨ تصميم أبو سحاب</i>"
    )
    send_telegram(message)

def run_scan():
    log.info("Starting scan for SPY options & signals...")
    bars = fetch_stock_bars("SPY", timeframe=TIMEFRAME, history_days=HISTORY_DAYS)
    if bars.empty:
        return
        
    pred = train_and_predict(bars)
    if not pred.get("ok"):
        return
        
    if pred["probability"] >= (MIN_PROBABILITY * 100):
        opt = fetch_best_0dte_option(pred["current_price"], pred["signal"])
        if opt and opt["mid"] >= MIN_OPTION_PREMIUM:
            send_option_recommendation_telegram(pred, opt)
            log.info("تم إرسال توصية خيارات SPY بنجاح.")

def main():
    log.info("=" * 60)
    log.info("%s — VERSION %s (FULL SPY & OPTIONS SCANNER)", BOT_NAME, BOT_VERSION)
    log.info("=" * 60)

    # إرسال تحليل المؤشر الصباحي
    try:
        bars = fetch_stock_bars("SPY", timeframe=TIMEFRAME, history_days=5)
        if not bars.empty:
            closes = bars["close"].tolist()
            highs = bars["high"].tolist()
            lows = bars["low"].tolist()
            spy_current = float(closes[-1])
            tr_list = [max(highs[i] - lows[i], abs(highs[i] - closes[i-1]), abs(lows[i] - closes[i-1])) for i in range(1, len(closes))]
            atr = sum(tr_list) / len(tr_list) if tr_list else 1.0
            
            if closes[-1] >= closes[-2]:
                send_market_outlook_telegram(spy_current, "up", spy_current + (atr * 0.5), f"+{round(atr * 0.5, 2)} نقاط")
            else:
                send_market_outlook_telegram(spy_current, "down", spy_current - (atr * 0.5), f"-{round(atr * 0.5, 2)} نقاط")
    except Exception as e:
        log.error(f"خطأ في إرسال التقرير الصباحي: {e}")

    while True:
        try:
            run_scan()
        except Exception:
            log.exception("Scan error in main loop.")
        time.sleep(POLL_SECONDS)

if __name__ == "__main__":
    main()
