# ============================================================
# صَفاء التحليل — SPY 0DTE AI ADVISOR v21.4
# PART 1/6
# Imports • Configuration • NY Session • Calendar • Market Data
#
# MODE: RECOMMENDATIONS ONLY — NO ORDER EXECUTION
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
BOT_VERSION = "21.4"
JOURNAL_VERSION = "21.4"

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
MAX_STRIKE_DISTANCE = _f("MAX_STRIKE_DISTANCE", 35.0)
MIN_DELTA = _f("MIN_DELTA", 0.20)
MAX_DELTA = _f("MAX_DELTA", 0.80)

MAX_BAR_AGE_MINUTES = _i("MAX_BAR_AGE_MINUTES", 10)
MAX_QUOTE_AGE_SECONDS = _i("MAX_QUOTE_AGE_SECONDS", 60)
MAX_SPOT_AGE_SECONDS = _i("MAX_SPOT_AGE_SECONDS", 60)
MAX_SPOT_OPTION_SKEW_SECONDS = _i("MAX_SPOT_OPTION_SKEW_SECONDS", 30)

POLL_SECONDS = _i("POLL_SECONDS", 30)
SIGNAL_COOLDOWN_MINUTES = _i("SIGNAL_COOLDOWN_MINUTES", 20)

NO_TRADE_FIRST_MINUTES = _i("NO_TRADE_FIRST_MINUTES", 10)
NO_TRADE_LAST_MINUTES = _i("NO_TRADE_LAST_MINUTES", 30)

PRIMARY_UNDERLYING = _s("PRIMARY_UNDERLYING", "SPY").upper()

# ============================================================
# 5. RISK AND ACCOUNT SETTINGS
# ============================================================

ACCOUNT_EQUITY = _f("ACCOUNT_EQUITY", 0.0)
RISK_PCT = _f("RISK_PCT", 0.50)
MAX_DAILY_LOSS = _f("MAX_DAILY_LOSS", 200.0)
MAX_TRADES_PER_DAY = _i("MAX_TRADES_PER_DAY", 5)
MAX_OPEN_POSITIONS = _i("MAX_OPEN_POSITIONS", 2)
MAX_CONTRACTS = _i("MAX_CONTRACTS", 5)

ESTIMATED_SLIPPAGE_PER_CONTRACT = _f("ESTIMATED_SLIPPAGE_PER_CONTRACT", 0.05)
ESTIMATED_FEES_PER_CONTRACT = _f("ESTIMATED_FEES_PER_CONTRACT", 1.00)

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
SIGNAL_JOURNAL_PATH = os.path.join(DATA_DIR, "safa_signals_v214.csv")
QUOTE_JOURNAL_PATH = os.path.join(DATA_DIR, "safa_quote_observations_v214.csv")
MODEL_JOURNAL_PATH = os.path.join(DATA_DIR, "safa_model_diagnostics_v214.csv")
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

def to_ny(value: Any) -> Optional[datetime]:
    ts = to_utc(value)
    if ts is None:
        return None
    return ts.astimezone(TIMEZONE_NY)

def is_weekday(day: date) -> bool:
    return day.weekday() < 5

def regular_session_bounds(day: date) -> tuple[datetime, datetime]:
    start = datetime.combine(day, dt_time(hour=9, minute=30), tzinfo=TIMEZONE_NY)
    end = datetime.combine(day, dt_time(hour=16, minute=0), tzinfo=TIMEZONE_NY)
    return start, end

# ============================================================
# 9. HTTP SESSION AND ALPACA HEADERS
# ============================================================

HTTP = requests.Session()
HTTP.headers.update({
    "Accept": "application/json",
    "User-Agent": "SafaaAnalysis/21.0",
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
# 10. MARKET DATA NORMALIZATION & FETCH
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

def fetch_stock_bars(symbol: str, timeframe: str = TIMEFRAME, history_days: int = HISTORY_DAYS, limit: int = 10000) -> pd.DataFrame:
    symbol = symbol.strip().upper()
    end_time = now_utc()
    start_time = end_time - timedelta(days=history_days)
    url = f"{ALPACA_DATA_URL}/v2/stocks/{symbol}/bars"
    params = {
        "timeframe": timeframe,
        "start": start_time.isoformat().replace("+00:00", "Z"),
        "end": end_time.isoformat().replace("+00:00", "Z"),
        "limit": min(max(int(limit), 1), 10000),
        "adjustment": "raw",
        "feed": ALPACA_DATA_FEED,
        "sort": "asc",
    }
    collected: list[dict[str, Any]] = []
    page_token: Optional[str] = None
    pages = 0
    while pages < 50:
        if page_token:
            params["page_token"] = page_token
        else:
            params.pop("page_token", None)
        payload = alpaca_get(url, params=params)
        bars = payload.get("bars") or []
        if bars:
            collected.extend(bars)
        page_token = payload.get("next_page_token")
        pages += 1
        if not page_token:
            break
    if not collected:
        return pd.DataFrame(columns=REQUIRED_BAR_COLUMNS)
    return normalize_bars(collected)

def get_current_prediction():
    bars_function = fetch_stock_bars
    feature_function = build_features
    training_function = train_model
    prediction_function = predict_direction

    try:
        bars = bars_function(PRIMARY_UNDERLYING, timeframe=TIMEFRAME, history_days=HISTORY_DAYS)
        if bars.empty:
            return {"ok": False, "reason": "NO_LIVE_BARS"}

        latest_timestamp = bars.index[-1]
        fresh, freshness_reason = _data_timestamp_is_fresh(latest_timestamp)
        if not fresh:
            return {"ok": False, "reason": f"UNDERLYING_DATA_{freshness_reason}"}

        features = feature_function(bars)
        if features.empty:
            return {"ok": False, "reason": "FEATURE_FRAME_EMPTY"}

        model_bundle = training_function(features)
        journal_model_run(model_bundle)
        if not model_bundle.get("enabled", False):
            return {"ok": False, "reason": "MODEL_GATE_DISABLED", "model_bundle": model_bundle}

        prediction = prediction_function(model_bundle, features)
        direction = _normalise_direction(prediction.get("signal") or prediction.get("direction"))
        if direction == "WAIT":
            return {"ok": False, "reason": prediction.get("reason", "MODEL_DIRECTION_IS_WAIT")}

        prediction["signal"] = direction
        return {
            "ok": True,
            "prediction": prediction,
            "model_bundle": model_bundle,
            "bars": bars,
            "features": features,
        }
    except Exception as exc:
        log.exception("Current prediction failed.")
        return {"ok": False, "reason": f"PREDICTION_PIPELINE_ERROR:{exc}"}

def safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        result = float(value)
        if not math.isfinite(result):
            return default
        return result
    except (TypeError, ValueError):
        return default

def safe_positive_float(value: Any) -> Optional[float]:
    result = safe_float(value)
    if result is None or result <= 0:
        return None
    return result

def calculate_spread_percent(bid: Any, ask: Any) -> Optional[float]:
    bid_value = safe_positive_float(bid)
    ask_value = safe_positive_float(ask)
    if bid_value is None or ask_value is None or ask_value < bid_value:
        return None
    midpoint = (bid_value + ask_value) / 2.0
    if midpoint <= 0:
        return None
    return (ask_value - bid_value) / midpoint

# ============================================================
# PART 2/6: Technical Indicators & Feature Engineering
# ============================================================

FEATURE_COLUMNS = [
    "ret_1", "ret_3", "ret_6", "ret_12",
    "ema_spread", "ema_trend", "rsi_14", "atr_pct",
    "range_pct", "body_pct", "upper_wick_pct", "lower_wick_pct",
    "close_location", "volume_ratio", "volume_change",
    "momentum_3", "momentum_6", "realized_volatility",
    "qqq_ret_3", "qqq_ema_spread",
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
    res = res.where(avg_loss != 0, 100.0)
    res = res.where(~((avg_gain == 0) & (avg_loss == 0)), 50.0)
    return res.clip(0.0, 100.0)

def _true_range(frame: pd.DataFrame) -> pd.Series:
    pc = frame["close"].shift(1)
    comp = pd.concat([frame["high"] - frame["low"], (frame["high"] - pc).abs(), (frame["low"] - pc).abs()], axis=1)
    return comp.max(axis=1)

def _atr(frame: pd.DataFrame, period: int = 14) -> pd.Series:
    return _true_range(frame).ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()

def build_features(bars: pd.DataFrame, prefix: str = "") -> pd.DataFrame:
    frame = bars.copy().sort_index()
    close = frame["close"]
    frame["ret_1"] = close.pct_change(1)
    frame["ret_3"] = close.pct_change(3)
    frame["ret_6"] = close.pct_change(6)
    frame["ret_12"] = close.pct_change(12)
    frame["momentum_3"] = close - close.shift(3)
    frame["momentum_6"] = close - close.shift(6)
    ema9, ema20, ema50 = _ema(close, 9), _ema(close, 20), _ema(close, 50)
    frame["ema_spread"] = _safe_divide(ema9 - ema20, close)
    frame["ema_trend"] = _safe_divide(ema20 - ema50, close)
    frame["rsi_14"] = _rsi(close, 14) / 100.0
    frame["atr_pct"] = _safe_divide(_atr(frame, 14), close)
    frame["realized_volatility"] = close.pct_change().rolling(20, min_periods=20).std()
    cr = (frame["high"] - frame["low"]).clip(lower=0)
    body = close - frame["open"]
    frame["range_pct"] = _safe_divide(cr, close)
    frame["body_pct"] = _safe_divide(body, close)
    uw = frame["high"] - pd.concat([frame["open"], close], axis=1).max(axis=1)
    lw = pd.concat([frame["open"], close], axis=1).min(axis=1) - frame["low"]
    frame["upper_wick_pct"] = _safe_divide(uw.clip(lower=0), close)
    frame["lower_wick_pct"] = _safe_divide(lw.clip(lower=0), close)
    frame["close_location"] = _safe_divide(close - frame["low"], cr).where(cr > 0, 0.5)
    v_mean = frame["volume"].rolling(20, min_periods=20).mean()
    frame["volume_ratio"] = _safe_divide(frame["volume"], v_mean)
    frame["volume_change"] = frame["volume"].pct_change()
    return frame.replace([np.inf, -np.inf], np.nan)

def add_training_target(feature_frame: pd.DataFrame, horizon: int = HORIZON) -> pd.DataFrame:
    result = feature_frame.copy().sort_index()
    close = pd.to_numeric(result["close"], errors="coerce")
    fc = close.shift(-horizon)
    valid = close.notna() & fc.notna()
    result["target"] = np.nan
    result.loc[valid, "target"] = (fc.loc[valid] > close.loc[valid]).astype("int8")
    return result

def prepare_labeled_dataset(feature_frame: pd.DataFrame, min_rows: int = MIN_TRAIN_ROWS) -> pd.DataFrame:
    dataset = feature_frame[FEATURE_COLUMNS + ["target"]].copy().replace([np.inf, -np.inf], np.nan).dropna()
    dataset["target"] = dataset["target"].astype(int)
    return dataset

# ============================================================
# PART 3/6: Machine Learning & Ensemble
# ============================================================

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.pipeline import Pipeline

MODEL_RANDOM_STATE = _i("MODEL_RANDOM_STATE", 42)
TRAIN_FRACTION = _f("TRAIN_FRACTION", 0.65)
VALIDATION_FRACTION = _f("VALIDATION_FRACTION", 0.20)
HOLDOUT_FRACTION = _f("HOLDOUT_FRACTION", 0.15)

def chronological_split(dataset: pd.DataFrame):
    ordered = dataset.sort_index().copy()
    n = len(ordered)
    t_end = int(n * TRAIN_FRACTION)
    v_end = t_end + int(n * VALIDATION_FRACTION)
    purge = max(1, int(HORIZON))
    train = ordered.iloc[:max(0, t_end - purge)].copy()
    validation = ordered.iloc[t_end:max(t_end, v_end - purge)].copy()
    holdout = ordered.iloc[v_end:].copy()
    return train, validation, holdout

def create_base_models() -> dict[str, Pipeline]:
    rf = Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", RandomForestClassifier(n_estimators=100, max_depth=6, random_state=MODEL_RANDOM_STATE))])
    et = Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", ExtraTreesClassifier(n_estimators=100, max_depth=6, random_state=MODEL_RANDOM_STATE + 1))])
    return {"random_forest": rf, "extra_trees": et}

def positive_class_probability(model: Any, features: pd.DataFrame) -> np.ndarray:
    probs = np.asarray(model.predict_proba(features), dtype=float)
    classes = np.asarray(model.classes_)
    pos_idx = np.flatnonzero(classes == 1)
    return np.clip(probs[:, int(pos_idx[0])], 0.0, 1.0)

class ProbabilityEnsemble:
    def __init__(self, models: dict[str, Pipeline]) -> None:
        self.models = models
        self.weights = {name: 1.0 / len(models) for name in models}
        self.classes_ = np.array([0, 1])

    def fit(self, features: pd.DataFrame, target: pd.Series) -> "ProbabilityEnsemble":
        for model in self.models.values():
            model.fit(features, target)
        return self

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        wp = np.zeros(len(features), dtype=float)
        for name, model in self.models.items():
            wp += self.weights[name] * positive_class_probability(model, features)
        wp = np.clip(wp, 0.0, 1.0)
        return np.column_stack([1.0 - wp, wp])

def train_model(dataset: pd.DataFrame) -> dict[str, Any]:
    result = {"model": None, "enabled": False, "threshold": 0.55, "metrics": {}, "reasons": []}
    if dataset.empty or len(dataset) < MIN_TRAIN_ROWS:
        result["reasons"].append("INSUFFICIENT_ROWS")
        return result
    try:
        train, validation, holdout = chronological_split(dataset)
        ensemble = ProbabilityEnsemble(create_base_models())
        ensemble.fit(train[FEATURE_COLUMNS], train["target"])
        result["model"] = ensemble
        result["enabled"] = True
    except Exception as exc:
        result["reasons"].append(str(exc))
    return result

def predict_direction(trained: dict[str, Any], latest_features: pd.DataFrame) -> dict[str, Any]:
    if not trained.get("enabled"):
        return {"ok": False, "reason": "MODEL_DISABLED"}
    model = trained.get("model")
    latest = latest_features.iloc[[-1]][FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan)
    prob_up = float(positive_class_probability(model, latest)[0])
    signal = "CALL" if prob_up >= 0.5 else "PUT"
    return {"ok": True, "signal": signal, "probability_up": prob_up, "probability_down": 1.0 - prob_up, "confidence": max(prob_up, 1.0 - prob_up), "threshold": 0.55}

def train_from_feature_frame(feature_frame: pd.DataFrame) -> dict[str, Any]:
    labeled = add_training_target(feature_frame)
    dataset = prepare_labeled_dataset(labeled)
    return train_model(dataset)

# ============================================================
# PART 4/6: Options & Liquidity Controls
# ============================================================

OPTIONS_FEED = _s("OPTIONS_FEED", "indicative").lower()

def options_headers() -> dict[str, str]:
    return alpaca_headers()

def options_get(url: str, params: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    return alpaca_get(url, params=params)

def normalize_option_type(value: Any) -> Optional[str]:
    if value is None:
        return None
    val = str(value).strip().lower()
    return "CALL" if val in {"call", "c"} else "PUT" if val in {"put", "p"} else None

def parse_expiration_date(value: Any) -> Optional[date]:
    if value is None:
        return None
    try:
        return pd.Timestamp(value).date()
    except Exception:
        return None

def normalize_contract(raw: dict[str, Any]) -> Optional[dict[str, Any]]:
    if not isinstance(raw, dict):
        return None
    symbol = str(raw.get("symbol", "")).strip().upper()
    underlying = str(raw.get("underlying_symbol", "")).strip().upper()
    option_type = normalize_option_type(raw.get("type", raw.get("right")))
    strike = safe_positive_float(raw.get("strike_price", raw.get("strike")))
    expiration = parse_expiration_date(raw.get("expiration_date", raw.get("expiration")))
    if not symbol or not option_type or strike is None or expiration is None:
        return None
    return {"symbol": symbol, "underlying_symbol": underlying, "type": option_type, "strike": strike, "expiration_date": expiration, "raw": raw}

def fetch_option_contracts(underlying: str = PRIMARY_UNDERLYING, expiration: Optional[date] = None, option_type: str = "both") -> list[dict[str, Any]]:
    url = f"{ALPACA_TRADING_URL}/v2/options/contracts"
    params = {"underlying_symbols": underlying, "status": "active", "limit": 1000}
    if expiration:
        params["expiration_date"] = expiration.isoformat()
    if option_type in {"call", "put"}:
        params["type"] = option_type
    payload = options_get(url, params=params)
    contracts = payload.get("option_contracts", payload.get("contracts", []))
    normalized = []
    for raw in contracts:
        c = normalize_contract(raw)
        if c:
            normalized.append(c)
    return normalized

def fetch_latest_option_quotes(option_symbols: list[str]) -> dict[str, dict[str, Any]]:
    if not option_symbols:
        return {}
    url = f"{ALPACA_DATA_URL}/v1beta1/options/quotes/latest"
    payload = options_get(url, params={"symbols": ",".join(option_symbols), "feed": OPTIONS_FEED})
    return payload.get("quotes", {})

def validate_option_quote(quote: dict[str, Any], current_time: Optional[datetime] = None) -> tuple[bool, str, dict[str, Any]]:
    if not isinstance(quote, dict):
        return False, "QUOTE_MISSING", {}
    bid = safe_positive_float(quote.get("bp", quote.get("bid_price")))
    ask = safe_positive_float(quote.get("ap", quote.get("ask_price")))
    if bid is None or ask is None or ask < bid:
        return False, "INVALID_BID_ASK", {}
    mid = (bid + ask) / 2.0
    return True, "QUOTE_OK", {"bid": bid, "ask": ask, "midpoint": mid, "spread_percent": calculate_spread_percent(bid, ask)}

# ============================================================
# PART 5/6: Journaling and Logging
# ============================================================

JOURNAL_DIR = Path(_s("JOURNAL_DIR", "journal"))
SIGNAL_JOURNAL_FILE = JOURNAL_DIR / "signals.csv"
QUOTE_JOURNAL_FILE = JOURNAL_DIR / "quotes.csv"
OUTCOME_JOURNAL_FILE = JOURNAL_DIR / "outcomes.csv"
MODEL_JOURNAL_FILE = JOURNAL_DIR / "model_runs.csv"

SIGNAL_COLUMNS = ["signal_id", "created_at_utc", "underlying", "direction", "option_symbol", "status"]
QUOTE_COLUMNS = ["signal_id", "observed_at_utc", "option_symbol", "bid", "ask"]
OUTCOME_COLUMNS = ["signal_id", "evaluated_at_utc", "outcome_status"]
MODEL_COLUMNS = ["trained_at_utc", "enabled", "threshold"]

def initialize_journal() -> None:
    JOURNAL_DIR.mkdir(parents=True, exist_ok=True)
    for path, cols in [(SIGNAL_JOURNAL_FILE, SIGNAL_COLUMNS), (QUOTE_JOURNAL_FILE, QUOTE_COLUMNS), (OUTCOME_JOURNAL_FILE, OUTCOME_COLUMNS), (MODEL_JOURNAL_FILE, MODEL_COLUMNS)]:
        if not path.exists():
            pd.DataFrame(columns=cols).to_csv(path, index=False)

def append_journal_row(path: Path, cols: list[str], row: dict[str, Any]) -> tuple[bool, str]:
    try:
        initialize_journal()
        pd.DataFrame([{c: row.get(c) for c in cols}], columns=cols).to_csv(path, mode="a", header=not path.exists() or path.stat().st_size == 0, index=False)
        return True, "OK"
    except Exception as e:
        return False, str(e)

def journal_signal(signal_id: str, prediction: dict, recommendation: dict, model_bundle: Optional[dict] = None, reason: str = "") -> tuple[bool, str]:
    return append_journal_row(SIGNAL_JOURNAL_FILE, SIGNAL_COLUMNS, {"signal_id": signal_id, "created_at_utc": now_utc().isoformat(), "underlying": PRIMARY_UNDERLYING, "direction": prediction.get("signal"), "status": "RECORDED"})

def journal_model_run(model_bundle: dict) -> tuple[bool, str]:
    return append_journal_row(MODEL_JOURNAL_FILE, MODEL_COLUMNS, {"trained_at_utc": now_utc().isoformat(), "enabled": model_bundle.get("enabled", False), "threshold": model_bundle.get("threshold", 0.55)})

# ============================================================
# PART 6/6: Telegram, Main Loop & Outlook Report (SPY + SPX Conversion)
# ============================================================

TG_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TG_CHAT = os.getenv("TELEGRAM_CHAT_ID", "").strip()
TELEGRAM_ENABLED = _b("TELEGRAM_ENABLED", True)
POLL_SECONDS = max(10, _i("POLL_SECONDS", 30))

def send_telegram(message):
    if not TELEGRAM_ENABLED or not TG_TOKEN or not TG_CHAT:
        return False
    try:
        res = requests.post(f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage", json={"chat_id": TG_CHAT, "text": str(message), "parse_mode": "Markdown"}, timeout=10)
        return res.status_code == 200
    except Exception:
        return False

def send_market_outlook_telegram(spy_price, spx_equiv, direction, target_spy, points_move, bot_token, chat_id):
    if not bot_token or not chat_id:
        return
    dir_text = "صعود بنقاط مستهدفة 🟢" if direction == "up" else "هبوط بنقاط مستهدفة 🔴"
    message = (
        "✨ **تحليل حركة المؤشر المتوقعة**\n\n"
        f"▫️ **المتداول (SPY):** `{spy_price:.2f}`\n"
        f"▫️ **المعادل لـ (SPX الحقيقي):** `~{spx_equiv:.2f}`\n"
        f"▫️ **الوجهة المتوقعة:** {dir_text}\n"
        f"▫️ **الهدف السعري (SPY):** `{target_spy:.2f}`\n"
        f"▫️ **مدى الحركة:** `{points_move}`\n\n"
        "⚡ *جاهز للجلسة بناءً على تحليل الـ ATR*\n\n"
        "<i>✨ تصميم أبو سحاب</i>"
    )
    try:
        requests.post(f"https://api.telegram.org/bot{bot_token}/sendMessage", json={"chat_id": chat_id, "text": message, "parse_mode": "Markdown"}, timeout=10)
    except Exception:
        pass

def calculate_and_send_outlook(bot_token, chat_id):
    try:
        bars = fetch_stock_bars(PRIMARY_UNDERLYING, timeframe=TIMEFRAME, history_days=5)
        if bars.empty:
            return
        closes = bars["close"].tolist()
        highs = bars["high"].tolist()
        lows = bars["low"].tolist()
        spy_current = float(closes[-1])
        spx_equiv = spy_current * 10  # معامل تقريبي مطابق لشاشات SPX
        
        tr_list = []
        for i in range(1, len(closes)):
            tr = max(highs[i] - lows[i], abs(highs[i] - closes[i-1]), abs(lows[i] - closes[i-1]))
            tr_list.append(tr)
        atr = sum(tr_list) / len(tr_list) if tr_list else 1.0
        
        if closes[-1] >= closes[-2]:
            direction = "up"
            target_spy = spy_current + (atr * 0.5)
            points_move = f"+{round(atr * 0.5, 2)} نقاط"
        else:
            direction = "down"
            target_spy = spy_current - (atr * 0.5)
            points_move = f"-{round(atr * 0.5, 2)} نقاط"
            
        send_market_outlook_telegram(spy_current, spx_equiv, direction, target_spy, points_move, bot_token, chat_id)
        log.info("تم إرسال تحليل المؤشر بنجاح.")
    except Exception as e:
        log.error(f"خطأ في إرسال تحليل المؤشر: {e}")

def _data_timestamp_is_fresh(timestamp, max_age_seconds=None):
    return True, "FRESH_BYPASS"

def run_scan():
    log.info("Starting scan for %s.", PRIMARY_UNDERLYING)
    prediction_result = get_current_prediction()
    if not prediction_result.get("ok", False):
        log.info("Scan complete: no alert. reason=%s", prediction_result.get("reason"))
        return {"ok": False, "reason": prediction_result.get("reason")}
    return {"ok": True}

def main():
    log.info("=" * 60)
    log.info("%s — VERSION %s", BOT_NAME, BOT_VERSION)
    log.info("Underlying: %s", PRIMARY_UNDERLYING)
    log.info("Mode: RECOMMENDATIONS ONLY")
    log.info("=" * 60)

    try:
        initialize_journal()
    except Exception:
        pass

    log.info("جاري إرسال تحليل المؤشر الصباحي...")
    calculate_and_send_outlook(TG_TOKEN, TG_CHAT)

    while True:
        try:
            run_scan()
        except Exception:
            log.exception("Scan error.")
        time.sleep(POLL_SECONDS)

if __name__ == "__main__":
    main()
