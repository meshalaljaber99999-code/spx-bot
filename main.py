
# ============================================================
# صَفاء التحليل — SPY 0DTE AI ADVISOR v21.4
# PART 1/6
# Imports • Configuration • NY Session • Calendar • Market Data
#
# MODE: RECOMMENDATIONS ONLY — NO ORDER EXECUTION
# IMPORTANT: SPY is not SPX. SPY data must never be presented
# as the actual SPX index price or as an SPXW option quote.
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
    """Read a string environment variable."""
    value = os.getenv(name)
    return default if value is None else value.strip()


def _i(name: str, default: int) -> int:
    """Read an integer environment variable safely."""
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    try:
        return int(value.strip())
    except (TypeError, ValueError):
        log.warning("%s is invalid; using default %s", name, default)
        return default


def _f(name: str, default: float) -> float:
    """Read a finite floating-point environment variable safely."""
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    try:
        result = float(value.strip())
        if not math.isfinite(result):
            raise ValueError("non-finite value")
        return result
    except (TypeError, ValueError):
        log.warning("%s is invalid; using default %s", name, default)
        return default


def _b(name: str, default: bool = False) -> bool:
    """Read a boolean environment variable safely."""
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return value.strip().lower() in {
        "1", "true", "yes", "y", "on"
    }


# ============================================================
# 3. API CONFIGURATION
# ============================================================

def _s(name, default=""):
    return os.getenv(name, default).strip()

APCA_API_KEY_ID = _s("APCA_API_KEY_ID")
APCA_API_SECRET_KEY = _s("APCA_API_SECRET_KEY")

APCA_API_KEY_ID = _s("APCA_API_KEY_ID")
APCA_API_SECRET_KEY = _s("APCA_API_SECRET_KEY")

log.info(
    "ENV DEBUG | "
    f"KEY_IN_ENV={bool(os.environ.get('APCA_API_KEY_ID'))} | "
    f"SECRET_IN_ENV={bool(os.environ.get('APCA_API_SECRET_KEY'))} | "
    f"KEY_LOADED={bool(APCA_API_KEY_ID)} | "
    f"SECRET_LOADED={bool(APCA_API_SECRET_KEY)}"
)

ALPACA_DATA_URL = _s(
    "ALPACA_DATA_URL",
    "https://data.alpaca.markets",
).rstrip("/")

ALPACA_DATA_URL = _s(
    "ALPACA_DATA_URL",
    "https://data.alpaca.markets",
).rstrip("/")

ALPACA_TRADING_URL = _s(
    "ALPACA_TRADING_URL",
    "https://paper-api.alpaca.markets",
).rstrip("/")


log.info(
    "ALPACA ENV CHECK | "
    f"KEY_ID_PRESENT={bool(APCA_API_KEY_ID)} | "
    f"SECRET_PRESENT={bool(APCA_API_SECRET_KEY)}"
)

# Common Alpaca stock market-data feeds include iex and sip.
# The account's permissions determine which feeds are accessible.
ALPACA_DATA_FEED = _s("ALPACA_DATA_FEED", "iex").lower()

REQUEST_TIMEOUT_SECONDS = _i("REQUEST_TIMEOUT_SECONDS", 15)
HTTP_RETRIES = _i("HTTP_RETRIES", 3)
HTTP_RETRY_SLEEP_SECONDS = _f("HTTP_RETRY_SLEEP_SECONDS", 1.5)

# This bot must not submit orders.
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

# These thresholds are filters, not guarantees of profitability.
MIN_PROBABILITY = _f("MIN_PROBABILITY", 0.64)
MIN_AUC = _f("MIN_AUC", 0.55)
MIN_SPY_SCORE = _f("MIN_SPY_SCORE", 78.0)
MIN_CONFIRMATIONS = _i("MIN_CONFIRMATIONS", 2)

# Conservative model-evaluation gate.
MIN_HOLDOUT_SIGNAL_N = _i("MIN_HOLDOUT_SIGNAL_N", 30)
MIN_HOLDOUT_SIGNAL_ACC = _f("MIN_HOLDOUT_SIGNAL_ACC", 0.55)
MAX_HOLDOUT_BRIER = _f("MAX_HOLDOUT_BRIER", 0.25)

# Volatility / movement settings.
ATR_TARGET = _f("ATR_TARGET", 0.50)

# Option quote filters.
MAX_SPREAD_PERCENT = _f("MAX_SPREAD_PERCENT", 0.12)
MIN_OPTION_PREMIUM = _f("MIN_OPTION_PREMIUM", 0.50)
MAX_OPTION_PREMIUM = _f("MAX_OPTION_PREMIUM", 50.00)
MAX_STRIKE_DISTANCE = _f("MAX_STRIKE_DISTANCE", 35.0)
MIN_DELTA = _f("MIN_DELTA", 0.20)
MAX_DELTA = _f("MAX_DELTA", 0.80)

# Freshness requirements.
MAX_BAR_AGE_MINUTES = _i("MAX_BAR_AGE_MINUTES", 10)
MAX_QUOTE_AGE_SECONDS = _i("MAX_QUOTE_AGE_SECONDS", 60)
MAX_SPOT_AGE_SECONDS = _i("MAX_SPOT_AGE_SECONDS", 60)
MAX_SPOT_OPTION_SKEW_SECONDS = _i(
    "MAX_SPOT_OPTION_SKEW_SECONDS", 30
)

# Signal cadence.
POLL_SECONDS = _i("POLL_SECONDS", 30)
SIGNAL_COOLDOWN_MINUTES = _i("SIGNAL_COOLDOWN_MINUTES", 20)

# Avoid the first and last minutes of the regular session.
NO_TRADE_FIRST_MINUTES = _i("NO_TRADE_FIRST_MINUTES", 10)
NO_TRADE_LAST_MINUTES = _i("NO_TRADE_LAST_MINUTES", 30)

# Confirmation instruments: market context only.
MARKET_SYMBOLS = ["SPY", "QQQ"]

CONFIRMATION_SYMBOLS = [
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

# Stock options are not automatically equivalent to SPXW options.
# The actual SPX index and SPXW option chain require suitable data access.
PRIMARY_UNDERLYING = _s("PRIMARY_UNDERLYING", "SPY").upper()

# ============================================================
# 5. RISK AND ACCOUNT SETTINGS
# ============================================================

# Zero equity means no risk-based contract sizing can be approved.
# Set this to your actual account equity through environment settings.
ACCOUNT_EQUITY = _f("ACCOUNT_EQUITY", 0.0)

# Percent units: 0.5 means 0.5%, not 50%.
RISK_PCT = _f("RISK_PCT", 0.50)

MAX_DAILY_LOSS = _f("MAX_DAILY_LOSS", 200.0)
MAX_TRADES_PER_DAY = _i("MAX_TRADES_PER_DAY", 5)
MAX_OPEN_POSITIONS = _i("MAX_OPEN_POSITIONS", 2)
MAX_CONTRACTS = _i("MAX_CONTRACTS", 5)

ESTIMATED_SLIPPAGE_PER_CONTRACT = _f(
    "ESTIMATED_SLIPPAGE_PER_CONTRACT", 0.05
)
ESTIMATED_FEES_PER_CONTRACT = _f(
    "ESTIMATED_FEES_PER_CONTRACT", 1.00
)

# ============================================================
# 6. TELEGRAM CONFIGURATION
# ============================================================

TELEGRAM_BOT_TOKEN = _s("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = _s("TELEGRAM_CHAT_ID")

TELEGRAM_ENABLED = bool(
    TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID
)

# ============================================================
# 7. JOURNAL FILES
# ============================================================

DATA_DIR = _s("DATA_DIR", ".")

SIGNAL_JOURNAL_PATH = os.path.join(
    DATA_DIR, "safa_signals_v214.csv"
)
QUOTE_JOURNAL_PATH = os.path.join(
    DATA_DIR, "safa_quote_observations_v214.csv"
)
MODEL_JOURNAL_PATH = os.path.join(
    DATA_DIR, "safa_model_diagnostics_v214.csv"
)

os.makedirs(DATA_DIR, exist_ok=True)

# ============================================================
# 8. SESSION TIME HELPERS
# ============================================================

def now_utc() -> datetime:
    """Current timezone-aware UTC time."""
    return datetime.now(TIMEZONE_UTC)


def now_ny() -> datetime:
    """Current timezone-aware New York time."""
    return datetime.now(TIMEZONE_NY)


def to_utc(value: Any) -> Optional[datetime]:
    """
    Convert a datetime or parseable timestamp to UTC.
    Return None for invalid or missing values.
    """
    if value is None:
        return None

    try:
        ts = pd.Timestamp(value)
        if pd.isna(ts):
            return None

        if ts.tzinfo is None:
            # Naive timestamps from market-data APIs are treated as UTC
            # only where the calling API explicitly documents that behavior.
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")

        return ts.to_pydatetime()
    except Exception:
        return None


def to_ny(value: Any) -> Optional[datetime]:
    """Convert a datetime or timestamp to New York time."""
    ts = to_utc(value)
    if ts is None:
        return None
    return ts.astimezone(TIMEZONE_NY)


def minutes_between(a: datetime, b: datetime) -> float:
    """Absolute time difference in minutes."""
    return abs((a - b).total_seconds()) / 60.0


def seconds_between(a: datetime, b: datetime) -> float:
    """Absolute time difference in seconds."""
    return abs((a - b).total_seconds())


def is_weekday(day: date) -> bool:
    return day.weekday() < 5


def nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    """Return the nth weekday of a month; Monday=0."""
    first = date(year, month, 1)
    offset = (weekday - first.weekday()) % 7
    return first + timedelta(days=offset + 7 * (n - 1))


def last_weekday(year: int, month: int, weekday: int) -> date:
    """Return the last weekday of a month; Monday=0."""
    if month == 12:
        next_month = date(year + 1, 1, 1)
    else:
        next_month = date(year, month + 1, 1)

    current = next_month - timedelta(days=1)
    offset = (current.weekday() - weekday) % 7
    return current - timedelta(days=offset)


def observed_fixed_holiday(day: date) -> date:
    """Apply common US federal holiday observation rules."""
    if day.weekday() == 5:
        return day - timedelta(days=1)
    if day.weekday() == 6:
        return day + timedelta(days=1)
    return day


def us_market_holidays(year: int) -> set[date]:
    """
    Approximate US equity-market holiday calendar.

    This is a conservative manual calendar for common holidays.
    It does not model every exceptional exchange closure or every
    special early-close session. For production trading, verify the
    official exchange calendar.
    """
    holidays: set[date] = set()

    # New Year's Day, including a Saturday-observed closure
    jan1 = date(year, 1, 1)
    holidays.add(observed_fixed_holiday(jan1))

    # If January 1 of the next year is Saturday, December 31 may close.
    next_jan1 = date(year + 1, 1, 1)
    if next_jan1.weekday() == 5:
        holidays.add(date(year, 12, 31))

    # Martin Luther King Jr. Day: third Monday in January
    holidays.add(nth_weekday(year, 1, 0, 3))

    # Presidents' Day: third Monday in February
    holidays.add(nth_weekday(year, 2, 0, 3))

    # Good Friday: two days before Easter Sunday
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
    easter_month = (h + l - 7 * m + 114) // 31
    easter_day = ((h + l - 7 * m + 114) % 31) + 1
    easter_sunday = date(year, easter_month, easter_day)
    holidays.add(easter_sunday - timedelta(days=2))

    # Memorial Day: last Monday in May
    holidays.add(last_weekday(year, 5, 0))

    # Juneteenth, observed from 2022 onward
    if year >= 2022:
        holidays.add(observed_fixed_holiday(date(year, 6, 19)))

    # Independence Day
    holidays.add(observed_fixed_holiday(date(year, 7, 4)))

    # Labor Day: first Monday in September
    holidays.add(nth_weekday(year, 9, 0, 1))

    # Thanksgiving: fourth Thursday in November
    holidays.add(nth_weekday(year, 11, 3, 4))

    # Christmas
    holidays.add(observed_fixed_holiday(date(year, 12, 25)))

    return holidays


def is_market_holiday(day: date) -> bool:
    """Check common holidays for this date and adjacent calendar years."""
    holidays = set()
    for year in (day.year - 1, day.year, day.year + 1):
        holidays.update(us_market_holidays(year))
    return day in holidays


def regular_session_bounds(day: date) -> tuple[datetime, datetime]:
    """Regular session bounds in New York time: 09:30–16:00."""
    start = datetime.combine(
        day, dt_time(hour=9, minute=30), tzinfo=TIMEZONE_NY
    )
    end = datetime.combine(
        day, dt_time(hour=16, minute=0), tzinfo=TIMEZONE_NY
    )
    return start, end


def is_regular_session(value: Optional[datetime] = None) -> bool:
    """
    Check weekday, common holiday calendar and regular session hours.
    Early closes are not fully represented by this manual calendar.
    """
    current = to_ny(value or now_ny())
    if current is None:
        return False

    if not is_weekday(current.date()):
        return False

    if is_market_holiday(current.date()):
        return False

    start, end = regular_session_bounds(current.date())
    return start <= current < end


def session_window_status(
    value: Optional[datetime] = None,
) -> tuple[bool, str]:
    """
    Return whether new signals are allowed by the regular-session
    time window, but permits pre-market warm-up and after-hours review.
    """
    current = to_ny(value or now_ny())
    if current is None:
        return False, "INVALID_TIME"

    if not is_weekday(current.date()):
        return False, "WEEKEND"

    if is_market_holiday(current.date()):
        return False, "MARKET_HOLIDAY"

    start, end = regular_session_bounds(current.date())

    # 1. فترة ما قبل الافتتاح (Pre-Market Warm-up): قبل الجرس بساعة
    pre_market_start = start - timedelta(hours=1)
    if pre_market_start <= current < start:
        return True, "PRE_MARKET_WARMUP"

    if current < start:
        return False, "PRE_MARKET"

    # 2. فترة ما بعد الإغلاق (After-Hours Review): بعد الإغلاق بساعتين
    after_market_end = end + timedelta(hours=2)
    if end <= current < after_market_end:
        return True, "AFTER_MARKET_REVIEW"

    if current >= end:
        return False, "AFTER_MARKET"

    elapsed_minutes = (current - start).total_seconds() / 60.0
    remaining_minutes = (end - current).total_seconds() / 60.0

    if elapsed_minutes < NO_TRADE_FIRST_MINUTES:
        return False, "OPENING_BUFFER"

    if remaining_minutes <= NO_TRADE_LAST_MINUTES:
        return False, "CLOSING_BUFFER"

    return True, "SESSION_OK"


# ============================================================
# 9. HTTP SESSION AND ALPACA HEADERS
# ============================================================


HTTP = requests.Session()

HTTP.headers.update({
    "Accept": "application/json",
    "User-Agent": "SafaaAnalysis/21.0",
})

def alpaca_headers() -> dict[str, str]:
    """Create authenticated Alpaca market-data headers."""

    key_id = str(APCA_API_KEY_ID or "").strip()
    secret_key = str(APCA_API_SECRET_KEY or "").strip()

    if not key_id or not secret_key:
        raise RuntimeError(
            "Alpaca credentials are empty. "
            f"KEY_ID_PRESENT={bool(key_id)}, "
            f"SECRET_PRESENT={bool(secret_key)}. "
            "Check environment variables and _s()."
        )

    return {
        "APCA-API-KEY-ID": key_id,
        "APCA-API-SECRET-KEY": secret_key,
        "Accept": "application/json",
    }



def alpaca_get(
    url: str,
    params: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """
    Perform a retry-limited Alpaca GET request.
    Raises on authentication, HTTP or malformed-response errors.
    """
    headers = alpaca_headers()
    last_error: Optional[Exception] = None

    for attempt in range(1, HTTP_RETRIES + 1):
        try:
            response = HTTP.get(
                url,
                headers=headers,
                params=params,
                timeout=REQUEST_TIMEOUT_SECONDS,
            )

            if response.status_code in (401, 403):
                raise RuntimeError(
                    f"Alpaca authentication/permission error "
                    f"{response.status_code}: {response.text[:300]}"
                )

            response.raise_for_status()

            payload = response.json()
            if not isinstance(payload, dict):
                raise RuntimeError("Unexpected Alpaca JSON response.")

            return payload

        except Exception as exc:
            last_error = exc
            if attempt < HTTP_RETRIES:
                sleep_for = HTTP_RETRY_SLEEP_SECONDS * attempt
                log.warning(
                    "Alpaca request failed (%s/%s): %s; retrying in %.1fs",
                    attempt,
                    HTTP_RETRIES,
                    exc,
                    sleep_for,
                )
                time.sleep(sleep_for)

    raise RuntimeError(
        f"Alpaca GET failed after {HTTP_RETRIES} attempts: {last_error}"
    )


# ============================================================
# 10. MARKET DATA NORMALIZATION
# ============================================================

REQUIRED_BAR_COLUMNS = [
    "open",
    "high",
    "low",
    "close",
    "volume",
]


def normalize_bars(
    raw_bars: list[dict[str, Any]],
) -> pd.DataFrame:
    """
    Normalize Alpaca stock bars to a UTC-indexed DataFrame.
    Invalid rows are removed; duplicate timestamps keep the last row.
    """
    if not raw_bars:
        return pd.DataFrame(columns=REQUIRED_BAR_COLUMNS)

    frame = pd.DataFrame(raw_bars)

    if "t" not in frame.columns:
        raise ValueError("Bar response has no timestamp field 't'.")

    rename_map = {
        "t": "timestamp",
        "o": "open",
        "h": "high",
        "l": "low",
        "c": "close",
        "v": "volume",
        "n": "trade_count",
        "vw": "vwap",
    }
    frame = frame.rename(columns=rename_map)

    if "timestamp" not in frame.columns:
        raise ValueError("Unable to identify bar timestamps.")

    frame["timestamp"] = pd.to_datetime(
        frame["timestamp"], utc=True, errors="coerce"
    )
    frame = frame.dropna(subset=["timestamp"])

    for column in REQUIRED_BAR_COLUMNS:
        if column not in frame.columns:
            frame[column] = np.nan

        frame[column] = pd.to_numeric(
            frame[column], errors="coerce"
        )

    frame = frame.dropna(subset=REQUIRED_BAR_COLUMNS)

    frame = frame[
        (frame["open"] > 0)
        & (frame["high"] > 0)
        & (frame["low"] > 0)
        & (frame["close"] > 0)
        & (frame["volume"] >= 0)
    ]

    frame = frame.sort_values("timestamp")
    frame = frame.drop_duplicates(
        subset=["timestamp"], keep="last"
    )
    frame = frame.set_index("timestamp")
    frame.index.name = "timestamp"

    for optional_column in ("trade_count", "vwap"):
        if optional_column in frame.columns:
            frame[optional_column] = pd.to_numeric(
                frame[optional_column], errors="coerce"
            )

    return frame


# ============================================================
# 11. FETCH HISTORICAL STOCK BARS
# ============================================================


def fetch_stock_bars(
    symbol: str,
    timeframe: str = TIMEFRAME,
    history_days: int = HISTORY_DAYS,
    limit: int = 10000,
) -> pd.DataFrame:
    """
    Fetch historical stock bars from Alpaca's market-data API.

    Pagination uses next_page_token where provided.
    Returns actual stock bars, not SPY multiplied by ten.
    """
    symbol = symbol.strip().upper()

    if not symbol:
        raise ValueError("Symbol cannot be empty.")

    if history_days < 1:
        raise ValueError("history_days must be >= 1.")

    end_time = now_utc()
    start_time = end_time - timedelta(days=history_days)

    url = f"{ALPACA_DATA_URL}/v2/stocks/{symbol}/bars"

    params: dict[str, Any] = {
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
    max_pages = 50

    while pages < max_pages:
        if page_token:
            params["page_token"] = page_token
        else:
            params.pop("page_token", None)

        payload = alpaca_get(url, params=params)
        bars = payload.get("bars") or []

        log.warning(
            "ALPACA CHECK | symbol=%s | count=%s | "
            "last_bar=%s | next_page=%s | feed=%s | end=%s",
            symbol,
            len(bars),
            bars[-1].get("t") if bars else None,
            bool(payload.get("next_page_token")),
            ALPACA_DATA_FEED,
            params.get("end"),
        )

        if bars:
            if not isinstance(bars, list):
                raise RuntimeError(
                    f"Unexpected bars payload for {symbol}."
                )
            collected.extend(bars)

        page_token = payload.get("next_page_token")
        pages += 1

        if not page_token:
            break

    if not collected:
        log.warning(
            "No bars returned for %s (feed=%s, timeframe=%s).",
            symbol,
            ALPACA_DATA_FEED,
            timeframe,
        )
        return pd.DataFrame(columns=REQUIRED_BAR_COLUMNS)

    log.warning(
        "DEBUG FETCH | symbol=%s | collected=%d | pages=%d",
        symbol,
        len(collected),
        pages,
    )

    frame = normalize_bars(collected)

    log.warning(
        "DEBUG NORMALIZE | symbol=%s | rows=%d | empty=%s",
        symbol,
        len(frame),
        frame.empty,
    )

    # DIAGNOSTIC: verify latest bar timestamp
    if not frame.empty:
        latest_bar_time = frame.index[-1]
        current_utc = pd.Timestamp.now(tz="UTC")

        if latest_bar_time.tzinfo is None:
            log.warning(
                "DATA TIMEZONE NAIVE | %s | timestamp=%s",
                symbol,
                latest_bar_time,
            )
        else:
            age_seconds = (
                current_utc - latest_bar_time.tz_convert("UTC")
            ).total_seconds()

            log.info(
                "DATA AGE CHECK | %s | last=%s | "
                "now_utc=%s | age_seconds=%.1f",
                symbol,
                latest_bar_time.isoformat(),
                current_utc.isoformat(),
                age_seconds,
            )

    if frame.empty:
        log.warning(
            "All returned bars were invalid for %s.",
            symbol,
        )
        return frame

    log.info(
        "DATA %s: %s bars | last=%s | feed=%s",
        symbol,
        len(frame),
        frame.index[-1].isoformat(),
        ALPACA_DATA_FEED,
    )

    return frame



# ============================================================
# 12. DATA FRESHNESS AND QUALITY CHECKS
# ============================================================


def get_current_prediction():
    """
    Model Prediction Adapter
    صَفاء التحليل — SPY 0DTE AI ADVISOR

    Uses the actual Alpaca stock-bars function when available.
    Recommendation-only; never submits orders.
    """

    # 1. Resolve market-data function.
    bars_function = _resolve_callable(
        "get_live_bars",
        "get_market_bars",
        "get_underlying_bars",
        "fetch_stock_bars",  # FIX: actual function in Part 11
    )

    # 2. Resolve feature-engineering function.
    feature_function = _resolve_callable(
        "make_features",
        "build_features",
        "create_features",
    )

    # 3. Resolve model-training function.
    training_function = _resolve_callable(
        "train_model",
        "train_direction_model",
        "get_model",
    )

    # 4. Resolve prediction function.
    prediction_function = _resolve_callable(
        "predict_direction",
        "calculate_prediction",
    )

    if bars_function is None:
        return {
            "ok": False,
            "reason": "LIVE_BARS_FUNCTION_NOT_FOUND",
        }

    if feature_function is None:
        return {
            "ok": False,
            "reason": "FEATURE_FUNCTION_NOT_FOUND",
        }

    if training_function is None:
        return {
            "ok": False,
            "reason": "MODEL_TRAINING_FUNCTION_NOT_FOUND",
        }

    if prediction_function is None:
        return {
            "ok": False,
            "reason": "PREDICTION_FUNCTION_NOT_FOUND",
        }

    try:
        # 5. Fetch historical market bars.
        bars = _call_compatible(
            bars_function,
            symbol=PRIMARY_UNDERLYING,
            timeframe=TIMEFRAME,
            history_days=HISTORY_DAYS,
            days=HISTORY_DAYS,
        )

        if not isinstance(bars, pd.DataFrame) or bars.empty:
            return {
                "ok": False,
                "reason": "NO_LIVE_BARS",
            }

        # 6. Verify the latest bar timestamp.
        latest_timestamp = None

        for column in ("timestamp", "time", "datetime", "t"):
            if column in bars.columns:
                latest_timestamp = bars[column].iloc[-1]
                break

        # fetch_stock_bars() normally uses a DatetimeIndex.
        if latest_timestamp is None:
            latest_timestamp = bars.index[-1]

        log.info(
            "UNDERLYING FRESHNESS DEBUG | symbol=%s | "
            "latest_timestamp=%s | max_age_seconds=%s | "
            "now_utc=%s",
            PRIMARY_UNDERLYING,
            latest_timestamp,
            MAX_DATA_AGE_SECONDS,
            now_utc().isoformat(),
        )

      
        fresh, freshness_reason = _data_timestamp_is_fresh(
            latest_timestamp
        )

        if not fresh:
            return {
                "ok": False,
                "reason": (
                    f"UNDERLYING_DATA_{freshness_reason}"
                ),
            }

        # 7. Generate features.
        features = _call_compatible(
            feature_function,
            bars=bars,
            df=bars,
            data=bars,
        )

        if not isinstance(features, pd.DataFrame) or features.empty:
            return {
                "ok": False,
                "reason": "FEATURE_FRAME_EMPTY",
            }

        
        # 8. Train or obtain the model.
        model_bundle = _call_compatible(
            training_function,
            dataset=features,
        )


        if isinstance(model_bundle, tuple):
            model = model_bundle[0] if model_bundle else None
            model_bundle = {
                "model": model,
                "enabled": model is not None,
                "reasons": [],
            }

        elif not isinstance(model_bundle, dict):
            model_bundle = {
                "model": model_bundle,
                "enabled": model_bundle is not None,
                "reasons": [],
            }

        journal_model_run(model_bundle)

        
        if not model_bundle.get("enabled", False):
            return {
                "ok": False,
                "reason": "MODEL_GATE_DISABLED",
                "model_bundle": model_bundle,
            }


        
        # 9. Train using the existing feature-frame pipeline.
        training_pipeline = _resolve_callable(
            "train_from_feature_frame"
        )

        if training_pipeline is None:
            return {
                "ok": False,
                "reason": "TRAINING_PIPELINE_NOT_FOUND",
            }

        model_bundle = _call_compatible(
            training_pipeline,
            feature_frame=features,
        )


        # 10. Normalize direction and fail closed on WAIT.
        direction = _normalise_direction(
            prediction.get("signal")
            or prediction.get("direction")
        )

        if direction == "WAIT":
            return {
                "ok": False,
                "reason": prediction.get(
                    "reason",
                    "MODEL_DIRECTION_IS_WAIT",
                ),
                "prediction": prediction,
                "model_bundle": model_bundle,
            }

        prediction["signal"] = direction

        return {
            "ok": True,
            "prediction": prediction,
            "model_bundle": model_bundle,
            "bars": bars,
            "features": features,
            "latest_features": latest_features,
        }

    except Exception as exc:
        log.exception("Current prediction failed.")
        return {
            "ok": False,
            "reason": f"PREDICTION_PIPELINE_ERROR:{exc}",
        }



# ============================================================
# 13. SAFE PRICE / NUMERIC HELPERS
# ============================================================

def safe_float(
    value: Any,
    default: Optional[float] = None,
) -> Optional[float]:
    try:
        result = float(value)
        if not math.isfinite(result):
            return default
        return result
    except (TypeError, ValueError):
        return default


def safe_positive_float(
    value: Any,
) -> Optional[float]:
    result = safe_float(value)
    if result is None or result <= 0:
        return None
    return result


def pct_change(
    current: Optional[float],
    previous: Optional[float],
) -> Optional[float]:
    if current is None or previous is None or previous == 0:
        return None
    return ((current / previous) - 1.0) * 100.0


def calculate_spread_percent(
    bid: Any,
    ask: Any,
) -> Optional[float]:
    """
    Calculate bid/ask spread as a fraction of midpoint.
    Example: 0.05 means 5%. Returns None for invalid quotes.
    """
    bid_value = safe_positive_float(bid)
    ask_value = safe_positive_float(ask)

    if bid_value is None or ask_value is None:
        return None

    if ask_value < bid_value:
        return None

    midpoint = (bid_value + ask_value) / 2.0
    if midpoint <= 0:
        return None

    return (ask_value - bid_value) / midpoint


# ============================================================
# END OF PART 1/6
# ============================================================

# ============================================================
# صَفاء التحليل — SPY 0DTE AI ADVISOR v21.4
# PART 2/6
# Technical Indicators • Price Action • Feature Engineering
# • Multi-Timeframe Confirmation • Training Labels
#
# Paste directly below PART 1/6.
# NO ORDER EXECUTION.
# ============================================================

# ============================================================
# 14. FEATURE DEFINITIONS
# ============================================================

FEATURE_COLUMNS = [
    "ret_1",
    "ret_3",
    "ret_6",
    "ret_12",
    "ema_spread",
    "ema_trend",
    "rsi_14",
    "atr_pct",
    "range_pct",
    "body_pct",
    "upper_wick_pct",
    "lower_wick_pct",
    "close_location",
    "volume_ratio",
    "volume_change",
    "momentum_3",
    "momentum_6",
    "realized_volatility",
    "qqq_ret_3",
    "qqq_ema_spread",
]

MODEL_FEATURE_COUNT = len(FEATURE_COLUMNS)


# ============================================================
# 15. CORE INDICATOR HELPERS
# ============================================================

def _safe_divide(
    numerator: pd.Series,
    denominator: pd.Series,
) -> pd.Series:
    """Vectorized division with invalid or zero denominators masked."""
    denominator = denominator.replace(0, np.nan)
    result = numerator / denominator
    return result.replace([np.inf, -np.inf], np.nan)


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(
        span=span,
        adjust=False,
        min_periods=span,
    ).mean()


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Wilder-style RSI using exponentially smoothed gains/losses."""
    delta = series.diff()

    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)

    avg_gain = gains.ewm(
        alpha=1.0 / period,
        adjust=False,
        min_periods=period,
    ).mean()

    avg_loss = losses.ewm(
        alpha=1.0 / period,
        adjust=False,
        min_periods=period,
    ).mean()

    relative_strength = _safe_divide(avg_gain, avg_loss)
    result = 100.0 - (100.0 / (1.0 + relative_strength))

    # If there were gains but no losses, RSI approaches 100.
    result = result.where(avg_loss != 0, 100.0)
    # If neither gains nor losses exist, RSI is neutral.
    result = result.where(
        ~((avg_gain == 0) & (avg_loss == 0)),
        50.0,
    )

    return result.clip(0.0, 100.0)


def _true_range(frame: pd.DataFrame) -> pd.Series:
    previous_close = frame["close"].shift(1)

    components = pd.concat(
        [
            frame["high"] - frame["low"],
            (frame["high"] - previous_close).abs(),
            (frame["low"] - previous_close).abs(),
        ],
        axis=1,
    )

    return components.max(axis=1)


def _atr(
    frame: pd.DataFrame,
    period: int = 14,
) -> pd.Series:
    true_range = _true_range(frame)

    return true_range.ewm(
        alpha=1.0 / period,
        adjust=False,
        min_periods=period,
    ).mean()


# ============================================================
# 16. DATAFRAME INPUT VALIDATION
# ============================================================

def validate_feature_input(
    bars: Optional[pd.DataFrame],
    symbol: str = "UNKNOWN",
    min_rows: int = 60,
) -> tuple[bool, str]:
    if bars is None:
        return False, f"{symbol}:NO_DATA"

    if not isinstance(bars, pd.DataFrame):
        return False, f"{symbol}:NOT_DATAFRAME"

    if bars.empty:
        return False, f"{symbol}:EMPTY_DATA"

    missing = [
        column for column in REQUIRED_BAR_COLUMNS
        if column not in bars.columns
    ]

    if missing:
        return False, (
            f"{symbol}:MISSING_COLUMNS:"
            + ",".join(missing)
        )

    if len(bars) < min_rows:
        return False, f"{symbol}:TOO_FEW_ROWS:{len(bars)}"

    if not isinstance(bars.index, pd.DatetimeIndex):
        return False, f"{symbol}:INVALID_INDEX"

    if bars.index.tz is None:
        return False, f"{symbol}:TIMEZONE_REQUIRED"

    if not bars.index.is_monotonic_increasing:
        return False, f"{symbol}:INDEX_NOT_SORTED"

    if bars.index.has_duplicates:
        return False, f"{symbol}:DUPLICATE_INDEX"

    return True, "INPUT_OK"


# ============================================================
# 17. PRICE-ACTION FEATURE ENGINEERING
# ============================================================

def build_features(
    bars: pd.DataFrame,
    prefix: str = "",
) -> pd.DataFrame:
    """
    Build features from historical OHLCV bars.

    All features at timestamp t use only information available
    at or before t. No future close is used in the feature values.

    The optional prefix allows the same feature calculation to be
    applied to confirmation instruments without column collisions.
    """
    valid, reason = validate_feature_input(
        bars,
        symbol=prefix or "PRIMARY",
        min_rows=30,
    )

    if not valid:
        raise ValueError(f"Cannot build features: {reason}")

    frame = bars.copy().sort_index()

    for column in REQUIRED_BAR_COLUMNS:
        frame[column] = pd.to_numeric(
            frame[column],
            errors="coerce",
        )

    open_price = frame["open"]
    high = frame["high"]
    low = frame["low"]
    close = frame["close"]
    volume = frame["volume"]

    safe_close = close.replace(0, np.nan)

    # --------------------------------------------------------
    # Returns / momentum
    # --------------------------------------------------------

    frame["ret_1"] = close.pct_change(1)
    frame["ret_3"] = close.pct_change(3)
    frame["ret_6"] = close.pct_change(6)
    frame["ret_12"] = close.pct_change(12)

    frame["momentum_3"] = close - close.shift(3)
    frame["momentum_6"] = close - close.shift(6)

    # --------------------------------------------------------
    # Moving averages / trend
    # --------------------------------------------------------

    ema9 = _ema(close, 9)
    ema20 = _ema(close, 20)
    ema50 = _ema(close, 50)

    frame["ema_spread"] = _safe_divide(
        ema9 - ema20,
        safe_close,
    )

    frame["ema_trend"] = _safe_divide(
        ema20 - ema50,
        safe_close,
    )

    # --------------------------------------------------------
    # RSI / ATR / volatility
    # --------------------------------------------------------

    frame["rsi_14"] = _rsi(close, 14) / 100.0

    atr14 = _atr(frame, 14)
    frame["atr_pct"] = _safe_divide(
        atr14,
        safe_close,
    )

    returns = close.pct_change()

    frame["realized_volatility"] = returns.rolling(
        20,
        min_periods=20,
    ).std()

    # --------------------------------------------------------
    # Candle structure / price action
    # --------------------------------------------------------

    candle_range = (high - low).clip(lower=0)
    body = close - open_price

    frame["range_pct"] = _safe_divide(
        candle_range,
        safe_close,
    )

    frame["body_pct"] = _safe_divide(
        body,
        safe_close,
    )

    upper_wick = high - pd.concat(
        [open_price, close], axis=1
    ).max(axis=1)

    lower_wick = pd.concat(
        [open_price, close], axis=1
    ).min(axis=1) - low

    frame["upper_wick_pct"] = _safe_divide(
        upper_wick.clip(lower=0),
        safe_close,
    )

    frame["lower_wick_pct"] = _safe_divide(
        lower_wick.clip(lower=0),
        safe_close,
    )

    frame["close_location"] = _safe_divide(
        close - low,
        candle_range,
    )

    # Flat candles have no directional close location.
    frame["close_location"] = frame[
        "close_location"
    ].where(candle_range > 0, 0.5)

    # --------------------------------------------------------
    # Volume
    # --------------------------------------------------------

    volume_mean20 = volume.rolling(
        20,
        min_periods=20,
    ).mean()

    frame["volume_ratio"] = _safe_divide(
        volume,
        volume_mean20,
    )

    frame["volume_change"] = volume.pct_change()

    # Replace infinities, but preserve NaN until final filtering.
    frame = frame.replace([np.inf, -np.inf], np.nan)

    if prefix:
        frame = frame.rename(
            columns={
                column: f"{prefix}{column}"
                for column in FEATURE_COLUMNS
                if column in frame.columns
            }
        )

    return frame


# ============================================================
# 18. MARKET CONFIRMATION FEATURES
# ============================================================

def build_confirmation_features(
    bars: pd.DataFrame,
    prefix: str,
) -> pd.DataFrame:
    """
    Build a compact confirmation feature set for QQQ or another
    market instrument. It does not create an SPX price.
    """
    if not prefix:
        raise ValueError("A non-empty prefix is required.")

    frame = build_features(bars, prefix=prefix)

    expected = [
        f"{prefix}ret_3",
        f"{prefix}ema_spread",
    ]

    missing = [
        column for column in expected
        if column not in frame.columns
    ]

    if missing:
        raise ValueError(
            "Confirmation features missing: "
            + ", ".join(missing)
        )

    return frame[expected].copy()


def align_confirmation_data(
    primary: pd.DataFrame,
    confirmation: pd.DataFrame,
    max_age_seconds: int = 600,
) -> pd.DataFrame:
    """
    Align confirmation data backward to primary timestamps.

    merge_asof(direction='backward') prevents a primary bar from
    receiving a confirmation bar that occurs in the future.

    Both indexes must be timezone-aware.
    """
    if primary.empty:
        return primary.copy()

    if confirmation.empty:
        result = primary.copy()
        for column in confirmation.columns:
            result[column] = np.nan
        return result

    left = primary.sort_index().copy()
    right = confirmation.sort_index().copy()

    if left.index.tz is None or right.index.tz is None:
        raise ValueError(
            "Both primary and confirmation indexes must be timezone-aware."
        )

    left = left.reset_index()
    right = right.reset_index()

    left_time = left.columns[0]
    right_time = right.columns[0]

    if left_time != "timestamp":
        left = left.rename(columns={left_time: "timestamp"})

    if right_time != "timestamp":
        right = right.rename(columns={right_time: "timestamp"})

    right = right.rename(
        columns={"timestamp": "confirmation_timestamp"}
    )

    merged = pd.merge_asof(
        left.sort_values("timestamp"),
        right.sort_values("confirmation_timestamp"),
        left_on="timestamp",
        right_on="confirmation_timestamp",
        direction="backward",
        tolerance=pd.Timedelta(seconds=max_age_seconds),
    )

    # Retain the matched confirmation timestamp for diagnostics.
    # The caller can decide whether to keep or drop this column.
    merged = merged.set_index("timestamp")
    merged.index.name = "timestamp"

    return merged


# ============================================================
# 19. 15-MINUTE FEATURES FROM 5-MINUTE BARS
# ============================================================

def build_15m_features(
    bars_5m: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aggregate 5-minute bars into 15-minute bars and calculate
    higher-timeframe trend features.

    Bars are labeled at their ending timestamp. With closed='left'
    and label='right', a 09:30–09:45 bar is labeled 09:45.

    Incomplete groups with fewer than three 5-minute bars are removed.
    """
    valid, reason = validate_feature_input(
        bars_5m,
        symbol="15M_SOURCE",
        min_rows=30,
    )

    if not valid:
        raise ValueError(
            f"Cannot build 15-minute features: {reason}"
        )

    frame = bars_5m.copy().sort_index()

    if frame.index.tz is None:
        raise ValueError(
            "5-minute bars must have timezone-aware timestamps."
        )

    frame = frame.tz_convert(TIMEZONE_NY)

    aggregated = frame.resample(
        "15min",
        closed="left",
        label="right",
        origin="start_day",
        offset="9h30min",
    ).agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    })

    counts = frame["close"].resample(
        "15min",
        closed="left",
        label="right",
        origin="start_day",
        offset="9h30min",
    ).count()

    aggregated = aggregated[counts >= 3].dropna(
        subset=REQUIRED_BAR_COLUMNS
    )

    if len(aggregated) < 30:
        raise ValueError(
            "Not enough complete 15-minute bars for trend features."
        )

    features = build_features(aggregated)

    result = features[
        [
            "ema_spread",
            "ema_trend",
            "rsi_14",
            "atr_pct",
            "ret_3",
        ]
    ].copy()

    result = result.rename(
        columns={
            "ema_spread": "m15_ema_spread",
            "ema_trend": "m15_ema_trend",
            "rsi_14": "m15_rsi_14",
            "atr_pct": "m15_atr_pct",
            "ret_3": "m15_ret_3",
        }
    )

    return result


def align_15m_features(
    primary_5m_features: pd.DataFrame,
    features_15m: pd.DataFrame,
) -> pd.DataFrame:
    """
    Backward-align completed 15-minute features to 5-minute bars.
    No future 15-minute feature is allowed to leak backward.
    """
    if primary_5m_features.empty:
        return primary_5m_features.copy()

    if features_15m.empty:
        result = primary_5m_features.copy()
        for column in features_15m.columns:
            result[column] = np.nan
        return result

    left = primary_5m_features.sort_index().copy()
    right = features_15m.sort_index().copy()

    if left.index.tz is None or right.index.tz is None:
        raise ValueError(
            "Both feature indexes must be timezone-aware."
        )

    left = left.reset_index()
    right = right.reset_index()

    left = left.rename(columns={left.columns[0]: "timestamp"})
    right = right.rename(
        columns={right.columns[0]: "feature_timestamp"}
    )

    merged = pd.merge_asof(
        left.sort_values("timestamp"),
        right.sort_values("feature_timestamp"),
        left_on="timestamp",
        right_on="feature_timestamp",
        direction="backward",
        tolerance=pd.Timedelta(minutes=30),
    )

    merged = merged.set_index("timestamp")
    merged.index.name = "timestamp"

    return merged


# ============================================================
# 20. BUILD THE COMPLETE PRIMARY FEATURE FRAME
# ============================================================

def build_model_feature_frame(
    primary_bars: pd.DataFrame,
    qqq_bars: Optional[pd.DataFrame] = None,
    include_15m: bool = True,
) -> pd.DataFrame:
    """
    Create the primary model feature frame.

    The configured base model uses FEATURE_COLUMNS. Optional 15-minute
    features are appended for diagnostics or for a future model version;
    they are not silently added to the current model's feature list.
    """
    primary = build_features(primary_bars)

    # QQQ is context/confirmation data, not a substitute for SPX.
    if qqq_bars is not None and not qqq_bars.empty:
        qqq_features = build_confirmation_features(
            qqq_bars,
            prefix="qqq_",
        )

        primary = align_confirmation_data(
            primary,
            qqq_features,
            max_age_seconds=600,
        )

    else:
        primary["qqq_ret_3"] = np.nan
        primary["qqq_ema_spread"] = np.nan

    # A separate multi-timeframe feature set is available for analysis.
    if include_15m:
        try:
            higher_timeframe = build_15m_features(primary_bars)
            primary = align_15m_features(
                primary,
                higher_timeframe,
            )
        except ValueError as exc:
            log.warning(
                "15-minute features unavailable: %s",
                exc,
            )

            for column in (
                "m15_ema_spread",
                "m15_ema_trend",
                "m15_rsi_14",
                "m15_atr_pct",
                "m15_ret_3",
            ):
                if column not in primary.columns:
                    primary[column] = np.nan

    # Confirm the model's required columns are present.
    missing = [
        column for column in FEATURE_COLUMNS
        if column not in primary.columns
    ]

    if missing:
        raise ValueError(
            "Required model features missing: "
            + ", ".join(missing)
        )

    return primary.sort_index()


# ============================================================
# 21. TRAINING LABELS — FUTURE DATA ONLY IN THE TARGET
# ============================================================


def add_training_target(
    feature_frame: pd.DataFrame,
    horizon: int = HORIZON,
) -> pd.DataFrame:
    """
    Create a binary future-direction target.

    1 = future close > current close
    0 = future close <= current close
    NaN = target unavailable or price invalid
    """

    if feature_frame is None or feature_frame.empty:
        raise ValueError("Cannot label an empty feature frame.")

    if not isinstance(horizon, int) or horizon < 1:
        raise ValueError("horizon must be a positive integer.")

    if "close" not in feature_frame.columns:
        raise ValueError("Feature frame must contain close.")

    result = feature_frame.copy().sort_index()

    if not result.index.is_monotonic_increasing:
        raise ValueError("Feature timestamps must be chronological.")

    close = pd.to_numeric(result["close"], errors="coerce")
    close = close.where(np.isfinite(close) & (close > 0))

    future_close = close.shift(-horizon)

    valid = close.notna() & future_close.notna()

    target = pd.Series(
        np.nan,
        index=result.index,
        dtype="float64",
        name="target",
    )

    target.loc[valid] = (
        future_close.loc[valid] > close.loc[valid]
    ).astype("int8")

    result["target"] = target

    return result


def prepare_labeled_dataset(
    feature_frame: pd.DataFrame,
    min_rows: int = MIN_TRAIN_ROWS,
) -> pd.DataFrame:
    """
    Return only rows with complete features and known targets.
    The caller must split chronologically before fitting a model.
    """
    if feature_frame is None or feature_frame.empty:
        raise ValueError("Feature frame is empty.")

    missing = [
        column for column in FEATURE_COLUMNS
        if column not in feature_frame.columns
    ]

    if missing:
        raise ValueError(
            "Cannot prepare dataset; missing features: "
            + ", ".join(missing)
        )

    if "target" not in feature_frame.columns:
        raise ValueError(
            "Training target is missing; call add_training_target first."
        )

    dataset = feature_frame[
        FEATURE_COLUMNS + ["target"]
    ].copy()

    dataset = dataset.replace([np.inf, -np.inf], np.nan)
    dataset = dataset.dropna(subset=FEATURE_COLUMNS + ["target"])

    dataset["target"] = dataset["target"].astype(int)

    if not dataset["target"].isin([0, 1]).all():
        raise ValueError("Training target must contain only 0 and 1.")

    if len(dataset) < min_rows:
        raise ValueError(
            f"Insufficient labeled rows: {len(dataset)} < {min_rows}"
        )

    if dataset["target"].nunique() < 2:
        raise ValueError(
            "Training target contains only one class."
        )

    return dataset


# ============================================================
# 22. FEATURE DIAGNOSTICS
# ============================================================

def feature_diagnostics(
    feature_frame: pd.DataFrame,
) -> dict[str, Any]:
    """
    Return basic feature-quality diagnostics without fitting a model.
    """
    if feature_frame is None or feature_frame.empty:
        return {
            "ok": False,
            "reason": "EMPTY_FEATURE_FRAME",
            "rows": 0,
            "missing_percent": {},
        }

    available = [
        column for column in FEATURE_COLUMNS
        if column in feature_frame.columns
    ]

    missing_columns = [
        column for column in FEATURE_COLUMNS
        if column not in feature_frame.columns
    ]

    missing_percent = {}

    for column in available:
        missing_percent[column] = round(
            float(feature_frame[column].isna().mean() * 100.0),
            2,
        )

    complete_rows = (
        feature_frame[available].notna().all(axis=1).sum()
        if available
        else 0
    )

    return {
        "ok": (
            not missing_columns
            and complete_rows >= MIN_TRAIN_ROWS
        ),
        "rows": int(len(feature_frame)),
        "complete_rows": int(complete_rows),
        "required_feature_count": int(MODEL_FEATURE_COUNT),
        "available_feature_count": int(len(available)),
        "missing_columns": missing_columns,
        "missing_percent": missing_percent,
    }


# ============================================================
# END OF PART 2/6
# ============================================================

# ============================================================
# صَفاء التحليل — SPY 0DTE AI ADVISOR v21.4
# PART 3/6
# Chronological Training • ML Ensemble • Holdout Validation
# AUC • Brier Score • Signal Threshold • Honest Model Gate
#
# Paste directly below PART 2/6.
# NO ORDER EXECUTION.
# ============================================================

# ============================================================
# 23. MACHINE-LEARNING IMPORTS
# ============================================================

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline


# ============================================================
# 24. MODEL CONFIGURATION
# ============================================================

MODEL_RANDOM_STATE = _i("MODEL_RANDOM_STATE", 42)

TRAIN_FRACTION = _f("TRAIN_FRACTION", 0.65)
VALIDATION_FRACTION = _f("VALIDATION_FRACTION", 0.20)
HOLDOUT_FRACTION = _f("HOLDOUT_FRACTION", 0.15)

MIN_VALIDATION_ROWS = _i("MIN_VALIDATION_ROWS", 50)
MIN_HOLDOUT_ROWS = _i("MIN_HOLDOUT_ROWS", 50)

MIN_THRESHOLD = _f("MIN_THRESHOLD", 0.50)
MAX_THRESHOLD = _f("MAX_THRESHOLD", 0.90)
THRESHOLD_STEP = _f("THRESHOLD_STEP", 0.01)

MIN_VALIDATION_SIGNAL_N = _i(
    "MIN_VALIDATION_SIGNAL_N", 20
)
MIN_VALIDATION_SIGNAL_ACC = _f(
    "MIN_VALIDATION_SIGNAL_ACC", 0.55
)

# A model must pass all required quality gates before it is enabled.
REQUIRE_HOLDOUT_GATE = _b("REQUIRE_HOLDOUT_GATE", True)


# ============================================================
# 25. VALIDATE MODEL CONFIGURATION
# ============================================================

def validate_model_configuration() -> tuple[bool, str]:
    if not 0.0 < TRAIN_FRACTION < 1.0:
        return False, "TRAIN_FRACTION_OUT_OF_RANGE"

    if not 0.0 < VALIDATION_FRACTION < 1.0:
        return False, "VALIDATION_FRACTION_OUT_OF_RANGE"

    if not 0.0 < HOLDOUT_FRACTION < 1.0:
        return False, "HOLDOUT_FRACTION_OUT_OF_RANGE"

    fraction_total = (
        TRAIN_FRACTION
        + VALIDATION_FRACTION
        + HOLDOUT_FRACTION
    )

    if not math.isclose(
        fraction_total,
        1.0,
        rel_tol=0.0,
        abs_tol=1e-6,
    ):
        return False, "SPLIT_FRACTIONS_MUST_SUM_TO_ONE"

    if MIN_THRESHOLD < 0.5 or MAX_THRESHOLD > 1.0:
        return False, "THRESHOLD_RANGE_INVALID"

    if MIN_THRESHOLD >= MAX_THRESHOLD:
        return False, "THRESHOLD_MIN_MUST_BE_BELOW_MAX"

    if THRESHOLD_STEP <= 0:
        return False, "THRESHOLD_STEP_MUST_BE_POSITIVE"

    if MIN_HOLDOUT_SIGNAL_N < 1:
        return False, "MIN_HOLDOUT_SIGNAL_N_MUST_BE_POSITIVE"

    return True, "CONFIGURATION_OK"


# ============================================================
# 26. CHRONOLOGICAL DATA SPLIT
# ============================================================


def chronological_split(
    dataset: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Chronological split with a purge gap between each segment."""

    if dataset is None or dataset.empty:
        raise ValueError("Cannot split an empty dataset.")

    if not isinstance(dataset.index, pd.DatetimeIndex):
        raise ValueError(
            "Dataset must have a DatetimeIndex for chronological splitting."
        )

    if dataset.index.tz is None:
        raise ValueError(
            "Dataset index must be timezone-aware."
        )

    ordered = dataset.sort_index().copy()

    if ordered.index.has_duplicates:
        raise ValueError(
            "Dataset has duplicate timestamps; resolve before training."
        )

    if not 0 < TRAIN_FRACTION < 1:
        raise ValueError("TRAIN_FRACTION must be between 0 and 1.")

    if not 0 < VALIDATION_FRACTION < 1:
        raise ValueError(
            "VALIDATION_FRACTION must be between 0 and 1."
        )

    if TRAIN_FRACTION + VALIDATION_FRACTION >= 1:
        raise ValueError(
            "TRAIN_FRACTION + VALIDATION_FRACTION must be below 1."
        )

    n_rows = len(ordered)
    train_end = int(n_rows * TRAIN_FRACTION)
    validation_end = train_end + int(
        n_rows * VALIDATION_FRACTION
    )

    purge = max(1, int(HORIZON))

    # Remove boundary rows whose future labels could cross
    # into the next chronological segment.
    train = ordered.iloc[
        :max(0, train_end - purge)
    ].copy()

    validation = ordered.iloc[
        train_end:max(train_end, validation_end - purge)
    ].copy()

    holdout = ordered.iloc[validation_end:].copy()

    if len(train) < MIN_TRAIN_ROWS:
        raise ValueError(
            f"Training split too small after purge: {len(train)} rows."
        )

    if len(validation) < MIN_VALIDATION_ROWS:
        raise ValueError(
            f"Validation split too small after purge: {len(validation)} rows."
        )

    if len(holdout) < MIN_HOLDOUT_ROWS:
        raise ValueError(
            f"Holdout split too small: {len(holdout)} rows."
        )

    for name, split in (
        ("train", train),
        ("validation", validation),
        ("holdout", holdout),
    ):
        if split["target"].nunique() < 2:
            raise ValueError(
                f"{name} split contains only one target class."
            )

    if not (
        train.index.max() < validation.index.min()
        and validation.index.max() < holdout.index.min()
    ):
        raise ValueError(
            "Chronological split ordering failed."
        )

    return train, validation, holdout


# ============================================================
# 27. CREATE THE BASE MODELS
# ============================================================

def create_base_models() -> dict[str, Pipeline]:
    """
    Build two tree-based classifiers.

    Median imputation is fitted inside each training pipeline,
    so missing-value statistics are learned from training data only.
    """
    random_forest = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=350,
                    max_depth=8,
                    min_samples_leaf=8,
                    min_samples_split=16,
                    max_features="sqrt",
                    class_weight="balanced_subsample",
                    random_state=MODEL_RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    extra_trees = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "model",
                ExtraTreesClassifier(
                    n_estimators=350,
                    max_depth=8,
                    min_samples_leaf=8,
                    min_samples_split=16,
                    max_features="sqrt",
                    class_weight="balanced",
                    random_state=MODEL_RANDOM_STATE + 1,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    return {
        "random_forest": random_forest,
        "extra_trees": extra_trees,
    }


# ============================================================
# 28. SAFE PROBABILITY EXTRACTION
# ============================================================

def positive_class_probability(
    model: Any,
    features: pd.DataFrame,
) -> np.ndarray:
    """
    Extract probability for target class 1.

    This explicitly checks class ordering instead of assuming
    probability column 1 always represents the positive class.
    """
    probabilities = np.asarray(
        model.predict_proba(features),
        dtype=float,
    )

    classes = np.asarray(model.classes_)
    positive_indices = np.flatnonzero(classes == 1)

    if len(positive_indices) != 1:
        raise ValueError(
            "Model does not contain exactly one positive class."
        )

    result = probabilities[:, int(positive_indices[0])]

    if not np.isfinite(result).all():
        raise ValueError(
            "Model produced invalid probabilities."
        )

    return np.clip(result, 0.0, 1.0)


# ============================================================
# 29. ENSEMBLE MODEL WRAPPER
# ============================================================

class ProbabilityEnsemble:
    """
    Equal-weight probability ensemble.

    The output is a model probability estimate, not a guarantee
    that a trade will win and not a calibrated expected return.
    """

    def __init__(
        self,
        models: dict[str, Pipeline],
        weights: Optional[dict[str, float]] = None,
    ) -> None:
        if not models:
            raise ValueError("At least one model is required.")

        self.models = models
        self.weights = weights or {
            name: 1.0 for name in models
        }

        total_weight = sum(
            max(0.0, float(self.weights.get(name, 0.0)))
            for name in models
        )

        if total_weight <= 0:
            raise ValueError(
                "Ensemble weights must sum to a positive value."
            )

        self.weights = {
            name: max(
                0.0,
                float(self.weights.get(name, 0.0)),
            ) / total_weight
            for name in models
        }

        self.classes_ = np.array([0, 1])

    def fit(
        self,
        features: pd.DataFrame,
        target: pd.Series,
    ) -> "ProbabilityEnsemble":
        for name, model in self.models.items():
            log.info(
                "Training model %s using %s rows.",
                name,
                len(features),
            )
            model.fit(features, target)

        return self

    def predict_proba(
        self,
        features: pd.DataFrame,
    ) -> np.ndarray:
        weighted_probability = np.zeros(
            len(features),
            dtype=float,
        )

        for name, model in self.models.items():
            probability = positive_class_probability(
                model,
                features,
            )

            weighted_probability += (
                self.weights[name] * probability
            )

        weighted_probability = np.clip(
            weighted_probability,
            0.0,
            1.0,
        )

        return np.column_stack(
            [
                1.0 - weighted_probability,
                weighted_probability,
            ]
        )


# ============================================================
# 30. MODEL METRICS
# ============================================================

def calculate_model_metrics(
    target: pd.Series,
    probabilities: np.ndarray,
    threshold: float = 0.50,
) -> dict[str, Any]:
    """
    Calculate classification and probability-quality metrics.

    AUC evaluates ranking ability. Brier evaluates probability
    error. Neither metric alone proves profitability.
    """
    y_true = np.asarray(target, dtype=int)
    probs = np.asarray(probabilities, dtype=float)

    if len(y_true) != len(probs):
        raise ValueError(
            "Target and probability arrays have different lengths."
        )

    if len(y_true) == 0:
        raise ValueError("Cannot calculate metrics on empty data.")

    if not np.isfinite(probs).all():
        raise ValueError("Probabilities contain invalid values.")

    probs = np.clip(probs, 0.0, 1.0)
    predicted = (probs >= threshold).astype(int)

    if len(np.unique(y_true)) >= 2:
        try:
            auc_value = float(
                roc_auc_score(y_true, probs)
            )
        except Exception:
            auc_value = None
    else:
        auc_value = None

    try:
        brier_value = float(
            brier_score_loss(y_true, probs)
        )
    except Exception:
        brier_value = None

    try:
        accuracy = float(
            accuracy_score(y_true, predicted)
        )
        balanced_accuracy = float(
            balanced_accuracy_score(y_true, predicted)
        )
        precision = float(
            precision_score(
                y_true,
                predicted,
                zero_division=0,
            )
        )
        recall = float(
            recall_score(
                y_true,
                predicted,
                zero_division=0,
            )
        )
    except Exception:
        accuracy = None
        balanced_accuracy = None
        precision = None
        recall = None

    return {
        "n": int(len(y_true)),
        "positive_rate": float(np.mean(y_true)),
        "auc": auc_value,
        "brier": brier_value,
        "accuracy_at_050": accuracy,
        "balanced_accuracy_at_050": balanced_accuracy,
        "precision_at_050": precision,
        "recall_at_050": recall,
    }


# ============================================================
# 31. SELECT THRESHOLD USING VALIDATION ONLY
# ============================================================

def choose_signal_threshold(
    target: pd.Series,
    probabilities: np.ndarray,
) -> tuple[Optional[float], dict[str, Any]]:
    """
    Choose a directional confidence threshold using validation data.

    The holdout set must NOT be used to choose this threshold.
    Both upward and downward confidence are considered.
    """
    y_true = np.asarray(target, dtype=int)
    probs = np.asarray(probabilities, dtype=float)

    if len(y_true) != len(probs) or len(y_true) == 0:
        return None, {
            "reason": "INVALID_VALIDATION_DATA",
        }

    if not np.isfinite(probs).all():
        return None, {
            "reason": "INVALID_VALIDATION_PROBABILITIES",
        }

    confidence = np.maximum(probs, 1.0 - probs)
    predicted_direction = (probs >= 0.5).astype(int)

    candidates = np.arange(
        MIN_THRESHOLD,
        MAX_THRESHOLD + THRESHOLD_STEP / 2.0,
        THRESHOLD_STEP,
    )

    best_threshold: Optional[float] = None
    best_stats: Optional[dict[str, Any]] = None
    best_key: Optional[tuple[float, float, int]] = None

    for candidate in candidates:
        selected = confidence >= candidate
        selected_count = int(selected.sum())

        if selected_count < MIN_VALIDATION_SIGNAL_N:
            continue

        selected_accuracy = float(
            np.mean(
                predicted_direction[selected]
                == y_true[selected]
            )
        )

        if selected_accuracy < MIN_VALIDATION_SIGNAL_ACC:
            continue

        # Prefer higher accuracy, then more qualifying observations.
        # This threshold selection uses validation data only.
        key = (
            selected_accuracy,
            math.log1p(selected_count),
            selected_count,
        )

        if best_key is None or key > best_key:
            best_key = key
            best_threshold = float(candidate)
            best_stats = {
                "threshold": float(candidate),
                "signal_count": selected_count,
                "signal_accuracy": selected_accuracy,
                "validation_rows": int(len(y_true)),
            }

    if best_threshold is None:
        return None, {
            "reason": "NO_VALIDATION_THRESHOLD_PASSED",
            "validation_rows": int(len(y_true)),
            "minimum_signal_count": MIN_VALIDATION_SIGNAL_N,
            "minimum_signal_accuracy": MIN_VALIDATION_SIGNAL_ACC,
        }

    return best_threshold, best_stats or {}


# ============================================================
# 32. EVALUATE THE SELECTED THRESHOLD ON HOLDOUT
# ============================================================

def evaluate_holdout_signals(
    target: pd.Series,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    """
    Evaluate the already-selected validation threshold on untouched
    holdout data. This function must not alter the threshold.
    """
    y_true = np.asarray(target, dtype=int)
    probs = np.asarray(probabilities, dtype=float)

    if len(y_true) != len(probs) or len(y_true) == 0:
        return {
            "passed": False,
            "reason": "INVALID_HOLDOUT_DATA",
            "signal_count": 0,
            "signal_accuracy": None,
        }

    if not np.isfinite(probs).all():
        return {
            "passed": False,
            "reason": "INVALID_HOLDOUT_PROBABILITIES",
            "signal_count": 0,
            "signal_accuracy": None,
        }

    confidence = np.maximum(probs, 1.0 - probs)
    predicted_direction = (probs >= 0.5).astype(int)
    selected = confidence >= threshold

    signal_count = int(selected.sum())

    if signal_count == 0:
        return {
            "passed": False,
            "reason": "NO_HOLDOUT_SIGNALS",
            "signal_count": 0,
            "signal_accuracy": None,
            "threshold": float(threshold),
        }

    signal_accuracy = float(
        np.mean(
            predicted_direction[selected]
            == y_true[selected]
        )
    )

    enough_signals = (
        signal_count >= MIN_HOLDOUT_SIGNAL_N
    )

    accuracy_passed = (
        signal_accuracy >= MIN_HOLDOUT_SIGNAL_ACC
    )

    return {
        "passed": bool(
            enough_signals and accuracy_passed
        ),
        "reason": (
            "HOLDOUT_GATE_PASSED"
            if enough_signals and accuracy_passed
            else "HOLDOUT_GATE_FAILED"
        ),
        "signal_count": signal_count,
        "signal_accuracy": signal_accuracy,
        "minimum_signal_count": MIN_HOLDOUT_SIGNAL_N,
        "minimum_signal_accuracy": MIN_HOLDOUT_SIGNAL_ACC,
        "enough_signals": bool(enough_signals),
        "accuracy_passed": bool(accuracy_passed),
        "threshold": float(threshold),
    }


# ============================================================
# 33. TRAIN AND EVALUATE THE ENSEMBLE
# ============================================================

def train_model(
    dataset: pd.DataFrame,
) -> dict[str, Any]:
    """
    Train the ensemble and run chronological validation.

    Required input:
        A prepared dataset containing FEATURE_COLUMNS + target,
        indexed by timezone-aware timestamps.

    Return dictionary:
        model, enabled, threshold, metrics, reasons, split sizes.

    A model that fails any mandatory quality gate remains disabled.
    """
    result: dict[str, Any] = {
        "model": None,
        "enabled": False,
        "threshold": None,
        "metrics": {},
        "reasons": [],
        "trained_at_utc": now_utc().isoformat(),
        "version": BOT_VERSION,
    }

    config_ok, config_reason = validate_model_configuration()

    if not config_ok:
        result["reasons"].append(config_reason)
        log.error("MODEL DISABLED: %s", config_reason)
        return result

    if dataset is None or dataset.empty:
        result["reasons"].append("EMPTY_DATASET")
        return result

    missing = [
        column for column in FEATURE_COLUMNS + ["target"]
        if column not in dataset.columns
    ]

    if missing:
        result["reasons"].append(
            "MISSING_COLUMNS:" + ",".join(missing)
        )
        return result

    clean = dataset.copy().sort_index()
    clean = clean.replace([np.inf, -np.inf], np.nan)
    clean = clean.dropna(
        subset=FEATURE_COLUMNS + ["target"]
    )

    if len(clean) < MIN_TRAIN_ROWS:
        result["reasons"].append(
            f"INSUFFICIENT_ROWS:{len(clean)}"
        )
        return result

    clean["target"] = clean["target"].astype(int)

    if not clean["target"].isin([0, 1]).all():
        result["reasons"].append("INVALID_TARGET_VALUES")
        return result

    if clean["target"].nunique() != 2:
        result["reasons"].append("ONLY_ONE_TARGET_CLASS")
        return result

    try:
        train, validation, holdout = chronological_split(clean)
    except Exception as exc:
        result["reasons"].append(
            f"CHRONOLOGICAL_SPLIT_FAILED:{exc}"
        )
        log.error("MODEL DISABLED: %s", result["reasons"][-1])
        return result

    result["split_rows"] = {
        "train": int(len(train)),
        "validation": int(len(validation)),
        "holdout": int(len(holdout)),
    }

    x_train = train[FEATURE_COLUMNS]
    y_train = train["target"]

    x_validation = validation[FEATURE_COLUMNS]
    y_validation = validation["target"]

    x_holdout = holdout[FEATURE_COLUMNS]
    y_holdout = holdout["target"]

    # Fit only on the oldest training segment.
    models = create_base_models()
    ensemble = ProbabilityEnsemble(models)

    try:
        ensemble.fit(x_train, y_train)

        validation_probability = positive_class_probability(
            ensemble,
            x_validation,
        )

        validation_metrics = calculate_model_metrics(
            y_validation,
            validation_probability,
        )

    except Exception as exc:
        result["reasons"].append(
            f"TRAIN_OR_VALIDATION_FAILED:{exc}"
        )
        log.exception("Model training failed.")
        return result

    result["metrics"]["validation"] = validation_metrics

    validation_auc = validation_metrics.get("auc")
    validation_brier = validation_metrics.get("brier")

    if validation_auc is None or validation_auc < MIN_AUC:
        result["reasons"].append(
            f"VALIDATION_AUC_BELOW_MIN:{validation_auc}"
        )

    if (
        validation_brier is None
        or validation_brier > MAX_HOLDOUT_BRIER
    ):
        result["reasons"].append(
            f"VALIDATION_BRIER_FAILED:{validation_brier}"
        )

    threshold, threshold_stats = choose_signal_threshold(
        y_validation,
        validation_probability,
    )

    result["threshold_selection"] = threshold_stats

    if threshold is None:
        result["reasons"].append(
            "NO_ACCEPTABLE_VALIDATION_THRESHOLD"
        )

        log.warning(
            "MODEL DISABLED: no threshold passed validation."
        )

        return result

    # --------------------------------------------------------
    # Untouched holdout evaluation
    # --------------------------------------------------------

    try:
        holdout_probability = positive_class_probability(
            ensemble,
            x_holdout,
        )

        holdout_metrics = calculate_model_metrics(
            y_holdout,
            holdout_probability,
            threshold=0.50,
        )

        holdout_signal_metrics = evaluate_holdout_signals(
            y_holdout,
            holdout_probability,
            threshold,
        )

    except Exception as exc:
        result["reasons"].append(
            f"HOLDOUT_EVALUATION_FAILED:{exc}"
        )
        log.exception("Holdout evaluation failed.")
        return result

    result["metrics"]["holdout"] = holdout_metrics
    result["metrics"]["holdout_signals"] = (
        holdout_signal_metrics
    )

    holdout_auc = holdout_metrics.get("auc")
    holdout_brier = holdout_metrics.get("brier")

    if holdout_auc is None or holdout_auc < MIN_AUC:
        result["reasons"].append(
            f"HOLDOUT_AUC_BELOW_MIN:{holdout_auc}"
        )

    if (
        holdout_brier is None
        or holdout_brier > MAX_HOLDOUT_BRIER
    ):
        result["reasons"].append(
            f"HOLDOUT_BRIER_FAILED:{holdout_brier}"
        )

    if REQUIRE_HOLDOUT_GATE and not holdout_signal_metrics.get(
        "passed", False
    ):
        result["reasons"].append(
            "HOLDOUT_SIGNAL_GATE_FAILED"
        )

    result["model"] = ensemble
    result["threshold"] = float(threshold)

    # Do not silently retrain on the holdout after evaluating it:
    # doing so would invalidate the reported holdout assessment.
    result["enabled"] = len(result["reasons"]) == 0

    if result["enabled"]:
        log.info(
            "MODEL ENABLED | threshold=%.2f | "
            "validation AUC=%s | holdout AUC=%s | "
            "holdout signal accuracy=%s | holdout signals=%s",
            threshold,
            validation_auc,
            holdout_auc,
            holdout_signal_metrics.get("signal_accuracy"),
            holdout_signal_metrics.get("signal_count"),
        )
    else:
        log.warning(
            "MODEL DISABLED | reasons=%s",
            " ; ".join(result["reasons"]),
        )

    return result


# ============================================================
# 34. PREDICT WITH AN ENABLED MODEL
# ============================================================

def predict_direction(
    trained: dict[str, Any],
    latest_features: pd.DataFrame,
) -> dict[str, Any]:
    """
    Return direction, probability and confidence for the newest row.

    The caller must still apply market-session, data freshness,
    contract-liquidity, risk and final revalidation checks.
    """
    if not isinstance(trained, dict):
        return {
            "ok": False,
            "reason": "INVALID_MODEL_BUNDLE",
        }

    if not trained.get("enabled", False):
        return {
            "ok": False,
            "reason": "MODEL_NOT_ENABLED",
            "details": trained.get("reasons", []),
        }

    model = trained.get("model")
    threshold = trained.get("threshold")

    if model is None or threshold is None:
        return {
            "ok": False,
            "reason": "MODEL_OR_THRESHOLD_MISSING",
        }

    if latest_features is None or latest_features.empty:
        return {
            "ok": False,
            "reason": "NO_LATEST_FEATURES",
        }

    missing = [
        column for column in FEATURE_COLUMNS
        if column not in latest_features.columns
    ]

    if missing:
        return {
            "ok": False,
            "reason": "MISSING_FEATURES",
            "details": missing,
        }

    latest = latest_features.iloc[[-1]][FEATURE_COLUMNS]
    latest = latest.replace([np.inf, -np.inf], np.nan)

    if latest.isna().any().any():
        return {
            "ok": False,
            "reason": "LATEST_FEATURES_INCOMPLETE",
        }

    try:
        probability_up = float(
            positive_class_probability(model, latest)[0]
        )
    except Exception as exc:
        return {
            "ok": False,
            "reason": f"PREDICTION_FAILED:{exc}",
        }

    probability_down = 1.0 - probability_up
    confidence = max(probability_up, probability_down)

    if confidence < float(threshold):
        return {
            "ok": True,
            "signal": "WAIT",
            "probability_up": probability_up,
            "probability_down": probability_down,
            "confidence": confidence,
            "threshold": float(threshold),
            "reason": "CONFIDENCE_BELOW_THRESHOLD",
        }

    signal = "CALL" if probability_up > probability_down else "PUT"

    return {
        "ok": True,
        "signal": signal,
        "probability_up": probability_up,
        "probability_down": probability_down,
        "confidence": confidence,
        "threshold": float(threshold),
        "reason": "MODEL_THRESHOLD_PASSED",
    }


# ============================================================
# 35. TRAINING PIPELINE ENTRY POINT
# ============================================================

def train_from_feature_frame(
    feature_frame: pd.DataFrame,
) -> dict[str, Any]:
    """
    Build labels and prepare the training dataset from an existing
    feature frame, then run the chronological model pipeline.
    """
    try:
        labeled = add_training_target(
            feature_frame,
            horizon=HORIZON,
        )

        dataset = prepare_labeled_dataset(
            labeled,
            min_rows=MIN_TRAIN_ROWS,
        )

        return train_model(dataset)

    except Exception as exc:
        log.exception("Training pipeline failed.")

        return {
            "model": None,
            "enabled": False,
            "threshold": None,
            "metrics": {},
            "reasons": [f"TRAINING_PIPELINE_FAILED:{exc}"],
            "trained_at_utc": now_utc().isoformat(),
            "version": BOT_VERSION,
        }


# ============================================================
# END OF PART 3/6
# ============================================================

# ============================================================
# صَفاء التحليل — SPY 0DTE AI ADVISOR v21.4
# PART 4/6
# Option Contracts • Live Quotes • Liquidity • Risk Controls
#
# Paste directly below PART 3/6.
# RECOMMENDATIONS ONLY — NO ORDER EXECUTION.
# ============================================================

# ============================================================
# 36. OPTIONS CONFIGURATION
# ============================================================

OPTIONS_DATA_URL = _s(
    "OPTIONS_DATA_URL",
    "https://data.alpaca.markets",
).rstrip("/")

OPTIONS_CONTRACTS_URL = _s(
    "OPTIONS_CONTRACTS_URL",
    f"{ALPACA_TRADING_URL}/v2/options/contracts",
).rstrip("/")

OPTIONS_FEED = _s("OPTIONS_FEED", "indicative").lower()

OPTION_EXPIRY_MODE = _s(
    "OPTION_EXPIRY_MODE", "same_day"
).lower()

OPTION_TYPE = _s("OPTION_TYPE", "both").lower()

OPTION_CHAIN_LIMIT = _i("OPTION_CHAIN_LIMIT", 1000)
MAX_OPTION_PAGES = _i("MAX_OPTION_PAGES", 10)

MIN_TIME_TO_EXPIRY_MINUTES = _i(
    "MIN_TIME_TO_EXPIRY_MINUTES", 5
)

MAX_TIME_TO_EXPIRY_MINUTES = _i(
    "MAX_TIME_TO_EXPIRY_MINUTES", 390
)

# Optional delta data can be unavailable depending on feed/endpoint.
# If delta is required but unavailable, the contract is rejected.
REQUIRE_DELTA = _b("REQUIRE_DELTA", False)

# When delta is available, require it to fall inside the configured band.
MIN_ABS_DELTA = _f("MIN_ABS_DELTA", MIN_DELTA)
MAX_ABS_DELTA = _f("MAX_ABS_DELTA", MAX_DELTA)

# Additional option-specific filters.
MIN_OPTION_BID = _f("MIN_OPTION_BID", 0.01)
MIN_OPTION_ASK = _f("MIN_OPTION_ASK", MIN_OPTION_PREMIUM)
MAX_OPTION_ASK = _f("MAX_OPTION_ASK", MAX_OPTION_PREMIUM)

# A conservative estimate of entry cost.
# One standard equity-option contract represents 100 shares.
OPTION_MULTIPLIER = _i("OPTION_MULTIPLIER", 100)

# A signal may be generated only if the account equity is configured.
REQUIRE_ACCOUNT_EQUITY = _b("REQUIRE_ACCOUNT_EQUITY", True)


# ============================================================
# 37. OPTION API HEADERS
# ============================================================

def options_headers() -> dict[str, str]:
    """Return authenticated headers for Alpaca option data."""
    headers = alpaca_headers()
    headers["Accept"] = "application/json"
    return headers


def options_get(
    url: str,
    params: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """
    GET an options endpoint with bounded retries.
    No order-submission endpoint is called by this module.
    """
    headers = options_headers()
    last_error: Optional[Exception] = None

    for attempt in range(1, HTTP_RETRIES + 1):
        try:
            response = HTTP.get(
                url,
                headers=headers,
                params=params,
                timeout=REQUEST_TIMEOUT_SECONDS,
            )

            if response.status_code in (401, 403):
                raise RuntimeError(
                    "Alpaca options permission/authentication error "
                    f"{response.status_code}: {response.text[:300]}"
                )

            response.raise_for_status()
            payload = response.json()

            if not isinstance(payload, dict):
                raise RuntimeError(
                    "Unexpected options API response format."
                )

            return payload

        except Exception as exc:
            last_error = exc

            if attempt < HTTP_RETRIES:
                time.sleep(
                    HTTP_RETRY_SLEEP_SECONDS * attempt
                )

    raise RuntimeError(
        f"Options GET failed after {HTTP_RETRIES} attempts: "
        f"{last_error}"
    )


# ============================================================
# 38. CONTRACT FIELD NORMALIZATION
# ============================================================

def normalize_option_type(value: Any) -> Optional[str]:
    """Normalize option type to CALL or PUT."""
    if value is None:
        return None

    text_value = str(value).strip().lower()

    if text_value in {"call", "c"}:
        return "CALL"

    if text_value in {"put", "p"}:
        return "PUT"

    return None


def parse_expiration_date(value: Any) -> Optional[date]:
    """Parse an option expiration date."""
    if value is None:
        return None

    try:
        return pd.Timestamp(value).date()
    except Exception:
        return None


def normalize_contract(
    raw: dict[str, Any],
) -> Optional[dict[str, Any]]:
    """
    Normalize an Alpaca option-contract object.

    Returns None if required contract information is missing.
    """
    if not isinstance(raw, dict):
        return None

    symbol = str(raw.get("symbol", "")).strip().upper()
    underlying = str(
        raw.get("underlying_symbol", "")
    ).strip().upper()

    option_type = normalize_option_type(
        raw.get("type", raw.get("right"))
    )

    strike = safe_positive_float(
        raw.get("strike_price", raw.get("strike"))
    )

    expiration = parse_expiration_date(
        raw.get("expiration_date", raw.get("expiration"))
    )

    if not symbol or not option_type:
        return None

    if strike is None or expiration is None:
        return None

    return {
        "symbol": symbol,
        "underlying_symbol": underlying,
        "type": option_type,
        "strike": float(strike),
        "expiration_date": expiration,
        "tradable": raw.get("tradable"),
        "raw": raw,
    }


# ============================================================
# 39. FETCH OPTION CONTRACTS
# ============================================================

def fetch_option_contracts(
    underlying: str = PRIMARY_UNDERLYING,
    expiration: Optional[date] = None,
    option_type: str = "both",
    limit: int = OPTION_CHAIN_LIMIT,
) -> list[dict[str, Any]]:
    """
    Fetch option contracts from Alpaca's contracts endpoint.

    This retrieves contract definitions, not executable orders.
    A contract's presence in the chain does not prove that it has
    a current, liquid or tradable quote.
    """
    underlying = underlying.strip().upper()

    if not underlying:
        raise ValueError("Underlying symbol cannot be empty.")

    if option_type.lower() not in {"both", "call", "put"}:
        raise ValueError(
            "option_type must be 'both', 'call' or 'put'."
        )

    params: dict[str, Any] = {
        "underlying_symbols": underlying,
        "status": "active",
        "limit": min(max(int(limit), 1), 1000),
    }

    if expiration is not None:
        params["expiration_date"] = expiration.isoformat()

    if option_type.lower() in {"call", "put"}:
        params["type"] = option_type.lower()

    collected: list[dict[str, Any]] = []
    page_token: Optional[str] = None

    for _ in range(MAX_OPTION_PAGES):
        if page_token:
            params["page_token"] = page_token
        else:
            params.pop("page_token", None)

        payload = options_get(
            OPTIONS_CONTRACTS_URL,
            params=params,
        )

        contracts = payload.get(
            "option_contracts",
            payload.get("contracts", []),
        )

        if not isinstance(contracts, list):
            raise RuntimeError(
                "Unexpected option-contract list format."
            )

        collected.extend(contracts)

        page_token = payload.get("next_page_token")
        if not page_token:
            break

    normalized: list[dict[str, Any]] = []

    for raw in collected:
        contract = normalize_contract(raw)

        if contract is None:
            continue

        if (
            contract["underlying_symbol"]
            and contract["underlying_symbol"] != underlying
        ):
            continue

        if option_type.lower() == "call":
            if contract["type"] != "CALL":
                continue

        elif option_type.lower() == "put":
            if contract["type"] != "PUT":
                continue

        normalized.append(contract)

    log.info(
        "OPTION CHAIN %s: %s normalized contracts",
        underlying,
        len(normalized),
    )

    return normalized


# ============================================================
# 40. LATEST OPTION QUOTES
# ============================================================

def fetch_latest_option_quotes(
    option_symbols: list[str],
) -> dict[str, dict[str, Any]]:
    """
    Fetch latest option quotes from Alpaca.

    The quote endpoint and available feed depend on the account's
    options-data permissions. API failure is treated as unavailable
    data, never as permission to fabricate a price.
    """
    symbols = sorted({
        str(symbol).strip().upper()
        for symbol in option_symbols
        if str(symbol).strip()
    })

    if not symbols:
        return {}

    quotes: dict[str, dict[str, Any]] = {}

    # Batch requests to avoid excessively long URLs.
    batch_size = 50

    for start in range(0, len(symbols), batch_size):
        batch = symbols[start:start + batch_size]

        url = f"{OPTIONS_DATA_URL}/v1beta1/options/quotes/latest"

        params = {
            "symbols": ",".join(batch),
            "feed": OPTIONS_FEED,
        }

        payload = options_get(url, params=params)

        quote_map = payload.get("quotes", {})

        if not isinstance(quote_map, dict):
            log.warning(
                "Unexpected quote-map format for option batch."
            )
            continue

        for symbol, quote in quote_map.items():
            if isinstance(quote, dict):
                quotes[str(symbol).upper()] = quote

    return quotes


# ============================================================
# 41. QUOTE TIMESTAMP / QUALITY
# ============================================================

def quote_timestamp(
    quote: dict[str, Any],
) -> Optional[datetime]:
    """Extract a quote timestamp and normalize it to UTC."""
    if not isinstance(quote, dict):
        return None

    for key in ("t", "timestamp", "time"):
        if quote.get(key) is not None:
            timestamp = to_utc(quote[key])
            if timestamp is not None:
                return timestamp

    return None


def quote_age_seconds(
    quote: dict[str, Any],
    current_time: Optional[datetime] = None,
) -> Optional[float]:
    timestamp = quote_timestamp(quote)
    current = to_utc(current_time or now_utc())

    if timestamp is None or current is None:
        return None

    age = (current - timestamp).total_seconds()

    # A materially future-dated quote is invalid.
    if age < -2.0:
        return None

    return max(0.0, age)


def extract_quote_prices(
    quote: dict[str, Any],
) -> tuple[Optional[float], Optional[float]]:
    """Extract bid and ask prices from an Alpaca quote."""
    if not isinstance(quote, dict):
        return None, None

    bid = safe_positive_float(
        quote.get("bp", quote.get("bid_price"))
    )

    ask = safe_positive_float(
        quote.get("ap", quote.get("ask_price"))
    )

    return bid, ask


def validate_option_quote(
    quote: dict[str, Any],
    current_time: Optional[datetime] = None,
) -> tuple[bool, str, dict[str, Any]]:
    """
    Validate quote freshness, bid/ask ordering and spread.

    The returned metrics are diagnostics only; a valid quote does
    not guarantee that the displayed price can be filled.
    """
    if not isinstance(quote, dict):
        return False, "QUOTE_MISSING", {}

    bid, ask = extract_quote_prices(quote)

    if bid is None or ask is None:
        return False, "BID_OR_ASK_MISSING", {}

    if bid < MIN_OPTION_BID:
        return False, "BID_BELOW_MINIMUM", {
            "bid": bid,
            "ask": ask,
        }

    if ask < MIN_OPTION_ASK:
        return False, "ASK_BELOW_MINIMUM", {
            "bid": bid,
            "ask": ask,
        }

    if ask > MAX_OPTION_ASK:
        return False, "ASK_ABOVE_MAXIMUM", {
            "bid": bid,
            "ask": ask,
        }

    if ask < bid:
        return False, "CROSSED_MARKET", {
            "bid": bid,
            "ask": ask,
        }

    age = quote_age_seconds(
        quote,
        current_time=current_time,
    )

    if age is None:
        return False, "QUOTE_TIMESTAMP_INVALID", {
            "bid": bid,
            "ask": ask,
        }

    if age > MAX_QUOTE_AGE_SECONDS:
        return False, "QUOTE_STALE", {
            "bid": bid,
            "ask": ask,
            "age_seconds": age,
        }

    spread = calculate_spread_percent(bid, ask)

    if spread is None:
        return False, "SPREAD_INVALID", {
            "bid": bid,
            "ask": ask,
        }

    if spread > MAX_SPREAD_PERCENT:
        return False, "SPREAD_TOO_WIDE", {
            "bid": bid,
            "ask": ask,
            "spread_percent": spread,
            "age_seconds": age,
        }

    midpoint = (bid + ask) / 2.0

    return True, "QUOTE_OK", {
        "bid": bid,
        "ask": ask,
        "midpoint": midpoint,
        "spread_percent": spread,
        "age_seconds": age,
        "quote_timestamp": quote_timestamp(quote),
    }


# ============================================================
# 42. OPTION EXPIRY FILTERS
# ============================================================

def minutes_until_expiry(
    expiration: date,
    current_time: Optional[datetime] = None,
) -> Optional[float]:
    """
    Approximate minutes remaining until the regular-session close
    on the expiration date.

    This is a conservative session-time estimate, not a full model
    of every product's settlement and exercise rules.
    """
    current = to_ny(current_time or now_ny())

    if current is None:
        return None

    if expiration < current.date():
        return -1.0

    _, session_end = regular_session_bounds(expiration)

    if expiration > current.date():
        return (
            (session_end - current).total_seconds() / 60.0
        )

    return (
        (session_end - current).total_seconds() / 60.0
    )


def contract_expiry_is_acceptable(
    contract: dict[str, Any],
    current_time: Optional[datetime] = None,
) -> tuple[bool, str, Optional[float]]:
    expiration = contract.get("expiration_date")

    if not isinstance(expiration, date):
        return False, "INVALID_EXPIRATION", None

    current = to_ny(current_time or now_ny())
    if current is None:
        return False, "INVALID_CURRENT_TIME", None

    if OPTION_EXPIRY_MODE == "same_day":
        if expiration != current.date():
            return False, "NOT_SAME_DAY_EXPIRY", None

    elif OPTION_EXPIRY_MODE == "any_active":
        if expiration < current.date():
            return False, "EXPIRED_CONTRACT", None

    else:
        return False, "INVALID_EXPIRY_MODE", None

    remaining = minutes_until_expiry(
        expiration,
        current_time=current,
    )

    if remaining is None:
        return False, "EXPIRY_TIME_UNAVAILABLE", None

    if remaining <= MIN_TIME_TO_EXPIRY_MINUTES:
        return False, "TOO_CLOSE_TO_EXPIRY", remaining

    if remaining > MAX_TIME_TO_EXPIRY_MINUTES:
        return False, "OUTSIDE_EXPIRY_WINDOW", remaining

    return True, "EXPIRY_OK", remaining


# ============================================================
# 43. OPTIONAL DELTA EXTRACTION
# ============================================================

def extract_delta(
    contract: dict[str, Any],
    quote: Optional[dict[str, Any]] = None,
) -> Optional[float]:
    """
    Extract delta only if the API response actually supplies it.

    Most ordinary quote objects do not contain Greeks. This function
    does not invent a delta from strike distance or underlying price.
    """
    sources = [
        contract.get("raw", {}),
        quote or {},
    ]

    for source in sources:
        if not isinstance(source, dict):
            continue

        if source.get("delta") is not None:
            delta = safe_float(source.get("delta"))
            if delta is not None and -1.0 <= delta <= 1.0:
                return delta

        greeks = source.get("greeks")
        if isinstance(greeks, dict) and greeks.get("delta") is not None:
            delta = safe_float(greeks.get("delta"))
            if delta is not None and -1.0 <= delta <= 1.0:
                return delta

    return None


def delta_is_acceptable(
    contract: dict[str, Any],
    quote: Optional[dict[str, Any]] = None,
) -> tuple[bool, str, Optional[float]]:
    delta = extract_delta(contract, quote)

    if delta is None:
        if REQUIRE_DELTA:
            return False, "DELTA_REQUIRED_BUT_UNAVAILABLE", None
        return True, "DELTA_UNAVAILABLE_NOT_FILTERED", None

    abs_delta = abs(delta)

    if abs_delta < MIN_ABS_DELTA:
        return False, "DELTA_BELOW_MINIMUM", delta

    if abs_delta > MAX_ABS_DELTA:
        return False, "DELTA_ABOVE_MAXIMUM", delta

    return True, "DELTA_OK", delta


# ============================================================
# 44. CONTRACT CANDIDATE FILTER
# ============================================================

def evaluate_contract_candidate(
    contract: dict[str, Any],
    quote: Optional[dict[str, Any]],
    underlying_price: float,
    expected_direction: str,
    current_time: Optional[datetime] = None,
) -> dict[str, Any]:
    """
    Evaluate one contract against the direction, strike distance,
    expiry, quote and optional delta filters.
    """
    result: dict[str, Any] = {
        "accepted": False,
        "reason": "NOT_EVALUATED",
        "contract": contract,
    }

    if not isinstance(contract, dict):
        result["reason"] = "INVALID_CONTRACT"
        return result

    direction = expected_direction.strip().upper()

    if direction not in {"CALL", "PUT"}:
        result["reason"] = "INVALID_DIRECTION"
        return result

    if contract.get("type") != direction:
        result["reason"] = "DIRECTION_MISMATCH"
        return result

    underlying = safe_positive_float(underlying_price)

    if underlying is None:
        result["reason"] = "INVALID_UNDERLYING_PRICE"
        return result

    strike = safe_positive_float(contract.get("strike"))

    if strike is None:
        result["reason"] = "INVALID_STRIKE"
        return result

    strike_distance = abs(strike - underlying)

    if strike_distance > MAX_STRIKE_DISTANCE:
        result["reason"] = "STRIKE_TOO_FAR"
        result["strike_distance"] = strike_distance
        return result

    expiry_ok, expiry_reason, remaining = (
        contract_expiry_is_acceptable(
            contract,
            current_time=current_time,
        )
    )

    if not expiry_ok:
        result["reason"] = expiry_reason
        result["minutes_to_expiry"] = remaining
        return result

    quote_ok, quote_reason, quote_metrics = validate_option_quote(
        quote or {},
        current_time=current_time,
    )

    if not quote_ok:
        result["reason"] = quote_reason
        result.update(quote_metrics)
        return result

    delta_ok, delta_reason, delta = delta_is_acceptable(
        contract,
        quote,
    )

    if not delta_ok:
        result["reason"] = delta_reason
        result["delta"] = delta
        return result

    result.update({
        "accepted": True,
        "reason": "CONTRACT_OK",
        "symbol": contract["symbol"],
        "type": contract["type"],
        "strike": strike,
        "expiration_date": contract["expiration_date"].isoformat(),
        "strike_distance": strike_distance,
        "minutes_to_expiry": remaining,
        "delta": delta,
        **quote_metrics,
    })

    return result


# ============================================================
# 45. CONTRACT SCORING
# ============================================================

def score_contract_candidate(
    candidate: dict[str, Any],
) -> float:
    """
    Rank already-valid contracts.

    This is a deterministic ranking heuristic, not an estimate
    of option profitability or probability of filling.
    """
    if not candidate.get("accepted", False):
        return -1.0

    spread = safe_float(candidate.get("spread_percent"))
    distance = safe_float(candidate.get("strike_distance"))
    midpoint = safe_positive_float(candidate.get("midpoint"))

    if spread is None or distance is None or midpoint is None:
        return -1.0

    # Prefer tighter spreads and strikes nearer the underlying.
    # Premium is not rewarded without limit.
    spread_score = max(0.0, 1.0 - spread / MAX_SPREAD_PERCENT)
    distance_score = max(
        0.0,
        1.0 - distance / max(MAX_STRIKE_DISTANCE, 0.01),
    )

    # Avoid favoring expensive contracts merely because they cost more.
    premium_score = 1.0 / (1.0 + midpoint)

    delta = candidate.get("delta")
    if delta is None:
        delta_score = 0.5
    else:
        abs_delta = abs(float(delta))
        midpoint_delta = (
            MIN_ABS_DELTA + MAX_ABS_DELTA
        ) / 2.0
        delta_score = max(
            0.0,
            1.0 - abs(abs_delta - midpoint_delta),
        )

    return (
        0.40 * spread_score
        + 0.25 * distance_score
        + 0.15 * premium_score
        + 0.20 * delta_score
    )


def select_best_contract(
    contracts: list[dict[str, Any]],
    quotes: dict[str, dict[str, Any]],
    underlying_price: float,
    expected_direction: str,
    current_time: Optional[datetime] = None,
) -> dict[str, Any]:
    """
    Evaluate the chain and return the highest-ranked valid contract.
    Returns a structured rejection if no contract passes.
    """
    candidates: list[dict[str, Any]] = []
    rejection_counts: dict[str, int] = {}

    for contract in contracts:
        symbol = contract.get("symbol", "")
        quote = quotes.get(symbol)

        candidate = evaluate_contract_candidate(
            contract=contract,
            quote=quote,
            underlying_price=underlying_price,
            expected_direction=expected_direction,
            current_time=current_time,
        )

        if not candidate.get("accepted", False):
            reason = str(candidate.get("reason", "UNKNOWN"))
            rejection_counts[reason] = (
                rejection_counts.get(reason, 0) + 1
            )
            continue

        candidate["rank_score"] = score_contract_candidate(
            candidate
        )
        candidates.append(candidate)

    if not candidates:
        return {
            "ok": False,
            "reason": "NO_CONTRACT_PASSED_FILTERS",
            "rejection_counts": rejection_counts,
        }

    candidates.sort(
        key=lambda item: item.get("rank_score", -1.0),
        reverse=True,
    )

    best = candidates[0]

    return {
        "ok": True,
        "reason": "BEST_CONTRACT_SELECTED",
        "contract": best,
        "candidate_count": len(candidates),
        "rejection_counts": rejection_counts,
    }


# ============================================================
# 46. RISK BUDGET AND CONTRACT SIZING
# ============================================================

def calculate_risk_budget(
    account_equity: float = ACCOUNT_EQUITY,
    risk_pct: float = RISK_PCT,
) -> tuple[bool, str, float]:
    """
    Convert a percentage risk setting into a monetary risk budget.

    This is a budget cap, not a forecast of maximum loss.
    """
    equity = safe_positive_float(account_equity)

    if equity is None:
        return False, "ACCOUNT_EQUITY_NOT_CONFIGURED", 0.0

    pct = safe_float(risk_pct)

    if pct is None or pct <= 0 or pct > 100:
        return False, "INVALID_RISK_PERCENT", 0.0

    risk_budget = equity * (pct / 100.0)

    if risk_budget <= 0:
        return False, "RISK_BUDGET_ZERO", 0.0

    return True, "RISK_BUDGET_OK", risk_budget


def calculate_contract_quantity(
    ask_price: float,
    account_equity: float = ACCOUNT_EQUITY,
    risk_pct: float = RISK_PCT,
    max_contracts: int = MAX_CONTRACTS,
    slippage_per_contract: float = ESTIMATED_SLIPPAGE_PER_CONTRACT,
    fees_per_contract: float = ESTIMATED_FEES_PER_CONTRACT,
) -> dict[str, Any]:
    """
    Estimate a contract quantity using the full premium cost plus
    estimated slippage and fees.

    For long options, the premium paid can be lost in full.
    This function deliberately budgets against estimated total entry
    cost rather than pretending a stop loss is guaranteed to fill.

    It does not place an order.
    """
    ask = safe_positive_float(ask_price)

    if ask is None:
        return {
            "ok": False,
            "reason": "INVALID_ASK_PRICE",
            "contracts": 0,
        }

    if REQUIRE_ACCOUNT_EQUITY and account_equity <= 0:
        return {
            "ok": False,
            "reason": "ACCOUNT_EQUITY_NOT_CONFIGURED",
            "contracts": 0,
        }

    budget_ok, budget_reason, risk_budget = calculate_risk_budget(
        account_equity=account_equity,
        risk_pct=risk_pct,
    )

    if not budget_ok:
        return {
            "ok": False,
            "reason": budget_reason,
            "contracts": 0,
        }

    slippage = safe_float(slippage_per_contract)
    fees = safe_float(fees_per_contract)

    if slippage is None or slippage < 0:
        return {
            "ok": False,
            "reason": "INVALID_SLIPPAGE_ESTIMATE",
            "contracts": 0,
        }

    if fees is None or fees < 0:
        return {
            "ok": False,
            "reason": "INVALID_FEE_ESTIMATE",
            "contracts": 0,
        }

    premium_cost = ask * OPTION_MULTIPLIER

    estimated_cost_per_contract = (
        premium_cost
        + slippage * OPTION_MULTIPLIER
        + fees
    )

    if estimated_cost_per_contract <= 0:
        return {
            "ok": False,
            "reason": "INVALID_ESTIMATED_COST",
            "contracts": 0,
        }

    quantity_from_budget = int(
        math.floor(risk_budget / estimated_cost_per_contract)
    )

    quantity = min(
        quantity_from_budget,
        max(0, int(max_contracts)),
    )

    if quantity < 1:
        return {
            "ok": False,
            "reason": "RISK_BUDGET_TOO_SMALL",
            "contracts": 0,
            "risk_budget": risk_budget,
            "estimated_cost_per_contract": estimated_cost_per_contract,
        }

    total_estimated_cost = (
        quantity * estimated_cost_per_contract
    )

    return {
        "ok": True,
        "reason": "SIZE_OK",
        "contracts": quantity,
        "risk_budget": risk_budget,
        "premium_per_contract": premium_cost,
        "estimated_cost_per_contract": estimated_cost_per_contract,
        "estimated_total_cost": total_estimated_cost,
        "estimated_budget_remaining": (
            risk_budget - total_estimated_cost
        ),
        "maximum_premium_loss_before_fees": (
            quantity * premium_cost
        ),
        "note": (
            "Estimate only; actual fills, fees and slippage may differ."
        ),
    }


# ============================================================
# 47. DAILY RISK GUARD
# ============================================================

def daily_risk_guard(
    realized_daily_pnl: float,
    trades_today: int,
    open_positions: int,
) -> tuple[bool, str]:
    """
    Check externally tracked daily limits.

    This bot does not execute trades, so P&L, trade count and open
    positions must come from a trustworthy journal or account source.
    Do not pass invented values.
    """
    pnl = safe_float(realized_daily_pnl)

    if pnl is None:
        return False, "DAILY_PNL_UNAVAILABLE"

    if pnl <= -abs(MAX_DAILY_LOSS):
        return False, "MAX_DAILY_LOSS_REACHED"

    if int(trades_today) >= MAX_TRADES_PER_DAY:
        return False, "MAX_TRADES_PER_DAY_REACHED"

    if int(open_positions) >= MAX_OPEN_POSITIONS:
        return False, "MAX_OPEN_POSITIONS_REACHED"

    return True, "DAILY_RISK_OK"


# ============================================================
# 48. COMPLETE CONTRACT SELECTION WORKFLOW
# ============================================================

def find_option_recommendation_candidate(
    underlying_price: float = 0.0,
    expected_direction: str = "CALL",
    underlying: str = PRIMARY_UNDERLYING,
    current_time: Optional[datetime] = None,
    *args,
    **kwargs,
) -> dict[str, Any]:
    """
    Fetch contracts and quotes, apply filters, rank candidates,
    then calculate an estimated affordable contract quantity.

    This function returns a recommendation candidate only.
    It never submits an order.
    """
    # التعامل مع أي وسيط إضافي قد يمرر عبر args أو kwargs
    if args:
        if len(args) > 0 and not underlying_price:
            underlying_price = args[0]
        if len(args) > 1 and not expected_direction:
            expected_direction = args[1]

    if kwargs.get("direction") and not expected_direction:
        expected_direction = kwargs.get("direction")

    direction = str(expected_direction).strip().upper()

    if direction not in {"CALL", "PUT"}:
        return {
            "ok": False,
            "reason": "INVALID_DIRECTION",
        }

    current = to_ny(current_time or now_ny())

    if current is None:
        return {
            "ok": False,
            "reason": "INVALID_CURRENT_TIME",
        }

    session_ok, session_reason = session_window_status(current)

    if not session_ok:
        return {
            "ok": False,
            "reason": "SESSION_BLOCKED",
            "details": session_reason,
        }

    underlying_value = safe_positive_float(underlying_price)

    if underlying_value is None or underlying_value <= 0:
        return {
            "ok": False,
            "reason": "INVALID_UNDERLYING_PRICE",
        }

    try:
        contracts = fetch_option_contracts(
            underlying=underlying,
            expiration=current.date(),
            option_type=direction.lower(),
        )
    except Exception as exc:
        log.exception("Option contract fetch failed.")
        return {
            "ok": False,
            "reason": f"CONTRACT_FETCH_FAILED:{exc}",
        }

    if not contracts:
        return {
            "ok": False,
            "reason": "NO_ACTIVE_CONTRACTS",
            "underlying": underlying,
            "expiration": current.date().isoformat(),
        }

    symbols = [
        contract["symbol"]
        for contract in contracts
        if contract.get("symbol")
    ]

    try:
        quotes = fetch_latest_option_quotes(symbols)
    except Exception as exc:
        log.exception("Option quote fetch failed.")
        return {
            "ok": False,
            "reason": f"QUOTE_FETCH_FAILED:{exc}",
        }

    selection = select_best_contract(
        contracts=contracts,
        quotes=quotes,
        underlying_price=underlying_value,
        expected_direction=direction,
        current_time=current,
    )

    if not selection.get("ok"):
        return {
            "ok": False,
            "reason": selection.get("reason"),
            "underlying": underlying,
            "direction": direction,
            "details": selection,
        }

    candidate = selection["contract"]

    sizing = calculate_contract_quantity(
        ask_price=candidate["ask"],
    )

    if not sizing.get("ok"):
        return {
            "ok": False,
            "reason": sizing.get("reason"),
            "underlying": underlying,
            "direction": direction,
            "candidate": candidate,
            "sizing": sizing,
        }

    return {
        "ok": True,
        "reason": "RECOMMENDATION_CANDIDATE_READY",
        "underlying": underlying,
        "underlying_price": underlying_value,
        "direction": direction,
        "candidate": candidate,
        "sizing": sizing,
        "candidate_count": selection.get("candidate_count", 0),
        "rejection_counts": selection.get(
            "rejection_counts", {}
        ),
        "generated_at_utc": now_utc().isoformat(),
        "mode": "RECOMMENDATION_ONLY",
    }

# ============================================================

# ============================================================
# 49. JOURNAL CONFIGURATION
# ============================================================

JOURNAL_VERSION = "21.4"

JOURNAL_DIR = Path(
    _s("JOURNAL_DIR", "journal")
)

SIGNAL_JOURNAL_FILE = JOURNAL_DIR / "signals.csv"
QUOTE_JOURNAL_FILE = JOURNAL_DIR / "quotes.csv"
OUTCOME_JOURNAL_FILE = JOURNAL_DIR / "outcomes.csv"
MODEL_JOURNAL_FILE = JOURNAL_DIR / "model_runs.csv"

JOURNAL_FLUSH_EVERY_WRITE = _b(
    "JOURNAL_FLUSH_EVERY_WRITE", True
)

# Track price observations for a signal after the alert is generated.
OBSERVATION_INTERVAL_SECONDS = _i(
    "OBSERVATION_INTERVAL_SECONDS", 30
)

OBSERVATION_HORIZON_MINUTES = _i(
    "OBSERVATION_HORIZON_MINUTES", 5
)

MAX_OBSERVATION_AGE_SECONDS = _i(
    "MAX_OBSERVATION_AGE_SECONDS", 60
)

# Never claim a paper outcome when the required quotes are missing.
MIN_OBSERVATIONS_FOR_OUTCOME = _i(
    "MIN_OBSERVATIONS_FOR_OUTCOME", 2
)


# ============================================================
# 50. JOURNAL SCHEMAS
# ============================================================

SIGNAL_COLUMNS = [
    "journal_version",
    "signal_id",
    "created_at_utc",
    "created_at_ny",
    "underlying",
    "underlying_price",
    "direction",
    "probability_up",
    "probability_down",
    "confidence",
    "model_threshold",
    "model_enabled",
    "option_symbol",
    "option_type",
    "strike",
    "expiration_date",
    "bid",
    "ask",
    "midpoint",
    "spread_percent",
    "quote_timestamp_utc",
    "quote_age_seconds",
    "delta",
    "estimated_contracts",
    "estimated_cost_per_contract",
    "estimated_total_cost",
    "reason",
    "status",
]

QUOTE_COLUMNS = [
    "journal_version",
    "signal_id",
    "observed_at_utc",
    "quote_timestamp_utc",
    "quote_age_seconds",
    "option_symbol",
    "bid",
    "ask",
    "midpoint",
    "spread_percent",
    "underlying_price",
    "observation_number",
    "observation_status",
]

OUTCOME_COLUMNS = [
    "journal_version",
    "signal_id",
    "created_at_utc",
    "evaluated_at_utc",
    "option_symbol",
    "entry_reference_ask",
    "entry_reference_bid",
    "entry_reference_midpoint",
    "final_bid",
    "final_ask",
    "final_midpoint",
    "observations_seen",
    "elapsed_seconds",
    "midpoint_change_percent",
    "bid_to_ask_change_percent",
    "outcome_status",
    "is_actual_execution",
    "notes",
]

MODEL_COLUMNS = [
    "journal_version",
    "trained_at_utc",
    "enabled",
    "threshold",
    "train_rows",
    "validation_rows",
    "holdout_rows",
    "validation_auc",
    "validation_brier",
    "holdout_auc",
    "holdout_brier",
    "holdout_signal_count",
    "holdout_signal_accuracy",
    "reasons",
]


# ============================================================
# 51. JOURNAL INITIALIZATION
# ============================================================

def initialize_journal() -> None:
    """
    Create journal directory and CSV headers when absent.
    Existing journals are not overwritten.
    """
    JOURNAL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    schemas = {
        SIGNAL_JOURNAL_FILE: SIGNAL_COLUMNS,
        QUOTE_JOURNAL_FILE: QUOTE_COLUMNS,
        OUTCOME_JOURNAL_FILE: OUTCOME_COLUMNS,
        MODEL_JOURNAL_FILE: MODEL_COLUMNS,
    }

    for path, columns in schemas.items():
        if not path.exists():
            pd.DataFrame(
                columns=columns
            ).to_csv(
                path,
                index=False,
                encoding="utf-8",
            )


# ============================================================
# 52. SAFE JOURNAL APPEND
# ============================================================

def append_journal_row(
    path: Path,
    columns: list[str],
    row: dict[str, Any],
) -> tuple[bool, str]:
    """
    Append one row without replacing prior history.

    Unknown fields are ignored; missing schema fields are blank.
    """
    try:
        initialize_journal()

        normalized = {
            column: row.get(column)
            for column in columns
        }

        frame = pd.DataFrame(
            [normalized],
            columns=columns,
        )

        frame.to_csv(
            path,
            mode="a",
            header=not path.exists() or path.stat().st_size == 0,
            index=False,
            encoding="utf-8",
        )

        if JOURNAL_FLUSH_EVERY_WRITE:
            # File writes are synchronous; no additional buffer is
            # intentionally retained by this function.
            pass

        return True, "JOURNAL_ROW_APPENDED"

    except Exception as exc:
        log.exception(
            "Journal write failed for %s.",
            path,
        )
        return False, f"JOURNAL_WRITE_FAILED:{exc}"


# ============================================================
# 53. SIGNAL ID GENERATION
# ============================================================

def create_signal_id(
    underlying: str,
    direction: str,
    option_symbol: str,
    created_at: Optional[datetime] = None,
) -> str:
    """
    Generate a deterministic-unique signal identifier using time
    and a short digest. This is an identifier, not a security token.
    """
    timestamp = to_utc(created_at or now_utc())

    if timestamp is None:
        timestamp = now_utc()

    material = "|".join([
        JOURNAL_VERSION,
        timestamp.isoformat(),
        str(underlying).upper(),
        str(direction).upper(),
        str(option_symbol).upper(),
        str(uuid.uuid4()),
    ])

    digest = hashlib.sha256(
        material.encode("utf-8")
    ).hexdigest()[:12].upper()

    compact_time = timestamp.strftime("%Y%m%dT%H%M%S")

    return f"SF-{compact_time}-{digest}"


# ============================================================
# 54. JOURNAL A SIGNAL
# ============================================================

def journal_signal(
    signal_id: str,
    prediction: dict[str, Any],
    recommendation: dict[str, Any],
    model_bundle: Optional[dict[str, Any]] = None,
    reason: str = "",
) -> tuple[bool, str]:
    """
    Save a recommendation and its reference quote.

    The record describes a recommendation, not an actual fill.
    """
    if not recommendation.get("ok", False):
        return False, "CANNOT_JOURNAL_INVALID_RECOMMENDATION"

    candidate = recommendation.get("candidate", {})
    sizing = recommendation.get("sizing", {})

    if not isinstance(candidate, dict):
        return False, "MISSING_CONTRACT_CANDIDATE"

    if not isinstance(sizing, dict):
        return False, "MISSING_SIZING_DATA"

    created_utc = now_utc()
    created_ny = to_ny(created_utc)

    model_bundle = model_bundle or {}

    row = {
        "journal_version": JOURNAL_VERSION,
        "signal_id": signal_id,
        "created_at_utc": created_utc.isoformat(),
        "created_at_ny": (
            created_ny.isoformat()
            if created_ny is not None
            else None
        ),
        "underlying": recommendation.get("underlying"),
        "underlying_price": recommendation.get(
            "underlying_price"
        ),
        "direction": prediction.get("signal"),
        "probability_up": prediction.get("probability_up"),
        "probability_down": prediction.get("probability_down"),
        "confidence": prediction.get("confidence"),
        "model_threshold": prediction.get("threshold"),
        "model_enabled": bool(model_bundle.get("enabled", False)),
        "option_symbol": candidate.get("symbol"),
        "option_type": candidate.get("type"),
        "strike": candidate.get("strike"),
        "expiration_date": candidate.get("expiration_date"),
        "bid": candidate.get("bid"),
        "ask": candidate.get("ask"),
        "midpoint": candidate.get("midpoint"),
        "spread_percent": candidate.get("spread_percent"),
        "quote_timestamp_utc": (
            candidate.get("quote_timestamp").isoformat()
            if isinstance(
                candidate.get("quote_timestamp"),
                datetime,
            )
            else candidate.get("quote_timestamp")
        ),
        "quote_age_seconds": candidate.get("age_seconds"),
        "delta": candidate.get("delta"),
        "estimated_contracts": sizing.get("contracts"),
        "estimated_cost_per_contract": sizing.get(
            "estimated_cost_per_contract"
        ),
        "estimated_total_cost": sizing.get(
            "estimated_total_cost"
        ),
        "reason": reason or prediction.get("reason", ""),
        "status": "RECOMMENDATION_RECORDED_NOT_EXECUTED",
    }

    return append_journal_row(
        SIGNAL_JOURNAL_FILE,
        SIGNAL_COLUMNS,
        row,
    )


# ============================================================
# 55. LOAD A RECORDED SIGNAL
# ============================================================

def load_signal_record(
    signal_id: str,
) -> Optional[dict[str, Any]]:
    """Return the newest journal row for the supplied signal ID."""
    try:
        if not SIGNAL_JOURNAL_FILE.exists():
            return None

        frame = pd.read_csv(
            SIGNAL_JOURNAL_FILE,
            dtype={"signal_id": str},
        )

        if frame.empty or "signal_id" not in frame.columns:
            return None

        matches = frame[
            frame["signal_id"].astype(str) == str(signal_id)
        ]

        if matches.empty:
            return None

        return matches.iloc[-1].to_dict()

    except Exception:
        log.exception("Unable to load signal record.")
        return None


# ============================================================
# 56. RECORD A QUOTE OBSERVATION
# ============================================================

def record_quote_observation(
    signal_id: str,
    option_symbol: str,
    quote: dict[str, Any],
    underlying_price: Optional[float] = None,
    observation_number: int = 1,
    target_time: Optional[datetime] = None,
    observed_at: Optional[datetime] = None,
) -> dict[str, Any]:
    """
    Record a fresh quote observation.

    A quote is rejected if it is missing, stale, materially future
    dated, crossed or otherwise invalid. The source quote timestamp
    is preserved separately from the local observation timestamp.
    """
    observed_time = to_utc(observed_at or now_utc())
    target = to_utc(target_time or observed_time)

    if observed_time is None or target is None:
        return {
            "ok": False,
            "reason": "INVALID_OBSERVATION_TIME",
        }

    if not isinstance(quote, dict):
        return {
            "ok": False,
            "reason": "QUOTE_MISSING",
        }

    quote_time = quote_timestamp(quote)

    if quote_time is None:
        return {
            "ok": False,
            "reason": "SOURCE_TIMESTAMP_MISSING",
        }

    # Never allow a quote timestamp to be meaningfully in the future.
    if (
        quote_time - observed_time
    ).total_seconds() > 2.0:
        return {
            "ok": False,
            "reason": "SOURCE_TIMESTAMP_IN_FUTURE",
        }

    age = (
        observed_time - quote_time
    ).total_seconds()

    if age < 0:
        age = 0.0

    if age > MAX_OBSERVATION_AGE_SECONDS:
        return {
            "ok": False,
            "reason": "SOURCE_QUOTE_TOO_OLD",
            "age_seconds": age,
        }

    # Do not silently accept a quote from far before the intended
    # observation time. This prevents stale data being presented as
    # if it had been observed at the target time.
    target_difference = abs(
        (quote_time - target).total_seconds()
    )

    if target_difference > MAX_OBSERVATION_AGE_SECONDS:
        return {
            "ok": False,
            "reason": "QUOTE_TOO_FAR_FROM_TARGET_TIME",
            "target_difference_seconds": target_difference,
        }

    quote_ok, quote_reason, metrics = validate_option_quote(
        quote,
        current_time=observed_time,
    )

    if not quote_ok:
        return {
            "ok": False,
            "reason": quote_reason,
            "details": metrics,
        }

    row = {
        "journal_version": JOURNAL_VERSION,
        "signal_id": signal_id,
        "observed_at_utc": observed_time.isoformat(),
        "quote_timestamp_utc": quote_time.isoformat(),
        "quote_age_seconds": age,
        "option_symbol": option_symbol,
        "bid": metrics.get("bid"),
        "ask": metrics.get("ask"),
        "midpoint": metrics.get("midpoint"),
        "spread_percent": metrics.get("spread_percent"),
        "underlying_price": safe_float(underlying_price),
        "observation_number": int(observation_number),
        "observation_status": "VALID_SOURCE_QUOTE",
    }

    ok, status = append_journal_row(
        QUOTE_JOURNAL_FILE,
        QUOTE_COLUMNS,
        row,
    )

    return {
        "ok": ok,
        "reason": status,
        "row": row if ok else None,
    }


# ============================================================
# 57. READ OBSERVATIONS FOR A SIGNAL
# ============================================================

def load_quote_observations(
    signal_id: str,
) -> pd.DataFrame:
    """Load observations for one signal, ordered by observation time."""
    if not QUOTE_JOURNAL_FILE.exists():
        return pd.DataFrame(columns=QUOTE_COLUMNS)

    try:
        frame = pd.read_csv(
            QUOTE_JOURNAL_FILE,
            dtype={"signal_id": str},
        )

        if frame.empty:
            return pd.DataFrame(columns=QUOTE_COLUMNS)

        matches = frame[
            frame["signal_id"].astype(str) == str(signal_id)
        ].copy()

        if matches.empty:
            return matches

        matches["_parsed_time"] = pd.to_datetime(
            matches["observed_at_utc"],
            utc=True,
            errors="coerce",
        )

        matches = matches.dropna(
            subset=["_parsed_time"]
        ).sort_values("_parsed_time")

        return matches.drop(
            columns=["_parsed_time"],
            errors="ignore",
        )

    except Exception:
        log.exception(
            "Unable to load quote observations for %s.",
            signal_id,
        )
        return pd.DataFrame(columns=QUOTE_COLUMNS)


# ============================================================
# 58. EVALUATE A PAPER OUTCOME
# ============================================================

def evaluate_observed_outcome(
    signal_id: str,
    minimum_observations: int = MIN_OBSERVATIONS_FOR_OUTCOME,
) -> dict[str, Any]:
    """
    Compare the initial reference quote with the latest valid quote.

    This is an observed-price comparison only. It is NOT an actual
    executed trade, realized P&L, or proof that a fill was possible.
    """
    signal = load_signal_record(signal_id)

    if signal is None:
        return {
            "ok": False,
            "reason": "SIGNAL_NOT_FOUND",
        }

    observations = load_quote_observations(signal_id)

    if observations.empty:
        return {
            "ok": False,
            "reason": "NO_VALID_QUOTE_OBSERVATIONS",
        }

    if len(observations) < max(1, int(minimum_observations)):
        return {
            "ok": False,
            "reason": "INSUFFICIENT_OBSERVATIONS",
            "observations_seen": int(len(observations)),
        }

    initial_bid = safe_positive_float(signal.get("bid"))
    initial_ask = safe_positive_float(signal.get("ask"))
    initial_midpoint = safe_positive_float(signal.get("midpoint"))

    if initial_ask is None or initial_midpoint is None:
        return {
            "ok": False,
            "reason": "INVALID_INITIAL_REFERENCE_QUOTE",
        }

    last = observations.iloc[-1]

    final_bid = safe_positive_float(last.get("bid"))
    final_ask = safe_positive_float(last.get("ask"))
    final_midpoint = safe_positive_float(last.get("midpoint"))

    if (
        final_bid is None
        or final_ask is None
        or final_midpoint is None
    ):
        return {
            "ok": False,
            "reason": "INVALID_FINAL_REFERENCE_QUOTE",
        }

    if initial_midpoint <= 0:
        return {
            "ok": False,
            "reason": "INITIAL_MIDPOINT_NOT_POSITIVE",
        }

    midpoint_change_pct = (
        (final_midpoint - initial_midpoint)
        / initial_midpoint
    ) * 100.0

    # This deliberately uses the initial ask and final bid as a
    # conservative reference for a hypothetical long-option entry
    # and exit. It still does not prove either fill was available.
    bid_to_ask_change_pct = (
        (final_bid - initial_ask)
        / initial_ask
    ) * 100.0

    created = to_utc(signal.get("created_at_utc"))
    evaluated = now_utc()

    elapsed_seconds = None

    if created is not None:
        elapsed_seconds = max(
            0.0,
            (evaluated - created).total_seconds(),
        )

    result = {
        "ok": True,
        "signal_id": signal_id,
        "option_symbol": signal.get("option_symbol"),
        "entry_reference_ask": initial_ask,
        "entry_reference_bid": initial_bid,
        "entry_reference_midpoint": initial_midpoint,
        "final_bid": final_bid,
        "final_ask": final_ask,
        "final_midpoint": final_midpoint,
        "observations_seen": int(len(observations)),
        "elapsed_seconds": elapsed_seconds,
        "midpoint_change_percent": midpoint_change_pct,
        "bid_to_ask_change_percent": bid_to_ask_change_pct,
        "outcome_status": "OBSERVED_PRICE_CHANGE_NOT_EXECUTED_PNL",
        "is_actual_execution": False,
        "notes": (
            "Quote-based estimate only. No order was submitted; "
            "fills, fees and slippage are not verified."
        ),
    }

    row = {
        "journal_version": JOURNAL_VERSION,
        "signal_id": signal_id,
        "created_at_utc": signal.get("created_at_utc"),
        "evaluated_at_utc": evaluated.isoformat(),
        "option_symbol": signal.get("option_symbol"),
        "entry_reference_ask": initial_ask,
        "entry_reference_bid": initial_bid,
        "entry_reference_midpoint": initial_midpoint,
        "final_bid": final_bid,
        "final_ask": final_ask,
        "final_midpoint": final_midpoint,
        "observations_seen": int(len(observations)),
        "elapsed_seconds": elapsed_seconds,
        "midpoint_change_percent": midpoint_change_pct,
        "bid_to_ask_change_percent": bid_to_ask_change_pct,
        "outcome_status": result["outcome_status"],
        "is_actual_execution": False,
        "notes": result["notes"],
    }

    ok, status = append_journal_row(
        OUTCOME_JOURNAL_FILE,
        OUTCOME_COLUMNS,
        row,
    )

    result["journal_saved"] = ok
    result["journal_status"] = status

    return result


# ============================================================
# 59. JOURNAL MODEL QUALITY RESULTS
# ============================================================

def journal_model_run(
    model_bundle: dict[str, Any],
) -> tuple[bool, str]:
    """Save model metrics and gate results for later diagnostics."""
    if not isinstance(model_bundle, dict):
        return False, "INVALID_MODEL_BUNDLE"

    metrics = model_bundle.get("metrics", {})
    if not isinstance(metrics, dict):
        metrics = {}

    validation = metrics.get("validation", {})
    if not isinstance(validation, dict):
        validation = {}

    holdout = metrics.get("holdout", {})
    if not isinstance(holdout, dict):
        holdout = {}

    holdout_signals = metrics.get("holdout_signals", {})
    if not isinstance(holdout_signals, dict):
        holdout_signals = {}

    split_rows = model_bundle.get("split_rows", {})
    if not isinstance(split_rows, dict):
        split_rows = {}

    row = {
        "journal_version": JOURNAL_VERSION,
        "trained_at_utc": model_bundle.get(
            "trained_at_utc",
            now_utc().isoformat(),
        ),
        "enabled": bool(model_bundle.get("enabled", False)),
        "threshold": model_bundle.get("threshold"),
        "train_rows": split_rows.get("train"),
        "validation_rows": split_rows.get("validation"),
        "holdout_rows": split_rows.get("holdout"),
        "validation_auc": validation.get("auc"),
        "validation_brier": validation.get("brier"),
        "holdout_auc": holdout.get("auc"),
        "holdout_brier": holdout.get("brier"),
        "holdout_signal_count": holdout_signals.get(
            "signal_count"
        ),
        "holdout_signal_accuracy": holdout_signals.get(
            "signal_accuracy"
        ),
        "reasons": " ; ".join(
            str(item)
            for item in model_bundle.get("reasons", [])
        ),
    }

    return append_journal_row(
        MODEL_JOURNAL_FILE,
        MODEL_COLUMNS,
        row,
    )


# ============================================================
# 60. JOURNAL SUMMARY
# ============================================================

def get_journal_summary() -> dict[str, Any]:
    """
    Return counts and descriptive outcome statistics.

    These are journal statistics, not a verified brokerage statement.
    """
    summary: dict[str, Any] = {
        "journal_version": JOURNAL_VERSION,
        "signals_recorded": 0,
        "quote_observations": 0,
        "outcomes_recorded": 0,
        "actual_executions": 0,
        "mean_midpoint_change_percent": None,
        "mean_bid_to_ask_change_percent": None,
    }

    try:
        if SIGNAL_JOURNAL_FILE.exists():
            signals = pd.read_csv(SIGNAL_JOURNAL_FILE)
            summary["signals_recorded"] = int(len(signals))

        if QUOTE_JOURNAL_FILE.exists():
            quotes = pd.read_csv(QUOTE_JOURNAL_FILE)
            summary["quote_observations"] = int(len(quotes))

        if OUTCOME_JOURNAL_FILE.exists():
            outcomes = pd.read_csv(OUTCOME_JOURNAL_FILE)
            summary["outcomes_recorded"] = int(len(outcomes))

            if "is_actual_execution" in outcomes.columns:
                actual_values = (
                    outcomes["is_actual_execution"]
                    .astype(str)
                    .str.strip()
                    .str.lower()
                )

                summary["actual_executions"] = int(
                    actual_values.isin(
                        {"true", "1", "yes"}
                    ).sum()
                )

            if "midpoint_change_percent" in outcomes.columns:
                values = pd.to_numeric(
                    outcomes["midpoint_change_percent"],
                    errors="coerce",
                ).dropna()

                if not values.empty:
                    summary["mean_midpoint_change_percent"] = (
                        float(values.mean())
                    )

            if "bid_to_ask_change_percent" in outcomes.columns:
                values = pd.to_numeric(
                    outcomes["bid_to_ask_change_percent"],
                    errors="coerce",
                ).dropna()

                if not values.empty:
                    summary["mean_bid_to_ask_change_percent"] = (
                        float(values.mean())
                    )

    except Exception:
        log.exception("Unable to build journal summary.")
        summary["error"] = "JOURNAL_SUMMARY_FAILED"

    return summary


# ============================================================
# END OF PART 5/6
# ============================================================

# ============================================================
# صَفاء التحليل — SPY 0DTE AI ADVISOR v21.4
# PART 6/6 — FINAL VALIDATION • TELEGRAM • MAIN LOOP
#
# RECOMMENDATIONS ONLY
# NO AUTOMATIC ORDER EXECUTION
# ============================================================

import os
import time
import inspect
import requests
from datetime import datetime, timedelta, timezone


# ============================================================
# 61. TELEGRAM CONFIGURATION
# ============================================================

TG_TOKEN = os.getenv(
    "TELEGRAM_BOT_TOKEN", ""
).strip()

TG_CHAT = os.getenv(
    "TELEGRAM_CHAT_ID", ""
).strip()

TELEGRAM_TIMEOUT_SECONDS = _i(
    "TELEGRAM_TIMEOUT_SECONDS", 12
)

TELEGRAM_ENABLED = _b(
    "TELEGRAM_ENABLED", True
)

POLL_SECONDS = max(
    10,
    _i("POLL_SECONDS", 30),
)

SIGNAL_COOLDOWN_MINUTES = max(
    1,
    _i("SIGNAL_COOLDOWN_MINUTES", 20),
)

MAX_DATA_AGE_SECONDS = max(
    30,
    _i("MAX_DATA_AGE_SECONDS", 600),
)

FINAL_SPOT_CHANGE_LIMIT_PERCENT = max(
    0.0,
    _f("FINAL_SPOT_CHANGE_LIMIT_PERCENT", 0.15),
)

# Local in-process state. Restarting the program resets this state;
# journal records remain available on disk.
_LAST_SIGNAL_AT = {}
_LAST_SIGNAL_KEY = None
_STOP_REQUESTED = False


# ============================================================
# 62. GENERAL HELPERS
# ============================================================

def _resolve_callable(*names):
    """Find the first available callable from a list of names."""
    for name in names:
        candidate = globals().get(name)

        if callable(candidate):
            return candidate

    return None


def _call_compatible(function, **kwargs):
    """
    Call a function using only keyword parameters it accepts.
    If signature inspection is unavailable, try the full keyword set.
    """
    try:
        signature = inspect.signature(function)

        accepts_kwargs = any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        )

        if accepts_kwargs:
            return function(**kwargs)

        accepted = {
            key: value
            for key, value in kwargs.items()
            if key in signature.parameters
        }

        return function(**accepted)

    except (TypeError, ValueError):
        return function(**kwargs)


def _normalise_direction(value):
    """Normalize common model direction labels."""
    text_value = str(value or "").strip().upper()

    if text_value in {
        "CALL", "BULL", "BULLISH", "UP", "BUY_CALL"
    }:
        return "CALL"

    if text_value in {
        "PUT", "BEAR", "BEARISH", "DOWN", "BUY_PUT"
    }:
        return "PUT"

    return "WAIT"


def _as_float(value):
    try:
        result = float(value)

        if not math.isfinite(result):
            return None

        return result

    except (TypeError, ValueError, OverflowError):
        return None


def _utc_datetime(value):
    if isinstance(value, datetime):
        return to_utc(value)

    if value is None:
        return None

    return to_utc(value)


def _data_timestamp_is_fresh(timestamp, max_age_seconds=None):
    """
    Fail closed when a timestamp is absent, invalid, or too old,
    unless we are in pre-market or after-hours review windows.
    """
    parsed = _utc_datetime(timestamp)

    if parsed is None:
        return False, "TIMESTAMP_MISSING_OR_INVALID"

    current = now_utc()
    age = (current - parsed).total_seconds()

    if age < -2:
        return False, "TIMESTAMP_IN_FUTURE"

    maximum = (
        max_age_seconds
        if max_age_seconds is not None
        else MAX_DATA_AGE_SECONDS
    )

    # --- استثناء أوقات الراحة (قبل وبعد السوق) ---
    # إذا كان السوق مغلقاً ونحن في فترات التحضير أو المراجعة، نتغاضى عن قدم البيانات
    ny_current = to_ny(current)
    if ny_current is not None:
        start, end = regular_session_bounds(ny_current.date())
        pre_start = start - timedelta(hours=1)
        after_end = end + timedelta(hours=2)
        
        # إذا كنا في فترة ما قبل الافتتاح أو بعد الإغلاق المسموحة، نقبل البيانات بهدوء
        if (pre_start <= ny_current < start) or (end <= ny_current < after_end):
            return True, "FRESH_OFF_HOURS"
    # ---------------------------------------------

    if age > maximum:
        return False, f"STALE_DATA_{age:.1f}_SECONDS"

    return True, "FRESH"

# ============================================================
# 63. TELEGRAM DELIVERY
# ============================================================

def send_telegram(message):
    """
    Send one Telegram message.
    Returns True only when Telegram confirms successful delivery.
    """
    if not TELEGRAM_ENABLED:
        log.info("Telegram is disabled.")
        return False

    if not TG_TOKEN or not TG_CHAT:
        log.warning(
            "Telegram unavailable: TELEGRAM_BOT_TOKEN or "
            "TELEGRAM_CHAT_ID is missing."
        )
        return False

    endpoint = (
        f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
    )

    payload = {
        "chat_id": TG_CHAT,
        "text": str(message),
        "disable_web_page_preview": True,
    }

    try:
        response = requests.post(
            endpoint,
            json=payload,
            timeout=TELEGRAM_TIMEOUT_SECONDS,
        )

        response.raise_for_status()

        body = response.json()

        if not body.get("ok", False):
            log.error(
                "Telegram rejected message: %s",
                body,
            )
            return False

        log.info("Telegram message delivered successfully.")
        return True

    except Exception:
        log.exception("Telegram delivery failed.")
        return False


# ============================================================
# 64. MESSAGE FORMATTING
# ============================================================

def build_telegram_message(
    signal_id,
    prediction,
    recommendation,
):
    """
    Format a concise recommendation.
    Values are observations, not guaranteed fills or profits.
    """
    candidate = recommendation.get("candidate", {})
    sizing = recommendation.get("sizing", {})

    direction = _normalise_direction(
        prediction.get("signal")
        or prediction.get("direction")
    )

    direction_ar = (
        "شراء كول (CALL)"
        if direction == "CALL"
        else "شراء بوت (PUT)"
        if direction == "PUT"
        else "انتظار"
    )

    spot = _as_float(
        recommendation.get("underlying_price")
    )

    confidence = _as_float(
        prediction.get("confidence")
    )

    if confidence is not None and confidence <= 1:
        confidence *= 100

    probability_up = _as_float(
        prediction.get("probability_up")
    )

    probability_down = _as_float(
        prediction.get("probability_down")
    )

    if probability_up is not None and probability_up <= 1:
        probability_up *= 100

    if probability_down is not None and probability_down <= 1:
        probability_down *= 100

    symbol = str(candidate.get("symbol") or "غير متاح")
    expiry = str(
        candidate.get("expiration_date") or "غير متاح"
    )

    strike = _as_float(candidate.get("strike"))
    bid = _as_float(candidate.get("bid"))
    ask = _as_float(candidate.get("ask"))
    midpoint = _as_float(candidate.get("midpoint"))
    spread = _as_float(candidate.get("spread_percent"))
    delta = _as_float(candidate.get("delta"))

    contracts = sizing.get("contracts")
    cost = _as_float(
        sizing.get("estimated_cost_per_contract")
    )
    total_cost = _as_float(
        sizing.get("estimated_total_cost")
    )

    def fmt(value, digits=2):
        return (
            f"{value:,.{digits}f}"
            if value is not None
            else "غير متاح"
        )

    lines = [
        "📊 صَفاء التحليل — SPY 0DTE",
        "━━━━━━━━━━━━━━━━━━",
        f"📌 الإشارة: {direction_ar}",
        f"🆔 رقم التوصية: {signal_id}",
        f"💹 سعر SPY وقت الفحص: {fmt(spot)}",
        f"🧠 ثقة النموذج: {fmt(confidence, 1)}%",
    ]

    if probability_up is not None:
        lines.append(
            f"📈 احتمال الصعود المقدّر: {fmt(probability_up, 1)}%"
        )

    if probability_down is not None:
        lines.append(
            f"📉 احتمال الهبوط المقدّر: {fmt(probability_down, 1)}%"
        )

    lines.extend([
        "",
        "📑 تفاصيل العقد المرصود",
        f"🔹 الرمز: {symbol}",
        f"🎯 سعر التنفيذ: {fmt(strike)}",
        f"📅 الانتهاء: {expiry}",
        f"🟢 Bid: {fmt(bid)}",
        f"🔴 Ask: {fmt(ask)}",
        f"💵 Midpoint: {fmt(midpoint)}",
        f"↔️ السبريد: {fmt(spread, 3)}%",
        f"Δ Delta: {fmt(delta, 3)}",
    ])

    if contracts is not None:
        lines.append(
            f"📦 الحجم التقديري: {contracts} عقد"
        )

    if cost is not None:
        lines.append(
            f"💰 تكلفة العقد التقديرية: ${fmt(cost)}"
        )

    if total_cost is not None:
        lines.append(
            f"💰 التكلفة الإجمالية التقديرية: ${fmt(total_cost)}"
        )

    reason = (
        prediction.get("reason")
        or recommendation.get("reason")
        or "اجتازت التوصية الفلاتر المتاحة"
    )

    lines.extend([
        "",
        f"📝 السبب: {reason}",
        "",
        "⚠️ توصية آلية وليست ضمانًا للربح.",
        "⚠️ الأسعار قد تتغير قبل تنفيذ أي شراء يدوي.",
        "ℹ️ البوت لا ينفّذ أوامر شراء أو بيع.",
    ])

    return "\n".join(lines)


# ============================================================
# 65. COOLDOWN / DUPLICATE PROTECTION
# ============================================================

def signal_key(
    direction,
    option_symbol,
):
    return (
        f"{str(direction).upper()}|"
        f"{str(option_symbol).upper()}"
    )


def cooldown_active(key):
    last_time = _LAST_SIGNAL_AT.get(key)

    if last_time is None:
        return False

    elapsed = (
        now_utc() - last_time
    ).total_seconds()

    return elapsed < SIGNAL_COOLDOWN_MINUTES * 60


def mark_signal_sent(key):
    _LAST_SIGNAL_AT[key] = now_utc()


# ============================================================
# 66. FINAL REVALIDATION
# ============================================================

def final_revalidation(
    recommendation,
    original_spot=None,
):
    """
    Final fail-closed validation before Telegram delivery.

    The underlying and option quote must be re-read using available
    data functions. If this code cannot identify a compatible live
    data function, it refuses to send rather than pretending the
    original quote is still current.
    """
    if not isinstance(recommendation, dict):
        return False, "INVALID_RECOMMENDATION", None

    if not recommendation.get("ok", False):
        return False, "RECOMMENDATION_NOT_APPROVED", None

    candidate = recommendation.get("candidate")

    if not isinstance(candidate, dict):
        return False, "MISSING_OPTION_CANDIDATE", None

    option_symbol = candidate.get("symbol")

    if not option_symbol:
        return False, "MISSING_OPTION_SYMBOL", None

    # Find a live option quote function from the earlier parts.
    quote_function = _resolve_callable(
        "get_option_quote",
        "fetch_option_quote",
        "get_latest_option_quote",
        "get_option_quotes",
    )

    if quote_function is None:
        return False, "NO_LIVE_OPTION_QUOTE_FUNCTION", None

    try:
        fresh_quote = _call_compatible(
            quote_function,
            symbol=option_symbol,
            option_symbol=option_symbol,
            symbols=[option_symbol],
            current_time=now_utc(),
        )

    except Exception as exc:
        log.exception("Final option quote recheck failed.")
        return False, f"FINAL_OPTION_QUOTE_ERROR:{exc}", None

    # Some implementations return a dictionary keyed by symbol.
    if (
        isinstance(fresh_quote, dict)
        and option_symbol in fresh_quote
        and isinstance(fresh_quote[option_symbol], dict)
    ):
        fresh_quote = fresh_quote[option_symbol]

    # Some implementations return {"quotes": {...}}.
    if isinstance(fresh_quote, dict):
        quote_container = fresh_quote.get("quotes")

        if isinstance(quote_container, dict):
            fresh_quote = quote_container.get(
                option_symbol,
                fresh_quote,
            )

        elif isinstance(quote_container, list):
            fresh_quote = next(
                (
                    item for item in quote_container
                    if isinstance(item, dict)
                    and str(
                        item.get("symbol")
                        or item.get("S")
                        or ""
                    ).upper() == str(option_symbol).upper()
                ),
                fresh_quote,
            )

    if not isinstance(fresh_quote, dict):
        return False, "FINAL_OPTION_QUOTE_NOT_FOUND", None

    current = now_utc()

    valid_quote, quote_reason, quote_metrics = (
        validate_option_quote(
            fresh_quote,
            current_time=current,
        )
    )

    if not valid_quote:
        return (
            False,
            f"FINAL_QUOTE_REJECTED:{quote_reason}",
            None,
        )

    # Reject a material quote deterioration compared with the
    # contract originally selected.
    original_ask = _as_float(candidate.get("ask"))
    fresh_ask = _as_float(quote_metrics.get("ask"))

    if original_ask is None or fresh_ask is None:
        return False, "MISSING_ASK_FOR_FINAL_COMPARISON", None

    if original_ask <= 0 or fresh_ask <= 0:
        return False, "INVALID_ASK_FOR_FINAL_COMPARISON", None

    ask_change_pct = (
        abs(fresh_ask - original_ask) / original_ask
    ) * 100.0

    # Use the configured limit as a percentage-point guard.
    # A zero limit means the ask must remain unchanged.
    if ask_change_pct > max(
        FINAL_SPOT_CHANGE_LIMIT_PERCENT,
        0.01,
    ):
        return (
            False,
            f"OPTION_ASK_CHANGED_{ask_change_pct:.3f}_PCT",
            None,
        )

    # Underlying spot must also be refreshed if a compatible
    # spot function exists. If unavailable, do not claim a fresh
    # underlying check was completed.
    spot_function = _resolve_callable(
        "get_spot",
        "get_underlying_spot",
        "get_latest_spot",
        "get_live_spot",
    )

    fresh_spot = None

    if spot_function is not None:
        try:
            fresh_spot = _call_compatible(
                spot_function,
                symbol=PRIMARY_UNDERLYING,
                underlying=PRIMARY_UNDERLYING,
                current_time=current,
            )

            if isinstance(fresh_spot, dict):
                fresh_spot = (
                    fresh_spot.get("price")
                    or fresh_spot.get("spot")
                    or fresh_spot.get("close")
                )

            fresh_spot = _as_float(fresh_spot)

        except Exception:
            log.exception("Final underlying spot recheck failed.")
            return False, "FINAL_SPOT_RECHECK_FAILED", None

        if fresh_spot is None or fresh_spot <= 0:
            return False, "FINAL_SPOT_INVALID", None

        baseline_spot = _as_float(original_spot)

        if baseline_spot is not None and baseline_spot > 0:
            spot_change_pct = (
                abs(fresh_spot - baseline_spot)
                / baseline_spot
            ) * 100.0

            if spot_change_pct > FINAL_SPOT_CHANGE_LIMIT_PERCENT:
                return (
                    False,
                    f"UNDERLYING_MOVED_{spot_change_pct:.3f}_PCT",
                    None,
                )

    else:
        return False, "NO_LIVE_SPOT_RECHECK_FUNCTION", None

    refreshed_candidate = dict(candidate)
    refreshed_candidate.update({
        "bid": quote_metrics.get("bid"),
        "ask": quote_metrics.get("ask"),
        "midpoint": quote_metrics.get("midpoint"),
        "spread_percent": quote_metrics.get("spread_percent"),
        "quote_timestamp": quote_timestamp(fresh_quote),
        "age_seconds": quote_metrics.get("age_seconds"),
    })

    refreshed = dict(recommendation)
    refreshed["candidate"] = refreshed_candidate
    refreshed["underlying_price"] = fresh_spot
    refreshed["final_revalidation_at_utc"] = current.isoformat()

    return True, "FINAL_REVALIDATION_PASSED", refreshed


# ============================================================
# 67. MODEL PREDICTION ADAPTER
# ============================================================

def get_current_prediction():
    """
    صَفاء التحليل — SPY 0DTE AI ADVISOR v21.4
    Fetch SPY + QQQ, align features, train and predict.
    Recommendation-only. No Telegram. No order execution.
    """

    bars_function = _resolve_callable(
        "fetch_stock_bars",
        "get_live_bars",
        "get_market_bars",
        "get_underlying_bars",
    )

    prediction_function = _resolve_callable(
        "predict_direction",
        "calculate_prediction",
    )

    training_pipeline = _resolve_callable(
        "train_from_feature_frame"
    )

    if bars_function is None:
        return {"ok": False, "reason": "LIVE_BARS_FUNCTION_NOT_FOUND"}

    if prediction_function is None:
        return {"ok": False, "reason": "PREDICTION_FUNCTION_NOT_FOUND"}

    if training_pipeline is None:
        return {"ok": False, "reason": "TRAINING_PIPELINE_NOT_FOUND"}

    try:
        # 1. Fetch actual SPY and QQQ bars.
        spy_bars = _call_compatible(
            bars_function,
            symbol=PRIMARY_UNDERLYING,
            timeframe=TIMEFRAME,
            history_days=HISTORY_DAYS,
            days=HISTORY_DAYS,
        )

        qqq_bars = _call_compatible(
            bars_function,
            symbol="QQQ",
            timeframe=TIMEFRAME,
            history_days=HISTORY_DAYS,
            days=HISTORY_DAYS,
        )

        if not isinstance(spy_bars, pd.DataFrame) or spy_bars.empty:
            return {"ok": False, "reason": "SPY_BARS_EMPTY"}

        if not isinstance(qqq_bars, pd.DataFrame) or qqq_bars.empty:
            return {"ok": False, "reason": "QQQ_BARS_EMPTY"}

        # 2. Validate freshness of both data sources.
        for symbol, bars in (
            (PRIMARY_UNDERLYING, spy_bars),
            ("QQQ", qqq_bars),
        ):
            timestamp = None

            for column in ("timestamp", "time", "datetime", "t"):
                if column in bars.columns:
                    timestamp = bars[column].iloc[-1]
                    break

            if timestamp is None:
                if bars.index.empty:
                    return {
                        "ok": False,
                        "reason": f"{symbol}_TIMESTAMP_MISSING",
                    }
                timestamp = bars.index[-1]

            fresh, freshness_reason = _data_timestamp_is_fresh(
                timestamp
            )

            if not fresh:
                return {
                    "ok": False,
                    "reason": (
                        f"{symbol}_DATA_{freshness_reason}"
                    ),
                }

        # 3. Build SPY features and QQQ confirmation features.
        spy_features = build_features(spy_bars)

        qqq_features = build_confirmation_features(
            qqq_bars,
            prefix="qqq_",
        )

        if spy_features.empty or qqq_features.empty:
            return {
                "ok": False,
                "reason": "FEATURE_FRAME_EMPTY",
            }

        # 4. Align QQQ backward to SPY timestamps.
        # This avoids using future QQQ bars.
        aligned_features = align_confirmation_data(
            spy_features,
            qqq_features,
            max_age_seconds=600,
        )

        if aligned_features.empty:
            return {
                "ok": False,
                "reason": "ALIGNED_FEATURE_FRAME_EMPTY",
            }

        # 5. Verify required model columns exist.
        missing_columns = [
            column
            for column in FEATURE_COLUMNS
            if column not in aligned_features.columns
        ]

        if missing_columns:
            return {
                "ok": False,
                "reason": (
                    "MISSING_FEATURE_COLUMNS:"
                    + ",".join(missing_columns)
                ),
            }

        # 6. Do not predict if the newest SPY row has
        # incomplete or stale QQQ confirmation features.
        latest_features = aligned_features[
            FEATURE_COLUMNS
        ].tail(1)

        if (
            latest_features.empty
            or latest_features.isna().any(axis=1).iloc[0]
            or not np.isfinite(
                latest_features.to_numpy(dtype=float)
            ).all()
        ):
            return {
                "ok": False,
                "reason": "LATEST_FEATURES_INCOMPLETE",
            }

        # 7. Train through the existing labeling and
        # dataset-preparation pipeline.
        model_bundle = _call_compatible(
            training_pipeline,
            feature_frame=aligned_features,
        )

        if not isinstance(model_bundle, dict):
            return {
                "ok": False,
                "reason": "TRAINING_RESULT_NOT_A_DICTIONARY",
            }

        journal_model_run(model_bundle)

        log.warning(
            "MODEL TRAIN RESULT | enabled=%s | reasons=%s | "
            "metrics=%s | threshold=%s | split_rows=%s",
            model_bundle.get("enabled"),
            model_bundle.get("reasons", []),
            model_bundle.get("metrics", {}),
            model_bundle.get("threshold"),
            model_bundle.get("split_rows", {}),
        )

        if not model_bundle.get("enabled", False):
            return {
                "ok": False,
                "reason": "MODEL_GATE_DISABLED",
                "model_bundle": model_bundle,
            }

        # 8. Predict using the actual predict_direction signature:
        # predict_direction(trained, latest_features)
        prediction = _call_compatible(
            prediction_function,
            trained=model_bundle,
            latest_features=latest_features,
        )

        if not isinstance(prediction, dict):
            return {
                "ok": False,
                "reason": "PREDICTION_RETURN_TYPE_UNSUPPORTED",
            }

        direction = _normalise_direction(
            prediction.get("signal")
            or prediction.get("direction")
        )

        if direction == "WAIT":
            return {
                "ok": False,
                "reason": prediction.get(
                    "reason",
                    "MODEL_DIRECTION_IS_WAIT",
                ),
                "prediction": prediction,
                "model_bundle": model_bundle,
            }

        prediction["signal"] = direction

        return {
            "ok": True,
            "prediction": prediction,
            "model_bundle": model_bundle,
            "bars": spy_bars,
            "qqq_bars": qqq_bars,
            "features": aligned_features,
            "latest_features": latest_features,
        }

    except Exception as exc:
        log.exception("Current prediction failed.")

        return {
            "ok": False,
            "reason": f"PREDICTION_PIPELINE_ERROR:{exc}",
        }



# ============================================================
# 68. UNDERLYING SPOT ADAPTER
# ============================================================

def get_current_underlying_spot():
    spot_function = _resolve_callable(
        "get_spot",
        "get_underlying_spot",
        "get_latest_spot",
        "get_live_spot",
    )

    if spot_function is None:
        return None, "SPOT_FUNCTION_NOT_FOUND"

    try:
        result = _call_compatible(
            spot_function,
            symbol=PRIMARY_UNDERLYING,
            underlying=PRIMARY_UNDERLYING,
            current_time=now_utc(),
        )

        if isinstance(result, dict):
            price = (
                result.get("price")
                or result.get("spot")
                or result.get("close")
            )
        else:
            price = result

        price = _as_float(price)

        if price is None or price <= 0:
            return None, "SPOT_INVALID"

        return price, "SPOT_OK"

    except Exception as exc:
        log.exception("Unable to fetch current spot.")
        return None, f"SPOT_ERROR:{exc}"


# ============================================================
# 69. CONTRACT SELECTION ADAPTER
# ============================================================

def get_recommendation_candidate(
    direction,
    spot,
):
    """
    Use the contract-selection function defined in earlier parts.
    No candidate means no alert.
    """
    selector = _resolve_callable(
        "find_option_recommendation_candidate",
        "choose_option_candidate",
        "select_option_candidate",
        "choose_contract",
    )

    if selector is None:
        return {
            "ok": False,
            "reason": "OPTION_SELECTION_FUNCTION_NOT_FOUND",
        }

    try:
        result = _call_compatible(
            selector,
            direction=direction,
            signal=direction,
            underlying=PRIMARY_UNDERLYING,
            underlying_price=spot,
            spot=spot,
            current_time=now_utc(),
        )

        if not isinstance(result, dict):
            return {
                "ok": False,
                "reason": "OPTION_SELECTOR_RETURN_TYPE_UNSUPPORTED",
            }

        # A selector may return a candidate directly rather than
        # the full {"ok": ..., "candidate": ...} structure.
        if (
            "candidate" not in result
            and result.get("symbol")
        ):
            result = {
                "ok": True,
                "candidate": result,
                "underlying": PRIMARY_UNDERLYING,
                "underlying_price": spot,
                "sizing": {},
            }

        if not result.get("ok", False):
            return result

        if not isinstance(result.get("candidate"), dict):
            return {
                "ok": False,
                "reason": "OPTION_CANDIDATE_MISSING",
            }

        result.setdefault("underlying", PRIMARY_UNDERLYING)
        result.setdefault("underlying_price", spot)
        result.setdefault("sizing", {})

        return result

    except Exception as exc:
        log.exception("Option candidate selection failed.")
        return {
            "ok": False,
            "reason": f"OPTION_SELECTION_ERROR:{exc}",
        }


# ============================================================
# 70. MAIN SCAN
# ============================================================

def run_scan():
    """
    One scan cycle.

    Workflow:
      1. Obtain fresh model prediction.
      2. Obtain fresh underlying spot.
      3. Select an option contract.
      4. Apply final quote and spot revalidation.
      5. Check duplicate cooldown.
      6. Journal recommendation.
      7. Send Telegram.

    No orders are submitted.
    """
    log.info("Starting scan for %s.", PRIMARY_UNDERLYING)

    try:
        calculate_and_send_outlook(bars, TG_TOKEN, TG_CHAT)
    except Exception:
        pass

    
    prediction_result = get_current_prediction()

    if not prediction_result.get("ok", False):
        log.info(
            "No signal: %s",
            prediction_result.get("reason"),
        )
        return {
            "ok": False,
            "stage": "PREDICTION",
            "reason": prediction_result.get("reason"),
        }

    prediction = prediction_result["prediction"]
    model_bundle = prediction_result.get("model_bundle", {})

    direction = _normalise_direction(
        prediction.get("signal")
        or prediction.get("direction")
    )

    if direction == "WAIT":
        log.info("Patient mode: model direction is WAIT.")
        return {
            "ok": False,
            "stage": "DIRECTION",
            "reason": "DIRECTION_WAIT",
        }

    spot, spot_reason = get_current_underlying_spot()

    if spot is None:
        log.warning("No recommendation: %s", spot_reason)
        return {
            "ok": False,
            "stage": "SPOT",
            "reason": spot_reason,
        }

    recommendation = get_recommendation_candidate(
        direction=direction,
        spot=spot,
    )
  try:
calculate_and_send_outlook(bars, TG_TOKEN, TG_CHAT)
except Exception:
    pass

    if not recommendation.get("ok", False):
        reason = recommendation.get(
            "reason",
            "NO_QUALIFIED_OPTION",
        )

        log.info("Patient mode: %s", reason)

        return {
            "ok": False,
            "stage": "CONTRACT",
            "reason": reason,
        }

    # Re-read spot and option quote immediately before any alert.
    passed, recheck_reason, refreshed = final_revalidation(
        recommendation=recommendation,
        original_spot=spot,
    )

    if not passed:
        log.info(
            "Final revalidation rejected the signal: %s",
            recheck_reason,
        )
        return {
            "ok": False,
            "stage": "FINAL_REVALIDATION",
            "reason": recheck_reason,
        }

    recommendation = refreshed

    candidate = recommendation["candidate"]

    key = signal_key(
        direction,
        candidate.get("symbol"),
    )

    if cooldown_active(key):
        log.info(
            "Duplicate signal suppressed by cooldown: %s",
            key,
        )
        return {
            "ok": False,
            "stage": "COOLDOWN",
            "reason": "DUPLICATE_SIGNAL_COOLDOWN",
        }

    signal_id = create_signal_id(
        underlying=PRIMARY_UNDERLYING,
        direction=direction,
        option_symbol=candidate.get("symbol"),
        created_at=now_utc(),
    )

    journal_ok, journal_status = journal_signal(
        signal_id=signal_id,
        prediction=prediction,
        recommendation=recommendation,
        model_bundle=model_bundle,
        reason=prediction.get("reason", ""),
    )

    # If the recommendation cannot be journaled, do not send an
    # alert that cannot be audited afterward.
    if not journal_ok:
        log.error(
            "Signal suppressed because journal write failed: %s",
            journal_status,
        )
        return {
            "ok": False,
            "stage": "JOURNAL",
            "reason": journal_status,
        }

    message = build_telegram_message(
        signal_id=signal_id,
        prediction=prediction,
        recommendation=recommendation,
    )

    delivered = send_telegram(message)

    if delivered:
        mark_signal_sent(key)

        log.info(
            "Recommendation sent. signal_id=%s symbol=%s",
            signal_id,
            candidate.get("symbol"),
        )

        return {
            "ok": True,
            "signal_id": signal_id,
            "symbol": candidate.get("symbol"),
            "direction": direction,
            "telegram_sent": True,
            "orders_submitted": False,
        }

    log.warning(
        "Telegram was not delivered. Signal remains journaled: %s",
        signal_id,
    )

    return {
        "ok": False,
        "stage": "TELEGRAM",
        "reason": "TELEGRAM_DELIVERY_FAILED",
        "signal_id": signal_id,
        "orders_submitted": False,
    }


# ============================================================
# 71. CONTINUOUS MAIN LOOP
# ============================================================

def main():
    """
    Start the recommendation-only advisor.
    """
    log.info("=" * 60)
    log.info("%s — VERSION %s", BOT_NAME, BOT_VERSION)
    log.info("Underlying: %s", PRIMARY_UNDERLYING)
    log.info("Mode: RECOMMENDATIONS ONLY")
    log.info("Automatic order execution: DISABLED")
    log.info("=" * 60)

    try:
        initialize_journal()
    except Exception:
        log.exception("Journal initialization failed.")
        raise

    if not TG_TOKEN or not TG_CHAT:
        log.warning(
            "Telegram credentials are missing. "
            "The scanner can run, but alerts cannot be delivered."
        )
    while not _STOP_REQUESTED:
        started = time.monotonic()

        try:
            result = run_scan()

            if result.get("ok"):
                log.info(
                    "Scan complete: recommendation delivered."
                )
            else:
                log.info(
                    "Scan complete: no alert. stage=%s reason=%s",
                    result.get("stage"),
                    result.get("reason"),
                )

        except KeyboardInterrupt:
            log.info("Shutdown requested by keyboard.")
            break

        except Exception:
            log.exception(
                "Unexpected scan error; waiting for next cycle."
            )

        elapsed = time.monotonic() - started

        sleep_seconds = max(
            1,
            POLL_SECONDS - int(elapsed),
        )

        try:
            time.sleep(sleep_seconds)
        except KeyboardInterrupt:
            log.info("Shutdown requested.")
            break

    log.info("صَفاء التحليل stopped.")


# ============================================================
# 72. ENTRY POINT
# ============================================================
def get_spot(symbol=PRIMARY_UNDERLYING, **kwargs):
    """دالة لجلب أحدث سعر إغلاق لحظي"""
    try:
        bars = fetch_stock_bars(symbol, timeframe=TIMEFRAME, history_days=2, limit=5)
        if bars is not None and not bars.empty:
            return float(bars["close"].iloc[-1])
    except Exception:
        pass
    return None

def fetch_latest_option_quotes(symbols, **kwargs):
    """دالة جلب أسعار عقود الخيارات الحية"""
    try:
        # محاولة جلب البيانات عبر مكتبة Alpaca المتاحة في البوت
        return rest.get_latest_option_quotes(symbols)
    except Exception:
        return {}

# تعريف أسماء بديلة قد يبحث عنها البوت في هذه المرحلة:
get_live_option_quote = fetch_latest_option_quotes
fetch_option_quotes = fetch_latest_option_quotes

if __name__ == "__main__":
    main()

import math
import requests

def send_market_outlook_telegram(current_price, direction, target_price, points_move, bot_token, chat_id):
    """
    إرسال تقرير تحليل حركة المؤشر المستقل عبر تليجرام بتصميم أبو سحاب
    """
    if not bot_token or not chat_id:
        return
    
    # تحديد الإيموجي والاتجاه بناءً على الحسابات
    dir_text = "صعود بنقاط مستهدفة 🟢" if direction == "up" else "هبوط بنقاط مستهدفة 🔴"
    
    message = (
        "✨ **تحليل حركة المؤشر المتوقعة**\n"
        "🤖\n\n"
        f"▫️ **المؤشر:** `S&P 500`\n"
        f"▫️ **السعر الحالي:** `{current_price:.2f}`\n"
        f"▫️ **الوجهة المتوقعة:** {dir_text}\n"
        f"▫️ **الهدف السعري:** `{target_price:.2f}`\n"
        f"▫️ **مدى الحركة:** `{points_move}`\n\n"
        "⚡ *جاهز للجلسة بناءً على تحليل الـ ATR*\n\n"
        "<i>✨ تصميم أبو سحاب</i>"
    )
    
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": message,
        "parse_mode": "Markdown"
    }
    
    try:
        response = requests.post(url, json=payload, timeout=10)
        if response.status_code == 200:
            print("تم إرسال تحليل المؤشر اليومي بنجاح.")
        else:
            print(f"فشل إرسال تحليل المؤشر: {response.text}")
    except Exception as e:
        print(f"خطأ في الاتصال بتليجرام: {e}")

def calculate_and_send_outlook(bars, bot_token, chat_id):
    """
    حساب الهدف السعري بناءً على آخر الشموع والـ ATR
    """
    if not bars or len(bars) < 2:
        return
    
    current_price = float(bars[-1]['close'])
    
    # حساب تقريبي للـ ATR والمدى بناءً على آخر الشموع (أو استخدام دالة الـ ATR الموجودة عندك في البوت)
    highs = [float(b['high']) for b in bars[-14:]]
    lows = [float(b['low']) for b in bars[-14:]]
    closes = [float(b['close']) for b in bars[-14:]]
    
    tr_list = []
    for i in range(1, len(closes)):
        tr = max(highs[i] - lows[i], abs(highs[i] - closes[i-1]), abs(lows[i] - closes[i-1]))
        tr_list.append(tr)
    
    atr = sum(tr_list) / len(tr_list) if tr_list else 5.0
    
    # تحديد الاتجاه المتوقع والهدف بناءً على الزخم الأخير
    if closes[-1] >= closes[-2]:
        direction = "up"
        target_price = current_price + (atr * 0.5)
        points_move = f"+{round(atr * 0.5, 2)} نقاط"
    else:
        direction = "down"
        target_price = current_price - (atr * 0.5)
        points_move = f"-{round(atr * 0.5, 2)} نقاط"
        
    # إرسال الرسالة عبر التليجرام
    send_market_outlook_telegram(current_price, direction, target_price, points_move, bot_token, chat_id)

# ============================================================
# END OF PART 6/6
# ============================================================
