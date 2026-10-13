# ============================================================
# صَفاء التحليل v22.1 | SPY 0DTE | RECOMMENDATIONS ONLY
# Completed-bar checks • Model validation • Option tracking fixes
# Daily outlook • Move alerts • CSV journal • Telegram
# NO AUTOMATIC ORDERS — simulated tracking, not actual P&L
# ============================================================
import os
import sys
import csv
import json
import time
import math
import logging
from pathlib import Path
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
import requests
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, brier_score_loss

# -------------------- CONFIG --------------------
NAME, VERSION = "صَفاء التحليل", "22.1-Flex"
NY, UTC = ZoneInfo("America/New_York"), timezone.utc
SPY, TF = "SPY", "5Min"
BAR_MINUTES = 5
DATA_URL = os.getenv(
    "ALPACA_DATA_URL", "https://data.alpaca.markets"
).rstrip("/")
TRADING_URL = os.getenv(
    "ALPACA_TRADING_URL", "https://paper-api.alpaca.markets"
).rstrip("/")
KEY = (
    os.getenv("APCA_API_KEY_ID")
    or os.getenv("ALPACA_API_KEY")
    or ""
).strip()
SECRET = (
    os.getenv("APCA_API_SECRET_KEY")
    or os.getenv("ALPACA_SECRET_KEY")
    or ""
).strip()
STOCK_FEED = os.getenv("ALPACA_DATA_FEED", "iex").lower()
OPTION_FEED = os.getenv("OPTIONS_FEED", "indicative").lower()
TG_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TG_CHAT = os.getenv("TELEGRAM_CHAT_ID", "").strip()
TG_ENABLED = bool(TG_TOKEN and TG_CHAT)

HISTORY_DAYS = 60
MIN_ROWS = 300
HORIZON = 6

# --- تعديلات المرونة لتقليل القيود وتوليد إشارات أكثر ---
MIN_PROB = 0.57            # تم تخفيضها من 0.64 إلى 0.57
MIN_AUC = 0.52             # تم تخفيضها قليلاً لتساهل أدنى مع النموذج
MIN_SIGNAL_ACCURACY = 0.52 # تم تخفيضها من 0.55
MIN_VALIDATION_SIGNALS = 5 # تم تخفيضها من 10
MAX_BRIER = 0.28           # تم رفعها قليلاً لتكون أكثر تساهلاً
MOVE_POINTS = 4.0          # تم خفض تنبيه الحركة من 5 إلى 4 نقاط

MAX_STRIKE_DISTANCE = 7.0  # تم توسيع نطاق السترايك من 5 إلى 7 نقاط
MIN_PREMIUM = 0.30         # تم خفض الحد الأدنى لسعر العقد من 0.50 إلى 0.30
MAX_PREMIUM = 50.0
MAX_SPREAD_PCT = 0.18      # تم توسيع السبريد المقبول من 12% إلى 18%
# ------------------------------------------------------

MAX_QUOTE_AGE_SECONDS = 90
MAX_BAR_AGE_MINUTES = 10
COOLDOWN_MINUTES = 15      # تم خفض وقت الانتظار بين التوصيات من 20 إلى 15 دقيقة
POLL_SECONDS = 30
PAPER_TAKE_PROFIT_PCT = 0.30
PAPER_STOP_LOSS_PCT = 0.25
PAPER_EXIT_TIME = "15:55"
MAX_DAILY_SIGNALS = 8      # رفع الحد الأقصى للتوصيات اليومية إلى 8

ROOT = Path(os.getenv("DATA_DIR", "."))
ROOT.mkdir(parents=True, exist_ok=True)
JOURNAL = ROOT / "safa_signals_v221.csv"
STATE_FILE = ROOT / "safa_state_v221.json"

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("safaa")
HTTP = requests.Session()
HTTP.headers.update({"User-Agent": f"SafaaAnalysis/{VERSION}"})

# -------------------- GENERAL HELPERS --------------------
def now_ny():
    return datetime.now(NY)

def stamp():
    return datetime.now(UTC).isoformat()

def num(value, default=np.nan):
    try:
        x = float(value)
        return x if math.isfinite(x) else default
    except (TypeError, ValueError):
        return default

def read_state():
    try:
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}

def save_state(state):
    tmp = STATE_FILE.with_suffix(".tmp")
    tmp.write_text(
        json.dumps(state, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    tmp.replace(STATE_FILE)

def parse_ts(value):
    try:
        t = pd.Timestamp(value)
        if t.tzinfo is None:
            t = t.tz_localize("UTC")
        return t.tz_convert("UTC")
    except Exception:
        return None

def alpaca_get(url, params=None):
    if not KEY or not SECRET:
        raise RuntimeError(
            "APCA_API_KEY_ID / APCA_API_SECRET_KEY غير مضبوطين"
        )
    last = None
    for attempt in range(3):
        try:
            r = HTTP.get(
                url,
                headers={
                    "APCA-API-KEY-ID": KEY,
                    "APCA-API-SECRET-KEY": SECRET,
                },
                params=params,
                timeout=15,
            )
            if r.status_code in (401, 403):
                raise RuntimeError(
                    f"Alpaca authentication/permission error: "
                    f"{r.status_code}"
                )
            r.raise_for_status()
            result = r.json()
            if not isinstance(result, dict):
                raise RuntimeError("Unexpected Alpaca response")
            return result
        except Exception as exc:
            last = exc
            if attempt < 2:
                time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"Alpaca request failed: {last}")

def telegram(message):
    if not TG_ENABLED:
        log.warning("Telegram غير مضبوط؛ لم يتم إرسال الرسالة.")
        return False
    try:
        r = HTTP.post(
            f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage",
            json={
                "chat_id": TG_CHAT,
                "text": message,
                "parse_mode": "HTML",
            },
            timeout=12,
        )
        if r.status_code != 200:
            log.error(
                "Telegram error %s: %s",
                r.status_code,
                r.text[:300],
            )
            return False
        result = r.json()
        if not result.get("ok", False):
            log.error("Telegram returned ok=false")
            return False
        return True
    except Exception:
        log.exception("Telegram send failed")
        return False

def market_open():
    n = now_ny()
    return (
        n.weekday() < 5
        and (n.hour, n.minute) >= (9, 30)
        and (n.hour, n.minute) < (16, 0)
    )

def in_signal_window():
    n = now_ny()
    return (
        market_open()
        and (n.hour, n.minute) >= (9, 40)
        and (n.hour, n.minute) < (15, 30)
    )

def exit_time_reached(n=None):
    n = n or now_ny()
    try:
        hour, minute = map(int, PAPER_EXIT_TIME.split(":"))
        return (n.hour, n.minute) >= (hour, minute)
    except (ValueError, AttributeError):
        log.error("PAPER_EXIT_TIME غير صالح: %r", PAPER_EXIT_TIME)
        return False

# -------------------- STOCK DATA --------------------
def fetch_bars(symbol=SPY, days=HISTORY_DAYS):
    end = datetime.now(UTC)
    params = {
        "timeframe": TF,
        "start": (end - timedelta(days=days)).isoformat(),
        "end": end.isoformat(),
        "limit": 10000,
        "adjustment": "raw",
        "feed": STOCK_FEED,
        "sort": "asc",
    }
    url = f"{DATA_URL}/v2/stocks/{symbol}/bars"
    all_bars = []
    token = None
    for _ in range(20):
        p = dict(params)
        if token:
            p["page_token"] = token
        data = alpaca_get(url, p)
        all_bars.extend(data.get("bars") or [])
        token = data.get("next_page_token")
        if not token:
            break
    else:
        if token:
            log.warning(
                "Bar pagination limit reached for %s; "
                "history may be incomplete.",
                symbol,
            )
    if not all_bars:
        return pd.DataFrame(
            columns=["open", "high", "low", "close", "volume"]
        )
    df = pd.DataFrame(all_bars).rename(
        columns={
            "t": "timestamp",
            "o": "open",
            "h": "high",
            "l": "low",
            "c": "close",
            "v": "volume",
        }
    )
    df["timestamp"] = pd.to_datetime(
        df["timestamp"], utc=True, errors="coerce"
    )
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = (
        df.dropna(
            subset=["timestamp", "open", "high", "low", "close"]
        )
        .drop_duplicates("timestamp", keep="last")
        .sort_values("timestamp")
        .set_index("timestamp")
    )
    now_utc = pd.Timestamp.now(tz="UTC")
    current_bucket_minute = (
        now_utc.minute // BAR_MINUTES
    ) * BAR_MINUTES
    current_bucket = now_utc.replace(
        minute=current_bucket_minute,
        second=0,
        microsecond=0,
    )
    df = df.loc[df.index < current_bucket]
    return df

FEATURES = [
    "ret1", "ret3", "ret6", "ret12",
    "ema_spread", "ema_trend", "rsi", "atr_pct",
    "range_pct", "body_pct", "close_location",
    "volume_ratio",
]

def features_for(bars):
    f = bars.copy()
    c = f["close"]
    f["ret1"] = c.pct_change()
    f["ret3"] = c.pct_change(3)
    f["ret6"] = c.pct_change(6)
    f["ret12"] = c.pct_change(12)
    e9 = c.ewm(
        span=9, adjust=False, min_periods=9
    ).mean()
    e20 = c.ewm(
        span=20, adjust=False, min_periods=20
    ).mean()
    e50 = c.ewm(
        span=50, adjust=False, min_periods=50
    ).mean()
    f["ema_spread"] = (e9 - e20) / c
    f["ema_trend"] = (e20 - e50) / c
    d = c.diff()
    gain = d.clip(lower=0).ewm(
        alpha=1 / 14, adjust=False, min_periods=14
    ).mean()
    loss = (-d.clip(upper=0)).ewm(
        alpha=1 / 14, adjust=False, min_periods=14
    ).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = (100 - 100 / (1 + rs)) / 100
    rsi = rsi.mask((loss == 0) & (gain > 0), 1.0)
    rsi = rsi.mask((gain == 0) & (loss > 0), 0.0)
    rsi = rsi.mask((gain == 0) & (loss == 0), 0.5)
    f["rsi"] = rsi
    prev = c.shift()
    tr = pd.concat(
        [
            f["high"] - f["low"],
            (f["high"] - prev).abs(),
            (f["low"] - prev).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = tr.ewm(
        alpha=1 / 14, adjust=False, min_periods=14
    ).mean()
    f["atr_pct"] = atr / c
    rng = (f["high"] - f["low"]).clip(lower=0)
    f["range_pct"] = rng / c
    f["body_pct"] = (f["close"] - f["open"]) / c
    f["close_location"] = (
        (c - f["low"]) / rng.replace(0, np.nan)
    ).fillna(0.5)
    vmean = f["volume"].rolling(
        20, min_periods=20
    ).mean()
    f["volume_ratio"] = (
        f["volume"] / vmean.replace(0, np.nan)
    )
    return f.replace([np.inf, -np.inf], np.nan)

# -------------------- MODEL VALIDATION --------------------
def build_model():
    return make_pipeline(
        SimpleImputer(strategy="median"),
        RandomForestClassifier(
            n_estimators=180,
            max_depth=7,
            min_samples_leaf=4,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
    )

def train_predict(bars):
    if bars.empty or len(bars) < MIN_ROWS:
        return {
            "ok": False,
            "reason": "INSUFFICIENT_ROWS",
        }
    f = features_for(bars)
    future = bars["close"].shift(-HORIZON)
    f["target"] = np.where(
        future.notna(),
        (future > bars["close"]).astype(float),
        np.nan,
    )
    ds = f[FEATURES + ["target"]].dropna()
    if len(ds) < MIN_ROWS or ds["target"].nunique() < 2:
        return {
            "ok": False,
            "reason": "INSUFFICIENT_LABELS",
        }
    split = int(len(ds) * 0.80)
    if split <= 0 or split >= len(ds):
        return {
            "ok": False,
            "reason": "INVALID_TIME_SPLIT",
        }
    valid_start_time = ds.index[split]
    embargo_cutoff = (
        valid_start_time
        - pd.Timedelta(minutes=BAR_MINUTES * HORIZON)
    )
    train = ds.loc[ds.index < embargo_cutoff]
    valid = ds.loc[ds.index >= valid_start_time]
    if len(train) < 100 or len(valid) < 30:
        return {
            "ok": False,
            "reason": "SMALL_HOLDOUT",
        }
    if train["target"].nunique() < 2:
        return {
            "ok": False,
            "reason": "TRAIN_HAS_ONE_CLASS",
        }
    model = build_model()
    model.fit(train[FEATURES], train["target"].astype(int))
    p = model.predict_proba(valid[FEATURES])[:, 1]
    y = valid["target"].astype(int).to_numpy()
    auc = (
        roc_auc_score(y, p)
        if len(np.unique(y)) == 2
        else 0.5
    )
    brier = brier_score_loss(y, p)
    chosen = (p >= MIN_PROB) | (p <= 1 - MIN_PROB)
    signal_count = int(chosen.sum())
    signal_accuracy = (
        float(
            (
                (p[chosen] >= 0.5).astype(int)
                == y[chosen]
            ).mean()
        )
        if signal_count
        else 0.0
    )
    metrics = {
        "auc": float(auc),
        "brier": float(brier),
        "signals": signal_count,
        "signal_accuracy": signal_accuracy,
        "validation_rows": int(len(valid)),
    }
    log.info("HOLDOUT %s", metrics)
    if auc < MIN_AUC or brier > MAX_BRIER:
        return {
            "ok": False,
            "reason": "HOLDOUT_QUALITY_FAILED",
            "metrics": metrics,
        }
    if (
        signal_count < MIN_VALIDATION_SIGNALS
        or signal_accuracy < MIN_SIGNAL_ACCURACY
    ):
        return {
            "ok": False,
            "reason": "HOLDOUT_SIGNALS_FAILED",
            "metrics": metrics,
        }
    final_model = build_model()
    final_model.fit(
        ds[FEATURES],
        ds["target"].astype(int),
    )
    latest_features = f[FEATURES].iloc[[-1]]
    if latest_features.isna().any(axis=None):
        return {
            "ok": False,
            "reason": "LATEST_FEATURES_INCOMPLETE",
            "metrics": metrics,
        }
    prob_up = float(
        final_model.predict_proba(latest_features)[0, 1]
    )
    atr_pct = num(f["atr_pct"].iloc[-1])
    price = num(bars["close"].iloc[-1])
    if not math.isfinite(price) or price <= 0:
        return {
            "ok": False,
            "reason": "INVALID_LATEST_PRICE",
            "metrics": metrics,
        }
    atr = (
        atr_pct * price
        if math.isfinite(atr_pct) and atr_pct > 0
        else 0.0
    )
    return {
        "ok": True,
        "signal": "CALL" if prob_up >= 0.5 else "PUT",
        "prob_up": prob_up,
        "probability": max(prob_up, 1 - prob_up) * 100,
        "current_price": price,
        "atr": atr,
        "metrics": metrics,
        "bar_time": bars.index[-1].isoformat(),
    }

# -------------------- DAILY OUTLOOK --------------------
def daily_bars(bars):
    if bars.empty:
        return pd.DataFrame()
    x = bars.copy()
    x.index = x.index.tz_convert(NY)
    x = x.between_time("09:30", "16:00", inclusive="left")
    if x.empty:
        return pd.DataFrame()
    x["session"] = x.index.date
    return x.groupby("session").agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
    ).dropna()

def send_daily_outlook(bars):
    n = now_ny()
    state = read_state()
    day = n.date().isoformat()
    if (
        state.get("daily_outlook_day") == day
        or (n.hour, n.minute) < (9, 0)
        or (n.hour, n.minute) >= (9, 30)
    ):
        return
    d = daily_bars(bars)
    if d.empty:
        log.warning("Daily outlook skipped: no regular-session bars.")
        return
    d = d.loc[d.index < n.date()]
    if len(d) < 10:
        log.warning(
            "Daily outlook skipped: insufficient prior sessions."
        )
        return
    prev_close = float(d["close"].iloc[-1])
    gaps = (
        d["open"] / d["close"].shift(1) - 1
    ).dropna().tail(20)
    moves = (
        d["close"] / d["open"] - 1
    ).dropna().tail(20)
    high_exc = (
        d["high"] / d["open"] - 1
    ).dropna().tail(20)
    low_exc = (
        d["low"] / d["open"] - 1
    ).dropna().tail(20)
    gap_med = float(gaps.median()) if len(gaps) else 0.0
    move_med = float(moves.median()) if len(moves) else 0.0
    open_est = prev_close * (1 + gap_med)
    close_est = open_est * (1 + move_med)
    high_est = open_est * (
        1 + max(0.0, float(high_exc.quantile(0.75)))
    )
    low_est = open_est * (
        1 + min(0.0, float(low_exc.quantile(0.25)))
    )
    high_range = (
        float(high_exc.quantile(0.25)),
        float(high_exc.quantile(0.90)),
    )
    low_range = (
        float(low_exc.quantile(0.10)),
        float(low_exc.quantile(0.75)),
    )
    msg = (
        f"🌅 <b>{NAME} | التوقع اليومي SPY</b>\n"
        f"📅 {day} — تقدير إحصائي من الجلسات السابقة\n\n"
        f"▫️ إغلاق الجلسة السابقة: <b>${prev_close:.2f}</b>\n"
        f"▫️ الافتتاح المتوقع: <b>${open_est:.2f}</b>\n"
        f"▫️ الإغلاق المتوقع: <b>${close_est:.2f}</b>\n"
        f"▫️ أعلى سعر تقديري: <b>${high_est:.2f}</b>\n"
        f"▫️ أدنى سعر تقديري: <b>${low_est:.2f}</b>\n\n"
        f"📈 نطاق تاريخي استرشادي للأعلى: "
        f"{100 * high_range[0]:+.2f}% إلى "
        f"{100 * high_range[1]:+.2f}% من الافتتاح\n"
        f"📉 نطاق تاريخي استرشادي للأدنى: "
        f"{100 * low_range[0]:+.2f}% إلى "
        f"{100 * low_range[1]:+.2f}% من الافتتاح\n\n"
        "⚠️ تقدير تاريخي وليس توقعًا مضمونًا أو نطاق ثقة "
        "معايرًا. لا يتضمن أخبار ما قبل الافتتاح أو بيانات "
        "ما قبل السوق."
    )
    if telegram(msg):
        state["daily_outlook_day"] = day
        save_state(state)

# -------------------- OPTIONS DATA --------------------
def fetch_contracts(direction, price):
    today = now_ny().date().isoformat()
    payload = alpaca_get(
        f"{TRADING_URL}/v2/options/contracts",
        {
            "underlying_symbols": SPY,
            "status": "active",
            "expiration_date": today,
            "type": "call" if direction == "CALL" else "put",
            "limit": 1000,
        },
    )
    contracts = (
        payload.get("option_contracts")
        or payload.get("contracts")
        or []
    )
    candidates = []
    for c in contracts:
        strike = num(c.get("strike_price"))
        symbol = c.get("symbol")
        expiration = str(c.get("expiration_date") or "")
        option_type = str(c.get("type") or "").lower()
        if not symbol or not math.isfinite(strike):
            continue
        if expiration and expiration[:10] != today:
            continue
        expected_type = "call" if direction == "CALL" else "put"
        if option_type and option_type != expected_type:
            continue
        distance = abs(strike - price)
        if distance <= MAX_STRIKE_DISTANCE:
            candidates.append((distance, c))
    candidates.sort(key=lambda item: item[0])
    return [c for _, c in candidates[:30]]

def _get_option_quote(symbol):
    payload = alpaca_get(
        f"{DATA_URL}/v1beta1/options/quotes/latest",
        {
            "symbols": symbol,
            "feed": OPTION_FEED,
        },
    )
    q = (payload.get("quotes") or {}).get(symbol, {})
    bid = num(q.get("bp"))
    ask = num(q.get("ap"))
    ts = parse_ts(q.get("t"))
    if ts is None:
        return None
    age = (
        pd.Timestamp.now(tz="UTC") - ts
    ).total_seconds()
    if age < -2 or age > MAX_QUOTE_AGE_SECONDS:
        return None
    if not math.isfinite(bid) or bid <= 0:
        return None
    if not math.isfinite(ask) or ask < bid:
        return None
    mid = (bid + ask) / 2
    if mid <= 0:
        return None
    return {
        "bid": bid,
        "ask": ask,
        "mid": mid,
        "time": ts.isoformat(),
        "age_seconds": age,
    }

def latest_option_quote(symbol):
    q = _get_option_quote(symbol)
    if not q:
        return None
    bid, ask, mid = q["bid"], q["ask"], q["mid"]
    if mid <= 0:
        return None
    if (ask - bid) / mid > MAX_SPREAD_PCT:
        return None
    if ask < MIN_PREMIUM or ask > MAX_PREMIUM:
        return None
    return q

def tracking_option_quote(symbol):
    return _get_option_quote(symbol)

def best_option(price, direction):
    try:
        contracts = fetch_contracts(direction, price)
        for c in contracts:
            symbol = c.get("symbol")
            if not symbol:
                continue
            q = latest_option_quote(symbol)
            if q is None:
                continue
            return {
                "symbol": symbol,
                "strike": num(c.get("strike_price")),
                "expiration": c.get("expiration_date"),
                **q,
            }
    except Exception:
        log.exception("Option scan failed")
    return None

# -------------------- JOURNAL & PAPER TRACKING --------------------
FIELDS = [
    "id", "signal_time", "signal", "spy_price", "probability",
    "option_symbol", "strike", "entry_bid", "entry_ask",
    "cost_per_contract", "max_bid", "min_bid", "last_bid",
    "max_pnl_per_contract", "min_pnl_per_contract",
    "last_pnl_per_contract", "status", "exit_time", "exit_bid",
    "exit_pnl_per_contract", "model_auc", "model_brier",
    "alert_status", "tracking_note",
]

def load_journal():
    if not JOURNAL.exists():
        return []
    try:
        with JOURNAL.open(
            "r", newline="", encoding="utf-8-sig"
        ) as f:
            return list(csv.DictReader(f))
    except Exception:
        log.exception("Journal read failed")
        return []

def save_journal(rows):
    tmp = JOURNAL.with_suffix(".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=FIELDS,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(JOURNAL)

def append_signal(pred, opt):
    rows = load_journal()
    day = now_ny().date().isoformat()
    today_count = 0
    for r in rows:
        ts = parse_ts(r.get("signal_time"))
        if ts is not None:
            if ts.to_pydatetime().astimezone(NY).date().isoformat() == day:
                today_count += 1
    if today_count >= MAX_DAILY_SIGNALS:
        log.info("Daily recommendation limit reached.")
        return False
    sid = (
        f"SF-{now_ny():%Y%m%d-%H%M%S}-"
        f"{int(time.time() * 1000) % 100000:05d}"
    )
    cost = round(float(opt["ask"]) * 100, 2)
    row = {
        "id": sid,
        "signal_time": stamp(),
        "signal": pred["signal"],
        "spy_price": f'{pred["current_price"]:.4f}',
        "probability": f'{pred["probability"]:.2f}',
        "option_symbol": opt["symbol"],
        "strike": opt["strike"],
        "entry_bid": opt["bid"],
        "entry_ask": opt["ask"],
        "cost_per_contract": cost,
        "max_bid": opt["bid"],
        "min_bid": opt["bid"],
        "last_bid": opt["bid"],
        "max_pnl_per_contract": 0.0,
        "min_pnl_per_contract": 0.0,
        "last_pnl_per_contract": 0.0,
        "status": "TRACKING",
        "exit_time": "",
        "exit_bid": "",
        "exit_pnl_per_contract": "",
        "model_auc": pred["metrics"]["auc"],
        "model_brier": pred["metrics"]["brier"],
        "alert_status": "PENDING",
        "tracking_note": "",
    }
    rows.append(row)
    save_journal(rows)
    message = (
        f"🎯 <b>{NAME} | توصية SPY 0DTE</b>\n"
        f"🆔 <code>{sid}</code>\n\n"
        f"الاتجاه: <b>{pred['signal']}</b>\n"
        f"سعر SPY وقت التحليل: "
        f"<b>${pred['current_price']:.2f}</b>\n"
        f"Strike: <b>${opt['strike']:.2f}</b>\n"
        f"العقد: <code>{opt['symbol']}</code>\n"
        f"Bid / Ask: ${opt['bid']:.2f} / ${opt['ask']:.2f}\n"
        f"السبريد النسبي: "
        f"{(opt['ask'] - opt['bid']) / opt['mid']:.1%}\n"
        f"تكلفة عقد واحد على Ask: <b>${cost:.2f}</b>\n"
        f"ثقة اتجاه النموذج غير المعايرة: "
        f"{pred['probability']:.1f}%\n"
        f"AUC الاختبار: {pred['metrics']['auc']:.3f}\n"
        f"Brier: {pred['metrics']['brier']:.3f}\n\n"
        "⚠️ تحقق من السعر والسبريد قبل أي قرار.\n"
        "هذه توصية فقط؛ لا ينفذ البوت أي أمر تداول.\n"
        "المتابعة افتراضية وليست ربحًا أو خسارة فعلية."
    )
    sent = telegram(message)
    rows = load_journal()
    for r in rows:
        if r.get("id") == sid:
            r["alert_status"] = "SENT" if sent else "FAILED"
            break
    save_journal(rows)
    if not sent:
        log.error(
            "Recommendation %s was recorded, but Telegram failed.",
            sid,
        )
    else:
        log.info("Recommendation and Telegram alert recorded: %s", sid)
    return True

def _finish_tracking(row, status, note="", bid=None):
    row["status"] = status
    row["exit_time"] = stamp()
    row["tracking_note"] = note
    if bid is not None and math.isfinite(num(bid)):
        entry = num(row.get("entry_ask"))
        if math.isfinite(entry):
            pnl = round((float(bid) - entry) * 100, 2)
            row["exit_bid"] = float(bid)
            row["exit_pnl_per_contract"] = pnl
        else:
            row["exit_bid"] = ""
            row["exit_pnl_per_contract"] = ""
    else:
        row["exit_bid"] = ""
        row["exit_pnl_per_contract"] = ""

def track_options():
    rows = load_journal()
    changed = False
    n = now_ny()
    for r in rows:
        if r.get("status") != "TRACKING":
            continue
        signal_ts = parse_ts(r.get("signal_time"))
        if signal_ts is None:
            _finish_tracking(
                r,
                "PAPER_REVIEW_REQUIRED",
                "Invalid signal timestamp; no P&L calculated.",
            )
            changed = True
            continue
        signal_day = (
            signal_ts.to_pydatetime().astimezone(NY).date()
        )
        if signal_day != n.date():
            _finish_tracking(
                r,
                "PAPER_EXPIRED_UNRESOLVED",
                "Previous session ended without a recorded valid exit quote.",
            )
            changed = True
            continue
        try:
            q = tracking_option_quote(r["option_symbol"])
        except Exception:
            log.exception(
                "Tracking quote failed for %s",
                r.get("option_symbol"),
            )
            q = None
        if q is None:
            if exit_time_reached(n):
                _finish_tracking(
                    r,
                    "PAPER_TIME_EXIT_NO_QUOTE",
                    "Exit time reached, but no valid fresh quote; P&L unresolved.",
                )
                changed = True
            continue
        bid = float(q["bid"])
        entry = num(r.get("entry_ask"))
        if not math.isfinite(entry) or entry <= 0:
            _finish_tracking(
                r,
                "PAPER_REVIEW_REQUIRED",
                "Invalid entry price; no P&L calculated.",
            )
            changed = True
            continue
        pnl = round((bid - entry) * 100, 2)
        old_max = num(r.get("max_bid"), bid)
        old_min = num(r.get("min_bid"), bid)
        r["last_bid"] = bid
        r["max_bid"] = max(old_max, bid)
        r["min_bid"] = min(old_min, bid)
        r["last_pnl_per_contract"] = pnl
        r["max_pnl_per_contract"] = round(
            (float(r["max_bid"]) - entry) * 100, 2
        )
        r["min_pnl_per_contract"] = round(
            (float(r["min_bid"]) - entry) * 100, 2
        )
        changed = True
        reason = None
        if bid >= entry * (1 + PAPER_TAKE_PROFIT_PCT):
            reason = "PAPER_TARGET"
        elif bid <= entry * (1 - PAPER_STOP_LOSS_PCT):
            reason = "PAPER_STOP"
        elif exit_time_reached(n):
            reason = "PAPER_TIME_EXIT"
        if reason:
            _finish_tracking(
                r,
                reason,
                "Hypothetical exit valued using the latest valid Bid.",
                bid=bid,
            )
            notice = (
                f"📒 <b>تحديث متابعة افتراضية</b>\n"
                f"🆔 <code>{r['id']}</code>\n"
                f"الحالة: <b>{reason}</b>\n"
                f"Bid الافتراضي للخروج: ${bid:.2f}\n"
                f"النتيجة الافتراضية لعقد واحد: ${pnl:.2f}\n\n"
                "لا تشمل هذه المحاكاة الرسوم أو الانزلاق، "
                "وليست صفقة فعلية أو ربحًا محققًا."
            )
            telegram(notice)
    if changed:
        save_journal(rows)

# -------------------- ALERTS & SCAN LOOP --------------------
def run_scan():
    if not in_signal_window():
        return
    bars = fetch_bars()
    if bars.empty:
        log.warning("No signal: no SPY bars returned.")
        return
    latest_bar = bars.index[-1]
    now_utc = pd.Timestamp.now(tz="UTC")
    age = (now_utc - latest_bar).total_seconds() / 60
    if age < -1 or age > MAX_BAR_AGE_MINUTES:
        log.warning(
            "No signal: SPY bar stale (%.1f minutes).",
            age,
        )
        return
    pred = train_predict(bars)
    if not pred.get("ok"):
        log.info(
            "No signal: %s | metrics=%s",
            pred.get("reason", "model unavailable"),
            pred.get("metrics", {}),
        )
        return
    pdir = (
        pred["prob_up"]
        if pred["signal"] == "CALL"
        else 1 - pred["prob_up"]
    )
    projected = pred["atr"] * math.sqrt(HORIZON) * 1.5
    state = read_state()
    today = now_ny().date().isoformat()
    last_time = parse_ts(state.get("last_signal_time"))
    cooldown = False
    if last_time is not None:
        elapsed = (
            pd.Timestamp.now(tz="UTC") - last_time
        ).total_seconds()
        cooldown = (
            0 <= elapsed < COOLDOWN_MINUTES * 60
        )
    if pdir < MIN_PROB:
        log.info(
            "No signal: direction confidence %.3f below %.3f",
            pdir,
            MIN_PROB,
        )
        return
    if projected >= MOVE_POINTS:
        move_key = f"move:{today}:{pred['signal']}"
        if state.get("last_move_key") != move_key:
            direction_ar = (
                "صعود" if pred["signal"] == "CALL" else "هبوط"
            )
            msg = (
                f"🚀 <b>{NAME} | تنبيه حركة تقديرية</b>\n"
                f"SPY: ${pred['current_price']:.2f}\n"
                f"الاتجاه النموذجي: <b>{direction_ar}</b>\n"
                f"تقدير الحركة الحسابي: {projected:.2f} نقطة\n"
                f"الأفق: {HORIZON * BAR_MINUTES} دقيقة\n"
                f"درجة اتجاه النموذج: {pdir * 100:.1f}%\n"
                f"AUC الاختبار: {pred['metrics']['auc']:.3f}\n\n"
                "⚠️ تقدير ATR تقريبي، وليس احتمالًا معايرًا "
                "لتحقق حركة بهذا الحجم."
            )
            if telegram(msg):
                state["last_move_key"] = move_key
                save_state(state)
    if cooldown:
        log.info("No new recommendation: cooldown active.")
        return
    signal_key = f'{pred["bar_time"]}:{pred["signal"]}'
    if state.get("last_signal_key") == signal_key:
        log.info("Duplicate signal skipped.")
        return
    opt = best_option(
        pred["current_price"],
        pred["signal"],
    )
    if not opt:
        log.info(
            "No recommendation: no contract passed entry quote filters."
        )
        return
    if append_signal(pred, opt):
        state["last_signal_key"] = signal_key
        state["last_signal_time"] = stamp()
        save_state(state)

def main():
    log.info(
        "%s v%s | SPY | recommendations only | no order execution",
        NAME,
        VERSION,
    )
    log.info(
        "Stock feed=%s | Options feed=%s | Min AUC=%.2f | "
        "Min directional confidence=%.2f",
        STOCK_FEED,
        OPTION_FEED,
        MIN_AUC,
        MIN_PROB,
    )
    if not KEY or not SECRET:
        raise SystemExit(
            "ضع APCA_API_KEY_ID و APCA_API_SECRET_KEY "
            "في متغيرات البيئة."
        )
    if not TG_ENABLED:
        log.warning(
            "Telegram غير مهيأ؛ سيعمل التحليل لكن لن تصل التنبيهات."
        )
    while True:
        try:
            n = now_ny()
            if (
                (n.hour, n.minute) >= (9, 0)
                and (n.hour, n.minute) < (9, 30)
            ):
                bars = fetch_bars(days=HISTORY_DAYS)
                if not bars.empty:
                    send_daily_outlook(bars)
            if market_open():
                track_options()
                run_scan()
        except KeyboardInterrupt:
            log.info("Stopped by user.")
            break
        except Exception:
            log.exception("Main loop error")
        time.sleep(max(5, POLL_SECONDS))

if __name__ == "__main__":
    main()
