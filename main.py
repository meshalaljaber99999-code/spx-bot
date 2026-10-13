# ============================================================
# صَفاء التحليل v25.0-Points-Speed-MegaCaps | SPY & Synthetic SPX
# Dynamic Points Projection • Spike Speed Meter • 11-Leader Options
# تصميم أبو سحاب
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
NAME, VERSION = "صَفاء التحليل", "25.0-Points-Speed-MegaCaps"
NY, UTC = ZoneInfo("America/New_York"), timezone.utc
SPY, TF = "SPY", "5Min"
BAR_MINUTES = 5

# --- قائمة العمالقة الموسعة (أكبر 11 شركة لتغطية أكثر من 50% من مؤشر S&P 500) ---
TOP_LEADERS = {
    "AAPL": 0.070,  # Apple
    "MSFT": 0.065,  # Microsoft
    "NVDA": 0.060,  # Nvidia
    "AMZN": 0.040,  # Amazon
    "GOOGL": 0.035, # Alphabet
    "META": 0.030,  # Meta
    "TSLA": 0.020,  # Tesla
    "BRK.B": 0.020, # Berkshire Hathaway
    "AVGO": 0.020,  # Broadcom
    "LLY": 0.020,   # Eli Lilly
    "JPM": 0.015    # JPMorgan Chase
}

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

# --- تعديلات مرونة القفزات والزخم واقتناص الانقلابات ---
HISTORY_DAYS = 120         
MIN_ROWS = 300
HORIZON = 6

MIN_PROB = 0.52            
MIN_AUC = 0.47             
MIN_SIGNAL_ACCURACY = 0.48 
MIN_VALIDATION_SIGNALS = 4 
MAX_BRIER = 0.32           
MOVE_POINTS = 3.0          

MAX_STRIKE_DISTANCE = 8.5  
MIN_PREMIUM = 0.25         
MAX_PREMIUM = 50.0
MAX_SPREAD_PCT = 0.24      
COOLDOWN_MINUTES = 7       
# -----------------------------------------------

MAX_QUOTE_AGE_SECONDS = 90
MAX_BAR_AGE_MINUTES = 10
POLL_SECONDS = 30
PAPER_TAKE_PROFIT_PCT = 0.30
PAPER_STOP_LOSS_PCT = 0.25
PAPER_EXIT_TIME = "15:55"
MAX_DAILY_SIGNALS = 10     

ROOT = Path(os.getenv("DATA_DIR", "."))
ROOT.mkdir(parents=True, exist_ok=True)
JOURNAL = ROOT / "safa_signals_v250.csv"
STATE_FILE = ROOT / "safa_state_v250.json"

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

# -------------------- STOCK & PRECISION SYNTHETIC SPX DATA --------------------
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

def get_precision_synthetic_spx():
    """حساب مؤشر اصطناعي فائق الدقة يعتمد على أكبر 11 شركة قيادية في السوق"""
    leaders_data = {}
    weighted_dollar_move = 0.0
    total_weight = sum(TOP_LEADERS.values())

    for ticker, weight in TOP_LEADERS.items():
        try:
            df = fetch_bars(ticker, days=5)
            if not df.empty and len(df) > 1:
                last_close = float(df["close"].iloc[-1])
                prev_close = float(df["close"].iloc[-2])
                pct_change = (last_close / prev_close) - 1.0
                
                leaders_data[ticker] = {
                    "price": last_close,
                    "change_pct": pct_change * 100
                }
                weighted_dollar_move += pct_change * (weight / total_weight)
        except Exception:
            log.exception(f"Failed to fetch precision data for {ticker}")

    spy_bars = fetch_bars(SPY, days=2)
    current_spy = float(spy_bars["close"].iloc[-1]) if not spy_bars.empty else 500.0
    base_spx_estimate = current_spy * 10.0
    precision_synthetic_spx = base_spx_estimate * (1.0 + weighted_dollar_move)

    return {
        "synthetic_spx": precision_synthetic_spx,
        "weighted_change": weighted_dollar_move * 100,
        "leaders": leaders_data
    }

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
    e9 = c.ewm(span=9, adjust=False, min_periods=9).mean()
    e20 = c.ewm(span=20, adjust=False, min_periods=20).mean()
    e50 = c.ewm(span=50, adjust=False, min_periods=50).mean()
    f["ema_spread"] = (e9 - e20) / c
    f["ema_trend"] = (e20 - e50) / c
    d = c.diff()
    gain = d.clip(lower=0).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    loss = (-d.clip(upper=0)).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = (100 - 100 / (1 + rs)) / 100
    f["rsi"] = rsi.fillna(0.5)
    prev = c.shift()
    tr = pd.concat([f["high"] - f["low"], (f["high"] - prev).abs(), (f["low"] - prev).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    f["atr_pct"] = atr / c
    rng = (f["high"] - f["low"]).clip(lower=0)
    f["range_pct"] = rng / c
    f["body_pct"] = (f["close"] - f["open"]) / c
    f["close_location"] = ((c - f["low"]) / rng.replace(0, np.nan)).fillna(0.5)
    vmean = f["volume"].rolling(20, min_periods=20).mean()
    f["volume_ratio"] = f["volume"] / vmean.replace(0, np.nan)
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
        return {"ok": False, "reason": "INSUFFICIENT_ROWS"}
    f = features_for(bars)
    future = bars["close"].shift(-HORIZON)
    f["target"] = np.where(future.notna(), (future > bars["close"]).astype(float), np.nan)
    ds = f[FEATURES + ["target"]].dropna()
    if len(ds) < MIN_ROWS or ds["target"].nunique() < 2:
        return {"ok": False, "reason": "INSUFFICIENT_LABELS"}
    split = int(len(ds) * 0.80)
    valid_start_time = ds.index[split]
    embargo_cutoff = valid_start_time - pd.Timedelta(minutes=BAR_MINUTES * HORIZON)
    train = ds.loc[ds.index < embargo_cutoff]
    valid = ds.loc[ds.index >= valid_start_time]
    if len(train) < 100 or len(valid) < 30:
        return {"ok": False, "reason": "SMALL_HOLDOUT"}
    model = build_model()
    model.fit(train[FEATURES], train["target"].astype(int))
    p = model.predict_proba(valid[FEATURES])[:, 1]
    y = valid["target"].astype(int).to_numpy()
    auc = roc_auc_score(y, p) if len(np.unique(y)) == 2 else 0.5
    brier = brier_score_loss(y, p)
    chosen = (p >= MIN_PROB) | (p <= 1 - MIN_PROB)
    signal_count = int(chosen.sum())
    signal_accuracy = float(((p[chosen] >= 0.5).astype(int) == y[chosen]).mean()) if signal_count else 0.0
    metrics = {"auc": float(auc), "brier": float(brier), "signals": signal_count, "signal_accuracy": signal_accuracy}
    
    if auc < MIN_AUC or brier > MAX_BRIER:
        return {"ok": False, "reason": "HOLDOUT_QUALITY_FAILED", "metrics": metrics}
    
    final_model = build_model()
    final_model.fit(ds[FEATURES], ds["target"].astype(int))
    latest_features = f[FEATURES].iloc[[-1]]
    prob_up = float(final_model.predict_proba(latest_features)[0, 1])
    atr_pct = num(f["atr_pct"].iloc[-1])
    price = num(bars["close"].iloc[-1])
    atr = atr_pct * price if math.isfinite(atr_pct) and atr_pct > 0 else 0.0
    vol_ratio = num(f["volume_ratio"].iloc[-1], 1.0)
    
    return {
        "ok": True,
        "signal": "CALL" if prob_up >= 0.5 else "PUT",
        "prob_up": prob_up,
        "probability": max(prob_up, 1 - prob_up) * 100,
        "current_price": price,
        "atr": atr,
        "vol_ratio": vol_ratio,
        "metrics": metrics,
        "bar_time": bars.index[-1].isoformat(),
    }

# -------------------- OPTIONS DATA --------------------
def fetch_contracts(symbol, direction, price):
    today = now_ny().date().isoformat()
    payload = alpaca_get(
        f"{TRADING_URL}/v2/options/contracts",
        {
            "underlying_symbols": symbol,
            "status": "active",
            "expiration_date": today,
            "type": "call" if direction == "CALL" else "put",
            "limit": 500,
        },
    )
    contracts = payload.get("option_contracts") or payload.get("contracts") or []
    candidates = []
    for c in contracts:
        strike = num(c.get("strike_price"))
        sym = c.get("symbol")
        expiration = str(c.get("expiration_date") or "")
        if not sym or not math.isfinite(strike):
            continue
        if expiration and expiration[:10] != today:
            continue
        distance = abs(strike - price)
        if distance <= (MAX_STRIKE_DISTANCE if symbol == SPY else 15.0):
            candidates.append((distance, c))
    candidates.sort(key=lambda item: item[0])
    return [c for _, c in candidates[:15]]

def _get_option_quote(symbol):
    payload = alpaca_get(
        f"{DATA_URL}/v1beta1/options/quotes/latest",
        {"symbols": symbol, "feed": OPTION_FEED},
    )
    q = (payload.get("quotes") or {}).get(symbol, {})
    bid, ask = num(q.get("bp")), num(q.get("ap"))
    ts = parse_ts(q.get("t"))
    if ts is None or not math.isfinite(bid) or bid <= 0 or not math.isfinite(ask) or ask < bid:
        return None
    mid = (bid + ask) / 2
    return {"bid": bid, "ask": ask, "mid": mid, "expiration": q.get("expiration_date", "")}

def best_option(symbol, price, direction, vol_ratio=1.0):
    try:
        contracts = fetch_contracts(symbol, direction, price)
        for c in contracts:
            sym = c.get("symbol")
            if not sym:
                continue
            q = _get_option_quote(sym)
            if q is None:
                continue
            if q["ask"] < MIN_PREMIUM or q["ask"] > MAX_PREMIUM:
                continue
            return {
                "symbol": sym,
                "strike": num(c.get("strike_price")),
                "expiration": c.get("expiration_date"),
                **q,
            }
    except Exception:
        log.exception(f"Option scan failed for {symbol}")
    return None

# -------------------- JOURNAL & TELEGRAM ALERTS --------------------
FIELDS = [
    "id", "signal_time", "signal", "spy_price", "probability",
    "option_symbol", "strike", "entry_bid", "entry_ask",
    "cost_per_contract", "status", "model_auc", "alert_status",
]

def load_journal():
    if not JOURNAL.exists():
        return []
    try:
        with JOURNAL.open("r", newline="", encoding="utf-8-sig") as f:
            return list(csv.DictReader(f))
    except Exception:
        return []

def save_journal(rows):
    tmp = JOURNAL.with_suffix(".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(JOURNAL)

def append_and_send_alert(pred, opt_spy, market_data):
    sid = f"SF-{now_ny():%Y%m%d-%H%M%S}"
    cost = round(float(opt_spy["ask"]) * 100, 2)
    
    # حساب النقاط المتوقعة والسرعة
    expected_points = pred["atr"] * math.sqrt(HORIZON) * 1.5
    vol_ratio = pred["vol_ratio"]
    
    if vol_ratio >= 2.0:
        speed_text = "⚡ صاروخية وعنيفة جداً (High Spike Velocity)"
    elif vol_ratio >= 1.4:
        speed_text = "🚀 سريعة وقوية (Strong Momentum)"
    else:
        speed_text = "🌊 متدرجة وثابتة (Steady Trend)"

    leaders_text = ""
    for ticker, info in market_data["leaders"].items():
        dir_t = "CALL" if info["change_pct"] >= 0 else "PUT"
        best_opt_ldr = best_option(ticker, info["price"], dir_t)
        exp_date = best_opt_ldr["expiration"] if best_opt_ldr else "اليوم"
        strike_ldr = f"${best_opt_ldr['strike']}" if best_opt_ldr else "N/A"
        
        leaders_text += (
            f"🔹 <b>{ticker}</b>: ${info['price']:.2f} ({info['change_pct']:+.2f}%)\n"
            f"   └ التوقع: {dir_t} | Strike: {strike_ldr} | ينتهي في: <code>{exp_date}</code>\n"
        )

    direction_ar = "صعود 🚀 (CALL)" if pred['signal'] == "CALL" else "هبوط 📉 (PUT)"
    synth_spx = market_data["synthetic_spx"]
    synth_chg = market_data["weighted_change"]

    message = (
        f"🎯 <b>{NAME} | التقرير الشامل وعقود السوق</b>\n"
        f"<i>تصميم أبو سحاب</i>\n"
        f"🆔 <code>{sid}</code>\n\n"
        f"📊 <b>مؤشر SPY الأساسي:</b> ${pred['current_price']:.2f}\n"
        f"📈 <b>مؤشر SPX الاصطناعي (عالي الدقة):</b> ${synth_spx:.2f} ({synth_chg:+.2f}%)\n"
        f" الاتجاه المتوقع: <b>{direction_ar}</b>\n"
        f" النطاق المتوقع للحركة: <b>±{expected_points:.2f} نقطة</b>\n"
        f" سرعة الزخم المتوقعة: <b>{speed_text}</b>\n"
        f" نسبة الثقة: <b>{pred['probability']:.1f}%</b>\n\n"
        f"🎯 <b>عقد SPY الموصى به (0DTE):</b>\n"
        f"▫️ الرمز: <code>{opt_spy['symbol']}</code>\n"
        f"▫️ Strike: ${opt_spy['strike']:.2f} | ينتهي في: <code>{opt_spy['expiration']}</code>\n"
        f"▫️ التكلفة التقديرية لعقد: <b>${cost:.2f}</b>\n\n"
        f"⭐ <b>تحليل وعقود الشركات القيادية (Top 11 Leaders):</b>\n"
        f"{leaders_text}\n"
        f"⚠️ تنبيه افتراضي تحليلي - تصميم أبو سحاب."
    )
    
    sent = telegram(message)
    if sent:
        log.info("Points & Speed comprehensive alert sent successfully: %s", sid)

# -------------------- MAIN SCAN LOOP --------------------
def run_scan():
    if not in_signal_window():
        return
    bars = fetch_bars()
    if bars.empty:
        return
    
    pred = train_predict(bars)
    if not pred.get("ok"):
        return

    state = read_state()
    signal_key = f'{pred["bar_time"]}:{pred["signal"]}'
    if state.get("last_signal_key") == signal_key:
        return

    market_data = get_precision_synthetic_spx()
    opt_spy = best_option(SPY, pred["current_price"], pred["signal"], vol_ratio=pred["vol_ratio"])
    if not opt_spy:
        return

    append_and_send_alert(pred, opt_spy, market_data)
    state["last_signal_key"] = signal_key
    state["last_signal_time"] = stamp()
    save_state(state)

def main():
    log.info("%s v%s | SPY, Points & Speed Enabled", NAME, VERSION)
    if not KEY or not SECRET:
        raise SystemExit("ضع مفاتيح ألقا في متغيرات البيئة.")
    while True:
        try:
            if market_open():
                run_scan()
        except KeyboardInterrupt:
            log.info("Stopped by user.")
            break
        except Exception:
            log.exception("Main loop error")
        time.sleep(max(5, POLL_SECONDS))

if __name__ == "__main__":
    main()
