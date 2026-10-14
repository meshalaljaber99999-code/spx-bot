# ============================================================
# صَفاء التحليل v25.4-Points-Speed-MegaCaps | Pre-Market & NY Session
# Pre-Market Reports • SPX Calls/Puts • 5-Point Spike • Open Premiums
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
import yfinance as yf

# -------------------- CONFIG --------------------
NAME, VERSION = "صَفاء التحليل", "25.4-Points-Speed-MegaCaps"
NY, UTC = ZoneInfo("America/New_York"), timezone.utc
SPY, TF = "SPY", "5Min"
BAR_MINUTES = 5

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

DATA_URL = os.getenv("ALPACA_DATA_URL", "https://data.alpaca.markets").rstrip("/")
TRADING_URL = os.getenv("ALPACA_TRADING_URL", "https://paper-api.alpaca.markets").rstrip("/")
KEY = (os.getenv("APCA_API_KEY_ID") or os.getenv("ALPACA_API_KEY") or "").strip()
SECRET = (os.getenv("APCA_API_SECRET_KEY") or os.getenv("ALPACA_SECRET_KEY") or "").strip()
STOCK_FEED = os.getenv("ALPACA_DATA_FEED", "iex").lower()
OPTION_FEED = os.getenv("OPTIONS_FEED", "indicative").lower()
TG_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TG_CHAT = os.getenv("TELEGRAM_CHAT_ID", "").strip()
TG_ENABLED = bool(TG_TOKEN and TG_CHAT)

HISTORY_DAYS = 120         
MIN_ROWS = 300
HORIZON = 6

MIN_PROB = 0.52            
MIN_AUC = 0.47             
MIN_SIGNAL_ACCURACY = 0.48 
MIN_VALIDATION_SIGNALS = 4 
MAX_BRIER = 0.32           
MOVE_POINTS = 3.0          
SPIKE_THRESHOLD_POINTS = 5.0 

MAX_STRIKE_DISTANCE = 10.0  
MAX_SPREAD_PCT = 0.35      
COOLDOWN_MINUTES = 7       

MAX_QUOTE_AGE_SECONDS = 90
MAX_BAR_AGE_MINUTES = 10
POLL_SECONDS = 30
PAPER_TAKE_PROFIT_PCT = 0.30
PAPER_STOP_LOSS_PCT = 0.25
PAPER_EXIT_TIME = "15:55"
MAX_DAILY_SIGNALS = 10     

ROOT = Path(os.getenv("DATA_DIR", "."))
ROOT.mkdir(parents=True, exist_ok=True)
JOURNAL = ROOT / "safa_signals_v254.csv"
STATE_FILE = ROOT / "safa_state_v254.json"

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
        raise RuntimeError("APCA_API_KEY_ID / APCA_API_SECRET_KEY غير مضبوطين")
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
                raise RuntimeError(f"Alpaca authentication/permission error: {r.status_code}")
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
            log.error("Telegram error %s: %s", r.status_code, r.text[:300])
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
        n.weekday() < 5
        and (n.hour, n.minute) >= (9, 40)
        and (n.hour, n.minute) < (15, 30)
    )

def is_pre_market_time():
    """تحديد وقت ما قبل السوق بساعة (مثلاً بين 8:30 و 8:35 صباحاً بتوقيت نيويورك)"""
    n = now_ny()
    return (
        n.weekday() < 5
        and n.hour == 8
        and 30 <= n.minute <= 35
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
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
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
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = (
        df.dropna(subset=["timestamp", "open", "high", "low", "close"])
        .drop_duplicates("timestamp", keep="last")
        .sort_values("timestamp")
        .set_index("timestamp")
    )
    now_utc = pd.Timestamp.now(tz="UTC")
    current_bucket_minute = (now_utc.minute // BAR_MINUTES) * BAR_MINUTES
    current_bucket = now_utc.replace(minute=current_bucket_minute, second=0, microsecond=0)
    df = df.loc[df.index < current_bucket]
    return df

def get_precision_synthetic_spx():
    leaders_data = {}
    weighted_dollar_move = 0.0
    total_weight = sum(TOP_LEADERS.values())

    try:
        tickers_list = list(TOP_LEADERS.keys())
        yf_data = yf.download(tickers_list, period="5d", interval="1d", progress=False)["Close"]
        
        for ticker, weight in TOP_LEADERS.items():
            try:
                if not yf_data.empty and ticker in yf_data:
                    series = yf_data[ticker].dropna()
                    if len(series) > 1:
                        last_close = float(series.iloc[-1])
                        prev_close = float(series.iloc[-2])
                        pct_change = (last_close / prev_close) - 1.0
                        
                        leaders_data[ticker] = {
                            "price": last_close,
                            "change_pct": pct_change * 100
                        }
                        weighted_dollar_move += pct_change * (weight / total_weight)
                        continue
            except Exception:
                pass
            
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
    except Exception:
        log.exception("Multi-source leaders fetch error")

    spy_bars = fetch_bars(SPY, days=2)
    current_spy = float(spy_bars["close"].iloc[-1]) if not spy_bars.empty else 500.0
    base_spx_estimate = current_spy * 10.0
    precision_synthetic_spx = base_spx_estimate * (1.0 + weighted_dollar_move)

    if weighted_dollar_move > 0.003:
        market_prediction = "اتجاه صاعد قوي ومؤشر SPX يستهدف الصعود 🚀 (CALL)"
    elif weighted_dollar_move > 0:
        market_prediction = "ميل طفيف للصعود واستقرار إيجابي 🟢"
    elif weighted_dollar_move < -0.003:
        market_prediction = "ضغوط بيعية واضحة ومؤشر SPX يتجه للهبوط 📉 (PUT)"
    else:
        market_prediction = "حركة عرضية وتماسك بانتظار العزم ⚪"

    return {
        "synthetic_spx": precision_synthetic_spx,
        "weighted_change": weighted_dollar_move * 100,
        "leaders": leaders_data,
        "market_prediction": market_prediction
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
def fetch_contracts(symbol, direction, price, strict_today=True):
    today = now_ny().date()
    params = {
        "underlying_symbols": symbol,
        "status": "active",
        "type": "call" if direction == "CALL" else "put",
        "limit": 500,
    }
    if strict_today:
        params["expiration_date"] = today.isoformat()
    
    payload = alpaca_get(f"{TRADING_URL}/v2/options/contracts", params)
    contracts = payload.get("option_contracts") or payload.get("contracts") or []
    candidates = []
    
    for c in contracts:
        strike = num(c.get("strike_price"))
        sym = c.get("symbol")
        expiration = str(c.get("expiration_date") or "")
        if not sym or not math.isfinite(strike):
            continue
        
        if not strict_today and expiration:
            exp_date = datetime.strptime(expiration[:10], "%Y-%m-%d").date()
            if exp_date < today or (exp_date - today).days > 35:
                continue
        elif strict_today and expiration and expiration[:10] != today.isoformat():
            continue
            
        distance = abs(strike - price)
        max_dist = MAX_STRIKE_DISTANCE if symbol == SPY else 25.0
        if distance <= max_dist:
            candidates.append((distance, c))
            
    candidates.sort(key=lambda item: item[0])
    return [c for _, c in candidates[:20]]

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

def best_option(symbol, price, direction, vol_ratio=1.0, strict_today=True):
    try:
        contracts = fetch_contracts(symbol, direction, price, strict_today=strict_today)
        for c in contracts:
            sym = c.get("symbol")
            if not sym:
                continue
            q = _get_option_quote(sym)
            if q is None:
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
def send_spike_alert(price_diff, current_price, synthetic_spx):
    direction = "صعوداً 🚀" if price_diff > 0 else "هبوطاً 📉"
    message = (
        f"⚡ <b>تنبيه حركة قوية (SPIKE ALERT)</b>\n"
        f"<i>تصميم أبو سحاب</i>\n\n"
        f"⚠️ تحرك مفاجئ في المؤشر بمقدار <b>{abs(price_diff):.2f} نقطة</b> {direction}!\n"
        f"📊 سعر SPY الحالي: <b>${current_price:.2f}</b>\n"
        f"📈 مؤشر SPX الاصطناعي: <b>${synthetic_spx:.2f}</b>\n\n"
        f"⏳ يرجى مراقبة السوق لاقتناص الفرصة اللحظية."
    )
    telegram(message)

def send_pre_market_report():
    """إرسال التقرير التحضيري قبل السوق بساعة (يشمل توقعات SPX وعقود الأوبشن للشركات والـ SPY)"""
    market_data = get_precision_synthetic_spx()
    synth_spx = market_data["synthetic_spx"]
    market_prediction = market_data["market_prediction"]
    
    spy_bars = fetch_bars(SPY, days=2)
    spy_price = float(spy_bars["close"].iloc[-1]) if not spy_bars.empty else 500.0

    # توقعات خيارات SPX قبل السوق
    spx_call_target = synth_spx + 15.0
    spx_put_target = synth_spx - 15.0

    # عقد SPY تقريبي قبل السوق
    spy_call_opt = best_option(SPY, spy_price, "CALL", strict_today=True)
    spy_put_opt = best_option(SPY, spy_price, "PUT", strict_today=True)

    leaders_text = ""
    for ticker, info in market_data["leaders"].items():
        dir_t = "CALL" if info["change_pct"] >= 0 else "PUT"
        best_opt_ldr = best_option(ticker, info["price"], dir_t, strict_today=False)
        strike_ldr = f"${best_opt_ldr['strike']}" if best_opt_ldr else "N/A"
        prem_ldr = f"${best_opt_ldr['ask']:.2f}" if best_opt_ldr else "N/A"
        exp_date = best_opt_ldr["expiration"] if best_opt_ldr else "غير متوفر"
        
        leaders_text += (
            f"🔹 <b>{ticker}</b>: ${info['price']:.2f} ({info['change_pct']:+.2f}%)\n"
            f"   └ توقع العقد: <b>{dir_t}</b> | Strike: {strike_ldr} | السعر: {prem_ldr} | ينتهي: <code>{exp_date}</code>\n"
        )

    spy_c_sym = spy_call_opt['symbol'] if spy_call_opt else "N/A"
    spy_p_sym = spy_put_opt['symbol'] if spy_put_opt else "N/A"

    message = (
        f"⏰ <b>{NAME} | تقرير ما قبل السوق (تحضيري)</b>\n"
        f"<i>تصميم أبو سحاب</i>\n\n"
        f"📊 <b>مؤشر SPY:</b> ${spy_price:.2f}\n"
        f"📈 <b>مؤشر SPX الاصطناعي:</b> ${synth_spx:.2f} ({market_data['weighted_change']:+.2f}%)\n"
        f" <b>توقع اتجاه المؤشر:</b> {market_prediction}\n\n"
        f"🎯 <b>توقعات خيارات مؤشر SPX (كول & بت) قبل الافتتاح:</b>\n"
        f"🟢 <b>SPX CALL (صعود):</b> مستهدف اختراق قرب <code>{spx_call_target:.2f}</code>\n"
        f"🔴 <b>SPX PUT (هبوط):</b> مستهدف كسر قرب <code>{spx_put_target:.2f}</code>\n\n"
        f"🎯 <b>توقعات عقود SPY (0DTE):</b>\n"
        f"▫️ كول مقترح: <code>{spy_c_sym}</code>\n"
        f"▫️ بت مقترح: <code>{spy_p_sym}</code>\n\n"
        f"⭐ <b>عقود الشركات القيادية المقترحة:</b>\n"
        f"{leaders_text}\n"
        f"⚠️ تقرير استباقي قبل الافتتاح - تصميم أبو سحاب."
    )
    
    sent = telegram(message)
    if sent:
        log.info("Pre-market report sent successfully.")

def append_and_send_alert(pred, opt_spy, market_data):
    sid = f"SF-{now_ny():%Y%m%d-%H%M%S}"
    cost = round(float(opt_spy["ask"]) * 100, 2)
    
    expected_points = pred["atr"] * math.sqrt(HORIZON) * 1.5
    vol_ratio = pred["vol_ratio"]
    
    if vol_ratio >= 2.0:
        speed_text = "⚡ صاروخية وعنيفة جداً (High Spike Velocity)"
    elif vol_ratio >= 1.4:
        speed_text = "🚀 سريعة وقوية (Strong Momentum)"
    else:
        speed_text = "🌊 متدرجة وثابتة (Steady Trend)"

    synth_spx = market_data["synthetic_spx"]
    spx_call_target = synth_spx + (expected_points * 10)
    spx_put_target = synth_spx - (expected_points * 10)

    leaders_text = ""
    for ticker, info in market_data["leaders"].items():
        dir_t = "CALL" if info["change_pct"] >= 0 else "PUT"
        best_opt_ldr = best_option(ticker, info["price"], dir_t, strict_today=False)
        
        if best_opt_ldr:
            exp_date = best_opt_ldr["expiration"]
            strike_ldr = f"${best_opt_ldr['strike']}"
            prem_ldr = f"${best_opt_ldr['ask']:.2f}"
        else:
            exp_date = "غير متوفر"
            strike_ldr = "N/A"
            prem_ldr = "N/A"
        
        leaders_text += (
            f"🔹 <b>{ticker}</b>: ${info['price']:.2f} ({info['change_pct']:+.2f}%)\n"
            f"   └ التوقع: <b>{dir_t}</b> | Strike: {strike_ldr} | السعر: {prem_ldr} | ينتهي: <code>{exp_date}</code>\n"
        )

    direction_ar = "صعود 🚀 (CALL)" if pred['signal'] == "CALL" else "هبوط 📉 (PUT)"
    synth_chg = market_data["weighted_change"]
    market_prediction = market_data["market_prediction"]

    message = (
        f"🎯 <b>{NAME} | التقرير الشامل وعقود السوق</b>\n"
        f"<i>تصميم أبو سحاب</i>\n"
        f"🆔 <code>{sid}</code>\n\n"
        f"📊 <b>مؤشر SPY الأساسي (0DTE):</b> ${pred['current_price']:.2f}\n"
        f"📈 <b>مؤشر SPX الاصطناعي:</b> ${synth_spx:.2f} ({synth_chg:+.2f}%)\n"
        f" <b>توقع حركة المؤشر أثناء السوق:</b> {market_prediction}\n\n"
        f"🎯 <b>توقعات خيارات مؤشر SPX (كول & بت):</b>\n"
        f"🟢 <b>SPX CALL (صعود):</b> مستهدف اختراق قرب <code>{spx_call_target:.2f}</code>\n"
        f"🔴 <b>SPX PUT (هبوط):</b> مستهدف كسر قرب <code>{spx_put_target:.2f}</code>\n\n"
        f" اتجاه نموذج البوت الذكي: <b>{direction_ar}</b>\n"
        f" النطاق المتوقع للحركة: <b>±{expected_points:.2f} نقطة</b>\n"
        f" سرعة الزخم: <b>{speed_text}</b> (الثقة: {pred['probability']:.1f}%)\n\n"
        f"🎯 <b>عقد SPY الموصى به (0DTE - سعر مفتوح):</b>\n"
        f"▫️ الرمز: <code>{opt_spy['symbol']}</code>\n"
        f"▫️ Strike: ${opt_spy['strike']:.2f} | ينتهي في: <code>{opt_spy['expiration']}</code>\n"
        f"▫️ التكلفة التقديرية لعقد: <b>${cost:.2f}</b>\n\n"
        f"⭐ <b>تحليل وعقود الشركات القيادية (عقود مرنة ومربحة):</b>\n"
        f"{leaders_text}\n"
        f"⚠️ تنبيه افتراضي تحليلي - تصميم أبو سحاب."
    )
    
    sent = telegram(message)
    if sent:
        log.info("Comprehensive alert sent successfully: %s", sid)

# -------------------- MAIN SCAN LOOP --------------------
def run_scan():
    state = read_state()
    current_time_str = now_ny().strftime("%Y-%m-%d %H:%M")

    # 1. إرسال تقرير ما قبل السوق (مرة واحدة يومياً في وقته)
    if is_pre_market_time():
        last_pre_market = state.get("last_pre_market_date", "")
        today_date_str = now_ny().strftime("%Y-%m-%d")
        if last_pre_market != today_date_str:
            send_pre_market_report()
            state["last_pre_market_date"] = today_date_str
            save_state(state)

    # 2. العمل أثناء الجلسة الرسمية
    if not in_signal_window():
        return

    bars = fetch_bars()
    if bars.empty:
        return
    
    current_price = float(bars["close"].iloc[-1])
    prev_price = float(bars["close"].iloc[-2]) if len(bars) > 1 else current_price
    price_diff = current_price - prev_price

    market_data = get_precision_synthetic_spx()

    last_spike_check_time = state.get("last_spike_time", "")
    if abs(price_diff) >= SPIKE_THRESHOLD_POINTS and last_spike_check_time != current_time_str:
        send_spike_alert(price_diff, current_price, market_data["synthetic_spx"])
        state["last_spike_time"] = current_time_str
        save_state(state)

    pred = train_predict(bars)
    if not pred.get("ok"):
        return

    signal_key = f'{pred["bar_time"]}:{pred["signal"]}'
    if state.get("last_signal_key") == signal_key:
        return

    opt_spy = best_option(SPY, pred["current_price"], pred["signal"], vol_ratio=pred["vol_ratio"], strict_today=True)
    if not opt_spy:
        return

    append_and_send_alert(pred, opt_spy, market_data)
    state["last_signal_key"] = signal_key
    state["last_signal_time"] = stamp()
    save_state(state)

def main():
    log.info("%s v%s | Pre-Market Reports, NY Session, 5-Point Spike & SPX Calls/Puts", NAME, VERSION)
    if not KEY or not SECRET:
        raise SystemExit("ضع مفاتيح ألقا في متغيرات البيئة.")
    while True:
        try:
            if market_open() or is_pre_market_time():
                run_scan()
        except KeyboardInterrupt:
            log.info("Stopped by user.")
            break
        except Exception:
            log.exception("Main loop error")
        time.sleep(max(5, POLL_SECONDS))

if __name__ == "__main__":
    main()
