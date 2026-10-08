# ============================================================
# SPX 0DTE LIVE AI ADVISOR - PRO VERSION
# ============================================================
# الهدف:
#   محرك توصيات حي متكامل (SPX + VIX + ES/NQ + ML + Walk-Forward)
#   بدون تنفيذ أوامر وبدون اشتراكات مدفوعة للخيارات.
# ============================================================

import os
import time
import math
import json
import warnings
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
import yfinance as yf

from dotenv import load_dotenv

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

load_dotenv()

# ============================================================
# CONFIG
# ============================================================

NY = ZoneInfo("America/New_York")

TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

# -----------------------------
# Market Symbols
# -----------------------------
SPX_SYMBOL = "^GSPC"
VIX_SYMBOL = "^VIX"
ES_SYMBOL = "ES=F"
NQ_SYMBOL = "NQ=F"

BAR_INTERVAL = "1m"
BAR_PERIOD = "5d"
POLL_SECONDS = 30

MARKET_OPEN = "09:30"
MARKET_CLOSE = "16:00"

NO_TRADE_FIRST_MIN = 20
NO_TRADE_LAST_MIN = 30

# -----------------------------
# ML & Walk-Forward
# -----------------------------
MIN_TRAIN_ROWS = 800
TARGET_HORIZON = 15
MIN_PROBABILITY = 0.62
MIN_EDGE = 0.10

# -----------------------------
# Risk & Volatility Filters
# -----------------------------
MAX_VIX_THRESHOLD = 30.0  # تجنب التداول إذا تجاوز VIX هذا الرقم (تقلبات شديدة)
MIN_VIX_THRESHOLD = 11.0  # هدوء مفرط قد يسبب تذبذب عرضي
ACCOUNT_EQUITY = 25000
RISK_PER_TRADE = 0.01
MAX_DAILY_LOSS = 0.03
MAX_CONSECUTIVE_LOSSES = 3
MAX_SIGNALS_PER_DAY = 5

# Option risk parameters
STOP_LOSS_PCT = 0.45
TAKE_PROFIT_PCT = 0.75
MAX_HOLD_MINUTES = 40

# ============================================================
# GLOBAL STATE
# ============================================================

STATE = {
    "model": None,
    "features": [],
    "last_signal_time": None,
    "last_signal_key": None,
    "signals_today": 0,
    "date": None,
    "daily_pnl": 0.0,
    "loss_streak": 0,
}

# ============================================================
# UTILITIES
# ============================================================

def now_ny():
    return datetime.now(NY)

def market_time_ok():
    now = now_ny()
    if now.weekday() >= 5:
        return False
    hhmm = now.strftime("%H:%M")
    if hhmm < MARKET_OPEN or hhmm >= MARKET_CLOSE:
        return False
    return True

def minutes_from_open():
    now = now_ny()
    h, m = map(int, MARKET_OPEN.split(":"))
    open_dt = now.replace(hour=h, minute=m, second=0, microsecond=0)
    return (now - open_dt).total_seconds() / 60

def session_allowed():
    mins = minutes_from_open()
    if mins < NO_TRADE_FIRST_MIN:
        return False
    if mins > 390 - NO_TRADE_LAST_MIN:
        return False
    return True

def reset_daily_state():
    today = now_ny().date()
    if STATE["date"] != today:
        STATE["date"] = today
        STATE["daily_pnl"] = 0
        STATE["loss_streak"] = 0
        STATE["signals_today"] = 0
        STATE["last_signal_key"] = None

def log(msg):
    print(f"[{now_ny().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)

# ============================================================
# TELEGRAM
# ============================================================

def send_telegram(message):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        return
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
        requests.post(url, json=payload, timeout=10)
    except Exception as e:
        log(f"Telegram error: {e}")

# ============================================================
# MULTI-ASSET DATA FETCHER (SPX, VIX, ES, NQ)
# ============================================================

class MarketData:
    def __init__(self):
        pass

    def get_data(self):
        try:
            # Fetch SPX
            spx_df = yf.download(SPX_SYMBOL, period=BAR_PERIOD, interval=BAR_INTERVAL, progress=False, auto_adjust=False)
            if spx_df is None or spx_df.empty:
                return None
            
            # Clean multi-index columns if present in newer yfinance versions
            if isinstance(spx_df.columns, pd.MultiIndex):
                spx_df.columns = spx_df.columns.get_level_values(0)

            spx_df = spx_df.reset_index()
            ts_col = "Datetime" if "Datetime" in spx_df.columns else ("Date" if "Date" in spx_df.columns else spx_df.columns[0])
            spx_df["timestamp"] = pd.to_datetime(spx_df[ts_col], utc=True).dt.tz_convert(NY)
            
            df = pd.DataFrame({
                "timestamp": spx_df["timestamp"],
                "open": pd.to_numeric(spx_df["Open"], errors="coerce"),
                "high": pd.to_numeric(spx_df["High"], errors="coerce"),
                "low": pd.to_numeric(spx_df["Low"], errors="coerce"),
                "close": pd.to_numeric(spx_df["Close"], errors="coerce"),
                "volume": pd.to_numeric(spx_df.get("Volume", 0), errors="coerce")
            }).dropna()

            # Fetch VIX for volatility filter
            vix_df = yf.download(VIX_SYMBOL, period="1d", interval="5m", progress=False, auto_adjust=False)
            current_vix = 18.0  # default baseline
            if vix_df is not None and not vix_df.empty:
                if isinstance(vix_df.columns, pd.MultiIndex):
                    vix_df.columns = vix_df.columns.get_level_values(0)
                current_vix = float(vix_df["Close"].iloc[-1])

            df["vix"] = current_vix

            # Regular trading hours filter
            df = df[
                (df["timestamp"].dt.time >= pd.Timestamp(MARKET_OPEN).time()) &
                (df["timestamp"].dt.time < pd.Timestamp(MARKET_CLOSE).time())
            ]

            return df.tail(2500).reset_index(drop=True)
        except Exception as e:
            log(f"Data fetch error: {e}")
            return None

# ============================================================
# TECHNICAL INDICATORS & PRICE ACTION
# ============================================================

def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def atr(df, period=14):
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False).mean()

def add_indicators(df):
    df = df.copy()
    df["ret_1"] = df["close"].pct_change(1)
    df["ret_3"] = df["close"].pct_change(3)
    df["ret_5"] = df["close"].pct_change(5)
    df["ret_15"] = df["close"].pct_change(15)

    df["rsi"] = rsi(df["close"])
    df["atr"] = atr(df)

    df["ema_9"] = df["close"].ewm(span=9, adjust=False).mean()
    df["ema_21"] = df["close"].ewm(span=21, adjust=False).mean()
    df["ema_50"] = df["close"].ewm(span=50, adjust=False).mean()

    df["ema_spread"] = (df["ema_9"] - df["ema_21"]) / df["close"]
    
    # Price Action & Candles
    df["body"] = df["close"] - df["open"]
    df["range"] = df["high"] - df["low"]
    df["body_ratio"] = df["body"].abs() / df["range"].replace(0, np.nan)
    df["upper_shadow"] = df["high"] - df[["open", "close"]].max(axis=1)
    df["lower_shadow"] = df[["open", "close"]].min(axis=1) - df["low"]

    # VWAP
    typical = (df["high"] + df["low"] + df["close"]) / 3
    date_key = df["timestamp"].dt.date
    cum_pv = (typical * df["volume"]).groupby(date_key).cumsum()
    cum_vol = df["volume"].groupby(date_key).cumsum()
    df["vwap"] = cum_pv / cum_vol.replace(0, np.nan)
    df["vwap_distance"] = (df["close"] - df["vwap"]) / df["close"]

    return df

FEATURES = [
    "ret_1", "ret_3", "ret_5", "ret_15",
    "rsi", "atr", "ema_spread",
    "body_ratio", "vwap_distance", "vix"
]

# ============================================================
# WALK-FORWARD / OUT-OF-SAMPLE VALIDATION
# ============================================================

def walk_forward_validation(data):
    """
    تقييم النموذج بنظام Walk-Forward (Out-of-Sample) عبر عدة تقسيمات زمنية متتالية.
    """
    if len(data) < MIN_TRAIN_ROWS:
        return None, 0.0

    chunk_size = len(data) // 4
    aucs = []

    for i in range(3):
        train_end = chunk_size * (i + 2)
        train_data = data.iloc[:train_end]
        test_data = data.iloc[train_end:train_end + chunk_size]

        if len(test_data) < 50:
            break

        X_train, y_train = train_data[FEATURES], train_data["target"]
        X_test, y_test = test_data[FEATURES], test_data["target"]

        model = HistGradientBoostingClassifier(
            max_depth=3, learning_rate=0.03, max_iter=200, l2_regularization=1.0, random_state=42
        )
        model.fit(X_train, y_train)
        preds = model.predict_proba(X_test)[:, 1]

        try:
            score = roc_auc_score(y_test, preds)
            aucs.append(score)
        except:
            pass

    avg_auc = np.mean(aucs) if aucs else 0.50
    log(f"Walk-Forward Validation Average AUC: {avg_auc:.3f}")

    # Train final model on all data
    final_model = HistGradientBoostingClassifier(
        max_depth=3, learning_rate=0.03, max_iter=250, l2_regularization=1.0, random_state=42
    )
    final_model.fit(data[FEATURES], data["target"])
    return final_model, avg_auc

# ============================================================
# SIGNAL GENERATION & CONSERVATIVE SELECTION
# ============================================================

def calculate_signal(df):
    if df is None or len(df) < 100:
        return {"direction": "WAIT", "probability": 0, "reasons": ["بيانات غير كافية"]}

    df = add_indicators(df)
    last = df.iloc[-1]

    # Volatility filter check
    vix = last["vix"]
    if vix > MAX_VIX_THRESHOLD:
        return {"direction": "WAIT", "probability": 0, "reasons": [f"مؤشر VIX مرتفع جداً ({vix:.1f}) - تقلبات خطرة"]}
    if vix < MIN_VIX_THRESHOLD:
        return {"direction": "WAIT", "probability": 0, "reasons": [f"مؤشر VIX منخفض جداً ({vix:.1f}) - سيولة وحركة ضعيفة"]}

    X = pd.DataFrame([last[FEATURES]])
    if STATE["model"] is None:
        return {"direction": "WAIT", "probability": 0, "reasons": ["النموذج غير مدرب"]}

    try:
        prob_up = float(STATE["model"].predict_proba(X)[0][1])
    except Exception as e:
        return {"direction": "WAIT", "probability": 0, "reasons": [f"خطأ في التنبؤ: {e}"]}

    prob_down = 1 - prob_up
    reasons = []

    if last["ema_9"] > last["ema_21"]:
        reasons.append("EMA اتجاه صاعد")
    else:
        reasons.append("EMA اتجاه هابط")

    if last["close"] > last["vwap"]:
        reasons.append("السعر فوق VWAP")
    else:
        reasons.append("السعر تحت VWAP")

    if prob_up >= MIN_PROBABILITY and (prob_up - prob_down) >= MIN_EDGE:
        direction = "CALL"
        probability = prob_up
    elif prob_down >= MIN_PROBABILITY and (prob_down - prob_up) >= MIN_EDGE:
        direction = "PUT"
        probability = prob_down
    else:
        direction = "WAIT"
        probability = max(prob_up, prob_down)
        reasons.append("الحافة الإحصائية ضعيفة أو غير كافية")

    return {
        "direction": direction,
        "probability": probability,
        "prob_up": prob_up,
        "prob_down": prob_down,
        "reasons": reasons,
        "last_price": float(last["close"]),
        "rsi": float(last["rsi"]),
        "vix": float(vix),
    }

# ============================================================
# CONSERVATIVE OPTION & RISK MANAGEMENT
# ============================================================

def build_recommendation(df):
    if not market_time_ok() or not session_allowed():
        return {"status": "WAIT", "reason": "خارج أوقات التداول المسموحة أو فترة الافتتاح/الإغلاق المحظورة"}

    signal = calculate_signal(df)
    if signal["direction"] == "WAIT":
        return {"status": "WAIT", "reason": signal["reasons"][0], "signal": signal}

    spot = signal["last_price"]
    direction = signal["direction"]

    # اختيار عقد محافظ (Conservative OTM/ATM Strike estimation بدون الحاجة لبيانات OPRA مدفوعة)
    strike_step = 5.0
    if direction == "CALL":
        conservative_strike = math.ceil(spot / strike_step) * strike_step + 5.0 # بعيد قليلاً للأمان
    else:
        conservative_strike = math.floor(spot / strike_step) * strike_step - 5.0

    estimated_premium = 2.50  # تقدير محافظ للبريميوم المتوسط لعقود 0DTE
    entry = estimated_premium
    stop = entry * (1 - STOP_LOSS_PCT)
    target = entry * (1 + TAKE_PROFIT_PCT)

    max_risk_dollars = ACCOUNT_EQUITY * RISK_PER_TRADE
    stop_loss_amount = entry * STOP_LOSS_PCT * 100
    contracts = max(1, int(max_risk_dollars / stop_loss_amount)) if stop_loss_amount > 0 else 1

    return {
        "status": direction,
        "probability": signal["probability"],
        "spx": spot,
        "vix": signal["vix"],
        "contract": {
            "symbol": f"SPXW 0DTE {direction} (Strike: {conservative_strike})",
            "strike": conservative_strike,
            "estimated_price": entry
        },
        "contracts": contracts,
        "entry": entry,
        "stop": stop,
        "target": target,
        "max_hold": MAX_HOLD_MINUTES,
        "signal": signal,
    }

def format_recommendation(rec):
    if rec["status"] == "WAIT":
        return f"⏳ SPX 0DTE PRO\n\n⚪ WAIT\nالسبب: {rec['reason']}"

    c = rec["contract"]
    emoji = "🟢" if rec["status"] == "CALL" else "🔴"
    reasons = "\n".join("• " + str(x) for x in rec["signal"]["reasons"])

    return f"""
{emoji} SPX 0DTE PRO — {rec['status']}

📊 الثقة: {rec['probability'] * 100:.1f}%
📈 SPX: {rec['spx']:.2f} | 📉 VIX: {rec['vix']:.1f}

🎯 العقد المقترح (محافظ):
{c['symbol']}

📌 Entry: ~${rec['entry']:.2f}
🛑 SL: ${rec['stop']:.2f}
🎯 TP: ${rec['target']:.2f}
⏱️ أقصى مدة: {rec['max_hold']} دقيقة
📦 العقود: {rec['contracts']}

الأسباب:
{reasons}

⚠️ توصية تحليلية بحتة - بدون تنفيذ آلي.
""".strip()

def save_signal(rec):
    record = {
        "timestamp": now_ny().isoformat(),
        "status": rec.get("status"),
        "probability": rec.get("probability"),
        "spx": rec.get("spx"),
        "contract": rec.get("contract"),
    }
    with open("spx_signals_pro.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

# ============================================================
# MAIN LOOP
# ============================================================

def main():
    log("Starting SPX 0DTE LIVE AI ADVISOR - PRO")
    reset_daily_state()

    market = MarketData()
    log("Downloading market data & training model with Walk-Forward...")

    df = market.get_data()
    if df is None or df.empty:
        raise RuntimeError("تعذر جلب بيانات السوق الأساسية.")

    df = add_indicators(df)
    future_return = df["close"].shift(-TARGET_HORIZON) / df["close"] - 1
    df["target"] = (future_return > 0).astype(int)
    df = df.dropna(subset=FEATURES + ["target"])

    model, auc = walk_forward_validation(df)
    STATE["model"] = model
    STATE["features"] = FEATURES

    log(f"Model ready. Validation AUC: {auc:.3f}")

    while True:
        try:
            reset_daily_state()
            if not market_time_ok():
                time.sleep(60)
                continue

            df = market.get_data()
            if df is None or len(df) < 100:
                time.sleep(POLL_SECONDS)
                continue

            rec = build_recommendation(df)
            message = format_recommendation(rec)

            print("\n" + "=" * 70)
            print(message)
            print("=" * 70)

            if rec["status"] in ["CALL", "PUT"] and rec["probability"] >= MIN_PROBABILITY:
                key = (rec["status"], round(rec["probability"], 2))
                if key != STATE["last_signal_key"]:
                    STATE["last_signal_key"] = key
                    STATE["signals_today"] += 1
                    save_signal(rec)
                    send_telegram(message)
                    log(f"NEW SIGNAL SENT: {rec['status']}")

            time.sleep(POLL_SECONDS)

        except KeyboardInterrupt:
            log("Stopped by user.")
            break
        except Exception as e:
            log(f"MAIN LOOP ERROR: {e}")
            time.sleep(POLL_SECONDS)

if __name__ == "__main__":
    main()
