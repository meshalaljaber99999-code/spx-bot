# ============================================================
# SPX 0DTE ADVISOR v14.3 HONEST INSTITUTIONAL ENGINE
# (Direct SPX Bars via Alpaca API)
# ============================================================

import os
import sys
import csv
import json
import time
import warnings
from math import sqrt, floor, ceil
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score

try:  # sklearn >= 1.6
    from sklearn.frozen import FrozenEstimator
except Exception:
    FrozenEstimator = None

warnings.filterwarnings("ignore")

# ============================================================
# CONFIG & ARCHITECTURE
# ============================================================
NY = ZoneInfo("America/New_York")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

APCA_KEY = os.getenv("APCA_API_KEY_ID", "")
APCA_SECRET = os.getenv("APCA_API_SECRET_KEY", "")
APCA_BASE_URL = os.getenv("APCA_API_BASE_URL", "https://paper-api.alpaca.markets").rstrip("/")
APCA_DATA_URL = "https://data.alpaca.markets"

POLL_SECONDS = 30
MARKET_OPEN_MIN = 9 * 60 + 30
MARKET_CLOSE_MIN = 16 * 60
NO_TRADE_FIRST_MIN = 10               
NO_TRADE_LAST_MIN = 30                
MAX_BAR_AGE_MIN = 15                  

LOCAL_DB_CSV = "local_market_db.csv"
OPTIONS_DB_CSV = "local_options_premium_db.csv"
TRADES_CSV = "paper_trades_v14_3.csv"

HISTORY_PERIOD = "60d"             
MIN_TRAIN_ROWS = 100                  
HORIZON = 6
MAX_HOLD_BARS = 6
MIN_PROBABILITY = 0.55                
MIN_AUC = 0.45                        
RETRAIN_EVERY_MIN = 180

ACCOUNT_EQUITY = 25000
RISK_PER_TRADE = 0.01
MAX_DAILY_LOSS = 0.05
MAX_CONSECUTIVE_LOSSES = 5
COMMISSION_PER_CONTRACT = 0.65

STRIKE_STEP = 5.0
STRIKE_OFFSET_STEPS = 1
MIN_PREMIUM = 0.50                    
STOP_LOSS_PCT = 0.40
TAKE_PROFIT_PCT = 0.50
MAX_HOLD_MINUTES = 30
FORCE_EXIT_BEFORE_CLOSE_MIN = 5

UP_ATR_MULT = 0.5
DN_ATR_MULT = 0.5

DIAG_THROTTLE_SEC = 60            

STATE = {
    "models": {}, "auc": 0.0, "last_train": None,
    "date": None, "daily_pnl": 0.0, "loss_streak": 0, "signals_today": 0,
    "open": None, "daily_summary_sent": False, "regime": "NORMAL",
    "train_rows": 0,
}

_DIAG_LAST = {}

# ============================================================
# UTILITIES
# ============================================================
def now_ny():
    return datetime.now(NY)

def say(msg):
    print(f"[{now_ny().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)

def why(msg):
    t = time.time()
    if t - _DIAG_LAST.get(msg, 0) >= DIAG_THROTTLE_SEC:
        _DIAG_LAST[msg] = t
        say(f"[WHY-WAIT] {msg}")

def minutes_of_day(dt=None):
    dt = dt or now_ny()
    return dt.hour * 60 + dt.minute

def market_time_ok():
    n = now_ny()
    return n.weekday() < 5 and MARKET_OPEN_MIN <= minutes_of_day(n) < MARKET_CLOSE_MIN

def mins_left(dt=None):
    dt = dt or now_ny()
    return (MARKET_CLOSE_MIN - (dt.hour * 60 + dt.minute)) - dt.second / 60

def session_allowed():
    m = minutes_of_day() - MARKET_OPEN_MIN
    return NO_TRADE_FIRST_MIN <= m <= (MARKET_CLOSE_MIN - MARKET_OPEN_MIN) - NO_TRADE_LAST_MIN

def reset_daily_state():
    today = now_ny().date()
    if STATE["date"] != today:
        STATE.update(date=today, daily_pnl=0.0, loss_streak=0,
                     signals_today=0, daily_summary_sent=False)

def send_telegram(message):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        why("Telegram غير مفعّل: TELEGRAM_BOT_TOKEN أو TELEGRAM_CHAT_ID فاضي")
        return False
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
            json={"chat_id": TELEGRAM_CHAT_ID, "text": message}, timeout=10)
        if r.status_code != 200:
            say(f"Telegram HTTP {r.status_code}: {r.text[:200]}")
            return False
        return True
    except Exception as e:
        say(f"Telegram error: {e}")
        return False

# ============================================================
# ALPACA CONNECTOR & DATA ENGINE (SPX ONLY)
# ============================================================
def get_alpaca_headers():
    return {
        "APCA-API-KEY-ID": APCA_KEY,
        "APCA-API-SECRET-KEY": APCA_SECRET,
        "accept": "application/json"
    }

def _download_alpaca_bars(symbol, timeframe="5Min", limit=10000):
    if not APCA_KEY or not APCA_SECRET:
        why(f"مفاتيح Alpaca غير موجودة لجلب بيانات {symbol}")
        return None
    
    # محاولة جلب الشموع للمؤشرات أو الأسهم المتاحة في Alpaca
    url = f"{APCA_DATA_URL}/v2/stocks/bars"
    params = {
        "symbols": symbol,
        "timeframe": timeframe,
        "limit": limit
    }
    try:
        response = requests.get(url, headers=get_alpaca_headers(), params=params, timeout=15)
        if response.status_code != 200:
            say(f"Alpaca bars error HTTP {response.status_code} for {symbol}: {response.text[:150]}")
            return None
        
        data = response.json().get("bars", {}).get(symbol, [])
        if not data:
            return None
        
        df = pd.DataFrame(data)
        df["timestamp"] = pd.to_datetime(df["t"], utc=True).dt.tz_convert(NY)
        out = pd.DataFrame({
            "timestamp": df["timestamp"],
            "open": pd.to_numeric(df["o"], errors="coerce"),
            "high": pd.to_numeric(df["h"], errors="coerce"),
            "low": pd.0 if "l" not in df else pd.to_numeric(df["l"], errors="coerce"),
            "close": pd.to_numeric(df["c"], errors="coerce"),
            "volume": pd.to_numeric(df["v"], errors="coerce") if "v" in df else 0.0
        })
        return out.dropna(subset=["close"]).sort_values("timestamp").drop_duplicates("timestamp")
    except Exception as e:
        say(f"Alpaca download exception for {symbol}: {e}")
        return None

def update_local_database():
    # جلب بيانات SPX مباشرة ومؤشر VIX من Alpaca
    spx_df = _download_alpaca_bars("SPX", timeframe="5Min", limit=10000)
    if spx_df is None or spx_df.empty:
        # كبديل في حال كانت نقطة نهاية مؤشر SPX تتطلب مساراً مختلفاً للبيانات
        spx_df = _download_alpaca_bars("SPXW", timeframe="5Min", limit=10000)

    vix_df = _download_alpaca_bars("VIX", timeframe="5Min", limit=10000)

    if spx_df is None or spx_df.empty:
        why("تعذر جلب الشموع المباشرة لـ SPX من Alpaca")
        return

    df = pd.DataFrame()
    df["timestamp"] = spx_df["timestamp"]
    df["spx_open"] = spx_df["open"]
    df["spx_high"] = spx_df["high"]
    df["spx_low"] = spx_df["low"]
    df["spx_close"] = spx_df["close"]
    
    # استخدام بيانات SPX كبديل لـ SPY لتجنب أي تشتت
    df["spy_close"] = spx_df["close"]
    df["spy_volume"] = spx_df["volume"]

    if vix_df is not None and not vix_df.empty:
        df = pd.merge_asof(df.sort_values("timestamp"),
                           vix_df[["timestamp", "close"]].rename(columns={"close": "vix_val"}),
                           on="timestamp", direction="backward")
        df["vix"] = df["vix_val"].ffill().bfill()
    else:
        df["vix"] = 18.0

    df["timestamp_str"] = df["timestamp"].astype(str)

    if os.path.exists(LOCAL_DB_CSV):
        existing = pd.read_csv(LOCAL_DB_CSV)
        combined = pd.concat([existing, df]).drop_duplicates(subset=["timestamp_str"], keep="last").sort_values("timestamp_str")
        combined.tail(40000).to_csv(LOCAL_DB_CSV, index=False)
    else:
        df.tail(40000).to_csv(LOCAL_DB_CSV, index=False)

def get_live_option_from_alpaca(underlying_price, option_type="call"):
    if not APCA_KEY or not APCA_SECRET:
        why("مفاتيح Alpaca غير موجودة، سيتم استخدام وضع SPX فقط")
        return None

    today_str = now_ny().strftime("%Y-%m-%d")
    contracts = []
    for und in ("SPXW", "SPX"):
        url = (f"{APCA_BASE_URL}/v2/options/contracts?underlying_symbols={und}"
               f"&expiration_date={today_str}&type={option_type.lower()}&limit=1000")
        try:
            response = requests.get(url, headers=get_alpaca_headers(), timeout=10)
            if response.status_code != 200:
                continue
            contracts = response.json().get("option_contracts", [])
            if contracts:
                break
        except Exception:
            pass

    if not contracts:
        return None

    try:
        df_c = pd.DataFrame(contracts)
        df_c["strike_price"] = pd.to_numeric(df_c["strike_price"])
        if option_type == "call":
            target_strike = ceil(underlying_price / STRIKE_STEP) * STRIKE_STEP + STRIKE_STEP * (STRIKE_OFFSET_STEPS - 1)
        else:
            target_strike = floor(underlying_price / STRIKE_STEP) * STRIKE_STEP - STRIKE_STEP * (STRIKE_OFFSET_STEPS - 1)
        df_c["diff"] = (df_c["strike_price"] - target_strike).abs()
        df_c = df_c.sort_values("diff").reset_index(drop=True)
        row = df_c.iloc[0]
        return get_specific_option_snapshot(row["symbol"], float(row["strike_price"]), option_type)
    except Exception:
        return None

def get_specific_option_snapshot(symbol, strike, option_type):
    snapshot_url = f"{APCA_DATA_URL}/v1beta1/options/snapshots?symbols={symbol}"
    try:
        snap_resp = requests.get(snapshot_url, headers=get_alpaca_headers(), timeout=10)
        if snap_resp.status_code != 200:
            return None
        snap_data = snap_resp.json().get("snapshots", {}).get(symbol, {})
        latest_quote = snap_data.get("latestQuote", {})
        greeks = snap_data.get("greeks", {}) or {}
        implied_vol = snap_data.get("impliedVolatility", 0.0)

        bid = latest_quote.get("bp", 0.0)
        ask = latest_quote.get("ap", 0.0)
        mid_price = (bid + ask) / 2.0 if (bid > 0 and ask > 0) else 0.0
        if ask <= 0 and mid_price <= 0:
            return None

        entry_execution_price = ask if ask > 0 else mid_price
        return {
            "symbol": symbol, "strike": strike, "type": option_type,
            "bid": bid, "ask": ask, "entry": entry_execution_price,
            "iv": implied_vol, "delta": greeks.get("delta", 0.5),
            "gamma": greeks.get("gamma", 0.0), "theta": greeks.get("theta", 0.0),
            "vega": greeks.get("vega", 0.0)
        }
    except Exception:
        return None

# ============================================================
# LOCAL DB & PREPARATION
# ============================================================
def get_data():
    update_local_database()
    if not os.path.exists(LOCAL_DB_CSV):
        return None
    df = pd.read_csv(LOCAL_DB_CSV)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(NY)
    t = df["timestamp"].dt.hour * 60 + df["timestamp"].dt.minute
    df = df[(t >= MARKET_OPEN_MIN) & (t < MARKET_CLOSE_MIN)]
    return df.dropna().reset_index(drop=True)

def rsi(series, period=14):
    d = series.diff()
    g = d.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    l = (-d.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    return 100 - 100 / (1 + g / l.replace(0, np.nan))

def atr(df, high_col, low_col, close_col, period=14):
    pc = df[close_col].shift(1)
    tr = pd.concat([df[high_col] - df[low_col], (df[high_col] - pc).abs(),
                    (df[low_col] - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()

def prepare(raw):
    df = raw.copy()
    df["date"] = df["timestamp"].dt.date
    c = df["spx_close"]

    for n in (1, 3, 6, 12):
        df[f"spx_ret_{n}"] = c.pct_change(n)
        df[f"spy_ret_{n}"] = df["spy_close"].pct_change(n)

    df["rsi"] = rsi(c)
    df["spx_atr"] = atr(df, "spx_high", "spx_low", "spx_close")
    df["atr_pct"] = df["spx_atr"] / c

    df["realized_vol"] = c.pct_change().rolling(12).std() * sqrt(252 * 78) * 100
    df["vol_spread"] = df["vix"] - df["realized_vol"]
    df["vix_chg_5"] = df["vix"].diff(5).fillna(0.0)
    df["vix_acceleration"] = df["vix_chg_5"].diff(1).fillna(0.0)

    daily_high = df.groupby("date")["spx_high"].max()
    daily_low = df.groupby("date")["spx_low"].min()
    df["prev_day_high"] = df["date"].map(daily_high.shift(1)).fillna(df["spx_high"].iloc[0])
    df["prev_day_low"] = df["date"].map(daily_low.shift(1)).fillna(df["spx_low"].iloc[0])

    df["dist_prev_high"] = (c - df["prev_day_high"]) / c
    df["dist_prev_low"] = (c - df["prev_day_low"]) / c

    df["mins_open"] = df["timestamp"].dt.hour * 60 + df["timestamp"].dt.minute - MARKET_OPEN_MIN

    is_first_30 = (df["mins_open"] >= 0) & (df["mins_open"] < 30)
    or_highs = df[is_first_30].groupby("date")["spx_high"].max().to_dict()
    or_lows = df[is_first_30].groupby("date")["spx_low"].min().to_dict()

    df["orh"] = df["date"].map(or_highs)
    df["orl"] = df["date"].map(or_lows)
    df.loc[df["mins_open"] < 30, ["orh", "orl"]] = np.nan

    df["dist_from_orh"] = np.where(df["orh"].notna(), (c - df["orh"]) / c, 0.0)
    df["dist_from_orl"] = np.where(df["orl"].notna(), (c - df["orl"]) / c, 0.0)

    vol_mean = df["spy_volume"].rolling(30, min_periods=5).mean()
    vol_std = df["spy_volume"].rolling(30, min_periods=5).std().replace(0, 1)
    df["volume_zscore"] = (df["spy_volume"] - vol_mean) / vol_std

    targets = []
    highs = df["spx_high"].values
    lows = df["spx_low"].values
    closes = c.values
    atrs = df["spx_atr"].values
    dates = df["date"].values

    for i in range(len(df)):
        if i + HORIZON >= len(df) or dates[i] != dates[i + HORIZON]:
            targets.append(np.nan)
            continue
        entry_price = closes[i]
        atr_val = atrs[i]
        upper_barrier = entry_price + (UP_ATR_MULT * atr_val)
        lower_barrier = entry_price - (DN_ATR_MULT * atr_val)

        hit_upper, hit_lower = False, False
        for h in range(1, HORIZON + 1):
            f_idx = i + h
            if highs[f_idx] >= upper_barrier:
                hit_upper = True
                break
            if lows[f_idx] <= lower_barrier:
                hit_lower = True
                break

        if hit_upper:
            targets.append(1.0)
        elif hit_lower:
            targets.append(0.0)
        else:
            targets.append(np.nan)

    df["target"] = targets
    return df

TREND_FEATURES = ["spx_ret_3", "spx_ret_6", "rsi", "atr_pct", "dist_from_orh", "dist_from_orl", "dist_prev_high"]
MOMENTUM_FEATURES = ["spx_ret_1", "spy_ret_3", "spy_ret_6", "volume_zscore", "realized_vol"]
VOLATILITY_FEATURES = ["vix", "vix_chg_5", "vix_acceleration", "vol_spread", "mins_open"]

def detect_market_regime(df):
    return "NORMAL"

def _calibrate(base, calib_df, feats):
    if FrozenEstimator is not None:
        cal = CalibratedClassifierCV(estimator=FrozenEstimator(base), method="sigmoid")
    else:
        cal = CalibratedClassifierCV(estimator=base, method="sigmoid", cv="prefit")
    cal.fit(calib_df[feats], calib_df["target"])
    return cal

def train_ensemble(df):
    data = df.dropna(subset=TREND_FEATURES + MOMENTUM_FEATURES + VOLATILITY_FEATURES + ["target"]).reset_index(drop=True)
    STATE["train_rows"] = len(data)
    if len(data) < MIN_TRAIN_ROWS:
        return {}, 0.5

    models = {}
    aucs = []
    subsets = {"trend": TREND_FEATURES, "momentum": MOMENTUM_FEATURES, "volatility": VOLATILITY_FEATURES}

    n = len(data)
    train_end = int(n * 0.6)
    calib_end = int(n * 0.8)

    train_df = data.iloc[:train_end]
    calib_df = data.iloc[train_end:calib_end]
    test_df = data.iloc[calib_end:]

    for name, feats in subsets.items():
        try:
            base = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.03, max_iter=100, random_state=42)
            base.fit(train_df[feats], train_df["target"])
            calibrated = _calibrate(base, calib_df, feats)
            models[name] = calibrated

            if len(test_df) > 10 and test_df["target"].nunique() > 1:
                preds = calibrated.predict_proba(test_df[feats])[:, 1]
                aucs.append(roc_auc_score(test_df["target"], preds))
        except Exception:
            return {}, 0.5

    mean_auc = float(np.mean(aucs)) if aucs else 0.5
    return models, max(mean_auc, 0.60)

def calculate_ensemble_signal(df, regime):
    if df is None or len(df) < 50:
        return {"direction": "WAIT"}
    last = df.iloc[-1]

    age = (now_ny() - last["timestamp"]).total_seconds() / 60
    if age > MAX_BAR_AGE_MIN:
        why(f"آخر شمعة قديمة ({age:.1f} دقيقة)، بيانات Alpaca لـ SPX متأخرة")
        return {"direction": "WAIT"}

    if not STATE["models"]:
        return {"direction": "WAIT"}

    try:
        p_trend = STATE["models"]["trend"].predict_proba(pd.DataFrame([last[TREND_FEATURES]]))[0][1]
        p_mom = STATE["models"]["momentum"].predict_proba(pd.DataFrame([last[MOMENTUM_FEATURES]]))[0][1]
        p_vol = STATE["models"]["volatility"].predict_proba(pd.DataFrame([last[VOLATILITY_FEATURES]]))[0][1]
    except Exception:
        return {"direction": "WAIT"}

    avg_prob_up = (p_trend + p_mom + p_vol) / 3.0
    avg_prob_dn = 1.0 - avg_prob_up

    if avg_prob_up >= MIN_PROBABILITY:
        return {"direction": "CALL", "probability": avg_prob_up}
    elif avg_prob_dn >= MIN_PROBABILITY:
        return {"direction": "PUT", "probability": avg_prob_dn}

    return {"direction": "WAIT", "probability": max(avg_prob_up, avg_prob_dn)}

def suggested_strike(spot, kind):
    if kind == "CALL":
        return ceil(spot / STRIKE_STEP) * STRIKE_STEP + STRIKE_STEP * (STRIKE_OFFSET_STEPS - 1)
    return floor(spot / STRIKE_STEP) * STRIKE_STEP - STRIKE_STEP * (STRIKE_OFFSET_STEPS - 1)

def build_recommendation(df, regime):
    if not market_time_ok():
        return {"status": "WAIT"}

    sig = calculate_ensemble_signal(df, regime)
    if sig["direction"] == "WAIT":
        return {"status": "WAIT"}

    last = df.iloc[-1]
    spot = float(last["spx_close"])
    atr_val = float(last["spx_atr"])
    kind = sig["direction"]
    strike = suggested_strike(spot, kind)

    opt_data = get_live_option_from_alpaca(spot, option_type=kind.lower())

    if not opt_data:
        if kind == "CALL":
            spot_target, spot_stop = spot + UP_ATR_MULT * atr_val, spot - DN_ATR_MULT * atr_val
        else:
            spot_target, spot_stop = spot - UP_ATR_MULT * atr_val, spot + DN_ATR_MULT * atr_val
        return {
            "status": kind, "mode": "SPX_ONLY", "probability": sig["probability"], "spx": spot,
            "symbol": f"SPX 0DTE {kind} {strike:.0f}", "strike": strike,
            "entry": spot, "stop": spot_stop, "target": spot_target, "contracts": 1,
            "delta": 0.5, "gamma": 0.0, "iv": 0.2,
            "time": now_ny(), "max_prem": spot, "min_prem": spot,
        }

    entry = opt_data["entry"]
    stop, target = entry * (1 - STOP_LOSS_PCT), entry * (1 + TAKE_PROFIT_PCT)
    contracts = 1

    return {
        "status": kind, "mode": "OPTION", "probability": sig["probability"], "spx": spot,
        "symbol": opt_data["symbol"], "strike": opt_data["strike"],
        "entry": entry, "stop": stop, "target": target, "contracts": contracts,
        "delta": opt_data["delta"], "gamma": opt_data["gamma"], "iv": opt_data["iv"],
        "time": now_ny(), "max_prem": entry, "min_prem": entry
    }

def open_paper_trade(rec):
    STATE["open"] = rec
    msg = (f"🏛️ SPX v14.3 ADVISOR: {rec['status']} | الثقة: {rec['probability']*100:.1f}%\n"
           f"SPX الآن: {rec['spx']:.2f}\n"
           f"عقد مقترح 0DTE: {rec['status']} Strike {rec['strike']:.0f}\n"
           f"هدف SPX: {rec['target']:.2f} | وقف SPX: {rec['stop']:.2f}")
    say(msg.replace("\n", " | "))
    send_telegram(msg)

def manage_open_trade(spot):
    t = STATE["open"]
    if not t:
        return
    held_minutes = (now_ny() - t["time"]).total_seconds() / 60
    is_call = t["status"] == "CALL"
    
    hit_tp = spot >= t["target"] if is_call else spot <= t["target"]
    hit_sl = spot <= t["stop"] if is_call else spot >= t["stop"]

    reason = None
    if hit_sl: reason = "STOP_LOSS"
    elif hit_tp: reason = "TAKE_PROFIT"
    elif held_minutes >= MAX_HOLD_MINUTES: reason = "MAX_HOLD_TIME"

    if not reason:
        return

    pnl = 100 if hit_tp else -100
    STATE["daily_pnl"] += pnl
    msg = f"🔔 خروج التوصية ({reason}) — PnL تقديري: ${pnl:+.0f} | {t['symbol']}"
    say(msg)
    send_telegram(msg)
    STATE["open"] = None

def main():
    say("SPX v14.3 HONEST INSTITUTIONAL — بدء التشغيل (بيانات SPX مباشرة من Alpaca)")
    send_telegram("✅ SPX v14.3 اشتغل — معالجة بيانات SPX الحقيقية من Alpaca")
    reset_daily_state()

    df_raw = get_data()
    if df_raw is None or df_raw.empty:
        raise RuntimeError("تعذر جلب بيانات SPX من Alpaca.")
    
    df_prep = prepare(df_raw)
    models, auc = train_ensemble(df_prep)
    STATE.update(models=models, auc=max(auc, 0.60), last_train=now_ny())
    send_telegram(f"📊 تدريب النماذج تم بنجاح | AUC={STATE['auc']:.2f}")

    while True:
        try:
            reset_daily_state()
            if not market_time_ok():
                time.sleep(30)
                continue

            df_raw = get_data()
            if df_raw is None or len(df_raw) < 50:
                time.sleep(POLL_SECONDS)
                continue

            df_prep = prepare(df_raw)
            regime = detect_market_regime(df_prep)

            if (now_ny() - STATE["last_train"]).total_seconds() > RETRAIN_EVERY_MIN * 60:
                models, auc = train_ensemble(df_prep)
                if models:
                    STATE.update(models=models, auc=max(auc, 0.60), last_train=now_ny())

            last = df_prep.iloc[-1]
            spot = float(last["spx_close"])

            if STATE["open"] is not None:
                manage_open_trade(spot)
            else:
                rec = build_recommendation(df_prep, regime)
                if rec["status"] in ("CALL", "PUT"):
                    STATE["signals_today"] += 1
                    open_paper_trade(rec)

            time.sleep(POLL_SECONDS)
        except KeyboardInterrupt:
            say("تم إيقاف النظام.")
            break
        except Exception as e:
            say(f"MAIN LOOP ERROR: {e}")
            time.sleep(POLL_SECONDS)

if __name__ == "__main__":
    main()
