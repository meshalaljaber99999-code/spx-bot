# ============================================================
# SPX 0DTE ADVISOR v14.2 HONEST INSTITUTIONAL ENGINE
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
import yfinance as yf
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

# ============================================================
# CONFIG & ARCHITECTURE
# ============================================================
NY = ZoneInfo("America/New_York")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

APCA_KEY = os.getenv("APCA_API_KEY_ID", "")
APCA_SECRET = os.getenv("APCA_API_SECRET_KEY", "")
APCA_BASE_URL = os.getenv("APCA_API_BASE_URL", "https://paper-api.alpaca.markets")

POLL_SECONDS = 30
MARKET_OPEN_MIN = 9 * 60 + 30
MARKET_CLOSE_MIN = 16 * 60
NO_TRADE_FIRST_MIN = 35
NO_TRADE_LAST_MIN = 60
MAX_BAR_AGE_MIN = 3

LOCAL_DB_CSV = "local_market_db.csv"
OPTIONS_DB_CSV = "local_options_premium_db.csv"
TRADES_CSV = "paper_trades_v14_2.csv"

MIN_TRAIN_ROWS = 600
HORIZON = 6                        
MAX_HOLD_BARS = 6
MIN_PROBABILITY = 0.75
MIN_AUC = 0.65
RETRAIN_EVERY_MIN = 180

ACCOUNT_EQUITY = 25000
RISK_PER_TRADE = 0.01
MAX_DAILY_LOSS = 0.03
MAX_CONSECUTIVE_LOSSES = 3
COMMISSION_PER_CONTRACT = 0.65

STRIKE_STEP = 5.0
STRIKE_OFFSET_STEPS = 1
MIN_PREMIUM = 1.30
STOP_LOSS_PCT = 0.35
TAKE_PROFIT_PCT = 0.65
MAX_HOLD_MINUTES = 30
FORCE_EXIT_BEFORE_CLOSE_MIN = 10

STATE = {
    "models": {}, "auc": 0.0, "last_train": None,
    "date": None, "daily_pnl": 0.0, "loss_streak": 0, "signals_today": 0,
    "open": None, "daily_summary_sent": False, "regime": "NORMAL",
}

# ============================================================
# UTILITIES & ALPACA OPTIONS CONNECTOR
# ============================================================
def now_ny():
    return datetime.now(NY)

def say(msg):
    print(f"[{now_ny().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)

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
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
            json={"chat_id": TELEGRAM_CHAT_ID, "text": message}, timeout=10)
    except Exception as e:
        say(f"Telegram error: {e}")

def get_alpaca_headers():
    return {
        "APCA-API-KEY-ID": APCA_KEY,
        "APCA-API-SECRET-KEY": APCA_SECRET,
        "accept": "application/json"
    }

def get_live_option_from_alpaca(underlying_price, option_type="call"):
    today_str = now_ny().strftime("%Y-%m-%d")
    contracts_url = f"https://data.alpaca.markets/v1beta1/options/contracts?underlying_symbol=SPX&expiration_date={today_str}"
    
    try:
        response = requests.get(contracts_url, headers=get_alpaca_headers(), timeout=10)
        if response.status_code != 200:
            return None
        
        contracts = response.json().get("option_contracts", [])
        if not contracts:
            return None
            
        df_contracts = pd.DataFrame(contracts)
        df_contracts = df_contracts[df_contracts["type"] == option_type.lower()]
        df_contracts["strike_price"] = pd.to_numeric(df_contracts["strike_price"])
        
        target_strike = ceil(underlying_price / STRIKE_STEP) * STRIKE_STEP + STRIKE_STEP * (STRIKE_OFFSET_STEPS - 1) if option_type == "call" \
            else floor(underlying_price / STRIKE_STEP) * STRIKE_STEP - STRIKE_STEP * (STRIKE_OFFSET_STEPS - 1)
            
        df_contracts["diff"] = (df_contracts["strike_price"] - target_strike).abs()
        df_contracts = df_contracts.sort_values("diff").reset_index(drop=True)
        
        if df_contracts.empty:
            return None
            
        target_contract = df_contracts.iloc[0]
        symbol = target_contract["symbol"]
        
        return get_specific_option_snapshot(symbol, target_contract["strike_price"], option_type)
    except Exception as e:
        say(f"Alpaca Option Exception: {e}")
        return None

def get_specific_option_snapshot(symbol, strike, option_type):
    snapshot_url = f"https://data.alpaca.markets/v1beta1/options/snapshots?symbols={symbol}"
    try:
        snap_resp = requests.get(snapshot_url, headers=get_alpaca_headers(), timeout=10)
        if snap_resp.status_code == 200:
            snap_data = snap_resp.json().get("snapshots", {}).get(symbol, {})
            latest_quote = snap_data.get("latestQuote", {})
            greeks = snap_data.get("greeks", {})
            implied_vol = snap_data.get("impliedVolatility", 0.0)
            
            bid = latest_quote.get("bp", 0.0)
            ask = latest_quote.get("ap", 0.0)
            mid_price = (bid + ask) / 2.0 if (bid > 0 and ask > 0) else 1.0
            
            entry_execution_price = ask if ask > 0 else mid_price

            record_option_premium(symbol, strike, option_type, bid, ask, mid_price, implied_vol, greeks)

            return {
                "symbol": symbol, "strike": strike, "type": option_type,
                "bid": bid, "ask": ask, "entry": entry_execution_price,
                "iv": implied_vol, "delta": greeks.get("delta", 0.0),
                "gamma": greeks.get("gamma", 0.0), "theta": greeks.get("theta", 0.0),
                "vega": greeks.get("vega", 0.0)
            }
    except Exception as e:
        say(f"Snapshot Exception for {symbol}: {e}")
    return None

def record_option_premium(symbol, strike, opt_type, bid, ask, mid, iv, greeks):
    row = {
        "timestamp": now_ny().isoformat(), "symbol": symbol, "strike": strike, "type": opt_type,
        "bid": bid, "ask": ask, "mid": mid, "iv": iv,
        "delta": greeks.get("delta", 0.0), "gamma": greeks.get("gamma", 0.0), "theta": greeks.get("theta", 0.0)
    }
    file_exists = os.path.exists(OPTIONS_DB_CSV)
    with open(OPTIONS_DB_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            w.writeheader()
        w.writerow(row)

# ============================================================
# LOCAL DB & PREPARATION (WITH OPENING RANGE FIX)
# ============================================================
def _download_safe(symbol, period, interval):
    try:
        df = yf.download(symbol, period=period, interval=interval, progress=False, auto_adjust=False)
        if df is None or df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.reset_index()
        ts_col = "Datetime" if "Datetime" in df.columns else df.columns[0]
        out = pd.DataFrame({"timestamp": pd.to_datetime(df[ts_col], utc=True).dt.tz_convert(NY)})
        for c in ["Open", "High", "Low", "Close", "Volume"]:
            out[c.lower()] = pd.to_numeric(df[c], errors="coerce") if c in df.columns else 0.0
        return out.dropna(subset=["close"]).sort_values("timestamp").drop_duplicates("timestamp")
    except Exception:
        return None

def update_local_database():
    spx_new = _download_safe("^GSPC", "5d", "5m")
    spy_new = _download_safe("SPY", "5d", "5m")
    vix_new = _download_safe("^VIX", "5d", "5m")
    
    if spx_new is None or spy_new is None or vix_new is None:
        return

    df = spx_new[["timestamp", "open", "high", "low", "close"]].rename(
        columns={"open": "spx_open", "high": "spx_high", "low": "spx_low", "close": "spx_close"}
    )
    df = df.merge(spy_new[["timestamp", "close", "volume"]].rename(
        columns={"close": "spy_close", "volume": "spy_volume"}
    ), on="timestamp", how="left")
    
    df = pd.merge_asof(df.sort_values("timestamp"),
                       vix_new[["timestamp", "close"]].rename(columns={"close": "vix"}),
                       on="timestamp", direction="backward")
    df["vix"] = df["vix"].ffill().bfill()
    df["timestamp_str"] = df["timestamp"].astype(str)

    if os.path.exists(LOCAL_DB_CSV):
        existing = pd.read_csv(LOCAL_DB_CSV)
        combined = pd.concat([existing, df]).drop_duplicates(subset=["timestamp_str"]).sort_values("timestamp_str")
        combined.tail(40000).to_csv(LOCAL_DB_CSV, index=False)
    else:
        df.tail(40000).to_csv(LOCAL_DB_CSV, index=False)

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

    daily_high = df.groupby("date")["spx_high"].transform("max")
    daily_low = df.groupby("date")["spx_low"].transform("min")

    prev_high = daily_high.groupby(df["date"]).first().shift(1)
    prev_low = daily_low.groupby(df["date"]).first().shift(1)

    df["prev_day_high"] = df["date"].map(prev_high).fillna(df["spx_high"].iloc[0])
    df["prev_day_low"] = df["date"].map(prev_low).fillna(df["spx_low"].iloc[0])

    df["dist_prev_high"] = (c - df["prev_day_high"]) / c
    df["dist_prev_low"] = (c - df["prev_day_low"]) / c

    df["mins_open"] = df["timestamp"].dt.hour * 60 + df["timestamp"].dt.minute - MARKET_OPEN_MIN

    # 🚨 التصحيح الدقيق: استبعاد الدقيقة 30 تماماً لضبط أول 30 دقيقة (9:30 - 10:00)
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
        upper_barrier = entry_price + (0.6 * atr_val)
        lower_barrier = entry_price - (0.4 * atr_val)

        hit_upper, hit_lower = False, False
        for h in range(1, HORIZON + 1):
            f_idx = i + h
            if highs[f_idx] >= upper_barrier:
                hit_upper = True
                break
            if lows[f_idx] <= lower_barrier:
                hit_lower = True
                break

        if hit_upper: targets.append(1.0)
        elif hit_lower: targets.append(0.0)
        else: targets.append(np.nan)
    
    df["target"] = targets
    return df

# ============================================================
# ENSEMBLE MODELS & REGIME FILTER
# ============================================================
TREND_FEATURES = ["spx_ret_3", "spx_ret_6", "rsi", "atr_pct", "dist_from_orh", "dist_from_orl", "dist_prev_high"]
MOMENTUM_FEATURES = ["spx_ret_1", "spy_ret_3", "spy_ret_6", "volume_zscore", "realized_vol"]
VOLATILITY_FEATURES = ["vix", "vix_chg_5", "vix_acceleration", "vol_spread", "mins_open"]

def detect_market_regime(df):
    if df is None or len(df) < 10:
        return "NORMAL"
    last_vix = df.iloc[-1]["vix"]
    last_atr = df.iloc[-1]["atr_pct"]
    vix_accel = df.iloc[-1]["vix_acceleration"]
    
    if last_vix > 25.0 or last_atr > 0.0040:
        return "HIGH_VOLATILITY"
    elif abs(vix_accel) > 0.5:
        return "VOLATILITY_EXPANSION"
    elif last_vix < 13.0:
        return "LOW_VOL_CHOPPY"
    return "NORMAL"

def train_ensemble(df):
    data = df.dropna(subset=TREND_FEATURES + MOMENTUM_FEATURES + VOLATILITY_FEATURES + ["target"]).reset_index(drop=True)
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
        base = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.03, max_iter=200, random_state=42)
        base.fit(train_df[feats], train_df["target"])
        
        calibrated = CalibratedClassifierCV(estimator=base, method='sigmoid', cv='prefit')
        calibrated.fit(calib_df[feats], calib_df["target"])
        models[name] = calibrated
        
        if len(test_df) > 30 and test_df["target"].nunique() > 1:
            preds = calibrated.predict_proba(test_df[feats])[:, 1]
            aucs.append(roc_auc_score(test_df["target"], preds))

    mean_auc = float(np.mean(aucs)) if aucs else 0.5
    say(f"Honest Institutional Ensemble Trained. Proxy AUC = {mean_auc:.3f}")
    return models, mean_auc

def calculate_ensemble_signal(df, regime):
    if regime in ("LOW_VOL_CHOPPY", "VOLATILITY_EXPANSION"):
        return {"direction": "WAIT"}

    if df is None or len(df) < 50:
        return {"direction": "WAIT"}
    last = df.iloc[-1]
    
    if (now_ny() - last["timestamp"]).total_seconds() / 60 > MAX_BAR_AGE_MIN:
        return {"direction": "WAIT"}

    if not STATE["models"] or STATE["auc"] < MIN_AUC:
        return {"direction": "WAIT"}

    p_trend = STATE["models"]["trend"].predict_proba(pd.DataFrame([last[TREND_FEATURES]]))[0][1]
    p_mom = STATE["models"]["momentum"].predict_proba(pd.DataFrame([last[MOMENTUM_FEATURES]]))[0][1]
    p_vol = STATE["models"]["volatility"].predict_proba(pd.DataFrame([last[VOLATILITY_FEATURES]]))[0][1]

    avg_prob_up = (p_trend + p_mom + p_vol) / 3.0
    avg_prob_dn = 1.0 - avg_prob_up

    if avg_prob_up >= MIN_PROBABILITY and (p_trend > 0.6 and p_mom > 0.6):
        return {"direction": "CALL", "probability": avg_prob_up}
    elif avg_prob_dn >= MIN_PROBABILITY and (p_trend < 0.4 and p_mom < 0.4):
        return {"direction": "PUT", "probability": avg_prob_dn}
    
    return {"direction": "WAIT", "probability": max(avg_prob_up, avg_prob_dn)}

def build_recommendation(df, regime):
    if not market_time_ok() or not session_allowed():
        return {"status": "WAIT"}
    if STATE["daily_pnl"] <= -ACCOUNT_EQUITY * MAX_DAILY_LOSS:
        return {"status": "WAIT"}
    if STATE["loss_streak"] >= MAX_CONSECUTIVE_LOSSES:
        return {"status": "WAIT"}

    sig = calculate_ensemble_signal(df, regime)
    if sig["direction"] == "WAIT":
        return {"status": "WAIT"}

    spot = float(df.iloc[-1]["spx_close"])
    kind = sig["direction"]

    opt_data = get_live_option_from_alpaca(spot, option_type=kind.lower())
    if not opt_data:
        return {"status": "WAIT"}

    entry = opt_data["entry"]
    if entry < MIN_PREMIUM:
        return {"status": "WAIT"}

    stop, target = entry * (1 - STOP_LOSS_PCT), entry * (1 + TAKE_PROFIT_PCT)
    contracts = int((ACCOUNT_EQUITY * RISK_PER_TRADE) / (entry * STOP_LOSS_PCT * 100))
    if contracts <= 0:
        return {"status": "WAIT"}

    return {
        "status": kind, "probability": sig["probability"], "spx": spot,
        "symbol": opt_data["symbol"], "strike": opt_data["strike"],
        "entry": entry, "stop": stop, "target": target, "contracts": contracts,
        "delta": opt_data["delta"], "gamma": opt_data["gamma"], "iv": opt_data["iv"],
        "time": now_ny(), "max_prem": entry, "min_prem": entry
    }

def open_paper_trade(rec):
    STATE["open"] = rec
    msg = (f"🏛️ SPX v14.2 ADVISOR: {rec['status']} | الثقة: {rec['probability']*100:.1f}% | "
           f"Symbol: {rec['symbol']} | Strike: {rec['strike']} | دخول (Ask): ${rec['entry']:.2f} | Delta: {rec['delta']:.2f}")
    say(msg)
    send_telegram(msg)

def manage_open_trade(spot):
    t = STATE["open"]
    if not t:
        return
    
    opt_data = get_specific_option_snapshot(t["symbol"], t["strike"], t["status"].lower())
    
    if opt_data:
        prem = opt_data["bid"] if opt_data["bid"] > 0 else opt_data["entry"]
    else:
        prem = t["entry"]
    
    t["max_prem"] = max(t["max_prem"], prem)
    t["min_prem"] = min(t["min_prem"], prem)

    ml = mins_left()
    held_minutes = (now_ny() - t["time"]).total_seconds() / 60
    reason = None

    if prem <= t["stop"]:
        reason = "STOP_LOSS"
    elif prem >= t["target"]:
        reason = "TAKE_PROFIT"
    elif held_minutes >= MAX_HOLD_MINUTES:
        reason = "MAX_HOLD_TIME"
    elif ml <= FORCE_EXIT_BEFORE_CLOSE_MIN:
        reason = "SESSION_CLOSE"

    if not reason:
        return

    pnl = (prem - t["entry"]) * 100 * t["contracts"] - (2 * COMMISSION_PER_CONTRACT * t["contracts"])
    STATE["daily_pnl"] += pnl
    STATE["loss_streak"] = STATE["loss_streak"] + 1 if pnl < 0 else 0

    file_exists = os.path.exists(TRADES_CSV)
    with open(TRADES_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if not file_exists:
            w.writerow(["entry_time", "exit_time", "symbol", "kind", "strike", "entry_ask", "exit_bid", "contracts", "pnl", "reason"])
        w.writerow([t["time"].isoformat(), now_ny().isoformat(), t["symbol"], t["status"], t["strike"],
                    round(t["entry"], 2), round(prem, 2), t["contracts"], round(pnl, 2), reason])

    msg = f"🔔 خروج التوصية ({reason}) — PnL: ${pnl:+.0f} | الرمز: {t['symbol']} | اليومي: ${STATE['daily_pnl']:+.0f}"
    say(msg)
    send_telegram(msg)
    STATE["open"] = None

def main():
    say("SPX v14.2 HONEST INSTITUTIONAL — بدء التشغيل مع إصلاح شمعة الـ Opening Range")
    reset_daily_state()
    
    df_raw = get_data()
    if df_raw is None or df_raw.empty:
        raise RuntimeError("تعذر جلب البيانات الأساسية.")
    
    df_prep = prepare(df_raw)
    regime = detect_market_regime(df_prep)
    STATE["regime"] = regime
    say(f"حالة نظام السوق (Market Regime): {regime}")

    models, auc = train_ensemble(df_prep)
    STATE.update(models=models, auc=auc, last_train=now_ny())
    say(f"النماذج جاهزة. Proxy AUC = {STATE['auc']:.3f}")

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
            STATE["regime"] = regime

            if (now_ny() - STATE["last_train"]).total_seconds() > RETRAIN_EVERY_MIN * 60:
                models, auc = train_ensemble(df_prep)
                if models:
                    STATE.update(models=models, auc=auc, last_train=now_ny())

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
