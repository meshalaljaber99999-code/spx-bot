# ============================================================
# SPX 0DTE ADVISOR v10 PRO PRECISION - النسخة المصححة هندسياً
# ============================================================

import os
import sys
import csv
import json
import time
import warnings
from math import erf, sqrt, floor, ceil
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from dotenv import load_dotenv
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, brier_score_loss

warnings.filterwarnings("ignore")
load_dotenv()

# ============================================================
# CONFIG & ARCHITECTURE
# ============================================================
NY = ZoneInfo("America/New_York")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

POLL_SECONDS = 30
MARKET_OPEN_MIN = 9 * 60 + 30
MARKET_CLOSE_MIN = 16 * 60
NO_TRADE_FIRST_MIN = 35
NO_TRADE_LAST_MIN = 60
MAX_BAR_AGE_MIN = 3

LOCAL_DB_CSV = "local_market_db.csv"
TRADES_CSV = "paper_trades_v10.csv"
SIGNALS_JSONL = "spx_signals_v10.jsonl"

MIN_TRAIN_ROWS = 600
HORIZON = 6                        # 6 شموع × 5 دقائق = 30 دقيقة בדיוק (مطابق تماماً لـ Max Hold)
MAX_HOLD_BARS = 6                  # عدد الشموع الأقصى للبقاء في الصفقة (30 دقيقة)
MIN_PROBABILITY = 0.72
MIN_AUC = 0.62
RETRAIN_EVERY_MIN = 180

ACCOUNT_EQUITY = 25000
RISK_PER_TRADE = 0.01
MAX_DAILY_LOSS = 0.03
MAX_CONSECUTIVE_LOSSES = 3
MAX_SIGNALS_PER_DAY = 3
COMMISSION_PER_CONTRACT = 0.65

STRIKE_STEP = 5.0
STRIKE_OFFSET_STEPS = 1
MIN_PREMIUM = 1.30
STOP_LOSS_PCT = 0.35
TAKE_PROFIT_PCT = 0.65
MAX_HOLD_MINUTES = 30
FORCE_EXIT_BEFORE_CLOSE_MIN = 10
SLIPPAGE_PENALTY_PCT = 0.10

STATE = {
    "models": {}, "auc": 0.0, "last_train": None,
    "date": None, "daily_pnl": 0.0, "loss_streak": 0, "signals_today": 0,
    "open": None, "daily_summary_sent": False, "extreme_market_alerted": False,
}

# ============================================================
# UTILITIES
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
                     signals_today=0, daily_summary_sent=False, extreme_market_alerted=False)

def send_telegram(message):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
            json={"chat_id": TELEGRAM_CHAT_ID, "text": message}, timeout=10)
    except Exception as e:
        say(f"Telegram error: {e}")

# ============================================================
# LOCAL DB ACCUMULATION
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
        combined.tail(35000).to_csv(LOCAL_DB_CSV, index=False)
    else:
        df.tail(35000).to_csv(LOCAL_DB_CSV, index=False)

def get_data():
    update_local_database()
    if not os.path.exists(LOCAL_DB_CSV):
        return None
    df = pd.read_csv(LOCAL_DB_CSV)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(NY)
    t = df["timestamp"].dt.hour * 60 + df["timestamp"].dt.minute
    df = df[(t >= MARKET_OPEN_MIN) & (t < MARKET_CLOSE_MIN)]
    return df.dropna().reset_index(drop=True)

# ============================================================
# ADVANCED FEATURES & CAUSAL OPENING RANGE
# ============================================================
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
    
    df["spy_vwap"] = (df["spy_close"] * df["spy_volume"]).groupby(df["date"]).cumsum() / df["spy_volume"].groupby(df["date"]).cumsum().replace(0, np.nan)
    df["spy_vwap"] = df["spy_vwap"].fillna(df["spy_close"])
    df["spy_vwap_dist"] = (df["spy_close"] - df["spy_vwap"]) / df["spy_close"]

    df["vix_chg_5"] = df["vix"].diff(5)
    df["vix_chg_15"] = df["vix"].diff(15)
    df["vix_spx_divergence"] = np.where((df["vix_chg_5"] > 0) & (df["spx_ret_1"] > 0), 1, 0)

    df["mins_open"] = df["timestamp"].dt.hour * 60 + df["timestamp"].dt.minute - MARKET_OPEN_MIN

    is_first_30 = (df["mins_open"] >= 0) & (df["mins_open"] <= 30)
    or_highs = df[is_first_30].groupby("date")["spx_high"].max().to_dict()
    or_lows = df[is_first_30].groupby("date")["spx_low"].min().to_dict()

    df["orh"] = df["date"].map(or_highs)
    df["orl"] = df["date"].map(or_lows)
    df.loc[df["mins_open"] <= 30, ["orh", "orl"]] = np.nan

    df["dist_from_orh"] = np.where(df["orh"].notna(), (c - df["orh"]) / c, 0.0)
    df["dist_from_orl"] = np.where(df["orl"].notna(), (c - df["orl"]) / c, 0.0)

    vol_mean = df["spy_volume"].rolling(30, min_periods=5).mean()
    vol_std = df["spy_volume"].rolling(30, min_periods=5).std().replace(0, 1)
    df["volume_zscore"] = (df["spy_volume"] - vol_mean) / vol_std

    # Target باستخدام HORIZON المحدث بدقة (6 شموع = 30 دقيقة)
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
        up_target = entry_price + (0.6 * atr_val)
        up_stop = entry_price - (0.4 * atr_val)

        hit_up_first = False
        hit_dn_first = False

        for h in range(1, HORIZON + 1):
            future_idx = i + h
            bar_high = highs[future_idx]
            bar_low = lows[future_idx]

            if not hit_up_first and not hit_dn_first:
                touched_up = bar_high >= up_target
                touched_dn = bar_low <= up_stop
                if touched_up and not touched_dn:
                    hit_up_first = True
                    break
                elif touched_dn and not touched_up:
                    hit_dn_first = True
                    break
                elif touched_up and touched_dn:
                    hit_dn_first = True
                    break

        if hit_up_first:
            targets.append(1.0)
        elif hit_dn_first:
            targets.append(0.0)
        else:
            targets.append(np.nan)

    df["target"] = targets
    return df

# ============================================================
# ENSEMBLE MODELS & CALIBRATION
# ============================================================
TREND_FEATURES = ["spx_ret_3", "spx_ret_6", "rsi", "atr_pct", "dist_from_orh", "dist_from_orl"]
MOMENTUM_FEATURES = ["spx_ret_1", "spy_ret_3", "spy_ret_6", "volume_zscore", "spy_vwap_dist"]
VOLATILITY_FEATURES = ["vix", "vix_chg_5", "vix_chg_15", "vix_spx_divergence", "mins_open"]

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
    say(f"Precision Ensemble Trained. Test AUC = {mean_auc:.3f}")
    return models, mean_auc

# ============================================================
# BLACK-SCHOLES
# ============================================================
def _cdf(x):
    return 0.5 * (1 + erf(x / sqrt(2)))

def bs_price(spot, strike, minutes_left, vix, kind):
    T = max(minutes_left, 0.5) / (252 * 390)
    sigma = max(vix, 5) / 100 * 1.05
    d1 = (np.log(spot / strike) + 0.5 * sigma ** 2 * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    if kind == "CALL":
        p = spot * _cdf(d1) - strike * _cdf(d2)
    else:
        p = strike * _cdf(-d2) - spot * _cdf(-d1)
    return max(float(p), 0.05)

# ============================================================
# SIGNAL & LIVE TRADE MANAGEMENT
# ============================================================
def calculate_ensemble_signal(df):
    if df is None or len(df) < 50:
        return {"direction": "WAIT"}
    last = df.iloc[-1]
    
    age = (now_ny() - last["timestamp"]).total_seconds() / 60
    if age > MAX_BAR_AGE_MIN:
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

def build_recommendation(df):
    if not market_time_ok() or not session_allowed():
        return {"status": "WAIT", "reason": "خارج النافذة"}
    
    if STATE["daily_pnl"] <= -ACCOUNT_EQUITY * MAX_DAILY_LOSS:
        return {"status": "WAIT", "reason": "بلوغ حد الخسارة اليومي"}

    sig = calculate_ensemble_signal(df)
    if sig["direction"] == "WAIT":
        return {"status": "WAIT"}

    spot = float(df.iloc[-1]["spx_close"])
    vix = float(df.iloc[-1]["vix"])
    kind = sig["direction"]
    strike = ceil(spot / STRIKE_STEP) * STRIKE_STEP + STRIKE_STEP * (STRIKE_OFFSET_STEPS - 1) if kind == "CALL" \
        else floor(spot / STRIKE_STEP) * STRIKE_STEP - STRIKE_STEP * (STRIKE_OFFSET_STEPS - 1)

    ml = mins_left()
    entry = bs_price(spot, strike, ml, vix, kind) * (1.0 + SLIPPAGE_PENALTY_PCT)
    if entry < MIN_PREMIUM:
        return {"status": "WAIT"}

    stop, target = entry * (1 - STOP_LOSS_PCT), entry * (1 + TAKE_PROFIT_PCT)
    contracts = int((ACCOUNT_EQUITY * RISK_PER_TRADE) / (entry * STOP_LOSS_PCT * 100))
    if contracts <= 0:
        return {"status": "WAIT"}

    return {
        "status": kind, "probability": sig["probability"], "spx": spot, "vix": vix,
        "strike": strike, "entry": entry, "stop": stop, "target": target, "contracts": contracts,
        "time": now_ny(), "max_prem": entry, "min_prem": entry
    }

def open_paper_trade(rec):
    STATE["open"] = rec
    msg = f"🎯 SPX v10 PRECISION SIGNAL: {rec['status']} | الثقة: {rec['probability']*100:.1f}% | Strike: {rec['strike']} | دخول: ${rec['entry']:.2f}"
    say(msg)
    send_telegram(msg)

def manage_open_trade(spot, vix):
    t = STATE["open"]
    if not t:
        return
    
    ml = mins_left()
    prem = bs_price(spot, t["strike"], ml, vix, t["status"]) * (1.0 - SLIPPAGE_PENALTY_PCT)
    t["max_prem"] = max(t["max_prem"], prem)
    t["min_prem"] = min(t["min_prem"], prem)

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

    pnl = (prem - t["entry"]) * 100 * t["contracts"] - 2 * COMMISSION_PER_CONTRACT * t["contracts"]
    STATE["daily_pnl"] += pnl
    STATE["loss_streak"] = STATE["loss_streak"] + 1 if pnl < 0 else 0

    new_file = not os.path.exists(TRADES_CSV)
    with open(TRADES_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["entry_time", "exit_time", "kind", "strike", "entry_prem", "exit_prem", "contracts", "pnl", "reason"])
        w.writerow([t["time"].isoformat(), now_ny().isoformat(), t["status"], t["strike"],
                    round(t["entry"], 2), round(prem, 2), t["contracts"], round(pnl, 2), reason])

    msg = f"🔔 خروج صفقة حي ({reason}) — PnL: ${pnl:+.0f} | اليومي: ${STATE['daily_pnl']:+.0f}"
    say(msg)
    send_telegram(msg)
    STATE["open"] = None

# ============================================================
# PRECISION 0DTE BACKTEST ENGINE (--backtest)
# ============================================================
def run_backtest():
    if not os.path.exists(LOCAL_DB_CSV):
        print("لا توجد بيانات تاريخية كافية.")
        return
    
    df = pd.read_csv(LOCAL_DB_CSV)
    df = prepare(df)
    data = df.dropna(subset=TREND_FEATURES + MOMENTUM_FEATURES + VOLATILITY_FEATURES + ["target"]).reset_index(drop=True)
    
    if len(data) < 1000:
        print(f"⚠️ تنبيه: حجم العينة صغير ({len(data)} صف).")
        return

    print("=" * 65)
    print("🚀 بدء محرك الـ Backtest بالمعايير الزمنية الدقيقة (v10)")
    print("=" * 65)
    
    split = int(len(data) * 0.7)
    train_set = data.iloc[:split]
    test_set = data.iloc[split:].reset_index(drop=True)
    
    subsets = {"trend": TREND_FEATURES, "momentum": MOMENTUM_FEATURES, "volatility": VOLATILITY_FEATURES}
    models = {}
    for name, feats in subsets.items():
        m = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.03, max_iter=200, random_state=42)
        m.fit(train_set[feats], train_set["target"])
        models[name] = m

    trades_results = []
    
    # المحاكاة تتطابق مع MAX_HOLD_BARS = 6 (أي 30 دقيقة بالضبط)
    for i in range(len(test_set) - MAX_HOLD_BARS):
        row = test_set.iloc[i]
        p_t = models["trend"].predict_proba(row[TREND_FEATURES].values.reshape(1, -1))[0][1]
        p_m = models["momentum"].predict_proba(row[MOMENTUM_FEATURES].values.reshape(1, -1))[0][1]
        p_v = models["volatility"].predict_proba(row[VOLATILITY_FEATURES].values.reshape(1, -1))[0][1]
        prob_up = (p_t + p_m + p_v) / 3.0
        prob_dn = 1.0 - prob_up

        kind = "WAIT"
        prob = max(prob_up, prob_dn)
        if prob_up >= MIN_PROBABILITY and (p_t > 0.6 and p_m > 0.6):
            kind = "CALL"
            prob = prob_up
        elif prob_dn >= MIN_PROBABILITY and (p_t < 0.4 and p_m < 0.4):
            kind = "PUT"
            prob = prob_dn

        if kind == "WAIT":
            continue

        spot = row["spx_close"]
        vix = row["vix"]
        strike = ceil(spot / STRIKE_STEP) * STRIKE_STEP + STRIKE_STEP if kind == "CALL" else floor(spot / STRIKE_STEP) * STRIKE_STEP
        
        t_dt = pd.to_datetime(row["timestamp"])
        mins_to_close = (16 * 60) - (t_dt.hour * 60 + t_dt.minute)
        if mins_to_close < 40:
            continue

        entry_prem = bs_price(spot, strike, mins_to_close, vix, kind) * (1.0 + SLIPPAGE_PENALTY_PCT)
        if entry_prem < MIN_PREMIUM:
            continue

        stop_prem = entry_prem * (1 - STOP_LOSS_PCT)
        target_prem = entry_prem * (1 + TAKE_PROFIT_PCT)

        exit_prem = entry_prem
        outcome = "EXPIRED"
        
        # حلقة المحاكاة تتطابق تماماً مع 6 شموع (30 دقيقة كحد أقصى)
        for h in range(1, MAX_HOLD_BARS + 1):
            if i + h >= len(test_set):
                break
            future_row = test_set.iloc[i + h]
            if future_row["date"] != row["date"]:
                break
            
            f_spot = future_row["spx_close"]
            f_vix = future_row["vix"]
            f_mins_left = mins_to_close - (h * 5)
            if f_mins_left <= 0:
                break

            current_prem = bs_price(f_spot, strike, f_mins_left, f_vix, kind) * (1.0 - SLIPPAGE_PENALTY_PCT)
            
            if current_prem <= stop_prem:
                exit_prem = stop_prem
                outcome = "LOSS"
                break
            elif current_prem >= target_prem:
                exit_prem = target_prem
                outcome = "WIN"
                break
        else:
            outcome = "TIME_EXIT"

        pnl = (exit_prem - entry_prem) * 100 - (2 * COMMISSION_PER_CONTRACT)
        trades_results.append({"pnl": pnl, "win": 1 if outcome == "WIN" else 0})

    if not trades_results:
        print("لا توجد صفقات كافية تم محاكاتها.")
        return

    res_df = pd.DataFrame(trades_results)
    wins = res_df[res_df.pnl > 0]
    losses = res_df[res_df.pnl <= 0]
    win_rate = len(wins) / len(res_df) * 100
    pf = wins.pnl.sum() / abs(losses.pnl.sum()) if len(losses) and losses.pnl.sum() != 0 else float("inf")
    
    avg_win = wins.pnl.mean() if len(wins) > 0 else 0
    avg_loss = abs(losses.pnl.mean()) if len(losses) > 0 else 0
    expectancy = (len(wins)/len(res_df) * avg_win) - (len(losses)/len(res_df) * avg_loss)

    print(f"• إجمالي صفقات الـ Backtest: {len(res_df)}")
    print(f"• نسبة النجاح (Win Rate): {win_rate:.1f}%")
    print(f"• عامل الربح (Profit Factor): {pf:.2f}")
    print(f"• القيمة المتوقعة (Expectancy): ${expectancy:+.2f}")
    print(f"• إجمالي PnL المحاكى: ${res_df.pnl.sum():+.2f}")
    print("=" * 65)

# ============================================================
# MAIN
# ============================================================
def main():
    if "--backtest" in sys.argv:
        run_backtest()
        return

    say("SPX 0DTE v10 PRO PRECISION — بدء التشغيل")
    reset_daily_state()
    
    df_raw = get_data()
    if df_raw is None or df_raw.empty:
        raise RuntimeError("تعذر جلب البيانات.")
    
    df_prep = prepare(df_raw)
    models, auc = train_ensemble(df_prep)
    STATE.update(models=models, auc=auc, last_train=now_ny())
    say(f"النظام جاهز بدقة زمنية متطابقة. AUC = {STATE['auc']:.3f}")

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
            if (now_ny() - STATE["last_train"]).total_seconds() > RETRAIN_EVERY_MIN * 60:
                models, auc = train_ensemble(df_prep)
                if models:
                    STATE.update(models=models, auc=auc, last_train=now_ny())

            last = df_prep.iloc[-1]
            spot, vix = float(last["spx_close"]), float(last["vix"])

            if STATE["open"] is not None:
                manage_open_trade(spot, vix)
            else:
                rec = build_recommendation(df_prep)
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
