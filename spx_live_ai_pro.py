# ============================================================
# SPX 0DTE ADVISOR v6 PRO QUANT - النسخة الكمية المؤسسية المتكاملة
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
NO_TRADE_FIRST_MIN = 30
NO_TRADE_LAST_MIN = 60
MAX_BAR_AGE_MIN = 3

LOCAL_DB_CSV = "local_market_db.csv"
TRADES_CSV = "paper_trades_v6.csv"
SIGNALS_JSONL = "spx_signals_v6.jsonl"

# ML & Strict Ensemble Rules
MIN_TRAIN_ROWS = 500
HORIZON = 6
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

def mins_left():
    n = now_ny()
    return (MARKET_CLOSE_MIN - minutes_of_day(n)) - n.second / 60

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
# DATA & LOCAL DB ACCUMULATION (تراكم البيانات محلياً)
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
    """جلب أحدث بيانات وتخزينها محلياً لعدم ضياع التاريخ مع قيود ياهو"""
    spx_new = _download_safe("^GSPC", "5d", "5m")
    spy_new = _download_safe("SPY", "5d", "5m")
    vix_new = _download_safe("^VIX", "5d", "5m")
    
    if spx_new is None or spy_new is None or vix_new is None:
        return

    # دمج البيانات
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
        combined.tail(15000).to_csv(LOCAL_DB_CSV, index=False)
    else:
        df.tail(15000).to_csv(LOCAL_DB_CSV, index=False)

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
# ADVANCED FEATURE ENGINEERING (مؤشرات متقدمة + SPY مستقل + VIX Divergence)
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
    
    # 1. Momentum متعدد الفترات
    for n in (1, 3, 6, 12):
        df[f"spx_ret_{n}"] = c.pct_change(n)
        df[f"spy_ret_{n}"] = df["spy_close"].pct_change(n)
    
    df["rsi"] = rsi(c)
    df["spx_atr"] = atr(df, "spx_high", "spx_low", "spx_close")
    df["atr_pct"] = df["spx_atr"] / c
    
    # 2. VWAP المستقل لـ SPY و SPX
    typical_spy = (df["spy_close"] + df["spy_close"] + df["spy_close"]) / 3 # تقريبي أو حقيقي
    df["spy_vwap"] = (df["spy_close"] * df["spy_volume"]).groupby(df["date"]).cumsum() / df["spy_volume"].groupby(df["date"]).cumsum().replace(0, np.nan)
    df["spy_vwap"] = df["spy_vwap"].fillna(df["spy_close"])
    df["spy_vwap_dist"] = (df["spy_close"] - df["spy_vwap"]) / df["spy_close"]

    # 3. VIX Momentum & Divergence
    df["vix_chg_5"] = df["vix"].diff(5)
    df["vix_chg_15"] = df["vix"].diff(15)
    df["vix_spx_divergence"] = np.where((df["vix_chg_5"] > 0) & (df["spx_ret_1"] > 0), 1, 0) # تعارض الخوف مع الصعود

    # 4. Opening Range (ORH / ORL) للأول 30 دقيقة
    df["mins_open"] = df["timestamp"].dt.hour * 60 + df["timestamp"].dt.minute - MARKET_OPEN_MIN
    is_or = df["mins_open"] <= 30
    df["orh"] = df.groupby("date")["spx_high"].transform(lambda x: x[is_or].max())
    df["orl"] = df.groupby("date")["spx_low"].transform(lambda x: x[is_or].min())
    df["dist_from_orh"] = (c - df["orh"]) / c
    df["dist_from_orl"] = (c - df["orl"]) / c

    # 5. Smart Volume Z-Score
    vol_mean = df["spy_volume"].rolling(30, min_periods=5).mean()
    vol_std = df["spy_volume"].rolling(30, min_periods=5).std().replace(0, 1)
    df["volume_zscore"] = (df["spy_volume"] - vol_mean) / vol_std

    # 6. Target: هل يحقق هدف الخيار قبل الوقف؟ (Trade Outcome Target)
    fut_max = df.groupby("date")["spx_high"].shift(-HORIZON)
    fut_min = df.groupby("date")["spx_low"].shift(-HORIZON)
    thr = 0.6 * df["spx_atr"]
    
    target = pd.Series(np.nan, index=df.index)
    # صعود يضرب الهدف أولاً
    target[(fut_max - c >= thr) & (c - fut_min < thr)] = 1.0
    # هبوط يضرب الهدف أولاً
    target[(c - fut_min >= thr) & (fut_max - c < thr)] = 0.0
    df["target"] = target
    return df

# ============================================================
# ENSEMBLE MODELS & CALIBRATION (النماذج المتعددة + المعايرة)
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
    
    subsets = {
        "trend": TREND_FEATURES,
        "momentum": MOMENTUM_FEATURES,
        "volatility": VOLATILITY_FEATURES
    }
    
    split_idx = int(len(data) * 0.8)
    train_df = data.iloc[:split_idx]
    test_df = data.iloc[split_idx:]

    for name, feats in subsets.items():
        base = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.03, max_iter=200, random_state=42)
        base.fit(train_df[feats], train_df["target"])
        
        # معايرة الاحتمالات (Probability Calibration)
        calibrated = CalibratedClassifierCV(estimator=base, method='sigmoid', cv='prefit')
        calibrated.fit(test_df[feats], test_df["target"])
        models[name] = calibrated
        
        preds = calibrated.predict_proba(test_df[feats])[:, 1]
        aucs.append(roc_auc_score(test_df["target"], preds))

    mean_auc = float(np.mean(aucs))
    say(f"Ensemble Trained. Average AUC = {mean_auc:.3f}")
    return models, mean_auc

# ============================================================
# BLACK-SCHOLES & PRICING
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
# SIGNAL & ENSEMBLE VOTING
# ============================================================
def calculate_ensemble_signal(df):
    if df is None or len(df) < 50:
        return {"direction": "WAIT", "reasons": ["بيانات غير كافية"]}
    last = df.iloc[-1]
    
    age = (now_ny() - last["timestamp"]).total_seconds() / 60
    if age > MAX_BAR_AGE_MIN:
        return {"direction": "WAIT", "reasons": [f"بيانات متأخرة ({age:.1f} دقيقة)"]}

    if not STATE["models"] or STATE["auc"] < MIN_AUC:
        return {"direction": "WAIT", "reasons": [f"النموذج غير جاهز أو AUC منخفض ({STATE['auc']:.3f})"]}

    # تصويت النماذج الثلاثة (Ensemble Voting)
    p_trend = STATE["models"]["trend"].predict_proba(pd.DataFrame([last[TREND_FEATURES]]))[0][1]
    p_mom = STATE["models"]["momentum"].predict_proba(pd.DataFrame([last[MOMENTUM_FEATURES]]))[0][1]
    p_vol = STATE["models"]["volatility"].predict_proba(pd.DataFrame([last[VOLATILITY_FEATURES]]))[0][1]

    # متوسط الاحتمالات المعايرة
    avg_prob_up = (p_trend + p_mom + p_vol) / 3.0
    avg_prob_dn = 1.0 - avg_prob_up

    # التحقق من الإجماع
    if avg_prob_up >= MIN_PROBABILITY and (p_trend > 0.6 and p_mom > 0.6):
        return {"direction": "CALL", "probability": avg_prob_up, "reasons": ["إجماع صعودي قوي للـ Ensemble"]}
    elif avg_prob_dn >= MIN_PROBABILITY and (p_trend < 0.4 and p_mom < 0.4):
        return {"direction": "PUT", "probability": avg_prob_dn, "reasons": ["إجماع هبوطي قوي للـ Ensemble"]}
    
    return {"direction": "WAIT", "probability": max(avg_prob_up, avg_prob_dn), "reasons": ["لا يوجد إجماع كافٍ بين النماذج"]}

def build_recommendation(df):
    if not market_time_ok() or not session_allowed():
        return {"status": "WAIT", "reason": "خارج النافذة المسموحة"}
    
    if STATE["daily_pnl"] <= -ACCOUNT_EQUITY * MAX_DAILY_LOSS:
        return {"status": "WAIT", "reason": "بلوغ الحد الأقصى للخسارة اليومية (3%)"}

    sig = calculate_ensemble_signal(df)
    if sig["direction"] == "WAIT":
        return {"status": "WAIT", "reason": sig["reasons"][0]}

    spot = float(df.iloc[-1]["spx_close"])
    vix = float(df.iloc[-1]["vix"])
    kind = sig["direction"]
    strike = ceil(spot / STRIKE_STEP) * STRIKE_STEP + STRIKE_STEP * (STRIKE_OFFSET_STEPS - 1) if kind == "CALL" \
        else floor(spot / STRIKE_STEP) * STRIKE_STEP - STRIKE_STEP * (STRIKE_OFFSET_STEPS - 1)

    ml = mins_left()
    entry = bs_price(spot, strike, ml, vix, kind) * (1.0 + SLIPPAGE_PENALTY_PCT)
    if entry < MIN_PREMIUM:
        return {"status": "WAIT", "reason": "البريميوم أقل من الحد الأدنى"}

    stop, target = entry * (1 - STOP_LOSS_PCT), entry * (1 + TAKE_PROFIT_PCT)
    contracts = int((ACCOUNT_EQUITY * RISK_PER_TRADE) / (entry * STOP_LOSS_PCT * 100))
    if contracts <= 0:
        return {"status": "WAIT", "reason": "المخاطرة لا تسمح بعقد واحد"}

    return {
        "status": kind, "probability": sig["probability"], "spx": spot, "vix": vix,
        "strike": strike, "entry": entry, "stop": stop, "target": target, "contracts": contracts
    }

# ============================================================
# TRUE HISTORICAL BACKTEST ENGINE (--backtest)
# ============================================================
def run_backtest():
    """محرك اختبار تاريخي حقيقي على قاعدة البيانات المتراكمة محلياً"""
    if not os.path.exists(LOCAL_DB_CSV):
        print("لا توجد بيانات تاريخية كافية في قاعدة البيانات المحلية.")
        return
    
    df = pd.read_csv(LOCAL_DB_CSV)
    df = prepare(df)
    data = df.dropna(subset=TREND_FEATURES + ["target"]).reset_index(drop=True)
    if len(data) < 200:
        print("البيانات المحلية قليلة جداً لإجراء اختبار تاريخي مفيد.")
        return

    print("=" * 65)
    print("🚀 بدء تشغيل محرك الـ Backtest التاريخي المؤسسي (v6 PRO QUANT)")
    print("=" * 65)
    
    # محاكاة إشارات تاريخية وتقييم الموثوقية (Reliability / Brier Score)
    results = []
    y_true, y_prob = [], []
    
    # تدريب نموذج اختبار على أول 70% وتقييم على الـ 30% الباقية (Out-of-Sample)
    split = int(len(data) * 0.7)
    train_set = data.iloc[:split]
    test_set = data.iloc[split:]
    
    m = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.03, max_iter=200, random_state=42)
    m.fit(train_set[TREND_FEATURES], train_set["target"])
    
    probs = m.predict_proba(test_set[TREND_FEATURES])[:, 1]
    trues = test_set["target"].values
    
    brier = brier_score_loss(trues, probs)
    auc = roc_auc_score(trues, probs)
    
    print(f"• إجمالي عينات الاختبار (Out-of-Sample): {len(test_set)}")
    print(f"• معدل الدقة الإحصائية (AUC): {auc:.3f}")
    print(f"• مقياس المعايرة (Brier Score - الأقل أفضل): {brier:.4f}")
    
    # فئات الثقة واختبار النسبة الفعلية للنجاح
    print("\n--- تحليل الموثوقية حسب فئات الثقة (Confidence Buckets) ---")
    for threshold in [0.60, 0.70, 0.75, 0.80]:
        mask = (probs >= threshold) | (probs <= (1 - threshold))
        if mask.sum() > 0:
            sub_true = trues[mask]
            sub_prob = probs[mask]
            # الحساب بناء على الاتجاه المتوقع
            predicted_dir = (sub_prob >= 0.5).astype(int)
            actual_win = (predicted_dir == sub_true).mean() * 100
            print(f"  * الثقة >= {int(threshold*100)}%: عدد الإشارات ({mask.sum()}) | نسبة النجاح الفعلي: {actual_win:.1f}%")
        else:
            print(f"  * الثقة >= {int(threshold*100)}%: لا توجد إشارات كافية")
    print("=" * 65)

# ============================================================
# MAIN LOOP
# ============================================================
def main():
    if "--backtest" in sys.argv:
        run_backtest()
        return

    say("SPX 0DTE v6 PRO QUANT — بدء التشغيل المؤسسي")
    reset_daily_state()
    
    df_raw = get_data()
    if df_raw is None or df_raw.empty:
        raise RuntimeError("تعذر جلب البيانات أو تهيئة القاعدة المحلية.")
    
    df_prep = prepare(df_raw)
    models, auc = train_ensemble(df_prep)
    STATE.update(models=models, auc=auc, last_train=now_ny())
    say(f"تم تدريب النماذج بنجاح. متوسط AUC = {STATE['auc']:.3f}")

    while True:
        try:
            reset_daily_state()
            if not market_time_ok():
                time.sleep(45)
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

            if STATE["open"] is None:
                rec = build_recommendation(df_prep)
                if rec["status"] in ("CALL", "PUT"):
                    STATE["signals_today"] += 1
                    STATE["open"] = rec
                    msg = f"🚀 SPX v6 QUANT SIGNAL: {rec['status']} | الثقة: {rec['probability']*100:.1f}% | Strike: {rec['strike']}"
                    say(msg)
                    send_telegram(msg)

            time.sleep(POLL_SECONDS)
        except KeyboardInterrupt:
            say("تم إيقاف النظام.")
            break
        except Exception as e:
            say(f"MAIN LOOP ERROR: {e}")
            time.sleep(POLL_SECONDS)

if __name__ == "__main__":
    main()
