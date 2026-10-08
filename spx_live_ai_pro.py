# ============================================================
# SPX 0DTE ADVISOR v2 PRO MAX - مع فلتر الطوارئ والهبوط العنيف
# ============================================================

import os
import sys
import csv
import json
import time
import warnings
from math import erf, sqrt, floor, ceil
from datetime import datetime, time as dtime
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

POLL_SECONDS = 30
MARKET_OPEN_MIN = 9 * 60 + 30      # 09:30
MARKET_CLOSE_MIN = 16 * 60         # 16:00
NO_TRADE_FIRST_MIN = 20
NO_TRADE_LAST_MIN = 60
MAX_BAR_AGE_MIN = 30

# ML
MIN_TRAIN_ROWS = 800
HORIZON = 15
MOVE_ATR_MULT = 0.5
MIN_PROBABILITY = 0.62
MIN_EDGE = 0.10
MIN_AUC = 0.54
RETRAIN_EVERY_MIN = 120

# Risk & Filters
MAX_VIX = 30.0
MIN_VIX = 11.0
VIX_EMERGENCY_JUMP = 3.0           # قفزة طوارئ VIX
EXTREME_MOVE_ATR_MULT = 2.0        # فلتر الهبوط/الصعود العنيف (أكثر من ضعفي الـ ATR في شمعة)

ACCOUNT_EQUITY = 25000
RISK_PER_TRADE = 0.01
MAX_DAILY_LOSS = 0.03
MAX_CONSECUTIVE_LOSSES = 3
MAX_SIGNALS_PER_DAY = 5
COOLDOWN_MIN = 10
COMMISSION_PER_CONTRACT = 0.65

# Options (Black-Scholes)
STRIKE_STEP = 5.0
STRIKE_OFFSET_STEPS = 1
IV_MULT = 1.0
MIN_PREMIUM = 1.00
STOP_LOSS_PCT = 0.45
TAKE_PROFIT_PCT = 0.75
MAX_HOLD_MINUTES = 40
FORCE_EXIT_BEFORE_CLOSE_MIN = 10

TRADES_CSV = "paper_trades.csv"
SIGNALS_JSONL = "spx_signals_v2.jsonl"

FEATURES = [
    "ret_1", "ret_3", "ret_5", "ret_15",
    "rsi", "atr_pct", "ema_spread", "body_ratio",
    "vwap_distance", "vix", "vix_chg", "vol_ratio", "mins_open",
]

STATE = {
    "model": None, "auc": 0.0, "last_train": None,
    "date": None, "daily_pnl": 0.0, "loss_streak": 0, "signals_today": 0,
    "open": None, "last_exit_time": None, "daily_summary_sent": False,
    "weekly_report_sent_date": None, "extreme_market_alerted": False,
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
# DATA
# ============================================================
def _download(symbol, period, interval):
    df = yf.download(symbol, period=period, interval=interval,
                     progress=False, auto_adjust=False)
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

def get_data():
    try:
        spx = _download("^GSPC", "5d", "1m")
        if spx is None:
            return None
        spy = _download("SPY", "5d", "1m")
        vix = _download("^VIX", "5d", "5m")

        df = spx[["timestamp", "open", "high", "low", "close"]].copy()
        if spy is not None:
            df = df.merge(spy[["timestamp", "volume"]], on="timestamp", how="left")
            df["volume"] = df["volume"].fillna(0.0)
        else:
            df["volume"] = 0.0

        if vix is not None:
            df = pd.merge_asof(df.sort_values("timestamp"),
                               vix[["timestamp", "close"]].rename(columns={"close": "vix"}),
                               on="timestamp", direction="backward")
            df["vix"] = df["vix"].ffill().bfill()
        else:
            df["vix"] = 18.0

        t = df["timestamp"].dt.hour * 60 + df["timestamp"].dt.minute
        df = df[(t >= MARKET_OPEN_MIN) & (t < MARKET_CLOSE_MIN)]
        return df.dropna().reset_index(drop=True)
    except Exception as e:
        say(f"Data fetch error: {e}")
        return None

# ============================================================
# FEATURES + TARGET
# ============================================================
def rsi(series, period=14):
    d = series.diff()
    g = d.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    l = (-d.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    return 100 - 100 / (1 + g / l.replace(0, np.nan))

def atr(df, period=14):
    pc = df["close"].shift(1)
    tr = pd.concat([df["high"] - df["low"], (df["high"] - pc).abs(),
                    (df["low"] - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()

def prepare(raw):
    df = raw.copy()
    df["date"] = df["timestamp"].dt.date
    c = df["close"]
    for n in (1, 3, 5, 15):
        df[f"ret_{n}"] = c.pct_change(n)
    df["rsi"] = rsi(c)
    df["atr"] = atr(df)
    df["atr_pct"] = df["atr"] / c
    ema9 = c.ewm(span=9, adjust=False).mean()
    ema21 = c.ewm(span=21, adjust=False).mean()
    df["ema_9"], df["ema_21"] = ema9, ema21
    df["ema_spread"] = (ema9 - ema21) / c
    rng = (df["high"] - df["low"]).replace(0, np.nan)
    df["body_ratio"] = (c - df["open"]).abs() / rng

    typical = (df["high"] + df["low"] + c) / 3
    g = df["date"]
    if df["volume"].sum() > 0:
        vwap = (typical * df["volume"]).groupby(g).cumsum() / df["volume"].groupby(g).cumsum().replace(0, np.nan)
        df["vwap"] = vwap.fillna(typical)
    else:
        df["vwap"] = typical.groupby(g).expanding().mean().reset_index(level=0, drop=True)
    df["vwap_distance"] = (c - df["vwap"]) / c

    df["vix_chg"] = df["vix"].diff(15)
    df["vol_ratio"] = df["volume"] / df["volume"].rolling(30, min_periods=5).mean().replace(0, np.nan)
    df["vol_ratio"] = df["vol_ratio"].fillna(1.0)
    df["mins_open"] = df["timestamp"].dt.hour * 60 + df["timestamp"].dt.minute - MARKET_OPEN_MIN

    fut = df.groupby("date")["close"].shift(-HORIZON) - c
    thr = MOVE_ATR_MULT * df["atr"]
    target = pd.Series(np.nan, index=df.index)
    target[fut > thr] = 1.0
    target[fut < -thr] = 0.0
    df["target"] = target
    return df

# ============================================================
# TRAINING + WALK-FORWARD
# ============================================================
def _model(iters):
    return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.03,
                                          max_iter=iters, l2_regularization=1.0,
                                          random_state=42)

def train(df):
    data = df.dropna(subset=FEATURES + ["target"]).reset_index(drop=True)
    if len(data) < MIN_TRAIN_ROWS:
        return None, 0.0
    chunk = len(data) // 4
    aucs = []
    for i in range(3):
        train_end = chunk * (i + 1)
        test_start = train_end + HORIZON
        test = data.iloc[test_start:test_start + chunk]
        tr = data.iloc[:train_end]
        if len(test) < 50 or tr["target"].nunique() < 2 or test["target"].nunique() < 2:
            continue
        m = _model(200).fit(tr[FEATURES], tr["target"])
        aucs.append(roc_auc_score(test["target"], m.predict_proba(test[FEATURES])[:, 1]))
    auc = float(np.mean(aucs)) if aucs else 0.5
    final = _model(250).fit(data[FEATURES], data["target"])
    return final, auc

# ============================================================
# BLACK-SCHOLES
# ============================================================
def _cdf(x):
    return 0.5 * (1 + erf(x / sqrt(2)))

def bs_price(spot, strike, minutes_left, vix, kind):
    T = max(minutes_left, 0.5) / (252 * 390)
    sigma = max(vix, 5) / 100 * IV_MULT
    d1 = (np.log(spot / strike) + 0.5 * sigma ** 2 * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    if kind == "CALL":
        p = spot * _cdf(d1) - strike * _cdf(d2)
    else:
        p = strike * _cdf(-d2) - spot * _cdf(-d1)
    return max(float(p), 0.05)

def spot_for_premium(target, strike, minutes_left, vix, kind, spot):
    lo, hi = spot - 150, spot + 150
    for _ in range(60):
        mid = (lo + hi) / 2
        p = bs_price(mid, strike, minutes_left, vix, kind)
        if (p < target) == (kind == "CALL"):
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2

# ============================================================
# SIGNAL & RECOMMENDATION (مع فلتر العنف والهبوط)
# ============================================================
def calculate_signal(df):
    if df is None or len(df) < 100:
        return {"direction": "WAIT", "reasons": ["بيانات غير كافية"]}
    last = df.iloc[-1]
    
    # 🚨 فلتر العنف والهبوط الاستثنائي (Market Crash / Extreme Volatility Filter)
    price_change_1m = abs(last["close"] - df.iloc[-2]["close"]) if len(df) > 1 else 0
    if price_change_1m > (EXTREME_MOVE_ATR_MULT * last["atr"]):
        if not STATE.get("extreme_market_alerted", False):
            send_telegram("⚠️ تنبيه طوارئ: تم رصد هبوط أو صعود عنيف استثنائي في السوق! تم تجميد إصدار التشارات مؤقتاً.")
            STATE["extreme_market_alerted"] = True
        return {"direction": "WAIT", "reasons": ["سوق عنيف / تذبذب استثنائي غير مأمون"]}

    age = (now_ny() - last["timestamp"]).total_seconds() / 60
    if age > MAX_BAR_AGE_MIN:
        return {"direction": "WAIT", "reasons": [f"بيانات قديمة ({age:.0f} دقيقة)"]}
    if STATE["model"] is None or STATE["auc"] < MIN_AUC:
        return {"direction": "WAIT", "reasons": ["النموذج غير جاهز أو بلا ميزة إحصائية"]}
    vix = float(last["vix"])
    if vix > MAX_VIX or vix < MIN_VIX:
        return {"direction": "WAIT", "reasons": [f"VIX غير مسموح ({vix:.1f})"]}

    p_up = float(STATE["model"].predict_proba(pd.DataFrame([last[FEATURES]]))[0][1])
    p_dn = 1 - p_up
    reasons = ["EMA صاعد" if last["ema_9"] > last["ema_21"] else "EMA هابط",
               "فوق VWAP" if last["close"] > last["vwap"] else "تحت VWAP"]
    if p_up >= MIN_PROBABILITY and p_up - p_dn >= MIN_EDGE:
        d, p = "CALL", p_up
    elif p_dn >= MIN_PROBABILITY and p_dn - p_up >= MIN_EDGE:
        d, p = "PUT", p_dn
    else:
        d, p = "WAIT", max(p_up, p_dn)
        reasons.append("الحافة ضعيفة")
    return {"direction": d, "probability": p, "reasons": reasons,
            "spot": float(last["close"]), "vix": vix, "atr": float(last["atr"])}

def build_recommendation(df):
    if not market_time_ok() or not session_allowed():
        return {"status": "WAIT", "reason": "خارج النافذة المسموحة"}
    if STATE["signals_today"] >= MAX_SIGNALS_PER_DAY or STATE["loss_streak"] >= MAX_CONSECUTIVE_LOSSES:
        return {"status": "WAIT", "reason": "توقف مؤقت حسب قواعد المخاطر"}

    s = calculate_signal(df)
    if s["direction"] == "WAIT":
        return {"status": "WAIT", "reason": s["reasons"][0], "signal": s}

    spot, kind = s["spot"], s["direction"]
    strike = ceil(spot / STRIKE_STEP) * STRIKE_STEP + STRIKE_STEP * (STRIKE_OFFSET_STEPS - 1) if kind == "CALL" \
        else floor(spot / STRIKE_STEP) * STRIKE_STEP - STRIKE_STEP * (STRIKE_OFFSET_STEPS - 1)

    ml = mins_left()
    entry = bs_price(spot, strike, ml, s["vix"], kind)
    if entry < MIN_PREMIUM:
        return {"status": "WAIT", "reason": "البريميوم أقل من الحد الأدنى", "signal": s}
    stop, target = entry * (1 - STOP_LOSS_PCT), entry * (1 + TAKE_PROFIT_PCT)
    contracts = max(1, int(ACCOUNT_EQUITY * RISK_PER_TRADE / (entry * STOP_LOSS_PCT * 100)))
    return {
        "status": kind, "probability": s["probability"], "spx": spot, "vix": s["vix"],
        "strike": strike, "entry": entry, "stop": stop, "target": target,
        "spx_target": spot_for_premium(target, strike, ml - 10, s["vix"], kind, spot),
        "spx_stop": spot_for_premium(stop, strike, ml - 10, s["vix"], kind, spot),
        "contracts": contracts, "signal": s,
    }

def format_recommendation(r):
    if r["status"] == "WAIT":
        return f"⏳ SPX 0DTE PRO MAX\n⚪ WAIT — {r['reason']}"
    emoji = "🟢" if r["status"] == "CALL" else "🔴"
    return f"""
{emoji} SPX 0DTE PRO MAX — {r['status']}
الثقة: {r['probability'] * 100:.1f}% | SPX {r['spx']:.2f} | VIX {r['vix']:.1f}
العقد: SPXW {r['status']} Strike {r['strike']:.0f}
اشترِ: ~${r['entry']:.2f} | هدف: ~${r['target']:.2f} | وقف: ~${r['stop']:.2f}
العقود: {r['contracts']} | أقصى مدة: {MAX_HOLD_MINUTES} دقيقة
""".strip()

# ============================================================
# PAPER TRADING & MANAGEMENT
# ============================================================
def open_paper_trade(r):
    STATE["open"] = {
        "kind": r["status"], "strike": r["strike"], "entry": r["entry"],
        "stop": r["stop"], "target": r["target"], "contracts": r["contracts"],
        "spx_entry": r["spx"], "time": now_ny(), "prob": r["probability"],
        "max_prem": r["entry"], "min_prem": r["entry"], "vix_at_entry": r["vix"]
    }

def manage_open_trade(spot, vix):
    t = STATE["open"]
    if not t:
        return
    ml = mins_left()
    prem = bs_price(spot, t["strike"], ml, vix, t["kind"])
    
    t["max_prem"] = max(t["max_prem"], prem)
    t["min_prem"] = min(t["min_prem"], prem)

    held = (now_ny() - t["time"]).total_seconds() / 60
    reason = None

    if abs(vix - t.get("vix_at_entry", vix)) >= VIX_EMERGENCY_JUMP:
        reason = "VIX_EMERGENCY"
    elif prem <= t["stop"]:
        reason = "STOP"
    elif prem >= t["target"]:
        reason = "TARGET"
    elif held >= MAX_HOLD_MINUTES:
        reason = "TIME"
    elif ml <= FORCE_EXIT_BEFORE_CLOSE_MIN:
        reason = "CLOSE"

    if not reason:
        return

    pnl = (prem - t["entry"]) * 100 * t["contracts"] - 2 * COMMISSION_PER_CONTRACT * t["contracts"]
    STATE["daily_pnl"] += pnl
    STATE["loss_streak"] = STATE["loss_streak"] + 1 if pnl < 0 else 0
    STATE["last_exit_time"] = now_ny()

    new = not os.path.exists(TRADES_CSV)
    with open(TRADES_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["entry_time", "exit_time", "kind", "strike", "spx_entry", "spx_exit",
                        "prem_entry", "prem_exit", "max_prem", "min_prem", "contracts", "pnl", "reason", "prob"])
        w.writerow([t["time"].isoformat(), now_ny().isoformat(), t["kind"], t["strike"],
                    round(t["spx_entry"], 2), round(spot, 2), round(t["entry"], 2),
                    round(prem, 2), round(t["max_prem"], 2), round(t["min_prem"], 2),
                    t["contracts"], round(pnl, 2), reason, round(t["prob"], 3)])

    msg = f"🔔 خروج صفقة ({reason}) — PnL: ${pnl:+.0f} (اليومي: ${STATE['daily_pnl']:+.0f})"
    say(msg)
    send_telegram(msg)
    STATE["open"] = None

def check_and_send_daily_summary():
    n = now_ny()
    if minutes_of_day(n) >= 16 * 0 and not STATE.get("daily_summary_sent", False):
        if os.path.exists(TRADES_CSV):
            d = pd.read_csv(TRADES_CSV)
            today_str = n.strftime("%Y-%m-%d")
            d_today = d[d["entry_time"].str.startswith(today_str)]
            if not d_today.empty:
                wins = len(d_today[d_today.pnl > 0])
                total = len(d_today)
                pnl_sum = d_today.pnl.sum()
                summary_msg = f"📊 **ملخص الأداء اليومي ({today_str}):**\n- عدد الصفقات: {total}\n- الرابحة: {wins} | الخاسرة: {total - wins}\n- صافي PnL: ${pnl_sum:+.0f}"
                send_telegram(summary_msg)
        STATE["daily_summary_sent"] = True

def check_and_send_weekly_report():
    n = now_ny()
    if n.weekday() == 4 and minutes_of_day(n) >= 16 * 60 + 15:
        today_str = n.strftime("%Y-%m-%d")
        if STATE.get("weekly_report_sent_date") != today_str:
            if os.path.exists(TRADES_CSV):
                d = pd.read_csv(TRADES_CSV)
                wins = d[d.pnl > 0]
                losses = d[d.pnl <= 0]
                pf = wins.pnl.sum() / abs(losses.pnl.sum()) if len(losses) and losses.pnl.sum() != 0 else float("inf")
                weekly_msg = f"📈 **تقرير الأداء الأسبوعي الآلي:**\n- إجمالي الصفقات: {len(d)}\n- نسبة النجاح: {len(wins)/len(d)*100:.1f}%\n- إجمالي PnL: ${d.pnl.sum():+.0f}\n- عامل الربح (Profit Factor): {pf:.2f}"
                send_telegram(weekly_msg)
            STATE["weekly_report_sent_date"] = today_str

# ============================================================
# MAIN
# ============================================================
def main():
    say("SPX 0DTE v2 PRO MAX — بدء التشغيل مع فلتر الطوارئ")
    reset_daily_state()
    raw = get_data()
    if raw is None or raw.empty:
        raise RuntimeError("تعذر جلب البيانات.")
    
    df_prep = prepare(raw)
    m, auc = train(df_prep)
    STATE.update(model=m, auc=auc, last_train=now_ny())
    say(f"النموذج جاهز. AUC = {STATE['auc']:.3f}")

    while True:
        try:
            reset_daily_state()
            check_and_send_daily_summary()
            check_and_send_weekly_report()

            if not market_time_ok():
                time.sleep(60)
                continue

            raw = get_data()
            if raw is None or len(raw) < 100:
                time.sleep(POLL_SECONDS)
                continue

            df = prepare(raw)
            if (now_ny() - STATE["last_train"]).total_seconds() > RETRAIN_EVERY_MIN * 60:
                m, auc = train(df)
                if m:
                    STATE.update(model=m, auc=auc, last_train=now_ny())

            last = df.iloc[-1]
            spot, vix = float(last["close"]), float(last["vix"])

            manage_open_trade(spot, vix)

            if STATE["open"] is None:
                rec = build_recommendation(df)
                if rec["status"] in ("CALL", "PUT"):
                    STATE["signals_today"] += 1
                    open_paper_trade(rec)
                    send_telegram(format_recommendation(rec))
                    with open(SIGNALS_JSONL, "a", encoding="utf-8") as f:
                        f.write(json.dumps({k: rec[k] for k in
                                ("status", "probability", "spx", "strike", "entry", "stop", "target")},
                                ensure_ascii=False) + "\n")

            time.sleep(POLL_SECONDS)
        except KeyboardInterrupt:
            say("تم الإيقاف.")
            break
        except Exception as e:
            say(f"MAIN LOOP ERROR: {e}")
            time.sleep(POLL_SECONDS)

if __name__ == "__main__":
    main()
