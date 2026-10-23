# -*- coding: utf-8 -*-
"""
صَفاء التحليل v59 — Dual Intermarket Liquidity Filter Edition (Nasdaq VWAP Confirmation)
توصيات تحليليّة وإشارات فقط — لا يوجد تنفيذ تلقائي للأوامر.
"""

import os
import time
import logging
import sqlite3
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import requests
import pandas as pd
import numpy as np
import yfinance as yf

# ================= ================= =================
# الإعدادات الأساسية
# ================= ================= =================

APP_NAME = "صَفاء التحليل"
APP_NAME_ASCII = "SafaaAnalysis"  # اسم متوافق مع ترويسات HTTP
VERSION = "v59-INTERMARKET"

NY = ZoneInfo("America/New_York")
UTC = timezone.utc

ALPACA_KEY = os.getenv("APCA_API_KEY_ID", "").strip()
ALPACA_SECRET = os.getenv("APCA_API_SECRET_KEY", "").strip()
TRADING_URL = os.getenv("ALPACA_TRADING_URL", "https://paper-api.alpaca.markets").rstrip("/")
DATA_URL = os.getenv("ALPACA_DATA_URL", "https://data.alpaca.markets").rstrip("/")
STOCK_FEED = os.getenv("ALPACA_STOCK_FEED", "iex").strip()
OPTIONS_FEED = os.getenv("ALPACA_OPTIONS_FEED", "indicative").strip().lower()
TG_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TG_CHAT = os.getenv("TELEGRAM_CHAT_ID", "").strip()
DB_PATH = os.getenv("DB_PATH", "./data/safaa_v59.db")

POLL_SECONDS = int(os.getenv("POLL_SECONDS", "20"))
MAX_SPREAD_PCT = float(os.getenv("MAX_SPREAD_PCT", "0.15"))
MIN_MOVE_POINTS = float(os.getenv("MIN_MOVE_POINTS", "2.5"))
MAX_ALERTS_PER_DAY = int(os.getenv("MAX_ALERTS_PER_DAY", "15"))

# قائمة المؤشرات والأسهم القيادية المفحوصة
SCAN_STOCKS = [s.strip().upper() for s in os.getenv(
    "SCAN_STOCKS", "NVDA,AAPL,TSLA,META,AMZN,AMD,QQQ,SPY,DIA"
).split(",") if s.strip()]

session = requests.Session()
# إصلاح خطأ latin-1: استخدام اسم باللغة الإنجليزية في User-Agent
session.headers.update({"User-Agent": f"{APP_NAME_ASCII}/{VERSION}"})

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("safaa")


def now_ny():
    return datetime.now(NY)


def get_target_expiration(symbol):
    today = now_ny().date()
    if symbol in ("SPY", "QQQ", "DIA", "SPX"):
        return today.isoformat(), "0DTE (ينتهي اليوم)"
    
    days_ahead = 4 - today.weekday()
    if days_ahead < 0:
        days_ahead += 7
    exp_date = today + timedelta(days=days_ahead)
    
    if exp_date == today:
        return today.isoformat(), "0DTE (ينتهي اليوم)"
    else:
        days_left = (exp_date - today).days
        return exp_date.isoformat(), f"أسبوعي (ينتهي بعد {days_left} أيام)"


def alpaca_headers():
    return {
        "APCA-API-KEY-ID": ALPACA_KEY,
        "APCA-API-SECRET-KEY": ALPACA_SECRET,
        "Accept": "application/json",
    }


def ensure_db():
    folder = os.path.dirname(DB_PATH)
    if folder:
        os.makedirs(folder, exist_ok=True)
    with sqlite3.connect(DB_PATH) as con:
        con.execute("""
            CREATE TABLE IF NOT EXISTS daily_state (
                day TEXT PRIMARY KEY,
                sent_count INTEGER NOT NULL DEFAULT 0,
                pm_report_sent INTEGER NOT NULL DEFAULT 0,
                eod_report_sent INTEGER NOT NULL DEFAULT 0
            )
        """)


def daily_state_get():
    today = now_ny().date().isoformat()
    with sqlite3.connect(DB_PATH) as con:
        row = con.execute("SELECT sent_count, pm_report_sent, eod_report_sent FROM daily_state WHERE day=?", (today,)).fetchone()
        if row:
            return {"sent_count": int(row[0]), "pm_report_sent": bool(row[1]), "eod_report_sent": bool(row[2])}
        return {"sent_count": 0, "pm_report_sent": False, "eod_report_sent": False}


def increment_alert_count():
    today = now_ny().date().isoformat()
    with sqlite3.connect(DB_PATH) as con:
        con.execute("""
            INSERT INTO daily_state(day, sent_count, pm_report_sent, eod_report_sent) VALUES (?, 1, 0, 0)
            ON CONFLICT(day) DO UPDATE SET sent_count=sent_count+1
        """, (today,))


def mark_pm_sent():
    today = now_ny().date().isoformat()
    with sqlite3.connect(DB_PATH) as con:
        con.execute("""
            INSERT INTO daily_state(day, sent_count, pm_report_sent, eod_report_sent) VALUES (?, 0, 1, 0)
            ON CONFLICT(day) DO UPDATE SET pm_report_sent=1
        """, (today,))


def mark_eod_sent():
    today = now_ny().date().isoformat()
    with sqlite3.connect(DB_PATH) as con:
        con.execute("""
            INSERT INTO daily_state(day, sent_count, pm_report_sent, eod_report_sent) VALUES (?, 0, 0, 1)
            ON CONFLICT(day) DO UPDATE SET eod_report_sent=1
        """, (today,))


def telegram_send(message):
    if not TG_TOKEN or not TG_CHAT:
        log.warning("بيانات التليجرام (TG_TOKEN أو TG_CHAT) غير مكتملة.")
        return False
    url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
    try:
        r = session.post(url, json={
            "chat_id": TG_CHAT,
            "text": message,
            "parse_mode": "Markdown",
            "disable_web_page_preview": True,
        }, timeout=15)
        if r.status_code != 200:
            log.error("خطأ تليجرام (%s): %s", r.status_code, r.text)
            return False
        return True
    except Exception as exc:
        log.error("استثناء تليجرام: %s", exc)
        return False


def market_phase():
    now = now_ny()
    if now.weekday() >= 5:
        return "CLOSED"
    minute = now.hour * 60 + now.minute
    # توسيع النافذة من 8:00 صباحاً حتى 9:30 صباحاً نيويورك لضمان عدم تفويت تقرير ما قبل الافتتاح
    if 8 * 60 <= minute < 9 * 60 + 30:
        return "PREMARKET_HOUR"
    elif 9 * 60 + 30 <= minute < 16 * 60:
        return "RTH"
    elif 16 * 60 <= minute < 16 * 60 + 30:
        return "POST_MARKET"
    return "CLOSED"


# ================= ================= =================
# الخوارزميات، السيولة المزدوجة، والتحليل الفني
# ================= ================= =================

def get_stock_bars(symbol, timeframe="1Min"):
    end = datetime.now(UTC)
    start = end - timedelta(days=2)
    url = f"{DATA_URL}/v2/stocks/{symbol}/bars"
    params = {
        "timeframe": timeframe,
        "start": start.isoformat().replace("+00:00", "Z"),
        "end": end.isoformat().replace("+00:00", "Z"),
        "limit": 300,
        "feed": STOCK_FEED,
    }
    try:
        r = session.get(url, headers=alpaca_headers(), params=params, timeout=15)
        if r.status_code != 200:
            log.warning("فشل جلب بيانات Alpaca لـ %s | الكود: %s", symbol, r.status_code)
            return pd.DataFrame()
        rows = r.json().get("bars", {}).get(symbol, [])
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows).rename(columns={"c":"close","h":"high","l":"low","v":"volume","t":"timestamp"})
        return df
    except Exception as exc:
        log.error("خطأ في جلب بيانات %s: %s", symbol, exc)
        return pd.DataFrame()


def filter_today_bars(df):
    """فلترة البيانات لليوم الحالي فقط لحساب VWAP اللحظي بدقة"""
    if df.empty or "timestamp" not in df.columns:
        return df
    try:
        df["dt"] = pd.to_datetime(df["timestamp"])
        latest_date = df["dt"].dt.date.max()
        df_today = df[df["dt"].dt.date == latest_date].copy()
        return df_today if len(df_today) >= 3 else df
    except Exception:
        return df


def get_qqq_liquidity_bias():
    """
    قراءة اتجاه السيولة المؤسسية في ناسداك (QQQ)
    لتأكيد اتجاه SPY و SPX وباقي الأسهم القيادية
    """
    df_qqq = get_stock_bars("QQQ", timeframe="1Min")
    df_qqq = filter_today_bars(df_qqq)
    
    if df_qqq.empty or len(df_qqq) < 10:
        return "NEUTRAL"

    close = df_qqq["close"].astype(float)
    volume = df_qqq["volume"].astype(float)

    # حساب VWAP اللحظي للجلسة الحالية
    vol_sum = volume.cumsum()
    vwap = (volume * close).cumsum() / np.where(vol_sum == 0, 1, vol_sum)

    if close.iloc[-1] > vwap.iloc[-1]:
        return "BULLISH"
    elif close.iloc[-1] < vwap.iloc[-1]:
        return "BEARISH"

    return "NEUTRAL"


def detect_qqq_early_setup(df):
    """خوارزمية مخصصة لمؤشر ناسداك QQQ تعتمد على VWAP والانضغاط السعري"""
    df_session = filter_today_bars(df)
    if df_session is None or len(df_session) < 15:
        df_session = df
        
    if len(df_session) < 15:
        return None, 0.0

    close = df_session["close"].astype(float)
    volume = df_session["volume"].astype(float)

    vol_sum = volume.cumsum()
    vwap = (volume * close).cumsum() / np.where(vol_sum == 0, 1, vol_sum)

    std_recent = close.tail(6).std()
    squeeze_ratio = (std_recent / close.iloc[-1]) * 100.0

    vol_prior = volume.iloc[-15:-3].mean()
    vol_surge = volume.tail(3).mean() / vol_prior if vol_prior > 0 else 1.0

    is_squeezing = squeeze_ratio <= 0.14
    has_volume = vol_surge >= 1.35

    if is_squeezing and has_volume:
        if close.iloc[-1] > vwap.iloc[-1]:
            return "CALL", min(96, int(83 + (vol_surge * 5)))
        elif close.iloc[-1] < vwap.iloc[-1]:
            return "PUT", min(96, int(83 + (vol_surge * 5)))

    return None, 0.0


def detect_strong_setup(df):
    """خوارزمية الأسهم والمؤشرات الأخرى (SPY, DIA, NVDA, AAPL, إلخ)"""
    if df is None or len(df) < 20:
        return None, 0.0

    close = df["close"].astype(float)
    volume = df["volume"].astype(float)

    std_recent = close.tail(6).std()
    squeeze_ratio = (std_recent / close.iloc[-1]) * 100.0

    vol_recent_mean = volume.tail(3).mean()
    vol_prior_mean = volume.iloc[-15:-3].mean()
    volume_surge = (vol_recent_mean / vol_prior_mean) if vol_prior_mean > 0 else 1.0

    ema5 = close.ewm(span=5, adjust=False).mean().iloc[-1]
    ema13 = close.ewm(span=13, adjust=False).mean().iloc[-1]

    is_squeezing = squeeze_ratio <= 0.16
    has_volume_buildup = volume_surge >= 1.30

    if is_squeezing and has_volume_buildup:
        direction = "CALL" if ema5 > ema13 else "PUT"
        confidence = min(96, int(80 + (volume_surge * 6)))
        return direction, confidence

    return None, 0.0


def get_premarket_forecast_all():
    spy_data = None
    try:
        # إضافة prepost=True لضمان جلب بيانات ما قبل الافتتاح
        df_spy = yf.download("SPY", period="5d", interval="5m", prepost=True, progress=False)
        if not df_spy.empty:
            if isinstance(df_spy.columns, pd.MultiIndex):
                df_spy.columns = df_spy.columns.get_level_values(0)
            
            latest_spy = float(df_spy["Close"].iloc[-1])
            returns_spy = df_spy["Close"].pct_change().dropna()
            vol_spy = returns_spy.std() * np.sqrt(78) * latest_spy
            move_spy = max(3.5, vol_spy)
            
            spy_low = round(latest_spy - move_spy, 2)
            spy_high = round(latest_spy + move_spy, 2)
            
            spx_low = round(spy_low * 10, 2)
            spx_high = round(spy_high * 10, 2)
            
            ma20 = df_spy["Close"].tail(20).mean()
            spy_trend = "صاعد 🟢" if latest_spy > ma20 else ("هابط 🔴" if latest_spy < ma20 else "عرضي ⚪")
            confidence = int(min(92, max(75, 80 + (abs(latest_spy - ma20) / ma20) * 1000)))
            
            spy_data = {
                "spy_low": spy_low,
                "spy_high": spy_high,
                "spx_low": spx_low,
                "spx_high": spx_high,
                "trend": spy_trend,
                "confidence": confidence
            }
    except Exception as exc:
        log.error("SPY forecast error: %s", exc)

    companies_data = []
    for sym in SCAN_STOCKS:
        if sym in ("SPY", "SPX"):
            continue
        try:
            df = get_stock_bars(sym, timeframe="5Min")
            if df.empty:
                continue
            spot = float(df["close"].iloc[-1])
            ret = df["close"].pct_change().dropna()
            vol = ret.std() * np.sqrt(78) * spot
            move = max(MIN_MOVE_POINTS, vol)
            
            low_p = round(spot - move, 2)
            high_p = round(spot + move, 2)
            
            ma15 = df["close"].tail(15).mean()
            direction = "CALL 🟢" if spot >= ma15 else "PUT 🔴"
            target_strike = round(spot + MIN_MOVE_POINTS, 1) if spot >= ma15 else round(spot - MIN_MOVE_POINTS, 1)
            
            companies_data.append({
                "symbol": sym,
                "spot": spot,
                "low": low_p,
                "high": high_p,
                "direction": direction,
                "target_strike": target_strike
            })
        except Exception as exc:
            log.warning("Company forecast skip %s: %s", sym, exc)

    return spy_data, companies_data


def find_high_conviction_option(symbol, direction, spot):
    target_move = 15.0 if symbol == "SPX" else MIN_MOVE_POINTS
    target_strike = round(spot + target_move) if direction == "CALL" else round(spot - target_move)
    
    exp_date, exp_type_label = get_target_expiration(symbol)

    url = f"{TRADING_URL}/v2/options/contracts"
    params = {
        "underlying_symbols": symbol,
        "expiration_date": exp_date,
        "type": direction.lower(),
        "status": "active",
        "limit": 100,
    }
    r = session.get(url, headers=alpaca_headers(), params=params, timeout=15)
    if r.status_code != 200:
        return None
        
    data = r.json()
    contracts = data.get("option_contracts", data.get("contracts", []))
    if isinstance(contracts, dict):
        contracts = contracts.get("option_contracts", [])
        
    best_contract = None
    min_diff = 999
    for c in contracts:
        s_price = float(c.get("strike_price", 0))
        diff = abs(s_price - target_strike)
        if diff < min_diff:
            min_diff = diff
            best_contract = c
            
    if not best_contract:
        return None

    q_url = f"{DATA_URL}/v1beta1/options/quotes/latest"
    q_r = session.get(q_url, headers=alpaca_headers(), params={"symbols": best_contract["symbol"], "feed": OPTIONS_FEED}, timeout=15)
    if q_r.status_code != 200:
        return None
    
    q_data = q_r.json().get("quotes", {}).get(best_contract["symbol"], {})
    ask = float(q_data.get("ap", 0) or 0)
    bid = float(q_data.get("bp", 0) or 0)
    if ask <= 0 or bid <= 0:
        return None
        
    mid = round((ask + bid) / 2, 2)
    spread = (ask - bid) / mid if mid > 0 else 1.0
    
    if spread > MAX_SPREAD_PCT:
        return None

    return {
        "symbol": symbol,
        "direction": direction,
        "spot": spot,
        "strike": float(best_contract["strike_price"]),
        "contract": best_contract["symbol"],
        "ask": ask,
        "bid": bid,
        "mid": mid,
        "target_move": target_move,
        "exp_date": exp_date,
        "exp_label": exp_type_label
    }


# ================= ================= =================
# تنسيق الرسائل
# ================= ================= =================

def format_premarket_report(spy_fc, comp_fc):
    lines = [
        f"🎯 *توقعات التداول اليومية (قبل الافتتاح)*",
        f"📅 `{now_ny().strftime('%Y-%m-%d')}`",
        f"━━━━━━━━━━━━━━━━━━"
    ]
    if spy_fc:
        lines.extend([
            f"📊 *SPX المتوقع:* `{spy_fc['spx_low']} — {spy_fc['spx_high']}`",
            f"📊 *SPY المتوقع:* `{spy_fc['spy_low']} — {spy_fc['spy_high']}`",
            f"📈 *أعلى قمة:* `{spy_fc['spy_high']}` (SPY) / `{spy_fc['spx_high']}` (SPX)",
            f"📉 *أدنى قمة:* `{spy_fc['spy_low']}` (SPY) / `{spy_fc['spx_low']}` (SPX)",
            f"🧭 *اتجاه السوق:* {spy_fc['trend']} | *الثقة:* `{spy_fc['confidence']}%`",
            f"━━━━━━━━━━━━━━━━━━"
        ])

    if comp_fc:
        lines.append("🏢 *توقعات عقود الشركات والمؤشرات:*")
        for c in comp_fc:
            lines.append(
                f"• *{c['symbol']}* (`{c['spot']:.2f}`): النطاق `{c['low']} — {c['high']}` | الاتجاه: {c['direction']} | السترايك: `{c['target_strike']}`"
            )
        lines.append("━━━━━━━━━━━━━━━━━━")
    return "\n".join(lines)


def format_conviction_alert(opt, confidence):
    dir_ar = "شراء CALL 🟢" if opt["direction"] == "CALL" else "شراء PUT 🔴"
    return (
        f"🔥 *توصية عالية القوة (توقع حركة {opt['target_move']}+ نقطة)*\n"
        f"━━━━━━━━━━━━━━━━━━\n"
        f"📌 *الأصل:* `{opt['symbol']}`\n"
        f"🧭 *الاتجاه المتوقع:* {dir_ar}\n"
        f"📅 *نوع العقد:* `{opt['exp_label']}` (تاريخ: `{opt['exp_date']}`)\n"
        f"💹 *السعر الحالي (قبل الكسر):* `{opt['spot']:.2f}`\n"
        f"🎯 *السترايك المستهدف:* `{opt['strike']:.2f}`\n"
        f"📄 *العقد:* `{opt['contract']}`\n"
        f"🟢 *Ask:* `${opt['ask']:.2f}` | ⚖️ *Mid:* `${opt['mid']:.2f}`\n"
        f"🎯 *نسبة ثقة الخوارزمية:* `{confidence}%`\n"
        f"━━━━━━━━━━━━━━━━━━"
    )


def format_eod_report():
    today_str = now_ny().strftime('%Y-%m-%d')
    state = daily_state_get()
    try:
        df_spy = get_stock_bars("SPY", timeframe="5Min")
        close_spy = df_spy["close"].iloc[-1] if not df_spy.empty else 0.0
        high_spy = df_spy["close"].max() if not df_spy.empty else 0.0
        low_spy = df_spy["close"].min() if not df_spy.empty else 0.0
        close_spx = close_spy * 10
    except Exception:
        close_spy, high_spy, low_spy, close_spx = 0.0, 0.0, 0.0, 0.0

    return (
        f"🔔 *تقرير إغلاق السوق وحصاد التوصيات*\n"
        f"📅 `{today_str}`\n"
        f"━━━━━━━━━━━━━━━━━━\n"
        f"📊 *إغلاق SPX:* `{close_spx:.2f}`\n"
        f"📊 *إغلاق SPY:* `{close_spy:.2f}`\n"
        f"📈 *أعلى سعر لـ SPY اليوم:* `{high_spy:.2f}`\n"
        f"📉 *أدنى سعر لـ SPY اليوم:* `{low_spy:.2f}`\n"
        f"━━━━━━━━━━━━━━━━━━\n"
        f"📬 *عدد التوصيات المرسلة اليوم:* `{state['sent_count']} توصيات`\n"
        f"✅ *حالة النظام:* مكتمل ومستقر.\n"
        f"━━━━━━━━━━━━━━━━━━"
    )


# ================= ================= =================
# تشغيل المسح وتطبيق فلتر سيولة ناسداك المزدوج
# ================= ================= =================

def run_premarket():
    state = daily_state_get()
    if state["pm_report_sent"]:
        log.info("تقرير ما قبل الافتتاح تم إرساله سابقاً لليوم.")
        return
        
    log.info("جاري إعداد تقرير ما قبل الافتتاح...")
    spy_fc, comp_fc = get_premarket_forecast_all()
    
    if spy_fc or comp_fc:
        msg = format_premarket_report(spy_fc, comp_fc)
        if telegram_send(msg):
            mark_pm_sent()
            log.info("تم إرسال تقرير التوقعات قبل الافتتاح بنجاح.")
        else:
            log.error("فشل إرسال تقرير ما قبل الافتتاح عبر التليجرام.")
    else:
        log.warning("لم يتم الحصول على بيانات كافية لإرسال تقرير ما قبل الافتتاح.")


def run_rth():
    state = daily_state_get()
    if state["sent_count"] >= MAX_ALERTS_PER_DAY:
        return

    qqq_bias = get_qqq_liquidity_bias()

    for symbol in SCAN_STOCKS:
        if daily_state_get()["sent_count"] >= MAX_ALERTS_PER_DAY:
            break
            
        df = get_stock_bars(symbol, timeframe="1Min")
        if df.empty or len(df) < 15:
            continue
            
        if symbol == "QQQ":
            direction, confidence = detect_qqq_early_setup(df)
        else:
            direction, confidence = detect_strong_setup(df)
            
        if not direction:
            continue

        if symbol != "QQQ":
            if direction == "CALL" and qqq_bias == "BEARISH":
                log.info("تم إلغاء توصية CALL على %s بسبب سلبية سيولة ناسداك.", symbol)
                continue
            elif direction == "PUT" and qqq_bias == "BULLISH":
                log.info("تم إلغاء توصية PUT على %s بسبب إيجابية سيولة ناسداك.", symbol)
                continue

        spot = float(df["close"].iloc[-1])
        opt = find_high_conviction_option(symbol, direction, spot)
        if opt:
            msg = format_conviction_alert(opt, confidence)
            if telegram_send(msg):
                increment_alert_count()
                log.info("تم إرسال توصية قوية (مؤكدة بسيولة ناسداك): %s %s", symbol, direction)


def run_post_market():
    state = daily_state_get()
    if state["eod_report_sent"]:
        return
    msg = format_eod_report()
    if telegram_send(msg):
        mark_eod_sent()
        log.info("تم إرسال تقرير إغلاق السوق.")


def main():
    ensure_db()
    log.info("تشغيل البوت %s %s...", APP_NAME, VERSION)

    startup_msg = (
        f"🚀 *تم تشغيل نظام {APP_NAME} ({VERSION}) بنجاح*\n"
        f"⏱️ *التوقيت:* `{now_ny().strftime('%Y-%m-%d %H:%M:%S %Z')}`\n"
        f"🛡️ *فلتر سيولة ناسداك المزدوج:* `مفعل 🟢`\n"
        f"✅ *الحالة:* النظام يعمل وفي انتظار نافذة السوق."
    )
    telegram_send(startup_msg)

    while True:
        try:
            phase = market_phase()
            if phase == "PREMARKET_HOUR":
                run_premarket()
            elif phase == "RTH":
                run_rth()
            elif phase == "POST_MARKET":
                run_post_market()
            else:
                log.info("السوق مغلق حالياً | التوقيت: %s", now_ny().strftime("%Y-%m-%d %H:%M:%S %Z"))
        except Exception as exc:
            log.error("خطأ تشغيلي: %s", exc)
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
