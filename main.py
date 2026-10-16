# -*- coding: utf-8 -*-
"""صَفاء التحليل v26.5 — Unified Bot (30 MegaCaps + Dynamic Option Pricing)"""
import os, time, json, math, sqlite3, logging, traceback, re
from datetime import datetime, timedelta, date, time as dtime
from zoneinfo import ZoneInfo
from pathlib import Path
from typing import Optional

import requests
import numpy as np
import pandas as pd
import yfinance as yf

APP_NAME = "صَفاء التحليل"
VERSION = "v26.5-30-MegaCaps"
ET = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("safaa")

ALPACA_KEY = os.getenv("APCA_API_KEY_ID", "").strip()
ALPACA_SECRET = os.getenv("APCA_API_SECRET_KEY", "").strip()
ALPACA_DATA_URL = os.getenv("ALPACA_DATA_URL", "https://data.alpaca.markets").rstrip("/")
TRADING_URL = os.getenv("ALPACA_TRADING_URL", "https://paper-api.alpaca.markets").rstrip("/")
STOCK_FEED = os.getenv("ALPACA_STOCK_FEED", "iex").strip().lower()
OPTIONS_FEED = os.getenv("ALPACA_OPTIONS_FEED", "indicative").strip().lower()
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()
DB_PATH = os.getenv("DB_PATH", "./data/safaa_v26.db")
ARTIFACT_DIR = Path(os.getenv("ARTIFACT_DIR", "./data/artifacts"))
POLL_SECONDS = max(60, int(os.getenv("POLL_SECONDS", "120")))

# القائمة المحدثة للـ 30 شركة الكبرى
TOP_LEADERS_MAP = {
    "NVDA": 0.07,  # إنفيديا
    "AAPL": 0.07,  # آبل
    "GOOGL": 0.05, # جوجل
    "MSFT": 0.06,  # مايكروسوفت
    "AMZN": 0.05,  # أمازون
    "TSM": 0.03,   # تايوان لأشباه الموصلات
    "META": 0.04,  # ميتا
    "AVGO": 0.03,  # برودكوم
    "TSLA": 0.04,  # تسلا
    "LLY": 0.03,   # إيلي ليلي
    "BRK-B": 0.03, # بيركشاير
    "MU": 0.02,    # ميكرون
    "AMD": 0.03,   # إيه إم دي
    "WMT": 0.02,   # وول مارت
    "JPM": 0.02,   # جي بي مورغان
    "V": 0.02,     # فيزا
    "XOM": 0.02,   # إكسون موبيل
    "JNJ": 0.02,   # جونسون آند جونسون
    "MA": 0.02,    # ماستركارد
    "ABBV": 0.02,  # أبفي
    "PLTR": 0.02,  # بالانتير
    "INTC": 0.02,  # إنتل
    "CVX": 0.02,   # شيفرون
    "COST": 0.02,  # كوستكو
    "CSCO": 0.02,  # سيسكو
    "KO": 0.02,    # كوكاكولا
    "ORCL": 0.02,  # أوراكل
    "UNH": 0.02,   # يونايتد هيلث
    "NFLX": 0.02,  # نتفلكس
    "GS": 0.02     # غولدمان ساكس
}

SESSION_OPEN = dtime(9, 30)
SESSION_CLOSE = dtime(16, 0)

session = requests.Session()
session.headers.update({"APCA-API-KEY-ID": ALPACA_KEY, "APCA-API-SECRET-KEY": ALPACA_SECRET, "accept": "application/json"})

def now_et() -> datetime:
    return datetime.now(ET)

def init_db():
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as con:
        con.execute("""CREATE TABLE IF NOT EXISTS daily_reports (
            report_date TEXT NOT NULL, report_type TEXT NOT NULL, sent_at TEXT NOT NULL,
            PRIMARY KEY(report_date, report_type))""")

def alpaca_get(path, params=None, timeout=20):
    if not ALPACA_KEY or not ALPACA_SECRET:
        raise RuntimeError("Missing Alpaca API keys")
    full_url = (TRADING_URL if "/v2/options" in path else ALPACA_DATA_URL) + path
    response = session.get(full_url, params=params or {}, timeout=timeout)
    if response.status_code == 429:
        raise RuntimeError("Alpaca rate limit reached (429)")
    response.raise_for_status()
    return response.json()

def tg_text(text: str):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        log.info("Telegram text skipped: %s", text[:100])
        return False
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    chunks = [text[i:i+3900] for i in range(0, len(text), 3900)] or [" "]
    for chunk in chunks:
        res = requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": chunk, "parse_mode": "HTML", "disable_web_page_preview": True}, timeout=30)
        res.raise_for_status()
    return True

def format_readable_option(raw_symbol: str, underlying: str, exp_date: str, opt_type: str, strike: float) -> str:
    try:
        dt_obj = datetime.strptime(exp_date, "%Y-%m-%d")
        date_str = dt_obj.strftime("%d%m%y")
        cp_char = "C" if opt_type.lower() in ["c", "call", "صعود 🚀"] else "P"
        return f"{underlying} {date_str} {cp_char} {int(strike) if strike.is_integer() else strike}"
    except Exception:
        return raw_symbol

def get_option_quote(symbol):
    try:
        payload = alpaca_get("/v1beta1/options/quotes/latest", {"symbols": symbol, "feed": OPTIONS_FEED})
        q = (payload.get("quotes") or {}).get(symbol, {})
        bid, ask = float(q.get("bp", 0)), float(q.get("ap", 0))
        if bid > 0 and ask >= bid:
            return {"bid": bid, "ask": ask, "mid": (bid + ask) / 2}
    except Exception:
        pass
    return None

# -------------------- SPY-DRIVEN ENGINE FOR SPXW --------------------
def fetch_spy_bars():
    end = datetime.now(UTC)
    start = end - timedelta(days=3)
    try:
        data = alpaca_get("/v2/stocks/bars", {
            "symbols": "SPY", "timeframe": "5Min",
            "start": start.isoformat(), "end": end.isoformat(),
            "limit": 1000, "feed": STOCK_FEED, "sort": "asc"
        })
        bars = data.get("bars", {}).get("SPY", [])
        if bars:
            df = pd.DataFrame(bars)
            df["timestamp"] = pd.to_datetime(df["t"], utc=True)
            df = df.set_index("timestamp").rename(columns={"o":"open", "h":"high", "l":"low", "c":"close", "v":"volume"})
            if len(df) > 3:
                df = df.iloc[:-2]
            return df[["open", "high", "low", "close", "volume"]].astype(float)
    except Exception as e:
        log.warning(f"Failed to fetch SPY bars: {e}")
    
    ticker = yf.Ticker("SPY")
    df = ticker.history(period="10d", interval="5m", auto_adjust=False, prepost=False)
    if df is not None and not df.empty:
        df = df.rename(columns={c: c.lower() for c in df.columns})
        df.index = pd.to_datetime(df.index, utc=True)
        cols = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
        df = df[cols].dropna(subset=["close"]).sort_index()
        if len(df) > 3:
            df = df.iloc[:-2]
        return df
    return pd.DataFrame()

def feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    close = x["close"].astype(float)
    x["r1"] = close.pct_change(1)
    x["r3"] = close.pct_change(3)
    x["r6"] = close.pct_change(6)
    x["r12"] = close.pct_change(12)
    x["vol12"] = x["r1"].rolling(12).std()
    x["ema_gap"] = close / close.ewm(span=12, adjust=False).mean() - 1
    x["range"] = (x["high"] - x["low"]) / close
    x["vchg"] = x["volume"].pct_change().replace([np.inf, -np.inf], np.nan).fillna(0)
    idx = x.index.tz_convert(ET)
    mins = np.array([i.hour * 60 + i.minute for i in idx])
    x["tod_sin"] = np.sin(2 * np.pi * mins / 1440)
    x["tod_cos"] = np.cos(2 * np.pi * mins / 1440)
    return x.replace([np.inf, -np.inf], np.nan)

FEATURES = ["r1", "r3", "r6", "r12", "vol12", "ema_gap", "range", "vchg", "tod_sin", "tod_cos"]

def get_spx_contract_signal_via_spy():
    try:
        df = fetch_spy_bars()
        if df.empty:
            return {"ok": False}
        x = feature_frame(df).dropna(subset=FEATURES)
        if x.empty:
            return {"ok": False}
        
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.pipeline import make_pipeline
        from sklearn.impute import SimpleImputer
        
        future = df["close"].shift(-6) / df["close"] - 1
        x["target"] = (future > 0).astype(int)
        ds = x[FEATURES + ["target"]].dropna()
        if len(ds) < 150:
            return {"ok": False}
            
        model = make_pipeline(SimpleImputer(strategy="median"), RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42))
        model.fit(ds[FEATURES], ds["target"])
        
        prob_up = float(model.predict_proba(x[FEATURES].iloc[[-1]])[0, 1])
        signal_dir = "كول (CALL) 🟢" if prob_up >= 0.5 else "بيت (PUT) 🔴"
        
        spy_current_price = float(df["close"].iloc[-1])
        spx_estimated_price = spy_current_price * 10.0
        
        today_str = now_et().date().isoformat()
        opt_type = "call" if prob_up >= 0.5 else "put"
        strike_price = round(spx_estimated_price / 5) * 5
        readable_symbol = format_readable_option("SPXW", "SPXW", today_str, opt_type, strike_price)
        
        entry_ask = 3.50  # قيمة افتراضية في حال تعذر جلب السعر الحي للمؤشر
        try:
            payload = alpaca_get("/v2/options/contracts", {
                "underlying_symbols": "SPX", "status": "active",
                "type": opt_type, "expiration_date": today_str, "limit": 30
            })
            contracts = payload.get("option_contracts") or payload.get("contracts") or []
            if contracts:
                contracts.sort(key=lambda c: abs(float(c.get("strike_price", 0)) - spx_estimated_price))
                best_contract = contracts[0]
                strike_price = float(best_contract.get("strike_price", strike_price))
                readable_symbol = format_readable_option(best_contract.get("symbol", ""), "SPXW", today_str, opt_type, strike_price)
                q = get_option_quote(best_contract.get("symbol"))
                if q:
                    entry_ask = q["ask"]
        except Exception:
            pass

        recent = df.tail(78)
        high_spx = float(recent["high"].max()) * 10.0
        low_spx = float(recent["low"].min()) * 10.0

        return {
            "ok": True, "signal": signal_dir, "prob_up": prob_up,
            "current_price": spx_estimated_price, "contract_symbol": readable_symbol,
            "strike": strike_price, "entry_price": entry_ask,
            "exit_short_term": round(entry_ask * 1.5, 2),  # هدف المدى القريب +50% ديناميكياً
            "exit_long_term": round(entry_ask * 3.0, 2),   # هدف المدى البعيد +200% ديناميكياً
            "high_day": high_spx, "low_day": low_spx,
            "short_term_text": "صعود تدريجي وتجاوز المقاومة القريبة" if prob_up >= 0.5 else "ضغط هبوطي واختبار الدعم القريب",
            "long_term_text": "اتجاه إيجابي قوي ومستهدفات عليا للمدى البعيد" if prob_up >= 0.5 else "استمرار الزخم السلبي وتعميق الهبوط للمدى البعيد"
        }
    except Exception as e:
        log.exception(f"SPX Error: {e}")
        return {"ok": False}

# -------------------- PRE-MARKET REPORT --------------------
def check_premarket_report():
    now = now_et()
    if now.weekday() < 5 and now.hour == 8 and 25 <= now.minute <= 35:
        day_str = now.date().isoformat()
        with sqlite3.connect(DB_PATH) as con:
            sent = con.execute("SELECT 1 FROM daily_reports WHERE report_date=? AND report_type='premarket'", (day_str,)).fetchone()
            if sent:
                return
            con.execute("INSERT INTO daily_reports(report_date,report_type,sent_at) VALUES(?,?,?)", (day_str, 'premarket', datetime.now(UTC).isoformat()))
        
        spx = get_spx_contract_signal_via_spy()
        if spx.get("ok"):
            msg = (
                f"🌅 <b>تقرير ما قبل السوق لمؤشر SPX</b>\n"
                f"📅 التاريخ: {day_str}\n\n"
                f"📊 <b>توقع السوق بشكل عام:</b> {spx['signal']}\n"
                f"📈 <b>أعلى قمة متوقعة للجلسة:</b> ${spx['high_day']:,.2f}\n"
                f"📉 <b>أقل قاع متوقع للجلسة:</b> ${spx['low_day']:,.2f}\n"
                f"🎯 <b>توقع المدى القريب:</b> {spx['short_term_text']}\n"
                f"🌐 <b>توقع المدى البعيد:</b> {spx['long_term_text']}\n\n"
                f"⚠️ تنبيه تحليلي افتراضي"
            )
            tg_text(msg)

# -------------------- MEGA CAPS CONTRACTS (DYNAMIC PRICING) --------------------
def scan_mega_caps_contracts():
    now = now_et()
    if now.weekday() >= 5 or not (SESSION_OPEN <= now.time() <= SESSION_CLOSE):
        return

    today = now.date()
    target_expiries = [today, today + timedelta(days=3)]

    for ticker in TOP_LEADERS_MAP.keys():
        try:
            bars = alpaca_get(f"/v2/stocks/bars", {"symbols": ticker, "timeframe": "1Min", "limit": 2})
            b_list = bars.get("bars", {}).get(ticker, [])
            if not b_list:
                continue
            price = float(b_list[-1]["c"])
            prev_price = float(b_list[0]["c"])
            pct_change = (price / prev_price - 1) * 100
            direction = "صعود 🚀" if pct_change >= 0 else "نزول 📉"
            opt_type = "call" if pct_change >= 0 else "put"

            for exp in target_expiries:
                exp_str = exp.isoformat()
                payload = alpaca_get(f"/v2/options/contracts", {
                    "underlying_symbols": ticker.replace("-", "."),
                    "status": "active", "type": opt_type,
                    "expiration_date": exp_str, "limit": 30
                })
                contracts = payload.get("option_contracts") or payload.get("contracts") or []
                if not contracts:
                    continue
                
                contracts.sort(key=lambda c: abs(float(c.get("strike_price", 0)) - price))
                best_c = contracts[0]
                contract_symbol = best_c.get("symbol")
                strike = float(best_c.get("strike_price", 0))
                readable_sym = format_readable_option(contract_symbol, ticker, exp_str, opt_type, strike)
                
                # جلب السعر الحي الحقيقي (Ask) للعقد ديناميكياً
                q = get_option_quote(contract_symbol)
                entry_ask = q["ask"] if q else 1.00 # قيمة احتياطية إذا لم يتوفر تسعير مباشر
                
                # حساب مستهدفات الخروج ديناميكياً بناءً على سعر الـ Ask الفعلي
                exit_short = round(entry_ask * 1.5, 2)  # هدف المدى القريب (+50%)
                exit_long = round(entry_ask * 3.0, 2)   # هدف المدى البعيد (+200%)

                exp_label = "عقد اليوم الواحد (0DTE)" if exp == today else f"عقد ينتهي بعد 3 أيام ({exp_str})"
                
                msg = (
                    f"🏢 <b>عقد الشركة الكبرى: {ticker}</b>\n\n"
                    f"🔹 <b>الشركة:</b> {ticker} (السعر: ${price:.2f} | {pct_change:+.2f}%)\n"
                    f"▫️ <b>رقم العقد:</b> <code>{readable_sym}</code>\n"
                    f"▫️ <b>تاريخ الانتهاء:</b> {exp_str} ({exp_label})\n"
                    f"▫️ <b>التوقع والاتجاه:</b> {direction}\n"
                    f"▫️ <b>سعر الـ Strike:</b> ${strike:.2f}\n"
                    f"▫️ <b>سعر الدخول الفعلي (Ask):</b> ${entry_ask:.2f}\n"
                    f"▫️ <b>الخروج (المدى القريب):</b> ${exit_short:.2f}\n"
                    f"▫️ <b>الخروج (المدى البعيد):</b> ${exit_long:.2f}\n\n"
                    f"⚠️ تنبيه تحليلي افتراضي"
                )
                tg_text(msg)
                time.sleep(1)
                break
        except Exception as e:
            log.warning(f"Error scanning mega cap {ticker}: {e}")

# -------------------- REAL-TIME TELEGRAM ALERT FOR SPXW --------------------
def run_spx_telegram_alert():
    now = now_et()
    if now.weekday() >= 5 or not (SESSION_OPEN <= now.time() <= SESSION_CLOSE):
        return

    spx = get_spx_contract_signal_via_spy()
    if not spx.get("ok"):
        return

    msg = (
        f"📊 <b>توقع مؤشر SPX / Spax (لحظي)</b>\n\n"
        f"▫️ <b>التوقع (كول / بوت):</b> {spx['signal']}\n"
        f"▫️ <b>رقم العقد:</b> <code>{spx['contract_symbol']}</code>\n"
        f"▫️ <b>سعر الـ Strike:</b> ${spx['strike']:,.2f}\n"
        f"▫️ <b>سعر المؤشر الحالي:</b> ${spx['current_price']:,.2f}\n"
        f"▫️ <b>سعر الدخول الفعلي:</b> ${spx['entry_price']:.2f}\n"
        f"▫️ <b>الخروج (المدى القريب):</b> ${spx['exit_short_term']:.2f} — ({spx['short_term_text']})\n"
        f"▫️ <b>الخروج (المدى البعيد):</b> ${spx['exit_long_term']:.2f} — ({spx['long_term_text']})\n\n"
        f"⚠️ تنبيه تحليلي افتراضي"
    )
    tg_text(msg)

# -------------------- MAIN LOOP --------------------
def main():
    init_db()
    log.info(f"Starting {APP_NAME} {VERSION}...")
    try:
        tg_text(f"🚀 تم تشغيل نظام {APP_NAME} ({VERSION}) بنجاح مع إضافة الـ 30 شركة وتفعيل الأسعار الديناميكية المباشرة للعقود!")
    except Exception:
        pass

    while True:
        try:
            check_premarket_report()
            run_spx_telegram_alert()
            scan_mega_caps_contracts()
        except Exception as e:
            log.exception(f"Main loop error: {e}")
        
        time.sleep(POLL_SECONDS)

if __name__ == "__main__":
    main()
