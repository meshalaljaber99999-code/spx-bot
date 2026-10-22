# -*- coding: utf-8 -*-
"""
صَفاء التحليل v49 — SPX/SPXW data fix
Recommendations only. NO automatic order execution.
- SPX spot: Yahoo Finance ^GSPC (validated for freshness)
- SPXW/stock option contracts: Alpaca Trading API, if enabled on the account
- Option quotes: Alpaca Options Market Data API
- Refuses to alert when data is missing, stale, or unreliable.
"""
import os
import time
import json
import logging
import sqlite3
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import requests
import pandas as pd
import numpy as np
import yfinance as yf

APP_NAME = "صَفاء التحليل"
VERSION = "v49-SPX-DATA-FIX"

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
DB_PATH = os.getenv("DB_PATH", "./data/safaa_v49.db")

POLL_SECONDS = int(os.getenv("POLL_SECONDS", "60"))
MAX_SPX_AGE_SECONDS = int(os.getenv("MAX_SPX_AGE_SECONDS", "240"))
MAX_STOCK_BAR_AGE_SECONDS = int(os.getenv("MAX_STOCK_BAR_AGE_SECONDS", "600"))
MAX_OPTION_QUOTE_AGE_SECONDS = int(os.getenv("MAX_OPTION_QUOTE_AGE_SECONDS", "120"))
MAX_SPREAD_PCT = float(os.getenv("MAX_SPREAD_PCT", "0.20"))
MIN_SIGNAL_SCORE = float(os.getenv("MIN_SIGNAL_SCORE", "72"))
MAX_SPX_STRIKE_DISTANCE = float(os.getenv("MAX_SPX_STRIKE_DISTANCE", "20"))
MAX_STOCK_STRIKE_DISTANCE_PCT = float(os.getenv("MAX_STOCK_STRIKE_DISTANCE_PCT", "0.025"))
MIN_OPTION_MID = float(os.getenv("MIN_OPTION_MID", "0.10"))
MAX_OPTION_MID = float(os.getenv("MAX_OPTION_MID", "30"))
MAX_ALERTS_PER_DAY = int(os.getenv("MAX_ALERTS_PER_DAY", "5"))
SCAN_STOCKS = [s.strip().upper() for s in os.getenv(
    "SCAN_STOCKS", "SPY,QQQ,NVDA,AAPL,MSFT,AMD,TSLA,META,AMZN"
).split(",") if s.strip()]
INCLUDE_SPXW = os.getenv("INCLUDE_SPXW", "true").lower() in ("1", "true", "yes")

session = requests.Session()
session.headers.update({"User-Agent": f"{APP_NAME}/{VERSION}"})

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("safaa")


def now_ny():
    return datetime.now(NY)


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
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                underlying TEXT NOT NULL,
                direction TEXT NOT NULL,
                option_symbol TEXT NOT NULL,
                strike REAL,
                spot REAL,
                bid REAL,
                ask REAL,
                mid REAL,
                spread_pct REAL,
                score REAL,
                data_source TEXT,
                status TEXT DEFAULT 'OPEN'
            )
        """)
        con.execute("""
            CREATE TABLE IF NOT EXISTS daily_state (
                day TEXT PRIMARY KEY,
                sent_count INTEGER NOT NULL DEFAULT 0
            )
        """)


def telegram_send(message):
    if not TG_TOKEN or not TG_CHAT:
        log.warning("Telegram variables missing; message not sent.")
        return False
    url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
    try:
        r = session.post(url, json={
            "chat_id": TG_CHAT,
            "text": message,
            "disable_web_page_preview": True,
        }, timeout=15)
        if r.status_code != 200:
            log.error("Telegram HTTP %s: %s", r.status_code, r.text[:250])
            return False
        return True
    except Exception as exc:
        log.error("Telegram error: %s", exc)
        return False


def get_spx_bars():
    """
    Yahoo symbol ^GSPC is used only as the SPX index reference.
    This is not a guaranteed exchange tick feed; freshness is checked.
    No SPY*10 proxy is ever used.
    """
    try:
        raw = yf.download(
            "^GSPC", period="1d", interval="1m",
            auto_adjust=False, progress=False, threads=False
        )
        if raw is None or raw.empty:
            raise RuntimeError("Yahoo returned no ^GSPC minute bars")
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        raw = raw.rename(columns={
            "Open": "open", "High": "high", "Low": "low",
            "Close": "close", "Volume": "volume",
        })
        raw.index = pd.to_datetime(raw.index)
        if raw.index.tz is None:
            # Yahoo intraday indexes are normally exchange-local; make this explicit.
            raw.index = raw.index.tz_localize(NY)
        else:
            raw.index = raw.index.tz_convert(NY)
        raw = raw.dropna(subset=["close"]).sort_index()
        if raw.empty:
            raise RuntimeError("^GSPC bars contain no close prices")
        stamp = raw.index[-1].to_pydatetime()
        age = (now_ny() - stamp).total_seconds()
        price = float(raw["close"].iloc[-1])
        log.info("SPX ^GSPC %.2f | bar=%s | age=%.0fs", price, stamp.isoformat(), age)
        if age < -30 or age > MAX_SPX_AGE_SECONDS:
            raise RuntimeError(f"SPX data stale/invalid (age={age:.0f}s)")
        return raw, price, stamp
    except Exception as exc:
        log.error("SPX data unavailable: %s", exc)
        return pd.DataFrame(), None, None


def get_stock_bars(symbol):
    end = datetime.now(UTC)
    start = end - timedelta(days=3)
    url = f"{DATA_URL}/v2/stocks/{symbol}/bars"
    params = {
        "timeframe": "5Min",
        "start": start.isoformat().replace("+00:00", "Z"),
        "end": end.isoformat().replace("+00:00", "Z"),
        "limit": 1000,
        "adjustment": "raw",
        "feed": STOCK_FEED,
        "sort": "asc",
    }
    r = session.get(url, headers=alpaca_headers(), params=params, timeout=20)
    if r.status_code >= 400:
        raise RuntimeError(f"Alpaca stock bars {symbol}: HTTP {r.status_code} {r.text[:160]}")
    payload = r.json()
    rows = payload.get("bars", {}).get(symbol, [])
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).rename(columns={"t":"timestamp","o":"open","h":"high","l":"low","c":"close","v":"volume"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(NY)
    df = df.set_index("timestamp").sort_index()
    for col in ("open","high","low","close","volume"):
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["close"])
    if df.empty:
        return df
    age = (now_ny() - df.index[-1].to_pydatetime()).total_seconds()
    if age < -30 or age > MAX_STOCK_BAR_AGE_SECONDS:
        raise RuntimeError(f"{symbol} stock bars stale (age={age:.0f}s)")
    return df


def signal_score(df):
    """
    A transparent directional score, not a probability or a guarantee.
    Returns (CALL/PUT/WAIT, score, explanation).
    """
    if df is None or len(df) < 30:
        return "WAIT", 0.0, "insufficient bars"
    close = df["close"].astype(float)
    ema9 = close.ewm(span=9, adjust=False).mean()
    ema21 = close.ewm(span=21, adjust=False).mean()
    r1 = close.pct_change(1).iloc[-1]
    r5 = close.pct_change(5).iloc[-1]
    r15 = close.pct_change(15).iloc[-1]
    gap = (ema9.iloc[-1] / ema21.iloc[-1]) - 1
    raw = 0.0
    raw += 1.0 if ema9.iloc[-1] > ema21.iloc[-1] else -1.0
    raw += 1.0 if r5 > 0 else -1.0
    raw += 1.0 if r15 > 0 else -1.0
    raw += 0.5 if r1 > 0 else -0.5
    raw += 0.5 if gap > 0 else -0.5
    # Score is a rule-alignment score, deliberately not called confidence.
    score = min(100.0, 50.0 + abs(raw) / 4.0 * 50.0)
    if raw >= 3.0:
        return "CALL", score, f"EMA alignment; 5-bar={r5*100:.2f}%; 15-bar={r15*100:.2f}%"
    if raw <= -3.0:
        return "PUT", score, f"EMA alignment; 5-bar={r5*100:.2f}%; 15-bar={r15*100:.2f}%"
    return "WAIT", score, "mixed direction"


def get_contracts(underlying, option_type, today):
    """
    Alpaca's contract endpoint may require account/product enablement.
    SPX weekly contracts are listed under underlying SPX; SPXW is the root.
    Fail closed if the API doesn't return contracts; never synthesize symbols.
    """
    url = f"{TRADING_URL}/v2/options/contracts"
    params = {
        "underlying_symbols": underlying,
        "expiration_date": today,
        "type": option_type.lower(),
        "status": "active",
        "limit": 1000,
    }
    r = session.get(url, headers=alpaca_headers(), params=params, timeout=25)
    if r.status_code >= 400:
        raise RuntimeError(
            f"Alpaca options contracts {underlying}: HTTP {r.status_code}; "
            f"check index-options enablement and API permissions. {r.text[:180]}"
        )
    data = r.json()
    contracts = data.get("option_contracts", data.get("contracts", []))
    if isinstance(contracts, dict):
        contracts = contracts.get("option_contracts", [])
    out = []
    for c in contracts or []:
        sym = str(c.get("symbol", "")).upper()
        exp = str(c.get("expiration_date", ""))[:10]
        typ = str(c.get("type", "")).lower()
        root = str(c.get("root_symbol", "")).upper()
        if not sym or exp != today or typ != option_type.lower():
            continue
        if underlying == "SPX":
            # Don't accept monthly SPX instead of same-day SPXW.
            if root and root != "SPXW":
                continue
            if not root and not sym.startswith("SPXW"):
                continue
        try:
            strike = float(c.get("strike_price"))
        except (TypeError, ValueError):
            continue
        out.append({**c, "symbol": sym, "strike_price": strike, "expiration_date": exp})
    return out


def get_option_quotes(symbols):
    if not symbols:
        return {}
    result = {}
    # Keep request sizes moderate.
    for offset in range(0, len(symbols), 100):
        batch = symbols[offset:offset+100]
        url = f"{DATA_URL}/v1beta1/options/quotes/latest"
        params = {"symbols": ",".join(batch), "feed": OPTIONS_FEED}
        r = session.get(url, headers=alpaca_headers(), params=params, timeout=25)
        if r.status_code >= 400:
            raise RuntimeError(
                f"Alpaca option quotes HTTP {r.status_code}; feed={OPTIONS_FEED}. "
                f"{r.text[:180]}"
            )
        payload = r.json()
        quotes = payload.get("quotes", {})
        for symbol, q in quotes.items():
            try:
                bid = float(q.get("bp", q.get("bid_price", 0)) or 0)
                ask = float(q.get("ap", q.get("ask_price", 0)) or 0)
                ts = q.get("t", q.get("timestamp"))
                stamp = pd.Timestamp(ts)
                if stamp.tzinfo is None:
                    stamp = stamp.tz_localize("UTC")
                stamp = stamp.tz_convert(UTC)
                age = (datetime.now(UTC) - stamp.to_pydatetime()).total_seconds()
                if bid <= 0 or ask <= 0 or ask < bid:
                    continue
                mid = (bid + ask) / 2
                spread = (ask - bid) / mid if mid else 9
                if age < -10 or age > MAX_OPTION_QUOTE_AGE_SECONDS:
                    continue
                if spread > MAX_SPREAD_PCT or not (MIN_OPTION_MID <= mid <= MAX_OPTION_MID):
                    continue
                result[symbol] = {
                    "bid": bid, "ask": ask, "mid": mid,
                    "spread_pct": spread, "timestamp": stamp.isoformat(), "age": age,
                }
            except Exception:
                continue
    return result


def find_option(underlying, direction, spot, today):
    option_type = "call" if direction == "CALL" else "put"
    contracts = get_contracts(underlying, option_type, today)
    if not contracts:
        return None
    max_dist = MAX_SPX_STRIKE_DISTANCE if underlying == "SPX" else max(0.5, spot * MAX_STOCK_STRIKE_DISTANCE_PCT)
    candidates = []
    for c in contracts:
        strike = c["strike_price"]
        dist = abs(strike - spot)
        if dist <= max_dist:
            candidates.append((dist, c))
    candidates.sort(key=lambda x: x[0])
    candidates = candidates[:100]
    quotes = get_option_quotes([c["symbol"] for _, c in candidates])
    best = None
    for dist, c in candidates:
        q = quotes.get(c["symbol"])
        if not q:
            continue
        # Prefer near-ATM and tight spread. This is a ranking score, not probability.
        rank = 100 - min(100, dist / max(max_dist, 0.01) * 65) - q["spread_pct"] * 100
        candidate = {
            "underlying": underlying, "symbol": c["symbol"],
            "strike": c["strike_price"], "direction": direction,
            "spot": spot, **q, "rank": rank, "distance": dist,
        }
        if best is None or candidate["rank"] > best["rank"]:
            best = candidate
    return best


def daily_count():
    today = now_ny().date().isoformat()
    with sqlite3.connect(DB_PATH) as con:
        row = con.execute("SELECT sent_count FROM daily_state WHERE day=?", (today,)).fetchone()
        return int(row[0]) if row else 0


def save_alert(item, score, source):
    today = now_ny().date().isoformat()
    with sqlite3.connect(DB_PATH) as con:
        con.execute("""
            INSERT INTO alerts
            (created_at, underlying, direction, option_symbol, strike, spot,
             bid, ask, mid, spread_pct, score, data_source)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            now_ny().isoformat(), item["underlying"], item["direction"],
            item["symbol"], item["strike"], item["spot"], item["bid"],
            item["ask"], item["mid"], item["spread_pct"], score, source,
        ))
        con.execute("""
            INSERT INTO daily_state(day, sent_count) VALUES (?, 1)
            ON CONFLICT(day) DO UPDATE SET sent_count=sent_count+1
        """, (today,))


def format_alert(item, score, explanation, spot_stamp=None):
    d_ar = "شراء CALL (توقع صعود)" if item["direction"] == "CALL" else "شراء PUT (توقع هبوط)"
    source_note = "SPX المرجعي: Yahoo Finance ^GSPC" if item["underlying"] == "SPX" else "السعر الأساسي: Alpaca"
    stamp = now_ny().strftime("%Y-%m-%d %H:%M:%S ET")
    return (
        f"🦅 {APP_NAME} {VERSION}\n"
        f"━━━━━━━━━━━━━━━━━━\n"
        f"📌 الأصل: {item['underlying']}\n"
        f"🧭 الاتجاه: {d_ar}\n"
        f"💹 سعر الأصل المرجعي: {item['spot']:.2f}\n"
        f"🎯 Strike: {item['strike']:.2f}\n"
        f"📄 رمز العقد الفعلي: {item['symbol']}\n"
        f"🟢 Ask: ${item['ask']:.2f} | 🔴 Bid: ${item['bid']:.2f}\n"
        f"⚖️ Mid: ${item['mid']:.2f} | السبريد: {item['spread_pct']*100:.1f}%\n"
        f"📊 درجة توافق الإشارة: {score:.0f}/100 (ليست احتمال ربح)\n"
        f"🧪 السبب: {explanation}\n"
        f"🕒 عمر سعر العقد: {item['age']:.0f} ثانية\n"
        f"🗂️ {source_note}\n"
        f"⏱️ وقت الإشارة: {stamp}\n"
        f"━━━━━━━━━━━━━━━━━━\n"
        "توصية احتمالية فقط — لا يوجد تنفيذ تلقائي ولا ضمان للربح."
    )


def market_window():
    now = now_ny()
    if now.weekday() >= 5:
        return False
    minute = now.hour * 60 + now.minute
    return 9 * 60 + 35 <= minute < 15 * 60 + 30


def scan_once():
    if not ALPACA_KEY or not ALPACA_SECRET:
        log.error("Missing APCA_API_KEY_ID / APCA_API_SECRET_KEY.")
        return
    if daily_count() >= MAX_ALERTS_PER_DAY:
        log.info("Daily alert limit reached.")
        return
    opportunities = []

    if INCLUDE_SPXW:
        spx_df, spx_price, spx_stamp = get_spx_bars()
        if spx_price is not None and len(spx_df) >= 30:
            direction, score, explanation = signal_score(spx_df.rename(columns={"close":"close"}))
            if direction in ("CALL", "PUT") and score >= MIN_SIGNAL_SCORE:
                try:
                    item = find_option("SPX", direction, spx_price, now_ny().date().isoformat())
                    if item:
                        item["signal_score"] = score
                        item["explanation"] = explanation
                        item["spot_stamp"] = spx_stamp
                        opportunities.append(item)
                    else:
                        log.warning("SPX direction present, but no fresh valid SPXW quote; no alert.")
                except Exception as exc:
                    log.error("SPXW chain/quote unavailable: %s", exc)

    for symbol in SCAN_STOCKS:
        if daily_count() + len(opportunities) >= MAX_ALERTS_PER_DAY:
            break
        try:
            df = get_stock_bars(symbol)
            if df.empty or len(df) < 30:
                continue
            direction, score, explanation = signal_score(df)
            if direction not in ("CALL", "PUT") or score < MIN_SIGNAL_SCORE:
                continue
            spot = float(df["close"].iloc[-1])
            item = find_option(symbol, direction, spot, now_ny().date().isoformat())
            if item:
                item["signal_score"] = score
                item["explanation"] = explanation
                opportunities.append(item)
        except Exception as exc:
            log.warning("%s scan skipped: %s", symbol, exc)

    opportunities.sort(key=lambda x: (x["signal_score"], x["rank"]), reverse=True)
    if not opportunities:
        log.info("No alert: no signal passed all data freshness, spread and direction filters.")
        return

    for item in opportunities:
        if daily_count() >= MAX_ALERTS_PER_DAY:
            break
        msg = format_alert(item, item["signal_score"], item["explanation"], item.get("spot_stamp"))
        if telegram_send(msg):
            save_alert(item, item["signal_score"], "Yahoo ^GSPC + Alpaca options" if item["underlying"] == "SPX" else "Alpaca")
            log.info("Alert sent: %s %s %s", item["underlying"], item["direction"], item["symbol"])


def main():
    ensure_db()
    log.info("Starting %s %s | recommendations only", APP_NAME, VERSION)
    log.info("SPX source=Yahoo ^GSPC | option feed=%s | stocks=%s", OPTIONS_FEED, ",".join(SCAN_STOCKS))
    log.info("No SPY*10 proxy; no synthetic option symbols; no automatic orders.")
    if not TG_TOKEN or not TG_CHAT:
        log.warning("Telegram config missing.")
    while True:
        try:
            if market_window():
                scan_once()
            else:
                log.info("Outside scanning window | New York time=%s", now_ny().strftime("%Y-%m-%d %H:%M:%S %Z"))
        except KeyboardInterrupt:
            log.info("Stopped.")
            break
        except Exception:
            log.exception("Unhandled scan error")
        time.sleep(max(15, POLL_SECONDS))


if __name__ == "__main__":
    main()
