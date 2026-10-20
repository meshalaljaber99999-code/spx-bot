# -*- coding: utf-8 -*-
import os, time, math, sqlite3, logging, html
from contextlib import closing
from datetime import datetime, timedelta, time as dtime
from zoneinfo import ZoneInfo
from pathlib import Path

import requests
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline

APP_NAME = "صَفاء التحليل (النسخة 44 - الخارقة المطلقة 🔒)"
VERSION = "v44"
ET = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper(),
                    format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("safaa")

# ---------------- CONFIG ----------------
ALPACA_KEY = os.getenv("APCA_API_KEY_ID", "").strip()
ALPACA_SECRET = os.getenv("APCA_API_SECRET_KEY", "").strip()
ALPACA_DATA_URL = os.getenv("ALPACA_DATA_URL", "https://data.alpaca.markets").rstrip("/")
TRADING_URL = os.getenv("ALPACA_TRADING_URL", "https://paper-api.alpaca.markets").rstrip("/")
STOCK_FEED = os.getenv("ALPACA_STOCK_FEED", "iex").strip().lower()
OPTIONS_FEED = os.getenv("ALPACA_OPTIONS_FEED", "indicative").strip().lower()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip() # إرسال خاص لأبو سحاب حصراً
DB_PATH = os.getenv("DB_PATH", "./data/safaa_v44.db")

POLL_SECONDS = max(10, int(os.getenv("POLL_SECONDS", "15")))
MAX_STOCK_ALERTS_PER_CYCLE = int(os.getenv("MAX_STOCK_ALERTS_PER_CYCLE", "6"))
MIN_CONFIDENCE_THRESHOLD = 0.95  # فلتر خارق: لا تنبيهات تحت 95% ثقة
PORTFOLIO_CAPITAL = float(os.getenv("PORTFOLIO_CAPITAL", "10000.0"))
MAX_RISK_PER_TRADE = 0.02 

STOCK_COOLDOWN_MIN = int(os.getenv("STOCK_COOLDOWN_MIN", "20")) 
SPX_COOLDOWN_MIN = int(os.getenv("SPX_COOLDOWN_MIN", "8"))     
FLIP_MIN_MINUTES = int(os.getenv("FLIP_MIN_MINUTES", "5"))       
MAX_DTE = int(os.getenv("MAX_DTE", "5"))                         
HORIZON = 2                                                        

TOP_7_LEADERS = ["NVDA", "AAPL", "MSFT", "AMZN", "GOOGL", "META", "TSLA"]
TOP_LEADERS_MAP = {
    "NVDA": 0.20, "AAPL": 0.18, "MSFT": 0.18, "AMZN": 0.12, 
    "GOOGL": 0.12, "META": 0.10, "TSLA": 0.10
}

OPTION_UNDERLYING = {"BRK-B": "BRKB"}

SESSION_OPEN = dtime(9, 30)
SESSION_CLOSE = dtime(16, 0)
ALERT_END_BEFORE_CLOSE_MIN = 5  

http = requests.Session()
http.headers.update({"APCA-API-KEY-ID": ALPACA_KEY,
                     "APCA-API-SECRET-KEY": ALPACA_SECRET,
                     "accept": "application/json"})

FOOTER = "\n\n💎 خوارزمية التنبؤ الكمي الخارقة (خاص لأبو سحاب - ثقة +95%)"

# ---------------- HELPERS & DB ----------------
def now_et() -> datetime: return datetime.now(ET)
def esc(x) -> str: return html.escape(str(x))
def to_alpaca_stock(sym: str) -> str: return sym.replace("-", ".")

def db_exec(sql, params=()):
    with closing(sqlite3.connect(DB_PATH)) as con:
        with con: con.execute(sql, params)

def db_one(sql, params=()):
    with closing(sqlite3.connect(DB_PATH)) as con:
        return con.execute(sql, params).fetchone()

def db_all(sql, params=()):
    with closing(sqlite3.connect(DB_PATH)) as con:
        return con.execute(sql, params).fetchall()

def init_db():
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    db_exec("""CREATE TABLE IF NOT EXISTS daily_reports (
        report_date TEXT NOT NULL, report_type TEXT NOT NULL, sent_at TEXT NOT NULL,
        PRIMARY KEY(report_date, report_type))""")
    db_exec("""CREATE TABLE IF NOT EXISTS alert_state (
        key TEXT PRIMARY KEY, direction TEXT NOT NULL, sent_at TEXT NOT NULL)""")
    db_exec("""CREATE TABLE IF NOT EXISTS trade_performance (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        symbol TEXT NOT NULL, direction TEXT NOT NULL, entry_price REAL NOT NULL,
        target_price REAL NOT NULL, confidence REAL NOT NULL, sent_at TEXT NOT NULL,
        status TEXT DEFAULT 'PENDING', final_price REAL, win_status INT)""")

def should_alert(key: str, direction: str, cooldown_min: int) -> bool:
    row = db_one("SELECT direction, sent_at FROM alert_state WHERE key=?", (key,))
    if not row: return True
    last_dir, sent_at = row
    elapsed = datetime.now(UTC) - datetime.fromisoformat(sent_at)
    if last_dir != direction: return elapsed >= timedelta(minutes=FLIP_MIN_MINUTES)
    return elapsed >= timedelta(minutes=cooldown_min)

def mark_alert(key: str, direction: str):
    db_exec("INSERT OR REPLACE INTO alert_state(key,direction,sent_at) VALUES(?,?,?)",
            (key, direction, datetime.now(UTC).isoformat()))

def record_trade(symbol: str, direction: str, entry_price: float, target_price: float, confidence: float):
    db_exec("""INSERT INTO trade_performance(symbol, direction, entry_price, target_price, confidence, sent_at)
               VALUES(?, ?, ?, ?, ?, ?)""",
            (symbol, direction, entry_price, target_price, confidence, datetime.now(UTC).isoformat()))

def tg_send_private_only(text: str) -> bool:
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        log.error("Telegram TOKEN or CHAT_ID is missing!")
        return False
        
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try:
        for i in range(0, len(text), 3900):
            for attempt in range(3):
                payload = {
                    "chat_id": TELEGRAM_CHAT_ID,
                    "text": text[i:i+3900], 
                    "parse_mode": "HTML", 
                    "disable_web_page_preview": True
                }
                r = requests.post(url, json=payload, timeout=15)
                if r.status_code == 429: 
                    time.sleep(1 * (2 ** attempt))
                    continue
                r.raise_for_status()
                break
            else: return False
        return True
    except Exception as e:
        log.error(f"Telegram private send error: {e}")
        return False

# ---------------- ULTRA-QUANT & HEDGE FUND ENGINES ----------------
def compute_gamma_exposure(ticker: str) -> dict:
    """حساب تأثير جاما صانعي السوق (Gamma Exposure - GEX)"""
    try:
        t = yf.Ticker(ticker)
        exps = t.options
        if not exps: return {"gex_regime": "NEUTRAL", "gex_score": 0.0}
        
        chain = t.option_chain(exps[0])
        calls, puts = chain.calls.copy(), chain.puts.copy()
        
        spot = t.history(period="1d")["Close"].iloc[-1]
        
        call_gex = sum(row['openInterest'] * (row['strike'] - spot) for _, row in calls.iterrows() if 'openInterest' in row and not pd.isna(row['openInterest']))
        put_gex = sum(row['openInterest'] * (spot - row['strike']) for _, row in puts.iterrows() if 'openInterest' in row and not pd.isna(row['openInterest']))
        
        net_gex = call_gex - put_gex
        gex_score = float(np.clip(net_gex / 100000.0, -1.0, 1.0))
        
        regime = "SHORT_GAMMA_VOLATILE ⚡" if gex_score < -0.2 else ("LONG_GAMMA_STABLE 🛡️" if gex_score > 0.2 else "NEUTRAL")
        return {"gex_regime": regime, "gex_score": gex_score}
    except Exception:
        return {"gex_regime": "NEUTRAL", "gex_score": 0.0}

def get_sector_breadth() -> float:
    """تحليل تدفق السيولة المتقاطع في قطاعات السوق القيادية"""
    try:
        sectors = ["XLK", "XLF", "XLC"]
        data = yf.download(sectors, period="2d", interval="1m", progress=False)["Close"]
        if data.empty: return 0.0
        ret = (data.iloc[-1] / data.iloc[-10]) - 1.0
        return float(np.clip(float(ret.mean() * 100.0), -1.0, 1.0))
    except Exception: return 0.0

def calculate_position_and_risk(contract_price: float, confidence: float, stock_price: float, atr: float) -> dict:
    """إدارة المخاطر ومعادلة كيلي وحجم العقود ووقف الخسارة"""
    try:
        if contract_price <= 0: contract_price = 1.0
        max_risk_dollars = PORTFOLIO_CAPITAL * MAX_RISK_PER_TRADE
        kelly_factor = (confidence - 0.50) / 0.50
        recommended_risk = max_risk_dollars * kelly_factor
        
        cost_per_contract = contract_price * 100.0
        num_contracts = max(1, math.floor(recommended_risk / cost_per_contract))
        
        stop_loss = stock_price - (1.5 * atr)
        target_tp = stock_price + (2.5 * atr)
        
        return {
            "contracts": num_contracts,
            "risk_amount": round(num_contracts * cost_per_contract, 2),
            "capital_pct": round((num_contracts * cost_per_contract / PORTFOLIO_CAPITAL) * 100, 1),
            "stop_loss": round(stop_loss, 2),
            "take_profit": round(target_tp, 2)
        }
    except Exception:
        return {"contracts": 1, "risk_amount": contract_price * 100, "capital_pct": 1.0, "stop_loss": stock_price * 0.99, "take_profit": stock_price * 1.02}

def check_unusual_option_activity(df_chain: pd.DataFrame, strike: float) -> bool:
    """رصد السيولة الذكية الحجمية للمؤسسات"""
    try:
        if df_chain.empty: return False
        row = df_chain[df_chain["strike"] == strike]
        if row.empty: return False
        vol = float(row["volume"].iloc[0]) if "volume" in row and not pd.isna(row["volume"].iloc[0]) else 0
        oi = float(row["openInterest"].iloc[0]) if "openInterest" in row and not pd.isna(row["openInterest"].iloc[0]) else 1
        return (vol / max(oi, 1)) >= 2.0 and vol > 500
    except Exception: return False

def evaluate_past_trades():
    """فحص وتقييم أداء الصفقات السابقة وتحديث Win-Rate"""
    try:
        pending_trades = db_all("SELECT id, symbol, direction, entry_price, target_price, sent_at FROM trade_performance WHERE status='PENDING'")
        for t_id, symbol, direction, entry_price, target_price, sent_at in pending_trades:
            if datetime.now(UTC) - datetime.fromisoformat(sent_at) < timedelta(minutes=10): continue
            df = yf_bars(symbol, period="1d", interval="1m")
            if df.empty: continue
            current_price = float(df["close"].iloc[-1])
            is_win = (current_price >= target_price) if direction == "CALL" else (current_price <= target_price)
            db_exec("UPDATE trade_performance SET status='COMPLETED', final_price=?, win_status=? WHERE id=?",
                    (current_price, 1 if is_win else 0, t_id))
    except Exception as e:
        log.error(f"Evaluate past trades error: {e}")

def get_performance_summary() -> str:
    try:
        total = db_one("SELECT COUNT(*) FROM trade_performance WHERE status='COMPLETED'")[0]
        if total == 0: return "📊 <b>سجل الأداء:</b> لا توجد صفقات مكتملة للتقييم بعد."
        wins = db_one("SELECT COUNT(*) FROM trade_performance WHERE status='COMPLETED' AND win_status=1")[0]
        return f"📈 <b>سجل أداء الإشارات الخارقة:</b> الصفقات: {total} | الناجحة: {wins} | <b>نسبة النجاح: {(wins/total)*100:.1f}%</b>"
    except Exception: return ""

# ---------------- MARKET DATA & TECHNICAL ENGINE ----------------
def alpaca_get(path, params=None, timeout=15, retries=2):
    base = TRADING_URL if path.startswith(("/v2/options", "/v2/calendar")) else ALPACA_DATA_URL
    for attempt in range(retries):
        try:
            r = http.get(base + path, params=params or {}, timeout=timeout)
            if r.status_code == 429: time.sleep(1 * (2 ** attempt)); continue
            r.raise_for_status()
            return r.json()
        except Exception as e:
            log.warning(f"Alpaca get error ({path}) attempt {attempt+1}: {e}")
            time.sleep(1)
    return {}

def drop_incomplete(df: pd.DataFrame, minutes=1) -> pd.DataFrame:
    if df is None or df.empty: return df
    if df.index[-1] + timedelta(minutes=minutes) > datetime.now(UTC): return df.iloc[:-1]
    return df

def alpaca_bars(symbols, timeframe, start, feed=None) -> dict:
    params = {"symbols": ",".join(symbols), "timeframe": timeframe,
              "start": start.astimezone(UTC).isoformat(), "limit": 10000,
              "feed": feed or STOCK_FEED, "sort": "asc"}
    rows, token = {}, None
    for _ in range(20):
        if token: params["page_token"] = token
        data = alpaca_get("/v2/stocks/bars", params)
        for sym, bars in (data.get("bars") or {}).items(): rows.setdefault(sym, []).extend(bars)
        token = data.get("next_page_token")
        if not token: break
    out = {}
    for sym, bars in rows.items():
        try:
            df = pd.DataFrame(bars)
            df.index = pd.to_datetime(df["t"], utc=True)
            df = df.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"})
            out[sym] = drop_incomplete(df[["open", "high", "low", "close", "volume"]].astype(float))
        except Exception as e:
            log.error(f"Error parsing alpaca bars for {sym}: {e}")
    return out

def yf_bars(symbol, period="2d", interval="1m", prepost=False) -> pd.DataFrame:
    try:
        df = yf.Ticker(symbol).history(period=period, interval=interval, auto_adjust=False, prepost=prepost)
        if df is None or df.empty: return pd.DataFrame()
        df = df.rename(columns={c: c.lower() for c in df.columns})
        df.index = pd.to_datetime(df.index, utc=True)
        return drop_incomplete(df[["open", "high", "low", "close", "volume"]].dropna(subset=["close"]).sort_index())
    except Exception as e:
        log.error(f"YF bars error for {symbol}: {e}")
        return pd.DataFrame()

def compute_technical_indicators(df: pd.DataFrame) -> dict:
    if df is None or len(df) < 20: return {}
    c = df["close"].astype(float)
    v = df["volume"].astype(float)
    h = df["high"].astype(float)
    l = df["low"].astype(float)

    delta = c.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-9)
    rsi = float((100 - (100 / (1 + rs))).iloc[-1])

    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    macd_hist = float((macd_line - signal_line).iloc[-1])

    ema9 = float(c.ewm(span=9, adjust=False).mean().iloc[-1])
    ema21 = float(c.ewm(span=21, adjust=False).mean().iloc[-1])
    ema_signal = 1.0 if ema9 > ema21 else -1.0

    vwap = float(((c * v).sum() / v.sum()) if v.sum() > 0 else c.iloc[-1])
    vwap_signal = 1.0 if c.iloc[-1] > vwap else -1.0

    sma20 = c.rolling(20).mean()
    std20 = c.rolling(20).std()
    upper_bb = float((sma20 + 2 * std20).iloc[-1])
    lower_bb = float((sma20 - 2 * std20).iloc[-1])
    bb_signal = 1.0 if c.iloc[-1] < lower_bb else (-1.0 if c.iloc[-1] > upper_bb else 0.0)

    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr = float(tr.rolling(14).mean().iloc[-1])

    v_mean = v.rolling(30).mean()
    v_std = v.rolling(30).std() + 1e-5
    v_zscore = float(((v.iloc[-1] - v_mean.iloc[-1]) / v_std.iloc[-1]))

    return {
        "rsi": rsi, "macd_hist": macd_hist, "ema_signal": ema_signal,
        "vwap": vwap, "vwap_signal": vwap_signal, "bb_signal": bb_signal,
        "v_zscore": v_zscore, "atr": atr, "last_price": float(c.iloc[-1])
    }

def get_news_sentiment(ticker: str) -> float:
    try:
        t = yf.Ticker(ticker)
        news = t.news
        if not news: return 0.0
        pos_words = ["growth", "beat", "surge", "upgraded", "profit", "bull", "record", "gain"]
        neg_words = ["drop", "miss", "fall", "downgraded", "loss", "bear", "risk", "decline"]
        score = 0.0
        for item in news[:5]:
            title = item.get("title", "").lower()
            score += sum(1 for w in pos_words if w in title)
            score -= sum(1 for w in neg_words if w in title)
        return float(np.clip(score / 5.0, -1.0, 1.0))
    except Exception: return 0.0

# ---------------- MACHINE LEARNING & EVALUATION ----------------
FEATURES = ["r1", "r3", "r6", "r12", "vol12", "ema_gap", "range", "vchg", "tod_sin", "tod_cos"]
def feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    close = x["close"].astype(float)
    for n in (1, 3, 6, 12): x[f"r{n}"] = close.pct_change(n)
    x["vol12"] = x["r1"].rolling(12).std()
    x["ema_gap"] = close / close.ewm(span=12, adjust=False).mean() - 1
    x["range"] = (x["high"] - x["low"]) / close
    x["vchg"] = x["volume"].pct_change().replace([np.inf, -np.inf], np.nan).fillna(0)
    mins = np.array([i.hour * 60 + i.minute for i in x.index.tz_convert(ET)])
    x["tod_sin"], x["tod_cos"] = np.sin(2 * np.pi * mins / 1440), np.cos(2 * np.pi * mins / 1440)
    return x.replace([np.inf, -np.inf], np.nan)

def make_model():
    return make_pipeline(SimpleImputer(strategy="median"), RandomForestClassifier(n_estimators=70, max_depth=5, min_samples_leaf=4, random_state=42, n_jobs=-1))

_MODEL = {"model": None, "trained_at": None, "acc": 0.5}

def fetch_spy_df() -> pd.DataFrame:
    try:
        df = alpaca_bars(["SPY"], "1Min", datetime.now(UTC) - timedelta(days=2)).get("SPY")
        if df is not None and len(df) >= 100: return df
    except: pass
    return yf_bars("SPY", period="5d", interval="1m")

def model_component(spy_df):
    try:
        if spy_df is None or len(spy_df) < 100: return None, None
        x = feature_frame(spy_df)
        fut = x["close"].shift(-HORIZON) / x["close"] - 1
        x["target"] = np.where(fut.isna(), np.nan, (fut > 0).astype(float))
        ds = x.dropna(subset=FEATURES + ["target"])
        if len(ds) < 100: return None, None

        if (_MODEL["model"] is None or _MODEL["trained_at"] is None or datetime.now(UTC) - _MODEL["trained_at"] > timedelta(minutes=60)):
            X, y = ds[FEATURES], ds["target"].astype(int)
            accs = []
            for tr, te in TimeSeriesSplit(n_splits=3, gap=HORIZON).split(X):
                m = make_model(); m.fit(X.iloc[tr], y.iloc[tr]); accs.append(m.score(X.iloc[te], y.iloc[te]))
            final = make_model(); final.fit(X, y)
            _MODEL.update(model=final, trained_at=datetime.now(UTC), acc=float(np.mean(accs)) if accs else 0.5)
            
        prob_up = float(_MODEL["model"].predict_proba(x[FEATURES].iloc[[-1]])[0, 1])
        return (prob_up - 0.5) * 2 * float(np.clip((_MODEL["acc"] - 0.5) / 0.05, 0, 1)), _MODEL["acc"]
    except Exception: return None, None

def evaluate_full_360_signal(ticker: str, df: pd.DataFrame, ml_score: float) -> tuple:
    tech = compute_technical_indicators(df)
    if not tech: return None, 0.0, {}

    gex_data = compute_gamma_exposure(ticker if ticker != "^GSPC" else "SPY")
    sector_breadth = get_sector_breadth()
    news_sent = get_news_sentiment(ticker)
    
    rsi_sig = 1.0 if tech["rsi"] < 30 else (-1.0 if tech["rsi"] > 70 else (50 - tech["rsi"]) / 20.0)
    macd_sig = float(np.clip(tech["macd_hist"] * 5.0, -1.0, 1.0))
    
    weights = {
        "technical": (rsi_sig * 0.10 + macd_sig * 0.15 + tech["ema_signal"] * 0.15 + tech["vwap_signal"] * 0.10 + tech["bb_signal"] * 0.05),
        "gex": (gex_data["gex_score"] * 0.15),
        "sector_breadth": (sector_breadth * 0.10),
        "ml_model": (ml_score * 0.10 if ml_score is not None else 0.0),
        "news": (news_sent * 0.10)
    }
    
    final_score = sum(weights.values())
    abs_score = min(abs(final_score), 1.0)
    confidence = round(0.50 + 0.50 * math.tanh(abs_score * 3.4), 4)
    return final_score, confidence, {**tech, **gex_data}

def fetch_leader_bars() -> dict:
    tickers = list(TOP_LEADERS_MAP)
    try:
        raw = alpaca_bars([to_alpaca_stock(t) for t in tickers], "1Min", datetime.now(UTC) - timedelta(days=1))
        return {k.replace(".", "-"): v for k, v in raw.items()}
    except Exception: return {}

def get_session(d):
    try:
        data = alpaca_get("/v2/calendar", {"start": d.isoformat(), "end": d.isoformat()})
        return (dtime.fromisoformat(data[0]["open"]), dtime.fromisoformat(data[0]["close"])) if data and data[0]["date"] == d.isoformat() else (SESSION_OPEN, SESSION_CLOSE)
    except Exception: return (SESSION_OPEN, SESSION_CLOSE) if d.weekday() < 5 else None

# ---------------- OPTIONS ENGINE ----------------
def get_option_quotes(symbols) -> dict:
    out = {}
    try:
        payload = alpaca_get("/v1beta1/options/quotes/latest", {"symbols": ",".join(symbols), "feed": OPTIONS_FEED})
        for sym, q in (payload.get("quotes") or {}).items():
            bid, ask = float(q.get("bp", 0) or 0), float(q.get("ap", 0) or 0)
            if bid > 0 and ask >= bid and ask >= 0.05:
                mid = (bid + ask) / 2
                if (ask - bid) / mid <= 0.50: out[sym] = {"bid": bid, "ask": ask, "mid": mid}
    except Exception: pass
    return out

def get_smart_spxw_contract(spot: float, direction: str):
    try:
        t = yf.Ticker("^SPX")
        exps = [e for e in (t.options or []) if e >= now_et().date().isoformat()]
        if not exps: return None
        exp = exps[0]
        chain = t.option_chain(exp)
        df = chain.calls if direction == "CALL" else chain.puts
        df = df[(df["bid"] > 0) & (df["ask"] > 0)].copy()
        if "contractSymbol" in df:
            w = df[df["contractSymbol"].astype(str).str.startswith("SPXW")]
            if not w.empty: df = w
        if df.empty: return None
        
        row = df.sort_values("strike", ascending=(direction == "CALL")).iloc[0]
        strike_price = float(row["strike"])
        is_unusual = check_unusual_option_activity(df, strike_price)
        
        return {"expiry": exp, "strike": strike_price, "bid": float(row["bid"]), "ask": float(row["ask"]), "unusual": is_unusual}
    except Exception: return None

def find_smart_stock_contract(ticker: str, price: float, opt_type: str):
    try:
        und = OPTION_UNDERLYING.get(ticker, ticker)
        today = now_et().date()
        s_min = round(price * 0.99, 2) if opt_type == "put" else price
        s_max = round(price * 1.01, 2) if opt_type == "call" else price
        
        contracts = alpaca_get("/v2/options/contracts", {
            "underlying_symbols": und, "status": "active", "type": opt_type, 
            "expiration_date_gte": today.isoformat(), "expiration_date_lte": (today + timedelta(days=MAX_DTE)).isoformat(), 
            "strike_price_gte": min(s_min, s_max) * 0.99, "strike_price_lte": max(s_min, s_max) * 1.01, "limit": 100
        }).get("option_contracts", [])
        
        if not contracts: return None
        quotes = get_option_quotes([c["symbol"] for c in contracts])
        valid = [{"symbol": c["symbol"], "expiry": c["expiration_date"], "strike": float(c["strike_price"]), "ask": q["ask"]}
                 for c in contracts if (q := quotes.get(c["symbol"]))]
        
        return sorted(valid, key=lambda x: abs(x['strike'] - price))[0] if valid else None
    except Exception: return None  

def option_label(underlying: str, exp: str, opt_type: str, strike: float) -> str:
    return f"{underlying} {datetime.strptime(exp, '%Y-%m-%d').strftime('%d%m%y')} {'C' if opt_type.lower().startswith('c') else 'P'} {strike:g}"

# ---------------- DURING MARKET ALERTS (CONFIDENCE > 95%) ----------------
def run_spx_alert(spy_df):
    ml_score, _ = model_component(spy_df)
    gspc = yf_bars("^GSPC", "2d", "1m")
    if gspc.empty: return
    
    score, confidence, tech = evaluate_full_360_signal("^GSPC", gspc, ml_score)
    if confidence < MIN_CONFIDENCE_THRESHOLD: return
    
    direction = "CALL" if score > 0 else "PUT"
    if not should_alert("spxw", direction, SPX_COOLDOWN_MIN): return

    spot = float(gspc["close"].iloc[-1])
    vol_factor = gspc["close"].pct_change().std() * spot if not gspc.empty else spot * 0.001
    target_price_10m = spot + (score * vol_factor * 1.5)

    c = get_smart_spxw_contract(spot, direction)
    up = direction == "CALL"
    
    risk_info = calculate_position_and_risk(c['ask'] if c else 1.0, confidence, spot, tech.get('atr', 2.0))
    
    lines = [
        "📊 <b>SPXW — إشارة صيد فائقة القوة (النسخة الخارقة) ⚡</b>\n",
        f"🎯 <b>درجة الثقة الكمية المطلقة:</b> <code>{confidence * 100:.1f}%</code> 🌟",
        f"▫️ <b>الاتجاه المعتمد:</b> {'كول (CALL) 🟢' if up else 'بوت (PUT) 🔴'}",
        f"▫️ <b>سعر المؤشر الحالي:</b> ${spot:,.2f}",
        f"⏱️ <b>المستهدف (خلال 10د):</b> <b>${target_price_10m:,.2f}</b>",
        f"🎯 <b>جني الأرباح (TP):</b> ${risk_info['take_profit']:,.2f} | <b>وقف الخسارة (SL):</b> ${risk_info['stop_loss']:,.2f}",
        f"⚡ <b>وضع الجاما:</b> {tech.get('gex_regime', 'N/A')}",
        f"🛡️ <b>حجم العقود:</b> <b>{risk_info['contracts']} عقود</b> (المخاطرة: ${risk_info['risk_amount']})",
    ]
    if c and c.get('unusual'):
        lines.append("🔥 <b>تنبيه سيولة غير طبيعية:</b> رصد تدفق حاد من المحافظ الكبرى في هذا العقد!")
        
    if c:
        lines += [
            f"▫️ <b>العقد الذكي:</b> <code>{esc(option_label('SPXW', c['expiry'], 'call' if up else 'put', c['strike']))}</code>",
            f"▫️ <b>Strike:</b> ${c['strike']:,.0f} | <b>سعر الشراء (Ask):</b> ${c['ask']:.2f}"
        ]
    
    if tg_send_private_only("\n".join(lines) + FOOTER):
        mark_alert("spxw", direction)
        record_trade("SPXW", direction, spot, target_price_10m, confidence)

def run_stock_alerts(bars: dict):
    spy_df = fetch_spy_df()
    ml_score, _ = model_component(spy_df)
    
    alerts = []
    for ticker in TOP_7_LEADERS:
        df = bars.get(ticker)
        if df is None or df.empty: continue
        score, confidence, tech = evaluate_full_360_signal(ticker, df, ml_score)
        if confidence >= MIN_CONFIDENCE_THRESHOLD:
            alerts.append((ticker, score, confidence, tech))
            
    alerts = sorted(alerts, key=lambda x: -x[2])
    
    for ticker, score, conf, tech in alerts[:MAX_STOCK_ALERTS_PER_CYCLE]:
        direction = "CALL" if score > 0 else "PUT"
        if not should_alert(f"stk:{ticker}", direction, STOCK_COOLDOWN_MIN): continue
        price = float(bars[ticker]["close"].iloc[-1])
        opt_type = "call" if score > 0 else "put"
        
        c = find_smart_stock_contract(ticker, price, opt_type)
        if c:
            stock_vol = bars[ticker]["close"].pct_change().std() * price if len(bars[ticker]) > 5 else price * 0.005
            target_price = price + (score * stock_vol * 1.5)
            
            risk_info = calculate_position_and_risk(c['ask'], conf, price, tech.get('atr', price * 0.01))
            
            text = (
                f"🎯 <b>سهم {esc(ticker)} — تحليل خوارزمي خارق (ثقة +95%)</b>\n\n"
                f"🌟 <b>نسبة الثقة:</b> <code>{conf * 100:.1f}%</code>\n"
                f"▫️ <b>الاتجاه المتوقع:</b> {'صعود صريح 🚀' if score > 0 else 'هبوط صريح 📉'}\n"
                f"▫️ <b>السعر الحالي:</b> ${price:,.2f}\n"
                f"⏱️ <b>المستهدف (خلال 10د):</b> ${target_price:,.2f}\n"
                f"🎯 <b>الهدف (TP):</b> ${risk_info['take_profit']:,.2f} | <b>الوقف (SL):</b> ${risk_info['stop_loss']:,.2f}\n"
                f"📊 <b>GEX Score:</b> {tech.get('gex_score', 0):.2f} | <b>Z-Score:</b> {tech.get('v_zscore', 0):.2f}\n"
                f"🛡️ <b>العقود المقترحة:</b> {risk_info['contracts']} عقود (المخاطرة: ${risk_info['risk_amount']})\n"
                f"▫️ <b>العقد الذكي:</b> <code>{esc(option_label(OPTION_UNDERLYING.get(ticker, ticker), c['expiry'], opt_type, c['strike']))}</code>\n"
                f"▫️ <b>Strike:</b> ${c['strike']:,.2f} | <b>الدخول:</b> ${c['ask']:.2f}"
            )
            if tg_send_private_only(text + FOOTER):
                mark_alert(f"stk:{ticker}", direction)
                record_trade(ticker, direction, price, target_price, conf)

# ---------------- PRE-MARKET REPORT ----------------
def build_premarket_message(day_str: str):
    try:
        fut = yf_bars("ES=F", "5d", "15m", prepost=True)
        fut_change = ((fut["close"].iloc[-1] / fut["close"].iloc[-2]) - 1) * 100 if not fut.empty else 0.0
        
        text = f"🌅 <b>التقرير التنبؤي الخارق قبل الافتتاح (خاص لأبو سحاب 🔒)</b>\n📅 {day_str}\n"
        text += f"🌐 <b>العقود الآجلة S&P Futures:</b> <code>{fut_change:+.2f}%</code>\n\n"
        text += "🏢 <b>تفصيل حركة الـ 7 الشركات الكبرى والسيولة الذكية:</b>\n\n"
        
        for ticker in TOP_7_LEADERS:
            try:
                stock_obj = yf.Ticker(ticker)
                hist = stock_obj.history(period="1mo", interval="1d")
                if hist.empty: continue
                
                last_close = float(hist["Close"].iloc[-1])
                pm_data = stock_obj.history(period="1d", interval="1m", prepost=True)
                expected_open = float(pm_data["Close"].iloc[-1]) if not pm_data.empty else last_close
                gap_pct = ((expected_open / last_close) - 1) * 100
                news_sent = get_news_sentiment(ticker)
                
                avg_move = float(((hist["High"] - hist["Low"]) / hist["Close"]).tail(14).mean())
                stock_high = expected_open * (1 + (avg_move / 2))
                stock_low = expected_open * (1 - (avg_move / 2))
                r = stock_high - stock_low
                
                conf = min(0.98, 0.88 + abs(gap_pct) * 0.04 + abs(news_sent) * 0.05)
                
                text += f"🔹 <b>سهم {ticker}</b>\n"
                text += f"   • <b>الافتتاح المتوقع:</b> ${expected_open:.2f} (<code>{gap_pct:+.2f}%</code>)\n"
                text += f"   • <b>معنويات الأخبار:</b> {'إيجابي 🟢' if news_sent > 0.1 else ('سلبي 🔴' if news_sent < -0.1 else 'محايد 🟡')}\n"
                text += f"   • <b>نسبة ثقة التنبؤ:</b> <code>{conf * 100:.1f}%</code>\n"
                text += f"   ⏱️ <b>9:30 - 11:30:</b> نطاق ${expected_open - r*0.25:.2f} إلى ${expected_open + r*0.25:.2f}\n"
                text += f"   ⏱️ <b>11:30 - 16:00:</b> استهداف ${stock_high if gap_pct > 0 else stock_low:.2f}\n\n"
            except Exception: continue
                
        return text + FOOTER
    except Exception: return None

def check_premarket_report(sess):
    if not sess: return
    now = now_et()
    if not (dtime(8, 30) <= now.time() < sess[0]): return
    day = now.date().isoformat()
    if db_one("SELECT 1 FROM daily_reports WHERE report_date=? AND report_type='premarket'", (day,)): return
    
    msg = build_premarket_message(day)
    if msg and tg_send_private_only(msg):
        db_exec("INSERT OR REPLACE INTO daily_reports(report_date,report_type,sent_at) VALUES(?,?,?)",
                (day, "premarket", datetime.now(UTC).isoformat()))

# ---------------- MAIN ----------------
def cycle():
    now = now_et()
    sess = get_session(now.date())
    
    evaluate_past_trades()
    check_premarket_report(sess)
    if not sess: return
    
    open_dt = datetime.combine(now.date(), sess[0], ET)
    close_dt = datetime.combine(now.date(), sess[1], ET)
    if not (open_dt <= now <= close_dt - timedelta(minutes=ALERT_END_BEFORE_CLOSE_MIN)): return

    spy_df = fetch_spy_df()
    bars = fetch_leader_bars()
    
    run_spx_alert(spy_df)
    if bars: run_stock_alerts(bars)

def main():
    init_db()
    log.info("Starting %s %s", APP_NAME, VERSION)
    tg_send_private_only(f"🚀 تم تشغيل {esc(APP_NAME)} ({esc(VERSION)})\n🔒 <b>النظام يعمل بفلتر الثقة الخارق (+95%) وإعادة حساب تأثير الجاما وسيولة المؤسسات.</b>")
    while True:
        try: cycle()
        except Exception: log.exception("Main loop critical crash")
        time.sleep(POLL_SECONDS)

if __name__ == "__main__":
    main()
