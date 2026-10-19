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

APP_NAME = "صَفاء التحليل (النسخة 37 - الدقة الفائقة والعقود الذكية)"
VERSION = "v37"
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
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()
DB_PATH = os.getenv("DB_PATH", "./data/safaa_v37.db")

POLL_SECONDS = max(10, int(os.getenv("POLL_SECONDS", "15")))

MAX_STOCK_ALERTS_PER_CYCLE = int(os.getenv("MAX_STOCK_ALERTS_PER_CYCLE", "6"))
STOCK_MIN_SCORE = float(os.getenv("STOCK_MIN_SCORE", "0.75"))     
SPX_MIN_SCORE = float(os.getenv("SPX_MIN_SCORE", "0.30"))         
STOCK_COOLDOWN_MIN = int(os.getenv("STOCK_COOLDOWN_MIN", "20")) 
SPX_COOLDOWN_MIN = int(os.getenv("SPX_COOLDOWN_MIN", "8"))     
FLIP_MIN_MINUTES = int(os.getenv("FLIP_MIN_MINUTES", "5"))       
MAX_DTE = int(os.getenv("MAX_DTE", "5"))                         
HORIZON = 2                                                        
MODEL_RETRAIN_MIN = 60

TOP_LEADERS_MAP = {
    "NVDA": 0.07, "AAPL": 0.07, "GOOGL": 0.05, "MSFT": 0.06, "AMZN": 0.05,
    "TSLA": 0.04, "META": 0.04
}

OPTION_UNDERLYING = {"BRK-B": "BRKB"}

SESSION_OPEN = dtime(9, 30)
SESSION_CLOSE = dtime(16, 0)
ALERT_START_DELAY_MIN = 0     
ALERT_END_BEFORE_CLOSE_MIN = 5  

http = requests.Session()
http.headers.update({"APCA-API-KEY-ID": ALPACA_KEY,
                     "APCA-API-SECRET-KEY": ALPACA_SECRET,
                     "accept": "application/json"})

FOOTER = "\n\n⚠️ تنبيه تحليلي (النسخة 37 - دقة عالية)"

# ---------------- HELPERS ----------------
def now_et() -> datetime: return datetime.now(ET)
def esc(x) -> str: return html.escape(str(x))
def to_alpaca_stock(sym: str) -> str: return sym.replace("-", ".")
def db_exec(sql, params=()):
    with closing(sqlite3.connect(DB_PATH)) as con:
        with con: con.execute(sql, params)
def db_one(sql, params=()):
    with closing(sqlite3.connect(DB_PATH)) as con:
        return con.execute(sql, params).fetchone()

def init_db():
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    db_exec("""CREATE TABLE IF NOT EXISTS daily_reports (
        report_date TEXT NOT NULL, report_type TEXT NOT NULL, sent_at TEXT NOT NULL,
        PRIMARY KEY(report_date, report_type))""")
    db_exec("""CREATE TABLE IF NOT EXISTS alert_state (
        key TEXT PRIMARY KEY, direction TEXT NOT NULL, sent_at TEXT NOT NULL)""")

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

# ---------------- ALPACA & YFINANCE ----------------
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

def get_option_quotes(symbols) -> dict:
    out = {}
    try:
        payload = alpaca_get("/v1beta1/options/quotes/latest", {"symbols": ",".join(symbols), "feed": OPTIONS_FEED})
        for sym, q in (payload.get("quotes") or {}).items():
            bid, ask = float(q.get("bp", 0) or 0), float(q.get("ap", 0) or 0)
            if bid > 0 and ask >= bid and ask >= 0.05:
                mid = (bid + ask) / 2
                if (ask - bid) / mid <= 0.50: out[sym] = {"bid": bid, "ask": ask, "mid": mid}
    except Exception as e:
        log.error(f"Error getting option quotes: {e}")
    return out

_cal_cache = {}
def get_session(d):
    if d in _cal_cache: return _cal_cache[d]
    try:
        data = alpaca_get("/v2/calendar", {"start": d.isoformat(), "end": d.isoformat()})
        sess = (dtime.fromisoformat(data[0]["open"]), dtime.fromisoformat(data[0]["close"])) if data and data[0]["date"] == d.isoformat() else None
        _cal_cache[d] = sess
        return sess
    except Exception as e:
        log.error(f"Error fetching calendar session for {d}: {e}")
        return (SESSION_OPEN, SESSION_CLOSE) if d.weekday() < 5 else None

def tg_send(text: str) -> bool:
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try:
        for i in range(0, len(text), 3900):
            for attempt in range(3):
                payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text[i:i+3900], "parse_mode": "HTML", "disable_web_page_preview": True}
                r = requests.post(url, json=payload, timeout=15)
                if r.status_code == 429: time.sleep(1 * (2 ** attempt)); continue
                r.raise_for_status()
                break
            else: return False
        return True
    except Exception as e:
        log.error(f"Telegram send error: {e}")
        return False

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

# ---------------- MOMENTUM / MODEL ----------------
def momentum_score(df, max_age_min=10):
    try:
        if df is None or len(df) < 10: return None
        if datetime.now(UTC) - df.index[-1] > timedelta(minutes=max_age_min): return None
        idx = df.index.tz_convert(ET)
        d = df[idx.date == idx[-1].date()]
        if len(d) < 2: return None
        c = d["close"].astype(float)
        k6, k12 = min(2, len(d) - 1), min(4, len(d) - 1)
        r6, r12 = c.iloc[-1] / c.iloc[-1 - k6] - 1, c.iloc[-1] / c.iloc[-1 - k12] - 1
        vol = d["volume"].astype(float)
        vgap = (c.iloc[-1] / ((c * vol).sum() / vol.sum()) - 1) if vol.sum() > 0 else 0.0
        sigma = df["close"].astype(float).pct_change().tail(78).std() * math.sqrt(6)
        if not sigma or math.isnan(sigma): return None
        return float(math.tanh((0.4 * r6 + 0.3 * r12 + 0.3 * vgap) / sigma))
    except Exception as e:
        log.error(f"Momentum score error: {e}")
        return None

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
    return make_pipeline(SimpleImputer(strategy="median"), RandomForestClassifier(n_estimators=60, max_depth=4, min_samples_leaf=5, random_state=42, n_jobs=-1))

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
    except Exception as e:
        log.error(f"Model component error: {e}")
        return None, None

def fetch_leader_bars() -> dict:
    tickers = list(TOP_LEADERS_MAP)
    try:
        raw = alpaca_bars([to_alpaca_stock(t) for t in tickers], "1Min", datetime.now(UTC) - timedelta(days=1))
        return {k.replace(".", "-"): v for k, v in raw.items()}
    except Exception as e:
        log.error(f"Fetch leader bars error: {e}")
        return {}

def leaders_component(bars: dict):
    scores = {t: s for t in TOP_LEADERS_MAP if (s := momentum_score(bars.get(t))) is not None}
    if len(scores) < 1: return None, scores
    tw = sum(TOP_LEADERS_MAP[t] for t in scores)
    return float(sum(scores[t] * TOP_LEADERS_MAP[t] for t in scores) / tw), scores

def combine_signal(leaders, model, index, vix):
    avail = {k: v for k, v in {"leaders": (leaders, 0.40), "model": (model, 0.25), "index": (index, 0.25), "vix": (vix, 0.10)}.items() if v[0] is not None}
    if len(avail) < 2 or ("leaders" not in avail and "index" not in avail): return None
    return float(sum(v * w for v, w in avail.values()) / sum(w for _, w in avail.values()))

def get_smart_spxw_contract(spot: float, direction: str):
    """
    اختيار عقود SPXW ذكية وواقعية (قريبة من السعر الحالي بمسافة آمنة ومدروسة لتحقيق العائد دون مخاطر مبالغ فيها)
    """
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
        
        # اختيار Strike قريب جداً ومنطقي (Near-The-Money / Smart OTM بمسافة صغيرة آمنة)
        if direction == "CALL":
            df = df[df["strike"] >= spot]  # يبدأ من السعر الحالي فصاعداً
            if df.empty: return None
            row = df.sort_values("strike", ascending=True).iloc[0] # أقرب ستريك (أعلى السعر مباشرة) لضمان الدقة والسيولة
        else:
            df = df[df["strike"] <= spot]  # يبدأ من السعر الحالي نازلاً
            if df.empty: return None
            row = df.sort_values("strike", ascending=False).iloc[0] # أقرب ستريك هبوطي

        return {"expiry": exp, "strike": float(row["strike"]), "bid": float(row["bid"]), "ask": float(row["ask"])}
    except Exception as e:
        log.error(f"Smart SPXW contract error: {e}")
        return None

def option_label(underlying: str, exp: str, opt_type: str, strike: float) -> str:
    return f"{underlying} {datetime.strptime(exp, '%Y-%m-%d').strftime('%d%m%y')} {'C' if opt_type.lower().startswith('c') else 'P'} {strike:g}"

def run_spx_alert(leaders_score):
    spy_df = fetch_spy_df()
    model_score, acc = model_component(spy_df)
    gspc = yf_bars("^GSPC", "2d", "1m")
    vix_s = momentum_score(yf_bars("^VIX", "2d", "1m"))
    total = combine_signal(leaders_score, model_score, momentum_score(gspc), -vix_s if vix_s is not None else None)
    
    if total is None or abs(total) < SPX_MIN_SCORE: return
    direction = "CALL" if total > 0 else "PUT"
    if not should_alert("spxw", direction, SPX_COOLDOWN_MIN): return

    estimated = False
    if not gspc.empty: spot = float(gspc["close"].iloc[-1])
    elif spy_df is not None and not spy_df.empty: spot, estimated = float(spy_df["close"].iloc[-1]) * 10.0, True
    else: return

    volatility_factor = gspc["close"].pct_change().std() * spot if not gspc.empty else spot * 0.001
    expected_change = total * volatility_factor * 1.5
    target_price_10m = spot + expected_change

    c = get_smart_spxw_contract(spot, direction)
    up = direction == "CALL"
    
    lines = [
        "📊 <b>SPXW — إشارة صيد ذكية عالية الدقة ⚡</b>\n",
        f"▫️ <b>الاتجاه المؤكد:</b> {'كول (CALL) 🟢' if up else 'بوت (PUT) 🔴'}",
        f"▫️ <b>سعر المؤشر الحالي:</b> ${spot:,.2f}{' (تقديري)' if estimated else ''}",
        f"⏱️ <b>التنبؤ المتقدم (خلال 10 دقائق):</b> مستهدف الوصول إلى <b>${target_price_10m:,.2f}</b> ({'صعود قوي 📈' if expected_change > 0 else 'هبوط حاد 📉'})",
        f"💎 <b>الهدف الاستراتيجي:</b> استهداف عقد دقيق وقريب لضمان نسبة نجاح عالية ومكاسب ممتازة.",
    ]
    if c: lines += [f"▫️ <b>رقم العقد الذكي:</b> <code>{esc(option_label('SPXW', c['expiry'], 'call' if up else 'put', c['strike']))}</code>", f"▫️ <b>Strike المختار:</b> ${c['strike']:,.0f} | <b>سعر الدخول (Ask):</b> ${c['ask']:.2f}"]
    
    if tg_send("\n".join(lines) + FOOTER): mark_alert("spxw", direction)

def find_smart_stock_contract(ticker: str, price: float, opt_type: str):
    try:
        und = OPTION_UNDERLYING.get(ticker, ticker)
        today = now_et().date()
        # نطاق ذكي وقريب جداً من سعر السهم الحالي (منعاً للبعد المبالغ فيه الذي يسبب الخسارة)
        s_min = round(price * 0.99, 2) if opt_type == "put" else price
        s_max = round(price * 1.01, 2) if opt_type == "call" else price
        
        contracts = alpaca_get("/v2/options/contracts", {
            "underlying_symbols": und, "status": "active", "type": opt_type, 
            "expiration_date_gte": today.isoformat(), "expiration_date_lte": (today + timedelta(days=MAX_DTE)).isoformat(), 
            "strike_price_gte": min(s_min, s_max) * 0.99, "strike_price_lte": max(s_min, s_max) * 1.01, "limit": 100
        }).get("option_contracts", [])
        
        if not contracts: return None
        quotes = get_option_quotes([c["symbol"] for c in contracts])
        valid_cands = []
        for c in contracts:
            if (q := quotes.get(c["symbol"])):
                valid_cands.append({"symbol": c["symbol"], "expiry": c["expiration_date"], "strike": float(c["strike_price"]), "ask": q["ask"]})
        
        if valid_cands:
            # اختيار العقد الأقرب للسعر الحالي (ATM / Near-ATM) لضمان تفاعل حقيقي وسريع مع حركة السهم
            return sorted(valid_cands, key=lambda x: abs(x['strike'] - price))[0]
    except Exception as e:
        log.error(f"Find smart stock contract error for {ticker}: {e}")
    return None  

def run_stock_alerts(bars: dict, scores: dict):
    cands = sorted([(t, s) for t, s in scores.items() if abs(s) >= STOCK_MIN_SCORE], key=lambda x: -abs(x[1]))
    blocks = []
    for t, s in cands:
        if len(blocks) >= MAX_STOCK_ALERTS_PER_CYCLE: break
        direction = "CALL" if s > 0 else "PUT"
        if not should_alert(f"stk:{t}", direction, STOCK_COOLDOWN_MIN): continue
        try:
            price = float(bars[t]["close"].iloc[-1])
            opt_type = "call" if s > 0 else "put"
            if (c := find_smart_stock_contract(t, price, opt_type)):
                stock_vol = bars[t]["close"].pct_change().std() * price if len(bars[t]) > 5 else price * 0.005
                exp_change = s * stock_vol * 1.5
                target_price = price + exp_change
                
                blocks.append({
                    "t": t, "dir": direction, 
                    "text": (f"🎯 <b>{esc(t)}</b> — {'صعود قوي 🚀' if s > 0 else 'هبوط قوي 📉'}\n"
                             f"▫️ <b>سعر السهم الحالي:</b> ${price:,.2f}\n"
                             f"⏱️ <b>المستهدف (خلال 10د):</b> ${target_price:,.2f}\n"
                             f"▫️ <b>عقد ذكي (Smart OTM):</b> <code>{esc(option_label(OPTION_UNDERLYING.get(t, t), c['expiry'], opt_type, c['strike']))}</code>\n"
                             f"▫️ <b>Strike:</b> ${c['strike']:,.2f} | <b>الدخول:</b> ${c['ask']:.2f}")
                })
        except Exception as e:
            log.error(f"Stock alert loop error for {t}: {e}")
            continue

    for i in range(0, len(blocks), 2):
        pair = blocks[i:i + 2]
        if tg_send("🔔 <b>عقود الأسهم الكبرى الذكية (عالية الدقة)</b>\n\n" + "\n\n──────────\n\n".join(b["text"] for b in pair) + FOOTER):
            for b in pair: mark_alert(f"stk:{b['t']}", b["dir"])

# ---------------- PRE-MARKET REPORT ----------------
def build_premarket_message(day_str: str):
    try:
        fut = yf_bars("ES=F", "5d", "15m", prepost=True)
        if fut.empty: return None
        
        mega_caps = ["NVDA", "AAPL", "MSFT", "TSLA", "META", "AMZN"]
        mega_text = "\n\n🏢 <b>تفصيل مسار الشركات الكبرى لليوم (حسب الوقت):</b>\n\n"
        
        for ticker in mega_caps:
            try:
                stock_data = yf.Ticker(ticker).history(period="1mo", interval="1d")
                if stock_data.empty: continue
                last_close = float(stock_data["Close"].iloc[-1])
                
                pm_data = yf.Ticker(ticker).history(period="1d", interval="1m", prepost=True)
                expected_open = float(pm_data["Close"].iloc[-1]) if not pm_data.empty else last_close
                
                high_low_pct = (stock_data["High"] - stock_data["Low"]) / stock_data["Close"]
                avg_move_pct = float(high_low_pct.tail(14).mean())
                stock_high = expected_open * (1 + (avg_move_pct / 2))
                stock_low = expected_open * (1 - (avg_move_pct / 2))
                move_range = stock_high - stock_low
                
                gap_pct = ((expected_open / last_close) - 1) * 100
                
                if gap_pct > 0.3:
                    m1_low, m1_high = sorted([expected_open - (move_range*0.2), stock_high])
                    m2_low, m2_high = sorted([expected_open, stock_high - (move_range*0.2)])
                    m3_low, m3_high = sorted([expected_open + (move_range*0.3), stock_high])
                elif gap_pct < -0.3:
                    m1_low, m1_high = sorted([stock_low, expected_open + (move_range*0.2)])
                    m2_low, m2_high = sorted([stock_low + (move_range*0.2), expected_open])
                    m3_low, m3_high = sorted([stock_low, expected_open - (move_range*0.3)])
                else:
                    m1_low, m1_high = sorted([expected_open - (move_range*0.3), expected_open + (move_range*0.3)])
                    m2_low, m2_high = sorted([expected_open - (move_range*0.15), expected_open + (move_range*0.15)])
                    m3_low, m3_high = sorted([stock_low, stock_high])

                mega_text += f"🔹 <b>سهم {ticker}</b> (الافتتاح المتوقع: ${expected_open:.2f})\n"
                mega_text += f"   ⏱️ من الساعة 9:30 إلى 11:30 صباحاً: بين ${m1_low:.2f} و ${m1_high:.2f}\n"
                mega_text += f"   ⏱️ من الساعة 11:30 إلى 14:00 ظهراً: بين ${m2_low:.2f} و ${m2_high:.2f}\n"
                mega_text += f"   ⏱️ من الساعة 14:00 إلى 16:00 مساءً: بين ${m3_low:.2f} و ${m3_high:.2f}\n\n"
            except Exception as e:
                log.error(f"Error building mega cap {ticker}: {e}")
                continue
                
        msg_body = (
            f"🌅 <b>التقرير المؤسساتي للاتجاة الدقيق قبل الافتتاح</b>\n"
            f"📅 {day_str}\n\n"
        )
        return msg_body + mega_text + FOOTER
    except Exception as e:
        log.error(f"Premarket message build error: {e}")
        return None

def check_premarket_report(sess):
    if not sess: return
    now = now_et()
    if not (dtime(8, 30) <= now.time() < sess[0]): return
    day = now.date().isoformat()
    if db_one("SELECT 1 FROM daily_reports WHERE report_date=? AND report_type='premarket'", (day,)): return
    msg = build_premarket_message(day)
    if msg and tg_send(msg):  
        db_exec("INSERT OR REPLACE INTO daily_reports(report_date,report_type,sent_at) VALUES(?,?,?)",
                (day, "premarket", datetime.now(UTC).isoformat()))

# ---------------- MAIN ----------------
def cycle():
    now = now_et()
    sess = get_session(now.date())
    check_premarket_report(sess)
    if not sess: return
    open_dt = datetime.combine(now.date(), sess[0], ET)
    close_dt = datetime.combine(now.date(), sess[1], ET)
    if not (open_dt + timedelta(minutes=ALERT_START_DELAY_MIN) <= now <= close_dt - timedelta(minutes=ALERT_END_BEFORE_CLOSE_MIN)): return

    bars = fetch_leader_bars()
    leaders_score, scores = leaders_component(bars)
    run_spx_alert(leaders_score)
    if scores: run_stock_alerts(bars, scores)

def main():
    init_db()
    log.info("Starting %s %s", APP_NAME, VERSION)
    tg_send(f"🚀 تم تشغيل {esc(APP_NAME)} ({esc(VERSION)}) بنظام الستريكات الذكية عالية الدقة 🎯")
    while True:
        try: cycle()
        except Exception: log.exception("Main loop critical crash")
        time.sleep(POLL_SECONDS)

if __name__ == "__main__":
    main()
