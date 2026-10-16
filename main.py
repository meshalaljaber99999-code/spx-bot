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

APP_NAME = "صَفاء التحليل"
VERSION = "v27"
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
DB_PATH = os.getenv("DB_PATH", "./data/safaa_v27.db")
POLL_SECONDS = max(60, int(os.getenv("POLL_SECONDS", "120")))

MAX_STOCK_ALERTS_PER_CYCLE = int(os.getenv("MAX_STOCK_ALERTS_PER_CYCLE", "6"))
STOCK_MIN_SCORE = float(os.getenv("STOCK_MIN_SCORE", "0.45"))     # قوة زخم السهم المطلوبة
SPX_MIN_SCORE = float(os.getenv("SPX_MIN_SCORE", "0.15"))         # أقل قوة لإشارة SPXW
STOCK_COOLDOWN_MIN = int(os.getenv("STOCK_COOLDOWN_MIN", "45"))
SPX_COOLDOWN_MIN = int(os.getenv("SPX_COOLDOWN_MIN", "30"))
FLIP_MIN_MINUTES = int(os.getenv("FLIP_MIN_MINUTES", "15"))       # أقل مدة قبل عكس الاتجاه
MAX_DTE = int(os.getenv("MAX_DTE", "7"))
HORIZON = 6                                                        # 6 شموع × 5د = 30د
MODEL_RETRAIN_MIN = 60

TOP_LEADERS_MAP = {
    "NVDA": 0.07, "AAPL": 0.07, "GOOGL": 0.05, "MSFT": 0.06, "AMZN": 0.05,
    "TSM": 0.03, "META": 0.04, "AVGO": 0.03, "TSLA": 0.04, "LLY": 0.03,
    "BRK-B": 0.03, "MU": 0.02, "AMD": 0.03, "WMT": 0.02, "JPM": 0.02,
    "V": 0.02, "XOM": 0.02, "JNJ": 0.02, "MA": 0.02, "ABBV": 0.02,
    "PLTR": 0.02, "INTC": 0.02, "CVX": 0.02, "COST": 0.02, "CSCO": 0.02,
    "KO": 0.02, "ORCL": 0.02, "UNH": 0.02, "NFLX": 0.02, "GS": 0.02,
}
‎# رمز الخيارات يختلف عن رمز السهم لبعض الشركات
OPTION_UNDERLYING = {"BRK-B": "BRKB"}

SESSION_OPEN = dtime(9, 30)
SESSION_CLOSE = dtime(16, 0)
ALERT_START_DELAY_MIN = 15      # لا تنبيهات أول 15 دقيقة
ALERT_END_BEFORE_CLOSE_MIN = 30 # ولا تنبيهات آخر 30 دقيقة

http = requests.Session()
http.headers.update({"APCA-API-KEY-ID": ALPACA_KEY,
                     "APCA-API-SECRET-KEY": ALPACA_SECRET,
                     "accept": "application/json"})

FOOTER = "\n\n⚠️ تنبيه تحليلي وليس توصية استثمارية"

# ---------------- HELPERS ----------------
def now_et() -> datetime:
    return datetime.now(ET)

def esc(x) -> str:
    return html.escape(str(x))

def to_alpaca_stock(sym: str) -> str:
    return sym.replace("-", ".")

def db_exec(sql, params=()):
    with closing(sqlite3.connect(DB_PATH)) as con:
        with con:
            con.execute(sql, params)

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
    if not row:
        return True
    last_dir, sent_at = row
    elapsed = datetime.now(UTC) - datetime.fromisoformat(sent_at)
    if last_dir != direction:
        return elapsed >= timedelta(minutes=FLIP_MIN_MINUTES)
    return elapsed >= timedelta(minutes=cooldown_min)

def mark_alert(key: str, direction: str):
    db_exec("INSERT OR REPLACE INTO alert_state(key,direction,sent_at) VALUES(?,?,?)",
            (key, direction, datetime.now(UTC).isoformat()))

# ---------------- ALPACA ----------------
def alpaca_get(path, params=None, timeout=20, retries=3):
    if not ALPACA_KEY or not ALPACA_SECRET:
        raise RuntimeError("Missing Alpaca API keys")
    base = TRADING_URL if path.startswith(("/v2/options", "/v2/calendar")) else ALPACA_DATA_URL
    for attempt in range(retries):
        r = http.get(base + path, params=params or {}, timeout=timeout)
        if r.status_code == 429:
            time.sleep(2 * (2 ** attempt))
            continue
        r.raise_for_status()
        return r.json()
    raise RuntimeError("Alpaca rate limit (429) after retries")

def drop_incomplete(df: pd.DataFrame, minutes=5) -> pd.DataFrame:
‎    """يحذف الشمعة الأخيرة فقط إذا لم تكتمل بعد (بدل حذف شمعتين)."""
    if df is None or df.empty:
        return df
    if df.index[-1] + timedelta(minutes=minutes) > datetime.now(UTC):
        return df.iloc[:-1]
    return df

def alpaca_bars(symbols, timeframe, start, feed=None) -> dict:
    params = {"symbols": ",".join(symbols), "timeframe": timeframe,
              "start": start.astimezone(UTC).isoformat(), "limit": 10000,
              "feed": feed or STOCK_FEED, "sort": "asc"}
    rows, token = {}, None
    for _ in range(20):
        if token:
            params["page_token"] = token
        data = alpaca_get("/v2/stocks/bars", params)
        for sym, bars in (data.get("bars") or {}).items():
            rows.setdefault(sym, []).extend(bars)
        token = data.get("next_page_token")
        if not token:
            break
    out = {}
    for sym, bars in rows.items():
        df = pd.DataFrame(bars)
        df.index = pd.to_datetime(df["t"], utc=True)
        df = df.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"})
        out[sym] = drop_incomplete(df[["open", "high", "low", "close", "volume"]].astype(float))
    return out

def get_option_quotes(symbols) -> dict:
‎    """يرجع فقط العروض الصالحة (bid/ask حقيقيين وسبريد معقول)."""
    out = {}
    try:
        payload = alpaca_get("/v1beta1/options/quotes/latest",
                             {"symbols": ",".join(symbols), "feed": OPTIONS_FEED})
        for sym, q in (payload.get("quotes") or {}).items():
            bid, ask = float(q.get("bp", 0) or 0), float(q.get("ap", 0) or 0)
            if bid > 0 and ask >= bid and ask >= 0.05:
                mid = (bid + ask) / 2
                if (ask - bid) / mid <= 0.35:
                    out[sym] = {"bid": bid, "ask": ask, "mid": mid}
    except Exception as e:
        log.warning("option quotes failed: %s", e)
    return out

# ---------------- MARKET CALENDAR ----------------
_cal_cache = {}

def get_session(d):
    """(open, close) لليوم أو None إذا عطلة. يدعم الإغلاق المبكر."""
    if d in _cal_cache:
        return _cal_cache[d]
    try:
        data = alpaca_get("/v2/calendar", {"start": d.isoformat(), "end": d.isoformat()})
        if data and data[0]["date"] == d.isoformat():
            sess = (dtime.fromisoformat(data[0]["open"]), dtime.fromisoformat(data[0]["close"]))
        else:
            sess = None
        _cal_cache[d] = sess
        return sess
    except Exception as e:
        log.warning("calendar fallback: %s", e)
        return (SESSION_OPEN, SESSION_CLOSE) if d.weekday() < 5 else None

# ---------------- TELEGRAM ----------------
def tg_send(text: str) -> bool:
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        log.info("Telegram skipped: %s", text[:100])
        return False
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try:
        for i in range(0, len(text), 3900):
            chunk = text[i:i + 3900]
            for attempt in range(3):
                r = requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": chunk,
                                             "parse_mode": "HTML",
                                             "disable_web_page_preview": True}, timeout=30)
                if r.status_code == 429:
                    wait = r.json().get("parameters", {}).get("retry_after", 5)
                    time.sleep(min(int(wait) + 1, 60))
                    continue
                r.raise_for_status()
                break
            else:
                return False
        return True
    except Exception as e:
        log.warning("Telegram send failed: %s", e)
        return False

# ---------------- YAHOO ----------------
def yf_bars(symbol, period="5d", interval="5m", prepost=False) -> pd.DataFrame:
    try:
        df = yf.Ticker(symbol).history(period=period, interval=interval,
                                       auto_adjust=False, prepost=prepost)
        if df is None or df.empty:
            return pd.DataFrame()
        df = df.rename(columns={c: c.lower() for c in df.columns})
        df.index = pd.to_datetime(df.index, utc=True)
        df = df[["open", "high", "low", "close", "volume"]].dropna(subset=["close"]).sort_index()
        return drop_incomplete(df)
    except Exception as e:
        log.warning("yahoo %s failed: %s", symbol, e)
        return pd.DataFrame()

def yf_batch(symbols) -> dict:
    out = {}
    try:
        raw = yf.download(list(symbols), period="5d", interval="5m", group_by="ticker",
                          auto_adjust=False, progress=False, threads=True)
        for s in symbols:
            try:
                d = raw[s].rename(columns=str.lower).dropna(subset=["close"])
                d.index = pd.to_datetime(d.index, utc=True)
                out[s] = drop_incomplete(d[["open", "high", "low", "close", "volume"]].sort_index())
            except Exception:
                continue
    except Exception as e:
        log.warning("yahoo batch failed: %s", e)
    return out

# ---------------- MOMENTUM / MODEL ----------------
def momentum_score(df, max_age_min=40):
‎    """زخم مُطبّع بالتذبذب في نطاق [-1, 1]. يعتمد على شموع اليوم فقط."""
    try:
        if df is None or len(df) < 30:
            return None
        if datetime.now(UTC) - df.index[-1] > timedelta(minutes=max_age_min):
            return None
        idx = df.index.tz_convert(ET)
        d = df[idx.date == idx[-1].date()]
        n = len(d)
        if n < 4:
            return None
        c = d["close"].astype(float)
        k6, k12 = min(6, n - 1), min(12, n - 1)
        r6 = c.iloc[-1] / c.iloc[-1 - k6] - 1
        r12 = c.iloc[-1] / c.iloc[-1 - k12] - 1
        vol = d["volume"].astype(float)
        vgap = 0.0
        if vol.sum() > 0:
            vwap = (c * vol).sum() / vol.sum()
            vgap = c.iloc[-1] / vwap - 1
        sigma = df["close"].astype(float).pct_change().tail(156).std() * math.sqrt(6)
        if not sigma or math.isnan(sigma):
            return None
        m = 0.4 * r6 + 0.3 * r12 + 0.3 * vgap
        return float(math.tanh(m / sigma))
    except Exception:
        return None

def day_change_pct(df) -> float:
    idx = df.index.tz_convert(ET)
    d = df[idx.date == idx[-1].date()]
    return (float(d["close"].iloc[-1]) / float(d["open"].iloc[0]) - 1) * 100

FEATURES = ["r1", "r3", "r6", "r12", "vol12", "ema_gap", "range", "vchg", "tod_sin", "tod_cos"]

def feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    close = x["close"].astype(float)
    for n in (1, 3, 6, 12):
        x[f"r{n}"] = close.pct_change(n)
    x["vol12"] = x["r1"].rolling(12).std()
    x["ema_gap"] = close / close.ewm(span=12, adjust=False).mean() - 1
    x["range"] = (x["high"] - x["low"]) / close
    x["vchg"] = x["volume"].pct_change().replace([np.inf, -np.inf], np.nan).fillna(0)
    idx = x.index.tz_convert(ET)
    mins = np.array([i.hour * 60 + i.minute for i in idx])
    x["tod_sin"] = np.sin(2 * np.pi * mins / 1440)
    x["tod_cos"] = np.cos(2 * np.pi * mins / 1440)
    return x.replace([np.inf, -np.inf], np.nan)

def make_model():
    return make_pipeline(SimpleImputer(strategy="median"),
                         RandomForestClassifier(n_estimators=120, max_depth=5,
                                                min_samples_leaf=10, random_state=42, n_jobs=-1))

_MODEL = {"model": None, "trained_at": None, "acc": 0.5}

def fetch_spy_df() -> pd.DataFrame:
    try:
        raw = alpaca_bars(["SPY"], "5Min", datetime.now(UTC) - timedelta(days=25))
        df = raw.get("SPY")
        if df is not None and len(df) >= 400:
            return df
    except Exception as e:
        log.warning("alpaca SPY failed: %s", e)
    return yf_bars("SPY", period="30d", interval="5m")

def model_component(spy_df):
‎    """يرجع (score في [-1,1], accuracy). الوزن يقل تلقائياً إذا الدقة ضعيفة."""
    try:
        if spy_df is None or len(spy_df) < 400:
            return None, None
        x = feature_frame(spy_df)
        fut = x["close"].shift(-HORIZON) / x["close"] - 1
        x["target"] = np.where(fut.isna(), np.nan, (fut > 0).astype(float))
        ds = x.dropna(subset=FEATURES + ["target"])
        if len(ds) < 300:
            return None, None

        stale = (_MODEL["model"] is None or _MODEL["trained_at"] is None or
                 datetime.now(UTC) - _MODEL["trained_at"] > timedelta(minutes=MODEL_RETRAIN_MIN))
        if stale:
            X, y = ds[FEATURES], ds["target"].astype(int)
            accs = []
            for tr, te in TimeSeriesSplit(n_splits=4, gap=HORIZON).split(X):
                m = make_model()
                m.fit(X.iloc[tr], y.iloc[tr])
                accs.append(m.score(X.iloc[te], y.iloc[te]))
            final = make_model()
            final.fit(X, y)
            _MODEL.update(model=final, trained_at=datetime.now(UTC),
                          acc=float(np.mean(accs)) if accs else 0.5)
            log.info("Model retrained | walk-forward acc=%.3f", _MODEL["acc"])

        prob_up = float(_MODEL["model"].predict_proba(x[FEATURES].iloc[[-1]])[0, 1])
        quality = float(np.clip((_MODEL["acc"] - 0.5) / 0.05, 0, 1))
        return (prob_up - 0.5) * 2 * quality, _MODEL["acc"]
    except Exception as e:
        log.warning("model failed: %s", e)
        return None, None

# ---------------- LEADERS ----------------
def fetch_leader_bars() -> dict:
    tickers = list(TOP_LEADERS_MAP)
    out = {}
    try:
        raw = alpaca_bars([to_alpaca_stock(t) for t in tickers], "5Min",
                          datetime.now(UTC) - timedelta(days=5))
        out = {k.replace(".", "-"): v for k, v in raw.items()}
    except Exception as e:
        log.warning("alpaca leaders failed: %s", e)
    missing = [t for t in tickers if t not in out or out[t].empty]
    if len(missing) > len(tickers) * 0.3:
        out.update(yf_batch(missing))
    return out

def leaders_component(bars: dict):
    scores = {}
    for t in TOP_LEADERS_MAP:
        s = momentum_score(bars.get(t))
        if s is not None:
            scores[t] = s
    if len(scores) < 10:
        return None, scores
    tw = sum(TOP_LEADERS_MAP[t] for t in scores)
    agg = sum(scores[t] * TOP_LEADERS_MAP[t] for t in scores) / tw
    return float(agg), scores

# ---------------- SPXW ENGINE ----------------
def combine_signal(leaders, model, index, vix):
    comps = {"leaders": (leaders, 0.40), "model": (model, 0.25),
             "index": (index, 0.25), "vix": (vix, 0.10)}
    avail = {k: v for k, v in comps.items() if v[0] is not None}
    if len(avail) < 2 or ("leaders" not in avail and "index" not in avail):
        return None
    tw = sum(w for _, w in avail.values())
    return float(sum(v * w for v, w in avail.values()) / tw)

def get_spxw_contract(spot: float, direction: str):
‎    """سلسلة خيارات SPX من ياهو (متأخرة ~15 دقيقة). لا نختلق أي سعر."""
    try:
        t = yf.Ticker("^SPX")
        today = now_et().date().isoformat()
        exps = [e for e in (t.options or []) if e >= today]
        if not exps:
            return None
        exp = exps[0]
        chain = t.option_chain(exp)
        df = chain.calls if direction == "CALL" else chain.puts
        df = df[(df["bid"] > 0) & (df["ask"] > 0)].copy()
        if "contractSymbol" in df:
            w = df[df["contractSymbol"].astype(str).str.startswith("SPXW")]
            if not w.empty:
                df = w
        if df.empty:
            return None
        df["spread"] = (df["ask"] - df["bid"]) / ((df["ask"] + df["bid"]) / 2)
        df = df[df["spread"] <= 0.30]
        if df.empty:
            return None
        df["dist"] = (df["strike"] - spot).abs()
        row = df.sort_values("dist").iloc[0]
        return {"expiry": exp, "strike": float(row["strike"]),
                "bid": float(row["bid"]), "ask": float(row["ask"])}
    except Exception as e:
        log.warning("SPXW chain failed: %s", e)
        return None

def option_label(underlying: str, exp: str, opt_type: str, strike: float) -> str:
    ds = datetime.strptime(exp, "%Y-%m-%d").strftime("%d%m%y")
    return f"{underlying} {ds} {'C' if opt_type.lower().startswith('c') else 'P'} {strike:g}"

def run_spx_alert(leaders_score):
    spy_df = fetch_spy_df()
    model_score, acc = model_component(spy_df)

    gspc = yf_bars("^GSPC", "5d", "5m")
    index_score = momentum_score(gspc)
    vix_s = momentum_score(yf_bars("^VIX", "5d", "5m"))
    vix_score = -vix_s if vix_s is not None else None  # ارتفاع VIX = ضغط هبوطي

    total = combine_signal(leaders_score, model_score, index_score, vix_score)
    if total is None:
        log.info("SPXW: insufficient data")
        return
    if abs(total) < SPX_MIN_SCORE:
        log.info("SPXW neutral (score=%.2f)", total)
        return

    direction = "CALL" if total > 0 else "PUT"
    if not should_alert("spxw", direction, SPX_COOLDOWN_MIN):
        return

‎    # سعر المؤشر: من ياهو مباشرة، وإلا تقدير من SPY (يُوسم تقديري)
    estimated = False
    if not gspc.empty:
        spot = float(gspc["close"].iloc[-1])
    elif spy_df is not None and not spy_df.empty:
        spot, estimated = float(spy_df["close"].iloc[-1]) * 10.0, True
    else:
        return

    c = get_spxw_contract(spot, direction)
    up = direction == "CALL"
    fmt = lambda v: "—" if v is None else f"{v:+.2f}"
    lines = [
        "📊 <b>SPXW — توقع لحظي</b>\n",
        f"▫️ <b>الاتجاه:</b> {'كول (CALL) 🟢' if up else 'بوت (PUT) 🔴'}",
        f"▫️ <b>قوة الإشارة:</b> {min(99, int(abs(total) * 100))}%",
        f"▫️ <b>سعر المؤشر:</b> ${spot:,.2f}{' (تقديري)' if estimated else ''}",
    ]
    if c:
        entry = c["ask"]
        lines += [
            f"▫️ <b>رقم العقد:</b> <code>{esc(option_label('SPXW', c['expiry'], 'call' if up else 'put', c['strike']))}</code>",
            f"▫️ <b>Strike:</b> ${c['strike']:,.0f}",
            f"▫️ <b>الدخول (Ask):</b> ${entry:.2f}",
            f"▫️ <b>خروج قريب (+50%):</b> ${entry * 1.5:.2f}",
            f"▫️ <b>خروج بعيد (+200%):</b> ${entry * 3.0:.2f}",
        ]
    else:
        lines.append("▫️ <b>العقد:</b> تعذر جلب سعر حقيقي للعقد الآن — الإشارة للاتجاه فقط")
    lines += [
        "",
        f"🧭 زخم الشركات الكبرى {fmt(leaders_score)} | النموذج {fmt(model_score)} | "
        f"حركة المؤشر {fmt(index_score)} | التذبذب {fmt(vix_score)}",
‎        "🎯 القريب: " + ("صعود تدريجي وتجاوز المقاومة" if up else "ضغط هبوطي واختبار الدعم"),
    ]
    if c:
        lines.append("⏱ أسعار العقد متأخرة ~15 دقيقة")
    if acc is not None and acc < 0.52:
        lines.append("ℹ️ دقة النموذج التاريخية ضعيفة، اعتمدت الإشارة على الزخم والمؤشر")
    if tg_send("\n".join(lines) + FOOTER):
        mark_alert("spxw", direction)

# ---------------- MEGA CAPS ----------------
def find_stock_contract(ticker: str, price: float, opt_type: str):
    und = OPTION_UNDERLYING.get(ticker, ticker)
    today = now_et().date()
    payload = alpaca_get("/v2/options/contracts", {
        "underlying_symbols": und, "status": "active", "type": opt_type,
        "expiration_date_gte": today.isoformat(),
        "expiration_date_lte": (today + timedelta(days=MAX_DTE)).isoformat(),
        "strike_price_gte": round(price * 0.97, 2),
        "strike_price_lte": round(price * 1.03, 2), "limit": 200})
    contracts = payload.get("option_contracts") or payload.get("contracts") or []
    by_exp = {}
    for c in contracts:
        by_exp.setdefault(c["expiration_date"], []).append(c)
    for exp in sorted(by_exp)[:2]:
        cs = sorted(by_exp[exp], key=lambda c: abs(float(c["strike_price"]) - price))[:3]
        quotes = get_option_quotes([c["symbol"] for c in cs])
        for c in cs:
            q = quotes.get(c["symbol"])
            if q:
                return {"symbol": c["symbol"], "expiry": exp,
                        "strike": float(c["strike_price"]), "ask": q["ask"]}
    return None  # لا نختلق سعر — نتخطى السهم

def run_stock_alerts(bars: dict, scores: dict):
    cands = sorted([(t, s) for t, s in scores.items() if abs(s) >= STOCK_MIN_SCORE],
                   key=lambda x: -abs(x[1]))
    today = now_et().date()
    blocks = []
    for t, s in cands:
        if len(blocks) >= MAX_STOCK_ALERTS_PER_CYCLE:
            break
        direction = "CALL" if s > 0 else "PUT"
        if not should_alert(f"stk:{t}", direction, STOCK_COOLDOWN_MIN):
            continue
        try:
            price = float(bars[t]["close"].iloc[-1])
            c = find_stock_contract(t, price, "call" if s > 0 else "put")
        except Exception as e:
            log.warning("contract lookup %s failed: %s", t, e)
            continue
        if not c:
            log.info("%s: no valid priced contract, skipped", t)
            continue
        entry = c["ask"]
        dte = (datetime.strptime(c["expiry"], "%Y-%m-%d").date() - today).days
        dte_label = "0DTE" if dte == 0 else f"بعد {dte} يوم"
        up = s > 0
        text = (
            f"🏢 <b>{esc(t)}</b> — {'صعود 🚀' if up else 'نزول 📉'} "
            f"(قوة {min(99, int(abs(s) * 100))}%)\n"
            f"▫️ السعر: ${price:.2f} ({day_change_pct(bars[t]):+.2f}% اليوم)\n"
            f"▫️ العقد: <code>{esc(option_label(OPTION_UNDERLYING.get(t, t), c['expiry'], 'call' if up else 'put', c['strike']))}</code>\n"
            f"▫️ الانتهاء: {c['expiry']} ({dte_label}) | Strike: ${c['strike']:g}\n"
            f"▫️ الدخول (Ask): ${entry:.2f}\n"
            f"▫️ خروج قريب (+50%): ${entry * 1.5:.2f} | بعيد (+200%): ${entry * 3.0:.2f}"
        )
        blocks.append({"t": t, "dir": direction, "text": text})
        time.sleep(0.3)

‎    # كل شركتين في رسالة واحدة
    for i in range(0, len(blocks), 2):
        pair = blocks[i:i + 2]
        msg = "🔔 <b>عقود الشركات الكبرى</b>\n\n" + "\n\n──────────\n\n".join(b["text"] for b in pair) + FOOTER
        if tg_send(msg):
            for b in pair:
                mark_alert(f"stk:{b['t']}", b["dir"])
        time.sleep(1)

# ---------------- PRE-MARKET REPORT ----------------
def build_premarket_message(day_str: str):
‎    # العقود المستقبلية ES=F تعطي حركة ما قبل السوق الحقيقية
    fut = yf_bars("ES=F", "5d", "15m", prepost=True)
    if fut.empty:
        return None
    fe = fut.index.tz_convert(ET)
    today = now_et().date()
    prior_mask = np.array([(d < today) and (t <= dtime(16, 0)) for d, t in zip(fe.date, fe.time)])
    if not prior_mask.any():
        return None
    prior_close = float(fut["close"][prior_mask].iloc[-1])
    fut_chg = float(fut["close"].iloc[-1]) / prior_close - 1
    fut_score = math.tanh(fut_chg / 0.005)

    vix_score = None
    try:
        v = yf.Ticker("^VIX").history(period="5d", interval="1d")
        if len(v) >= 2:
            vix_score = -math.tanh((float(v["Close"].iloc[-1]) / float(v["Close"].iloc[-2]) - 1) / 0.05)
    except Exception:
        pass
    total = fut_score if vix_score is None else 0.7 * fut_score + 0.3 * vix_score

    d = yf.Ticker("^GSPC").history(period="20d", interval="1d")
    if d.empty:
        return None
    avg_range = float((d["High"] - d["Low"]).tail(10).mean())
    center = float(d["Close"].iloc[-1]) * (1 + fut_chg)
    high, low = center + avg_range * 0.5, center - avg_range * 0.5

    if abs(total) < SPX_MIN_SCORE:
        view, short_t, long_t = "محايد ⚪", "تذبذب بدون اتجاه واضح بعد الافتتاح", "انتظار محفز واضح"
    elif total > 0:
        view, short_t, long_t = "صعود 🟢", "فتح إيجابي ومحاولة تجاوز المقاومة القريبة", "اتجاه إيجابي ما دام الزخم محافظاً على قوته"
    else:
        view, short_t, long_t = "هبوط 🔴", "ضغط هبوطي واختبار الدعم القريب", "حذر من استمرار الزخم السلبي"

    return (
        f"🌅 <b>تقرير ما قبل السوق — SPXW</b>\n📅 {day_str}\n\n"
        f"📊 <b>التوقع العام:</b> {view} (العقود المستقبلية {fut_chg * 100:+.2f}%)\n"
        f"📈 <b>أعلى قمة متوقعة:</b> ${high:,.2f}\n"
        f"📉 <b>أقل قاع متوقع:</b> ${low:,.2f}\n"
        f"🎯 <b>المدى القريب:</b> {short_t}\n"
        f"🌐 <b>المدى البعيد:</b> {long_t}" + FOOTER
    )

def check_premarket_report(sess):
    if not sess:
        return
    now = now_et()
    if not (dtime(8, 30) <= now.time() < sess[0]):
        return
    day = now.date().isoformat()
    if db_one("SELECT 1 FROM daily_reports WHERE report_date=? AND report_type='premarket'", (day,)):
        return
    msg = build_premarket_message(day)
    if msg and tg_send(msg):  # نسجل بعد نجاح الإرسال فقط
        db_exec("INSERT OR REPLACE INTO daily_reports(report_date,report_type,sent_at) VALUES(?,?,?)",
                (day, "premarket", datetime.now(UTC).isoformat()))

# ---------------- MAIN ----------------
def cycle():
    now = now_et()
    sess = get_session(now.date())
    check_premarket_report(sess)
    if not sess:
        return
    open_dt = datetime.combine(now.date(), sess[0], ET)
    close_dt = datetime.combine(now.date(), sess[1], ET)
    if not (open_dt + timedelta(minutes=ALERT_START_DELAY_MIN) <= now
            <= close_dt - timedelta(minutes=ALERT_END_BEFORE_CLOSE_MIN)):
        return

    bars = fetch_leader_bars()
    leaders_score, scores = leaders_component(bars)
    run_spx_alert(leaders_score)
    if scores:
        run_stock_alerts(bars, scores)

def main():
    init_db()
    log.info("Starting %s %s", APP_NAME, VERSION)
    tg_send(f"🚀 تم تشغيل {esc(APP_NAME)} ({esc(VERSION)})")
    while True:
        try:
            cycle()
        except Exception:
            log.exception("Main loop error")
        time.sleep(POLL_SECONDS)

if __name__ == "__main__":
    main()
sed -i 's/\xe2\x80\x8e//g; s/\xe2\x80\x8f//g; s/\xef\xbb\xbf//g' main.py

