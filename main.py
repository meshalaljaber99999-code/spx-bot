# -*- coding: utf-8 -*-
"""
صَفاء التحليل SPX — النسخة الكمّية v2.2
تصميم أبو سحاب
تحليل احتمالي فقط — لا تنفيذ تلقائي — Fail Closed
"""
import os, re, sys, json, gzip, math, time, sqlite3, logging, uuid
from datetime import datetime, timedelta, timezone, time as dtime
from zoneinfo import ZoneInfo

import requests
import numpy as np
import pandas as pd
import yfinance as yf

try:
    import lightgbm as lgb
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False

# ============================ الإعدادات ============================
NY = ZoneInfo("America/New_York")


def env(k, d=""):
    return os.getenv(k, d).strip()


ALPACA_KEY = env("APCA_API_KEY_ID")
ALPACA_SECRET = env("APCA_API_SECRET_KEY")
DATA_URL = env("ALPACA_DATA_URL", "https://data.alpaca.markets").rstrip("/")
STOCK_FEED = env("ALPACA_STOCK_FEED", "iex")
FRED_KEY = env("FRED_API_KEY")
TG_TOKEN = env("TELEGRAM_BOT_TOKEN")
TG_CHAT = env("TELEGRAM_CHAT_ID")
DB_PATH = env("DB_PATH", "./data/safaa_spx.db")
MODEL_PATH = env("MODEL_PATH", "./data/safaa_lgbm_model.txt")
ML_WEIGHT = float(env("ML_WEIGHT", "0.0"))          # 0 = Shadow فقط
MAX_STALE_MIN = int(env("MAX_STALE_MIN", "35"))
UPDATE_MINUTES = int(env("UPDATE_MINUTES", "60"))
_DEF_W = "NVDA:7.5,AAPL:6.5,MSFT:6.5,AMZN:4,GOOGL:4,META:3,AVGO:2.5,TSLA:2,JPM:1.5,LLY:1.3"
LEADER_W = {k.strip().upper(): float(v) for k, v in
            (p.split(":") for p in env("LEADER_WEIGHTS", _DEF_W).split(",") if ":" in p)}
LEADERS = list(LEADER_W.keys())
SECTORS = env("SECTORS", "XLK,XLF,XLE,XLV,XLY,XLP,XLI,XLU").split(",")
SIGNATURE = "⚡ تصميم أبو سحاب"
CRITICAL = ["SPX", "ES", "VIX", "CHAIN"]
ML_FEATURES = ["time_of_day", "vix_score", "gex_score", "tech_score", "breadth_score",
               "micro_score", "credit_score", "futures_score", "leaders_score"]
BASE_W = {"futures": 0.27, "vix": 0.13, "gex": 0.13, "technical": 0.12, "leaders": 0.15,
          "breadth": 0.08, "macro": 0.08, "nasdaq": 0.07, "microstructure": 0.10}

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("safaa_spx")


class SourceError(Exception):
    pass


def now_ny():
    return datetime.now(NY)


def is_rth(now):
    m = now.hour * 60 + now.minute
    return now.weekday() < 5 and 570 <= m < 960


def n0(v):
    return "—" if v is None else f"{v:,.0f}"


def clip(x, lo=-1.0, hi=1.0):
    return max(lo, min(hi, x))


# ============================ التخزين ============================
def init_db():
    folder = os.path.dirname(DB_PATH)
    if folder:
        os.makedirs(folder, exist_ok=True)
    with sqlite3.connect(DB_PATH) as c:
        c.execute("CREATE TABLE IF NOT EXISTS kv (k TEXT PRIMARY KEY, v TEXT)")
        c.execute("""CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, day TEXT, kind TEXT,
            anchor REAL, hi REAL, lo REAL, hi_near REAL, hi_far REAL, lo_near REAL, lo_far REAL,
            p_up REAL, p_down REAL, p_neutral REAL, score REAL,
            data_quality REAL, agreement REAL, payload TEXT, ml_id TEXT)""")
        try:
            c.execute("ALTER TABLE predictions ADD COLUMN ml_id TEXT")
        except sqlite3.OperationalError:
            pass
        c.execute("CREATE TABLE IF NOT EXISTS realized (day TEXT PRIMARY KEY, high REAL, low REAL, close REAL)")
        c.execute("CREATE TABLE IF NOT EXISTS data_quality (ts TEXT, source TEXT, ok INTEGER, detail TEXT)")
        c.execute("""CREATE TABLE IF NOT EXISTS gex_levels (
            ts TEXT, spot REAL, flip REAL, call_wall REAL, put_wall REAL, net_gex REAL, regime TEXT)""")
        c.execute("CREATE TABLE IF NOT EXISTS options_snapshots (ts TEXT, blob BLOB)")
        c.execute("""CREATE TABLE IF NOT EXISTS bars_1m (
            symbol TEXT, ts TEXT, open REAL, high REAL, low REAL, close REAL, volume REAL,
            PRIMARY KEY (symbol, ts))""")
        c.execute("""CREATE TABLE IF NOT EXISTS ml_features (
            ml_id TEXT PRIMARY KEY, ts TEXT, day TEXT, time_of_day INTEGER, anchor REAL,
            vix_score REAL, gex_score REAL, tech_score REAL, breadth_score REAL,
            micro_score REAL, credit_score REAL, futures_score REAL, leaders_score REAL,
            prediction REAL, actual_target INTEGER DEFAULT -1)""")
        for col in ("futures_score", "leaders_score"):
            try:
                c.execute(f"ALTER TABLE ml_features ADD COLUMN {col} REAL")
            except sqlite3.OperationalError:
                pass
        c.execute("CREATE TABLE IF NOT EXISTS drop_log (day TEXT PRIMARY KEY, risk REAL, level TEXT, happened INTEGER)")
        c.execute("""CREATE TABLE IF NOT EXISTS hourly_pred (
            day TEXT, made_at TEXT, lead INTEGER, w_start INTEGER, w_end INTEGER,
            center REAL, lo50 REAL, hi50 REAL, lo80 REAL, hi80 REAL,
            actual REAL, in50 INTEGER, in80 INTEGER)""")


def kv_get(k):
    with sqlite3.connect(DB_PATH) as c:
        r = c.execute("SELECT v FROM kv WHERE k=?", (k,)).fetchone()
    return r[0] if r else None


def kv_set(k, v):
    with sqlite3.connect(DB_PATH) as c:
        c.execute("INSERT INTO kv(k,v) VALUES(?,?) ON CONFLICT(k) DO UPDATE SET v=excluded.v", (k, str(v)))


def save_quality(now, q):
    with sqlite3.connect(DB_PATH) as c:
        c.executemany("INSERT INTO data_quality VALUES (?,?,?,?)",
                      [(now.isoformat(timespec="seconds"), n, int(ok), msg) for n, (ok, msg) in q.items()])


def save_prediction(now, kind, fc):
    ts, day = now.isoformat(timespec="seconds"), now.date().isoformat()
    with sqlite3.connect(DB_PATH) as c:
        c.execute("""INSERT INTO predictions (ts,day,kind,anchor,hi,lo,hi_near,hi_far,lo_near,lo_far,
                     p_up,p_down,p_neutral,score,data_quality,agreement,payload,ml_id)
                     VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                  (ts, day, kind, fc["anchor"], fc["hi"], fc["lo"],
                   fc["hi_near"], fc["hi_far"], fc["lo_near"], fc["lo_far"],
                   fc["probs"]["up"], fc["probs"]["down"], fc["probs"]["neutral"], fc["score"],
                   fc["data_quality"], fc["agreement"],
                   json.dumps(fc, default=float, ensure_ascii=False), fc["ml_id"]))
        g = fc["gex"]
        c.execute("INSERT INTO gex_levels VALUES (?,?,?,?,?,?,?)",
                  (ts, g["spot"], g["flip"], g["call_wall"], g["put_wall"], g["net"], g["regime"]))
        f = fc["ml_features"]
        c.execute("""INSERT INTO ml_features (ml_id, ts, day, time_of_day, anchor, vix_score, gex_score,
                     tech_score, breadth_score, micro_score, credit_score, futures_score, leaders_score,
                     prediction) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                  (fc["ml_id"], ts, day, f["time_of_day"], fc["anchor"], f["vix_score"], f["gex_score"],
                   f["tech_score"], f["breadth_score"], f["micro_score"], f["credit_score"],
                   f["futures_score"], f["leaders_score"], fc["score"]))


def save_snapshot(now, chain):
    """لقطة مضغوطة للسلسلة — بيانات تدريب مستقبلية."""
    try:
        df, spot, today = chain["df"], chain["spot"], now.date()
        x = df[(df.exp >= today) & (df.exp <= today + timedelta(days=7)) &
               (df.strike.between(spot * 0.94, spot * 1.06))].copy()
        x["exp"] = x["exp"].astype(str)
        blob = gzip.compress(x.to_json(orient="records").encode())
        with sqlite3.connect(DB_PATH) as c:
            c.execute("INSERT INTO options_snapshots VALUES (?,?)", (now.isoformat(timespec="seconds"), blob))
    except Exception as e:
        log.warning("تعذر حفظ اللقطة: %s", e)


# ============================ التيليجرام ============================
def tg(text):
    if not TG_TOKEN or not TG_CHAT:
        log.warning("بيانات التيليجرام غير مكتملة")
        return False
    try:
        r = requests.post(f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage",
                          json={"chat_id": TG_CHAT, "text": text[:4000], "disable_web_page_preview": True},
                          timeout=15)
        return r.status_code == 200
    except Exception as e:
        log.error("خطأ تيليجرام: %s", e)
        return False


# ============================ طبقة جمع البيانات ============================
def yf_hist(sym, period, interval):
    try:
        df = yf.Ticker(sym).history(period=period, interval=interval,
                                    prepost=(interval != "1d"), auto_adjust=False)
    except Exception as e:
        raise SourceError(f"{sym}: {str(e)[:80]}")
    if df is None or df.empty:
        raise SourceError(f"{sym}: لا توجد بيانات")
    df = df.dropna(subset=["Close"]).copy()
    idx = df.index
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    df.index = idx.tz_convert(NY)
    return df


def age_min(df):
    return (now_ny() - df.index[-1].to_pydatetime()).total_seconds() / 60.0


def daily_completed(sym, today):
    df = yf_hist(sym, "3mo", "1d")
    df = df[df.index.date < today]
    if len(df) < 15:
        raise SourceError(f"{sym}: تاريخ يومي غير كافٍ")
    return df


def session_close(df, day):
    m = df.index.hour * 60 + df.index.minute
    d = df[(df.index.date == day) & (m < 960)]
    return float(d["Close"].iloc[-1]) if not d.empty else None


def get_spx(now):
    today = now.date()
    dd = daily_completed("^GSPC", today)
    tr = pd.concat([dd.High - dd.Low, (dd.High - dd.Close.shift()).abs(),
                    (dd.Low - dd.Close.shift()).abs()], axis=1).max(axis=1)
    prev = float(dd.Close.iloc[-1])
    out = dict(prev_close=prev, atr=float(tr.tail(14).mean()), spot=prev,
               day_high=None, day_low=None, closes=None)
    if is_rth(now):
        df = yf_hist("^GSPC", "5d", "5m")
        a = age_min(df)
        t = df[df.index.date == today]
        if t.empty or a > MAX_STALE_MIN:
            raise SourceError(f"SPX متأخر {a:.0f} دقيقة")
        out.update(spot=float(t.Close.iloc[-1]), day_high=float(t.High.max()),
                   day_low=float(t.Low.min()), closes=df.Close.tail(60))
    return out


def get_future(sym, now):
    df = yf_hist(sym, "5d", "5m")
    a = age_min(df)
    if a > MAX_STALE_MIN:
        raise SourceError(f"{sym} متأخر {a:.0f} دقيقة")
    td = now.date()
    days = sorted({d for d in df.index.date if d < td})
    if not days:
        raise SourceError(f"{sym}: لا يوجد إغلاق سابق")
    pdv, pc = None, None
    for d_ in reversed(days):
        pc = session_close(df, d_)
        if pc:
            pdv = d_
            break
    if not pc:
        raise SourceError(f"{sym}: تعذر تحديد الإغلاق السابق")
    last = float(df.Close.iloc[-1])
    cut = datetime.combine(pdv, dtime(16, 0), tzinfo=NY)
    on = df[df.index > cut]
    return dict(last=last, prev_close=pc, ret_pct=(last / pc - 1) * 100,
                on_high=float(on.High.max()), on_low=float(on.Low.min()), closes=df.Close.tail(60))


def get_vix(now):
    today = now.date()
    dd = daily_completed("^VIX", today)
    prev = float(dd.Close.iloc[-1])
    vix = prev
    rth = is_rth(now)
    if rth:
        df = yf_hist("^VIX", "5d", "5m")
        if age_min(df) > MAX_STALE_MIN:
            raise SourceError("VIX متأخر")
        vix = float(df.Close.iloc[-1])
    v9 = None
    try:
        d9 = daily_completed("^VIX9D", today)
        v9 = float(d9.Close.iloc[-1])
        if rth:
            v9 = float(yf_hist("^VIX9D", "5d", "5m").Close.iloc[-1])
    except SourceError:
        pass
    return dict(vix=vix, prev=prev, chg_pct=(vix / prev - 1) * 100, vix9d=v9)


CBOE_URL = "https://cdn.cboe.com/api/global/delayed_quotes/options/_SPX.json"
OPT_RE = re.compile(r"^([A-Z]+)(\d{6})([CP])(\d{8})$")


def gex_levels(spot, df, today, max_dte=7, band=0.06):
    x = df.copy()
    x["dte"] = x["exp"].map(lambda e: (e - today).days)
    z = x[x.dte == 0]
    z_ratio = float(z[z.cp == "C"].vol.sum() / max(1.0, z[z.cp == "P"].vol.sum())) if not z.empty else None
    x = x[(x.dte >= 0) & (x.dte <= max_dte) & (x.strike.between(spot * (1 - band), spot * (1 + band)))
          & (x.gamma > 0) & (x.oi > 0)].copy()
    if x.empty:
        raise SourceError("GEX: لا توجد عقود صالحة")
    sign = np.where(x.cp == "C", 1.0, -1.0)
    x["gex"] = x.gamma * x.oi * 100 * spot * spot * 0.01 * sign
    by = x.groupby("strike").gex.sum().sort_index()
    calls = x[x.cp == "C"].groupby("strike").gex.sum()
    puts = x[x.cp == "P"].groupby("strike").gex.sum()
    above = calls[calls.index >= spot]
    below = puts[puts.index <= spot]
    call_wall = float(above.idxmax()) if not above.empty else None
    put_wall = float(below.idxmin()) if not below.empty else None
    cum = by.cumsum()
    s = np.sign(cum.values)
    cross = [i for i in range(1, len(s)) if s[i] != s[i - 1] and s[i] != 0]
    flip = float(cum.index[min(cross, key=lambda i: abs(cum.index[i] - spot))]) if cross else None
    net = float(by.sum())
    regime = ("POSITIVE" if spot > flip else "NEGATIVE") if flip else ("POSITIVE" if net > 0 else "NEGATIVE")
    tot = x.gex.abs().sum()
    z_share = float(x[x.dte == 0].gex.abs().sum() / tot) if tot > 0 else 0.0
    return dict(spot=float(spot), flip=flip, call_wall=call_wall, put_wall=put_wall, net=net,
                regime=regime, zero_dte_share=z_share, zero_dte_cp_vol=z_ratio)


def get_chain(now):
    try:
        r = requests.get(CBOE_URL, headers={"User-Agent": "Mozilla/5.0"}, timeout=25)
    except Exception as e:
        raise SourceError(f"Cboe: {str(e)[:80]}")
    if r.status_code != 200:
        raise SourceError(f"Cboe HTTP {r.status_code}")
    j = r.json().get("data", {})
    spot = float(j.get("current_price") or 0)
    rows = []
    for o in j.get("options", []):
        m = OPT_RE.match(o.get("option", ""))
        if not m:
            continue
        root, ymd, cp, k = m.groups()
        rows.append(dict(root=root, exp=datetime.strptime(ymd, "%y%m%d").date(), cp=cp, strike=int(k) / 1000.0,
                         oi=float(o.get("open_interest") or 0), vol=float(o.get("volume") or 0),
                         gamma=float(o.get("gamma") or 0), delta=float(o.get("delta") or 0),
                         iv=float(o.get("iv") or 0)))
    if not rows or spot <= 0:
        raise SourceError("Cboe: بيانات فارغة")
    df = pd.DataFrame(rows)
    return dict(spot=spot, df=df, gex=gex_levels(spot, df, now.date()))


def ah():
    if not (ALPACA_KEY and ALPACA_SECRET):
        raise SourceError("مفاتيح Alpaca غير مضبوطة")
    return {"APCA-API-KEY-ID": ALPACA_KEY, "APCA-API-SECRET-KEY": ALPACA_SECRET}


def get_breadth(now):
    if not is_rth(now):
        raise SourceError("غير مؤكد قبل الافتتاح")
    r = requests.get(f"{DATA_URL}/v2/stocks/snapshots", headers=ah(),
                     params={"symbols": ",".join(LEADERS + SECTORS), "feed": STOCK_FEED}, timeout=15)
    if r.status_code != 200:
        raise SourceError(f"Alpaca HTTP {r.status_code}")
    j = r.json()
    j = j.get("snapshots", j)

    def rets(names):
        out = []
        for s in names:
            sn = j.get(s) or {}
            p = (sn.get("latestTrade") or {}).get("p")
            pc = (sn.get("prevDailyBar") or {}).get("c")
            if p and pc:
                out.append((p / pc - 1) * 100)
        return out

    L, S = rets(LEADERS), rets(SECTORS)
    if len(L) < 5:
        raise SourceError("بيانات القياديات غير كافية")
    return dict(leaders_up=sum(v > 0 for v in L) / len(L), leaders_avg=float(np.mean(L)),
                sectors_up=(sum(v > 0 for v in S) / len(S)) if S else None)


def spy_bars_today(now):
    if not is_rth(now):
        raise SourceError("خارج الجلسة")
    s = datetime.combine(now.date(), dtime(9, 30), tzinfo=NY).astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    r = requests.get(f"{DATA_URL}/v2/stocks/SPY/bars", headers=ah(),
                     params={"timeframe": "1Min", "start": s, "limit": 1000, "feed": STOCK_FEED}, timeout=15)
    bars = (r.json().get("bars") or []) if r.status_code == 200 else []
    if len(bars) < 10:
        raise SourceError("شموع SPY غير كافية")
    df = pd.DataFrame(bars)
    for col in ("o", "h", "l", "c", "v"):
        df[col] = df[col].astype(float)
    return df


def get_spy_vwap(now):
    df = spy_bars_today(now)
    tp = (df.h + df.l + df.c) / 3
    cv = df.v.cumsum()
    if cv.iloc[-1] <= 0:
        raise SourceError("VWAP: حجم صفري")
    vwap = (tp * df.v).cumsum() / cv
    var = ((df.c - vwap) ** 2 * df.v).cumsum() / cv
    std = float(np.sqrt(var.iloc[-1]))
    dist = float((df.c.iloc[-1] / vwap.iloc[-1] - 1) * 100)
    z = float((df.c.iloc[-1] - vwap.iloc[-1]) / std) if std > 0 else 0.0
    return dict(dist_pct=dist, z_score=z)


def get_microstructure(now):
    """VPOC وCVD تقديري من شموع SPY الدقيقة (Alpaca IEX)."""
    df = spy_bars_today(now)
    tot = float(df.v.sum())
    if tot <= 0:
        raise SourceError("Micro: حجم صفري")
    df["bin"] = (df.c * 2).round() / 2
    vpoc = float(df.groupby("bin").v.sum().idxmax())
    rng = (df.h - df.l).replace(0, np.nan)
    pos = ((df.c - df.l) / rng).fillna(0.5)
    signed = np.where(pos > 0.66, df.v, np.where(pos < 0.33, -df.v, 0.0))
    cvd = float(signed.sum())
    return dict(vpoc=vpoc, cvd=cvd, cvd_ratio=cvd / tot, last=float(df.c.iloc[-1]))


def fred_last(series):
    if not FRED_KEY:
        raise SourceError("FRED_API_KEY غير مضبوط")
    r = requests.get("https://api.stlouisfed.org/fred/series/observations",
                     params=dict(series_id=series, api_key=FRED_KEY, file_type="json",
                                 sort_order="desc", limit=6), timeout=15)
    obs = [o for o in r.json().get("observations", []) if o.get("value") not in (".", "", None)]
    if len(obs) < 2:
        raise SourceError(f"FRED {series}: بيانات غير كافية")
    return float(obs[0]["value"]), float(obs[1]["value"])


def get_macro(now):
    out = {}
    try:
        y10, y10p = fred_last("DGS10")
        out.update(y10=y10, y10_bps=(y10 - y10p) * 100)
    except Exception:
        pass
    try:
        dx = yf_hist("DX-Y.NYB", "10d", "1d")
        out.update(dxy=float(dx.Close.iloc[-1]),
                   dxy_chg=(float(dx.Close.iloc[-1] / dx.Close.iloc[-2]) - 1) * 100)
    except Exception:
        pass
    if not out:
        raise SourceError("لا توجد بيانات ماكرو")
    return out


def get_news(now):
    r = requests.get(f"{DATA_URL}/v1beta1/news", headers=ah(), params={"symbols": "SPY", "limit": 3}, timeout=15)
    if r.status_code != 200:
        raise SourceError(f"News HTTP {r.status_code}")
    return [n.get("headline", "")[:110] for n in r.json().get("news", [])]


def get_leaders(now):
    syms = list(LEADER_W)
    try:
        raw = yf.download(syms, period="5d", interval="5m", prepost=True, group_by="ticker",
                          progress=False, auto_adjust=False, threads=True)
    except Exception as e:
        raise SourceError(f"القياديات: {str(e)[:80]}")
    if raw is None or raw.empty:
        raise SourceError("القياديات: لا بيانات")
    today = now.date()
    items = []
    for s in syms:
        try:
            sub = raw[s].dropna(subset=["Close"]).copy()
        except KeyError:
            continue
        if sub.empty:
            continue
        idx = sub.index
        if idx.tz is None:
            idx = idx.tz_localize("UTC")
        sub.index = idx.tz_convert(NY)
        if age_min(sub) > 90:
            continue
        days = sorted({dd for dd in sub.index.date if dd < today})
        if not days:
            continue
        pc = session_close(sub, days[-1])
        if not pc:
            continue
        items.append(dict(sym=s, ret=(float(sub.Close.iloc[-1]) / pc - 1) * 100, w=LEADER_W[s]))
    if len(items) < 5:
        raise SourceError("القياديات: بيانات غير كافية")
    tw = sum(i["w"] for i in items)
    return dict(items=items,
                wavg=sum(i["ret"] * i["w"] for i in items) / tw,
                wup=sum(i["w"] for i in items if i["ret"] > 0) / tw,
                top=sorted(items, key=lambda i: -i["w"])[:5])


def get_nasdaq(now):
    today = now.date()
    dd = daily_completed("^NDX", today)
    prev = float(dd.Close.iloc[-1])
    last = prev
    live = is_rth(now)
    if live:
        df = yf_hist("^NDX", "5d", "5m")
        if age_min(df) > MAX_STALE_MIN:
            raise SourceError("NDX متأخر")
        last = float(df.Close.iloc[-1])
    return dict(last=last, prev=prev, ret_pct=(last / prev - 1) * 100, live=live)


def get_nyse_breadth(now):
    df = yf_hist("^ADD", "5d", "5m")
    if age_min(df) > MAX_STALE_MIN:
        raise SourceError("ADD متأخر")
    v = float(df.Close.iloc[-1])
    return dict(add=v, score=math.tanh(v / 1500.0))


def get_credit_spread(now):
    hyg = yf_hist("HYG", "5d", "5m")
    tlt = yf_hist("TLT", "5d", "5m")
    ratio = (hyg.Close / tlt.Close).dropna()
    if len(ratio) < 8:
        raise SourceError("Credit: بيانات غير كافية")
    roc = float((ratio.iloc[-1] / ratio.iloc[-6] - 1) * 100)
    return dict(ratio=float(ratio.iloc[-1]), roc=roc)


SOURCES = [
    ("SPX", get_spx), ("ES", lambda n: get_future("ES=F", n)), ("NQ", lambda n: get_future("NQ=F", n)),
    ("VIX", get_vix), ("CHAIN", get_chain), ("LEADERS", get_leaders), ("NASDAQ", get_nasdaq),
    ("BREADTH", get_breadth), ("VWAP", get_spy_vwap), ("MACRO", get_macro), ("NEWS", get_news),
    ("NYSE_BREADTH", get_nyse_breadth), ("MICROSTRUCTURE", get_microstructure), ("CREDIT", get_credit_spread),
]


# ============================ طبقة التحقق ============================
def collect(now):
    d, q = {}, {}
    for name, fn in SOURCES:
        try:
            d[name] = fn(now)
            q[name] = (True, "")
        except Exception as e:
            d[name] = None
            q[name] = (False, str(e)[:120])
    if q["CHAIN"][0] and q["SPX"][0]:
        ref = d["SPX"]["spot"] if is_rth(now) else d["SPX"]["prev_close"]
        if abs(d["CHAIN"]["spot"] / ref - 1) > 0.008:
            q["CHAIN"] = (False, f"Cboe {d['CHAIN']['spot']:.0f} لا يطابق SPX {ref:.0f} (بيانات قديمة؟)")
            d["CHAIN"] = None
    return d, q


# ============================ الدمج والتوقع ============================
def lab(v, thr, up, down, flat):
    return up if v > thr else (down if v < -thr else flat)


def build_forecast(d, now):
    spx, es, vix, ch = d["SPX"], d["ES"], d["VIX"], d["CHAIN"]
    nq, br, vw, mc = d.get("NQ"), d.get("BREADTH"), d.get("VWAP"), d.get("MACRO")
    ld, nd, nyse = d.get("LEADERS"), d.get("NASDAQ"), d.get("NYSE_BREADTH")
    micro, credit = d.get("MICROSTRUCTURE"), d.get("CREDIT")
    gex = ch["gex"]
    rth = is_rth(now)
    m_now = now.hour * 60 + now.minute
    anchor = spx["spot"] if rth else spx["prev_close"] * es["last"] / es["prev_close"]
    warn = []

    # --- العقود الآجلة
    f, w = [math.tanh(es["ret_pct"] / 0.5)], [0.7]
    if nq:
        f.append(math.tanh(nq["ret_pct"] / 0.7))
        w.append(0.3)
    fut_score = float(np.average(f, weights=w))

    # --- هيكل VIX
    vix_score = -math.tanh(vix["chg_pct"] / 6.0)
    vratio = None
    if vix.get("vix9d"):
        vratio = vix["vix9d"] / vix["vix"]
        if vratio > 1.05:
            vix_score -= 0.4
        elif vratio < 0.88:
            vix_score += 0.2
    vix_score = clip(vix_score)

    # --- الجاما: جدران حية + جدران الصباح المثبتة
    g = 0.0
    cw, pw = gex["call_wall"], gex["put_wall"]
    if cw and 0 <= (cw - anchor) / anchor < 0.0015:
        g -= 0.4
    if pw and 0 <= (anchor - pw) / anchor < 0.0015:
        g += 0.4
    walls0 = None
    raw = kv_get(f"walls:{now.date()}")
    if raw:
        try:
            walls0 = json.loads(raw)
        except ValueError:
            walls0 = None
    if walls0:
        if walls0.get("c") and anchor > walls0["c"] + 2:
            g += 0.3
            warn.append("🔺 السعر فوق جدار الكول المثبت صباحاً (احتمال تسارع صعودي)")
        if walls0.get("p") and anchor < walls0["p"] - 2:
            g -= 0.3
            warn.append("🔻 السعر تحت جدار البوت المثبت صباحاً (احتمال تسارع هبوطي)")
    g = clip(g)

    # --- البنية الدقيقة (SPY)
    m_score = 0.0
    if micro:
        m_score = clip(0.4 * (1 if micro["last"] > micro["vpoc"] else -1) + 0.6 * math.tanh(micro["cvd_ratio"] * 4))

    # --- الائتمان HYG/TLT
    c_score = 0.0
    if credit:
        c_score = math.tanh(credit["roc"] / 0.1) * 0.5
        if credit["roc"] < -0.1 and es["ret_pct"] > 0.1:
            c_score -= 0.8
            warn.append("⚠️ تباين: الائتمان يضعف (HYG/TLT يتراجع) بينما الأسهم تصعد")
        c_score = clip(c_score)

    # --- الأوزان الديناميكية
    vix_val = vix["vix"]
    if vix_val < 15:
        w_tech, w_gex, w_macro, w_micro = 0.15, 0.10, 0.05, 0.12
    elif vix_val > 20:
        w_tech, w_gex, w_macro, w_micro = 0.05, 0.25, 0.15, 0.08
    else:
        w_tech, w_gex, w_macro, w_micro = 0.10, 0.15, 0.10, 0.10
    w_gex *= (1 + (gex.get("zero_dte_share") or 0.0))
    if rth:
        if 630 <= m_now < 690:
            w_macro *= 1.5
            warn.append("⏰ فترة إغلاق أوروبا (احتمال انعكاس الاتجاه الصباحي)")
        elif 900 <= m_now < 960:
            w_tech *= 0.3
            w_gex *= 1.5
            warn.append("⏰ الساعة الأخيرة: تحوط 0DTE يسيطر وقد يزيد التذبذب")
    w_gex = min(w_gex, 0.30)

    # --- الفني
    cl = spx["closes"] if rth and spx["closes"] is not None else es["closes"]
    e5 = cl.ewm(span=5, adjust=False).mean().iloc[-1]
    e13 = cl.ewm(span=13, adjust=False).mean().iloc[-1]
    tech = math.tanh((e5 - e13) / cl.iloc[-1] * 1500)
    if vw:
        tech = 0.6 * tech + 0.4 * math.tanh(vw["z_score"] / 2.0)

    # --- الاتساع
    br_score = None
    if nyse and br and br.get("sectors_up") is not None:
        br_score = 0.6 * nyse["score"] + 0.4 * math.tanh((br["sectors_up"] - 0.5) * 4)
    elif nyse:
        br_score = nyse["score"]
    elif br and br.get("sectors_up") is not None:
        br_score = math.tanh((br["sectors_up"] - 0.5) * 4)

    # --- القياديات
    ld_score = None
    if ld:
        ld_score = 0.5 * math.tanh(ld["wavg"] / 0.5) + 0.5 * math.tanh((ld["wup"] - 0.5) * 4)

    # --- الماكرو
    parts = []
    if mc:
        if "dxy_chg" in mc:
            parts.append(-math.tanh(mc["dxy_chg"] / 0.4))
        if "y10_bps" in mc:
            parts.append(-math.tanh(mc["y10_bps"] / 6.0))
    if credit:
        parts.append(c_score)

    comps = {"futures": (fut_score, BASE_W["futures"]), "vix": (vix_score, BASE_W["vix"]),
             "gex": (g, w_gex), "technical": (tech, w_tech)}
    if ld_score is not None:
        comps["leaders"] = (ld_score, BASE_W["leaders"])
    if br_score is not None:
        comps["breadth"] = (br_score, BASE_W["breadth"])
    if parts:
        comps["macro"] = (float(np.mean(parts)), w_macro)
    if micro:
        comps["microstructure"] = (m_score, w_micro)

    rel, rel_src = None, ""
    if nd and nd["live"]:
        spx_ret = (spx["spot"] / spx["prev_close"] - 1) * 100
        rel, rel_src = nd["ret_pct"] - spx_ret, "NDX"
        comps["nasdaq"] = (math.tanh(nd["ret_pct"] / 0.7), BASE_W["nasdaq"])
    elif nq:
        rel, rel_src = nq["ret_pct"] - es["ret_pct"], "NQ"

    tw = sum(wt for _, wt in comps.values())
    S = heur_S = sum(sc * wt for sc, wt in comps.values()) / tw
    quality = sum(BASE_W[k] for k in comps if k in BASE_W) / sum(BASE_W.values())

    # --- نموذج ML (Shadow افتراضياً)
    ml_features = {"time_of_day": m_now, "vix_score": vix_score, "gex_score": g, "tech_score": tech,
                   "breadth_score": br_score if br_score is not None else 0.0,
                   "micro_score": m_score, "credit_score": c_score, "futures_score": fut_score,
                   "leaders_score": ld_score if ld_score is not None else 0.0}
    ml_line = None
    if LGBM_AVAILABLE and os.path.exists(MODEL_PATH):
        try:
            bst = lgb.Booster(model_file=MODEL_PATH)
            ml_prob = float(bst.predict(np.array([[ml_features[k] for k in ML_FEATURES]]))[0])
            if ML_WEIGHT > 0:
                S = (1 - ML_WEIGHT) * heur_S + ML_WEIGHT * (ml_prob - 0.5) * 2.0
                ml_line = f"🤖 LightGBM (مدمج بوزن {ML_WEIGHT:.2f}): صعود {ml_prob*100:.0f}%"
            else:
                ml_line = f"🤖 LightGBM (Shadow): صعود {ml_prob*100:.0f}%"
        except Exception as e:
            log.warning("ML predict: %s", e)

    sig_n = [sc for sc, _ in comps.values() if abs(sc) > 0.1]
    agreement = (sum(1 for sc in sig_n if np.sign(sc) == np.sign(S)) / len(sig_n)) if sig_n else 0.0
    conflicts = sum(1 for sc, _ in comps.values() if abs(sc) > 0.15 and np.sign(sc) != np.sign(S))
    near_wall = bool((cw and S > 0 and 0 <= (cw - anchor) / anchor < 0.002) or
                     (pw and S < 0 and 0 <= (anchor - pw) / anchor < 0.002))
    divergence = False
    if micro and rth:
        up_day = spx["spot"] > spx["prev_close"]
        if (up_day and micro["cvd_ratio"] < -0.05) or ((not up_day) and micro["cvd_ratio"] > 0.05):
            divergence = True
            warn.append("⚠️ تباين بين السعر والسيولة (CVD عكس الاتجاه)")
    rev_score = conflicts + (1 if near_wall else 0) + (2 if divergence else 0)
    reversal = "مرتفع" if rev_score >= 3 else ("متوسط" if rev_score >= 1 else "منخفض")

    # --- خطر الهبوط الحاد
    dr, dr_why = 0, []

    def add(p, why):
        nonlocal dr
        dr += p
        dr_why.append(why)

    if gex["regime"] == "NEGATIVE":
        add(15, "جاما سلبية")
    if gex.get("flip") and anchor < gex["flip"]:
        add(10, "تحت نقطة Flip")
    if vix["chg_pct"] > 10:
        add(20, "VIX يقفز")
    elif vix["chg_pct"] > 5:
        add(12, "VIX يرتفع")
    if vratio and vratio > 1.05:
        add(15, "VIX9D فوق VIX (ذعر قريب)")
    elif vratio and vratio > 1.0:
        add(8, "هيكل VIX مقلوب")
    if es["ret_pct"] < -1.0:
        add(15, "العقود هابطة بقوة")
    elif es["ret_pct"] < -0.5:
        add(10, "العقود هابطة")
    if walls0 and walls0.get("p") and anchor < walls0["p"] - 2:
        add(15, "كسر جدار البوت")
    if micro and micro["cvd_ratio"] < -0.10:
        add(10, "سيولة بيعية")
    if credit and credit["roc"] < -0.15:
        add(10, "ضعف الائتمان")
    if br_score is not None and br_score < -0.5:
        add(10, "اتساع السوق ضعيف")
    if ld and ld["wavg"] < -0.7:
        add(10, "الشركات الكبرى هابطة")
    dr = min(100, dr)
    dr_level = "منخفض" if dr < 30 else ("متوسط" if dr < 55 else ("مرتفع" if dr < 75 else "مرتفع جداً"))

    neutral = 0.4 * math.exp(-(S / 0.2) ** 2)
    sg = 1 / (1 + math.exp(-4 * S))
    probs = dict(up=(1 - neutral) * sg, down=(1 - neutral) * (1 - sg), neutral=neutral)
    label = max(probs, key=probs.get)

    iv = vix["vix"] if not vix["vix9d"] else 0.5 * vix["vix"] + 0.5 * vix["vix9d"]
    sig_day = 0.6 * (anchor * iv / 100 / math.sqrt(252)) + 0.4 * (spx["atr"] / 1.596)
    sig_day *= 0.9 if gex["regime"] == "POSITIVE" else 1.12
    frac = 1.0 if m_now < 570 else max(0.05, (960 - m_now) / 390.0)
    sig = sig_day * math.sqrt(frac)
    c = anchor + S * 0.15 * sig
    hi, lo = c + 0.798 * sig, c - 0.798 * sig
    hi_near, hi_far = c + 0.126 * sig, c + 1.645 * sig
    lo_near, lo_far = c - 0.126 * sig, c - 1.645 * sig
    if rth:
        dh, dl = spx["day_high"], spx["day_low"]
        hi, hi_near, hi_far = max(hi, dh), max(hi_near, dh), max(hi_far, dh)
        lo, lo_near, lo_far = min(lo, dl), min(lo_near, dl), min(lo_far, dl)

    # --- سطور القراءة
    L = list(warn)
    L.append(f"ES: {lab(es['ret_pct'], 0.15, 'داعم للصعود', 'داعم للهبوط', 'محايد')} ({es['ret_pct']:+.2f}%)")
    L.append(f"NQ: {lab(nq['ret_pct'], 0.2, 'داعم للصعود', 'داعم للهبوط', 'محايد')} ({nq['ret_pct']:+.2f}%)"
             if nq else "NQ: غير متاح")
    if rel is not None:
        tag = lab(rel, 0.15, "قيادة تقنية", "تأخر تقنية", "متوافق مع SPX")
        base = f"NDX {nd['ret_pct']:+.2f}% | " if rel_src == "NDX" else ""
        L.append(f"ناسداك: {base}الفرق عن SPX {rel:+.2f}% ({tag})")
    v9 = f" | VIX9D {vix['vix9d']:.1f}" if vix["vix9d"] else ""
    vr = f" | نسبة 9D/30D {vratio:.2f}" if vratio else ""
    L.append(f"VIX: {vix['vix']:.1f} ({vix['chg_pct']:+.1f}%) "
             f"{lab(vix['chg_pct'], 3, 'ارتفاع التقلب', 'انخفاض التقلب', 'مستقر')}{v9}{vr}")
    L.append(f"GEX: جاما {'إيجابية' if gex['regime']=='POSITIVE' else 'سلبية'} | Flip {n0(gex['flip'])} | "
             f"مقاومة {n0(gex['call_wall'])} | دعم {n0(gex['put_wall'])}")
    zr = gex["zero_dte_cp_vol"]
    L.append(f"0DTE: {gex['zero_dte_share']*100:.0f}% من الجاما" + (f" | حجم Call/Put {zr:.2f}" if zr else ""))
    if ld:
        L.append(f"الشركات الكبرى (موزونة): {ld['wavg']:+.2f}% | {ld['wup']*100:.0f}% من الوزن صاعد")
        L.append("الأكبر وزناً: " + " | ".join(f"{i['sym']} {i['ret']:+.1f}%" for i in ld["top"]))
    else:
        L.append("الشركات الكبرى: غير متاح")
    if nyse:
        L.append(f"اتساع NYSE (ADD): {nyse['add']:.0f}")
    if br and br.get("sectors_up") is not None:
        L.append(f"Breadth (قطاعات): {br['sectors_up']*100:.0f}% صاعدة")
    elif not nyse:
        L.append("Breadth: " + ("غير مؤكد قبل الافتتاح" if not rth else "غير متاح"))
    if micro:
        L.append(f"السيولة (SPY): VPOC {micro['vpoc']:.1f} | CVD {micro['cvd_ratio']*100:+.1f}%")
    if vw:
        L.append(f"VWAP Z-Score: {vw['z_score']:+.2f}")
    if credit:
        L.append(f"الائتمان HYG/TLT: {credit['roc']:+.2f}% (30 دقيقة)")
    if mc:
        bits = []
        if "y10" in mc:
            bits.append(f"10Y {mc['y10']:.2f}% ({mc.get('y10_bps', 0):+.0f}bp)")
        if "dxy" in mc:
            bits.append(f"DXY {mc['dxy']:.1f} ({mc['dxy_chg']:+.2f}%)")
        if bits:
            L.append("Macro: " + " | ".join(bits))
    if ml_line:
        L.append(ml_line)
    if d.get("NEWS"):
        L.append("أخبار: " + d["NEWS"][0])

    levels = dict(r1=cw if cw and cw > anchor else None, r2=hi_far,
                  s1=pw if pw and pw < anchor else None, s2=lo_far)
    return dict(anchor=anchor, spot=spx["spot"], hi=hi, lo=lo, hi_near=hi_near, hi_far=hi_far,
                lo_near=lo_near, lo_far=lo_far, probs=probs, label=label, score=S, reversal=reversal,
                data_quality=quality, agreement=agreement,
                gex={k: gex[k] for k in ("spot", "flip", "call_wall", "put_wall", "net", "regime")},
                lines=L, levels=levels, comps={k: round(v[0], 3) for k, v in comps.items()},
                ml_id=str(uuid.uuid4()), ml_features=ml_features,
                drop_risk=dict(score=dr, level=dr_level, why=dr_why[:4]))


# ============================ التقارير ============================
def format_report(fc, title):
    p, lv, dk = fc["probs"], fc["levels"], fc["drop_risk"]
    names = {"up": "صاعد 🟢", "down": "هابط 🔴", "neutral": "عرضي ⚪"}
    now = now_ny()
    ksa = now.astimezone(ZoneInfo("Asia/Riyadh"))
    sep = "━━━━━━━━━━━━"
    out = [
        "💎 صَفاء التحليل", f"📊 {title}",
        f"🕘 {now:%H:%M} نيويورك | {ksa:%H:%M} السعودية", f"📅 {now:%Y-%m-%d}",
        sep, "🎯 التوقع", f"الاتجاه: {names[fc['label']]}",
        f"القمة المتوقعة: {n0(fc['hi'])}", f"القاع المتوقع: {n0(fc['lo'])}", "",
        "📏 النطاق المحتمل (80%)",
        f"القمة بين {n0(fc['hi_near'])} و {n0(fc['hi_far'])}",
        f"القاع بين {n0(fc['lo_far'])} و {n0(fc['lo_near'])}",
        sep, "📈 احتمالات الاتجاه (تقديرية)",
        f"🟢 صعود: {p['up']*100:.0f}%", f"🔴 هبوط: {p['down']*100:.0f}%", f"⚪ عرضي: {p['neutral']*100:.0f}%",
        f"⚠️ خطر الانعكاس: {fc['reversal']}",
        f"📉 خطر هبوط حاد: {dk['level']} ({dk['score']}/100)",
        ("   الأسباب: " + " | ".join(dk["why"])) if dk["why"] else "",
        sep, "🧭 قراءة الأسواق"] + ["• " + ln for ln in fc["lines"]] + [
        sep, "🎚️ مستويات المراقبة",
        f"🔴 مقاومة 1: {n0(lv['r1'])}", f"🔴 مقاومة 2: {n0(lv['r2'])}",
        f"🟢 دعم 1: {n0(lv['s1'])}", f"🟢 دعم 2: {n0(lv['s2'])}",
        sep, f"🔎 جودة البيانات {fc['data_quality']*100:.0f}% | اتفاق الإشارات {fc['agreement']*100:.0f}%",
        "ℹ️ بيانات مجانية متأخرة ~15 دقيقة", "تحليل احتمالي — لا يوجد ضمان للربح", SIGNATURE]
    return "\n".join(out)


def format_close_risk(fc):
    g = fc["gex"]
    return "\n".join(["🔔 مخاطر الإغلاق (15:45 نيويورك)", f"جاما: {'إيجابية' if g['regime']=='POSITIVE' else 'سلبية'}",
                      f"Flip {n0(g['flip'])} | مقاومة {n0(g['call_wall'])} | دعم {n0(g['put_wall'])}",
                      f"السعر {n0(fc['spot'])} | خطر الانعكاس: {fc['reversal']}",
                      "قرب الإغلاق وانتهاء 0DTE يزيدان تذبذب الحركة", SIGNATURE])


def eval_summary():
    with sqlite3.connect(DB_PATH) as c:
        rows = c.execute("""SELECT p.anchor,p.hi,p.lo,p.hi_near,p.hi_far,p.lo_near,p.lo_far,p.p_up,p.p_down,
                            r.high,r.low,r.close FROM predictions p JOIN realized r ON p.day=r.day
                            WHERE p.kind='premarket'""").fetchall()
    n = len(rows)
    if n == 0:
        return "لا توجد بيانات تقييم كافية بعد"
    mh = np.mean([abs(r[1] - r[9]) for r in rows])
    ml = np.mean([abs(r[2] - r[10]) for r in rows])
    da = np.mean([(r[11] > r[0]) == (r[7] > r[8]) for r in rows])
    ch = np.mean([r[3] <= r[9] <= r[4] for r in rows])
    cl = np.mean([r[6] <= r[10] <= r[5] for r in rows])
    note = " (عينة صغيرة — لا يُعتمد عليها)" if n < 20 else ""
    return (f"التقييم التراكمي ({n} يوم){note}\nخطأ القمة: {mh:.1f} نقطة | خطأ القاع: {ml:.1f} نقطة\n"
            f"دقة الاتجاه: {da*100:.0f}% | تغطية نطاق 80%: قمة {ch*100:.0f}% / قاع {cl*100:.0f}%")


# ============================ توقع الساعات ============================
WINDOWS = [(570, 630), (630, 690), (690, 750), (750, 810), (810, 870), (870, 930), (930, 960)]
RIYADH = ZoneInfo("Asia/Riyadh")
_PROFILE = {"day": None, "data": None}


def fetch_spy_1m(start, end):
    out, token = [], None
    for _ in range(12):
        params = {"timeframe": "1Min", "start": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
                  "end": end.strftime("%Y-%m-%dT%H:%M:%SZ"), "limit": 10000,
                  "feed": STOCK_FEED, "adjustment": "raw"}
        if token:
            params["page_token"] = token
        r = requests.get(f"{DATA_URL}/v2/stocks/SPY/bars", headers=ah(), params=params, timeout=30)
        if r.status_code != 200:
            raise SourceError(f"Alpaca bars HTTP {r.status_code}")
        j = r.json()
        out += j.get("bars") or []
        token = j.get("next_page_token")
        if not token:
            break
    if not out:
        raise SourceError("SPY: لا شموع")
    df = pd.DataFrame(out)
    df["ts"] = pd.to_datetime(df["t"], utc=True).dt.tz_convert(NY)
    for col in ("o", "h", "l", "c", "v"):
        df[col] = df[col].astype(float)
    return df.set_index("ts").sort_index()


def build_hour_profile(now):
    if _PROFILE["day"] == now.date() and _PROFILE["data"]:
        return _PROFILE["data"]
    end = datetime.now(timezone.utc)
    df = fetch_spy_1m(end - timedelta(days=40), end)
    mm = df.index.hour * 60 + df.index.minute
    df = df[(mm >= 570) & (mm < 960)]
    ch = {i: [] for i in range(len(WINDOWS))}
    sd = {i: [] for i in range(len(WINDOWS))}
    ndays = 0
    for day, g in df.groupby(df.index.date):
        if day >= now.date() or len(g) < 200:
            continue
        ndays += 1
        s = g["c"].resample("1min").last().ffill()
        first_open = float(g["o"].iloc[0])
        lr = np.log(s).diff().dropna()
        for i, (a, b) in enumerate(WINDOWS):
            t0 = datetime.combine(day, dtime(a // 60, a % 60), tzinfo=NY)
            t1 = datetime.combine(day, dtime(b // 60, b % 60), tzinfo=NY)
            lvl0 = first_open if a == 570 else s.asof(t0 - timedelta(minutes=1))
            lvl1 = s.asof(t1 - timedelta(minutes=1))
            if pd.isna(lvl0) or pd.isna(lvl1):
                continue
            ch[i].append(math.log(lvl1 / lvl0))
            seg = lr[(lr.index >= t0) & (lr.index < t1)]
            if len(seg) > 5:
                sd[i].append(float(seg.std()))
    sig, sd1 = [], []
    for i in range(len(WINDOWS)):
        a = np.array(ch[i])
        if len(a) < 8:
            raise SourceError("توقع الساعات: أيام تاريخية غير كافية")
        mad = 1.4826 * float(np.median(np.abs(a - np.median(a))))
        sig.append(0.5 * float(a.std(ddof=1)) + 0.5 * mad)
        sd1.append(float(np.mean(sd[i])) if sd[i] else 0.0003)
    data = dict(sig=sig, sd1=sd1, days=ndays)
    _PROFILE["day"], _PROFILE["data"] = now.date(), data
    log.info("ملف الساعات جاهز | أيام=%s", ndays)
    return data


def vol_multiplier(prof, spy):
    try:
        s = spy["c"].resample("1min").last().ffill()
        lr = np.log(s).diff().dropna().tail(60)
        if len(lr) < 20:
            return 1.0
        exp = []
        for ts in lr.index:
            m = ts.hour * 60 + ts.minute
            for i, (a, b) in enumerate(WINDOWS):
                if a <= m < b:
                    exp.append(prof["sd1"][i])
                    break
        if not exp:
            return 1.0
        ratio = float(lr.std()) / max(float(np.mean(exp)), 1e-6)
        return clip(0.6 * ratio + 0.4, 0.7, 1.6)
    except Exception:
        return 1.0


def hour_forecast(fc, d, now):
    prof = build_hour_profile(now)
    anchor, mult, src = fc["anchor"], 1.0, "SPX"
    if is_rth(now):
        try:
            st = datetime.combine(now.date(), dtime(9, 30), tzinfo=NY).astimezone(timezone.utc)
            spy = fetch_spy_1m(st, datetime.now(timezone.utc))
            if len(spy) >= 10 and (now - spy.index[-1].to_pydatetime()).total_seconds() < 240:
                mult = vol_multiplier(prof, spy)
                cl = d["SPX"]["closes"]
                spy_c = spy["c"].resample("1min").last().ffill()
                ref = float(spy_c.asof(cl.index[-1] + timedelta(minutes=4)))
                if ref > 0:
                    live = float(spy_c.iloc[-1]) * float(cl.iloc[-1]) / ref
                    if abs(live / anchor - 1) < 0.006:
                        anchor, src = live, "SPY"
        except Exception as e:
            log.warning("SPY اللحظي: %s", e)
    m_now = now.hour * 60 + now.minute
    cal = float(kv_get("hour_scale") or 1.0)
    gf = 0.92 if fc["gex"]["regime"] == "POSITIVE" else 1.10
    S = fc["score"]
    rows, var, lead = [], 0.0, 0
    for i, (a, b) in enumerate(WINDOWS):
        if b <= m_now:
            continue
        frac = 1.0 if m_now <= a else (b - m_now) / (b - a)
        var += (prof["sig"][i] * mult) ** 2 * frac
        sg = math.sqrt(var) * anchor * gf * cal
        c = anchor + S * 0.12 * sg
        rows.append(dict(a=a, b=b, lead=lead, center=c, lo50=c - 0.674 * sg, hi50=c + 0.674 * sg,
                         lo80=c - 1.282 * sg, hi80=c + 1.282 * sg))
        lead += 1
    return dict(rows=rows, anchor=anchor, src=src, mult=mult, cal=cal)


def save_hourly(now, hf):
    with sqlite3.connect(DB_PATH) as c:
        c.executemany("""INSERT INTO hourly_pred (day,made_at,lead,w_start,w_end,center,lo50,hi50,lo80,hi80)
                         VALUES (?,?,?,?,?,?,?,?,?,?)""",
                      [(now.date().isoformat(), now.isoformat(timespec="seconds"), r["lead"], r["a"], r["b"],
                        r["center"], r["lo50"], r["hi50"], r["lo80"], r["hi80"]) for r in hf["rows"]])


def fmt_hm(day, minute):
    t = datetime.combine(day, dtime(minute // 60, minute % 60), tzinfo=NY)
    return t.astimezone(RIYADH).strftime("%H:%M")


def hourly_stats():
    with sqlite3.connect(DB_PATH) as c:
        r = c.execute("""SELECT COUNT(*), AVG(in50), AVG(in80), AVG(hi50-lo50) FROM hourly_pred
                         WHERE actual IS NOT NULL AND lead=0""").fetchone()
    if not r or not r[0]:
        return "معايرة الساعات: لا بيانات بعد"
    return (f"معايرة الساعات (n={r[0]}): داخل الأرجح {r[1]*100:.0f}% | داخل الأوسع {r[2]*100:.0f}% | "
            f"متوسط عرض الأرجح {r[3]:.0f} نقطة")


def format_hourly(hf, fc, now):
    S = fc["score"]
    arrow = "↗" if S > 0.1 else ("↘" if S < -0.1 else "➡")
    sep = "━━━━━━━━━━━━"
    out = ["🕐 توقع الساعات (بتوقيت السعودية)",
           f"المؤشر الآن ≈ {hf['anchor']:,.0f} ({hf['src']}) | الميل {arrow}", sep,
           "الساعة | الأرجح (50%) | الأوسع (80%)"]
    for r in hf["rows"]:
        out.append(f"{fmt_hm(now.date(), r['a'])}–{fmt_hm(now.date(), r['b'])} | "
                   f"{n0(r['lo50'])}–{n0(r['hi50'])} | {n0(r['lo80'])}–{n0(r['hi80'])}")
    out += [sep, "الأرجح: يقع فيه السعر عند نهاية الساعة في نصف الحالات تقريباً.",
            "الساعات البعيدة أوسع لأن عدم اليقين يتراكم.", hourly_stats(),
            "تحليل احتمالي — لا يوجد ضمان للربح", SIGNATURE]
    return "\n".join(out)


def send_hourly(fc, d, now):
    try:
        hf = hour_forecast(fc, d, now)
        if not hf["rows"]:
            return
        save_hourly(now, hf)
        tg(format_hourly(hf, fc, now))
    except Exception as e:
        log.warning("توقع الساعات: %s", e)


def eval_hourly(day, df):
    with sqlite3.connect(DB_PATH) as c:
        rows = c.execute("""SELECT rowid,w_end,lo50,hi50,lo80,hi80 FROM hourly_pred
                            WHERE day=? AND actual IS NULL""", (day.isoformat(),)).fetchall()
        for rid, wend, lo50, hi50, lo80, hi80 in rows:
            t1 = datetime.combine(day, dtime(wend // 60, wend % 60), tzinfo=NY)
            sub = df[(df.index < t1) & (df.index.date == day)]
            if sub.empty or sub.index[-1] < t1 - timedelta(minutes=10):
                continue
            act = float(sub.Close.iloc[-1])
            c.execute("UPDATE hourly_pred SET actual=?, in50=?, in80=? WHERE rowid=?",
                      (act, int(lo50 <= act <= hi50), int(lo80 <= act <= hi80), rid))


def calibrate_hourly():
    with sqlite3.connect(DB_PATH) as c:
        n, h = c.execute("""SELECT COUNT(*), AVG(in50) FROM (SELECT in50 FROM hourly_pred
                            WHERE actual IS NOT NULL AND lead=0 ORDER BY rowid DESC LIMIT 21)""").fetchone()
    if not n or n < 21 or h is None:
        return
    cur = float(kv_get("hour_scale") or 1.0)
    kv_set("hour_scale", round(clip(cur * (1 + (0.5 - h) * 0.3), 0.7, 1.5), 3))


# ============================ الجدولة والتشغيل ============================
def halt(bad):
    last = kv_get("last_halt")
    if last and (now_ny() - datetime.fromisoformat(last)).total_seconds() < 1800:
        return
    kv_set("last_halt", now_ny().isoformat())
    tg("⛔ توقف إصدار التوقعات (Fail Closed)\n" + "\n".join(bad) +
       "\nلن تصدر توقعات حتى تعود المصادر الأساسية.\n" + SIGNATURE)


def run_forecast(kind):
    now = now_ny()
    d, q = collect(now)
    save_quality(now, q)
    bad = [f"{n}: {q[n][1]}" for n in CRITICAL if not q[n][0]]
    if bad:
        log.warning("توقف: %s", bad)
        halt(bad)
        return None
    if not kv_get("weights_v2_start"):
        kv_set("weights_v2_start", now.date().isoformat())
    fc = build_forecast(d, now)
    save_prediction(now, kind, fc)
    key = f"dropmax:{now.date()}"
    if fc["drop_risk"]["score"] > float(kv_get(key) or 0):
        kv_set(key, fc["drop_risk"]["score"])
    if kind == "premarket":
        g0 = fc["gex"]
        kv_set(f"walls:{now.date()}", json.dumps({"c": g0["call_wall"], "p": g0["put_wall"]}))
    return fc, d, now


def job_pre_update():
    return bool(run_forecast("premarket_update"))


def job_pre_final():
    r = run_forecast("premarket")
    if not r:
        return False
    fc, d, now = r
    tg(format_report(fc, "تقرير توقعات S&P 500 — قبل الافتتاح"))
    send_hourly(fc, d, now)
    save_snapshot(now, d["CHAIN"])
    return True


def job_rth(now):
    r = run_forecast("intraday")
    if not r:
        return
    fc, d, now = r
    if fc["drop_risk"]["score"] >= 70 and not kv_get(f"dropalert:{now.date()}"):
        if tg(f"🚨 خطر هبوط حاد مرتفع ({fc['drop_risk']['score']}/100)\n" +
              " | ".join(fc["drop_risk"]["why"]) +
              f"\nأقرب دعم: {n0(fc['levels']['s1'])} ثم {n0(fc['levels']['s2'])}\n" + SIGNATURE):
            kv_set(f"dropalert:{now.date()}", "1")
    if now.minute < 5 or now.minute in range(30, 35):
        save_snapshot(now, d["CHAIN"])
    last_label, last_send = kv_get("last_label"), kv_get("last_send")
    due = (not last_send) or (now - datetime.fromisoformat(last_send)).total_seconds() >= UPDATE_MINUTES * 60
    if fc["label"] != last_label or due:
        if tg(format_report(fc, "تحديث توقعات S&P 500 — أثناء الجلسة")):
            kv_set("last_label", fc["label"])
            kv_set("last_send", now.isoformat())


def job_close_risk():
    r = run_forecast("close_risk")
    if not r:
        return False
    tg(format_close_risk(r[0]))
    return True


def job_hourly():
    r = run_forecast("hourly")
    if not r:
        return False
    fc, d, now = r
    send_hourly(fc, d, now)
    return True


def job_eval():
    now = now_ny()
    try:
        df = yf_hist("^GSPC", "5d", "5m")
    except SourceError:
        return False
    m = df.index.hour * 60 + df.index.minute
    t = df[(df.index.date == now.date()) & (m >= 570) & (m < 960)]
    if t.empty:
        return False
    hi, lo = float(t.High.max()), float(t.Low.min())
    cl, op = float(t.Close.iloc[-1]), float(t.Open.iloc[0])
    target = 1 if cl > op else 0
    drop_happened = 1 if (op - lo) / op >= 0.01 else 0
    mx = float(kv_get(f"dropmax:{now.date()}") or 0)
    day = now.date().isoformat()
    with sqlite3.connect(DB_PATH) as c:
        c.execute("INSERT OR REPLACE INTO realized VALUES (?,?,?,?)", (day, hi, lo, cl))
        c.execute("UPDATE ml_features SET actual_target = ? WHERE day = ?", (target, day))
        c.execute("INSERT OR REPLACE INTO drop_log VALUES (?,?,?,?)", (day, mx, "", drop_happened))
        row = c.execute("SELECT hi,lo FROM predictions WHERE day=? AND kind='premarket'", (day,)).fetchone()
    try:
        eval_hourly(now.date(), df)
        calibrate_hourly()
    except Exception as e:
        log.warning("تقييم الساعات: %s", e)
    today_line = (f"توقع القمة {n0(row[0])} مقابل الفعلي {n0(hi)} | توقع القاع {n0(row[1])} مقابل الفعلي {n0(lo)}"
                  if row else "لا يوجد توقع قبل الافتتاح اليوم")
    tg("\n".join(["📊 حفظ النتائج والتقييم", f"إغلاق SPX: {cl:,.2f}", today_line,
                  eval_summary(), hourly_stats(), SIGNATURE]))
    return True


def once(key, now, start, window, fn):
    m = now.hour * 60 + now.minute
    k = f"{now.date()}:{key}"
    if start <= m < start + window and not kv_get(k):
        if fn():
            kv_set(k, "1")


def tick():
    now = now_ny()
    if now.weekday() >= 5:
        return
    m = now.hour * 60 + now.minute
    once("pre_update", now, 9 * 60, 15, job_pre_update)
    once("pre_final", now, 9 * 60 + 20, 12, job_pre_final)
    if 575 <= m < 955:
        k = f"{now.date()}:rth:{m // 5}"
        if not kv_get(k):
            kv_set(k, "1")
            job_rth(now)
    once("close_risk", now, 15 * 60 + 45, 10, job_close_risk)
    for s in (630, 690, 750, 810, 870, 930):
        once(f"hr{s}", now, s + 1, 6, job_hourly)
    once("eval", now, 16 * 60 + 15, 90, job_eval)


def dump_1m_once():
    if kv_get("dump_1m_done"):
        return
    stored = []
    for sym, name in [("^GSPC", "SPX"), ("ES=F", "ES"), ("NQ=F", "NQ"), ("^VIX", "VIX")]:
        try:
            df = yf.Ticker(sym).history(period="7d", interval="1m", prepost=True, auto_adjust=False)
            if df is None or df.empty:
                continue
            rows = []
            for ts, row in df.iterrows():
                t = ts.tz_convert(NY).isoformat(timespec="seconds") if ts.tzinfo else str(ts)
                rows.append((name, t, float(row["Open"]), float(row["High"]), float(row["Low"]),
                             float(row["Close"]), float(row.get("Volume", 0) or 0)))
            with sqlite3.connect(DB_PATH) as c:
                c.executemany("""INSERT OR REPLACE INTO bars_1m
                                 (symbol, ts, open, high, low, close, volume) VALUES (?,?,?,?,?,?,?)""", rows)
            stored.append(f"{name}:{len(rows)}")
        except Exception as e:
            log.warning("تعذر تجهيز %s: %s", sym, e)
    if stored:
        kv_set("dump_1m_done", "1")
        tg("📦 تم تخزين بيانات الدقيقة:\n" + "\n".join(stored))


def train_model():
    """تدريب يدوي: python safaa_spx.py train — يرفض إن كانت البيانات قليلة."""
    if not LGBM_AVAILABLE:
        print("lightgbm غير مثبت.")
        return
    with sqlite3.connect(DB_PATH) as c:
        df = pd.read_sql_query("SELECT * FROM ml_features WHERE actual_target >= 0", c)
    days = sorted(df.day.unique())
    if len(days) < 60 or len(df) < 500:
        print(f"البيانات غير كافية: {len(days)} يوم، {len(df)} صف (المطلوب 60 يوماً و500 صف على الأقل).")
        return
    cut = days[int(len(days) * 0.8)]
    tr, va = df[df.day < cut], df[df.day >= cut]
    params = dict(objective="binary", num_leaves=7, learning_rate=0.05, min_data_in_leaf=50,
                  feature_fraction=0.8, verbose=-1)
    dt = lgb.Dataset(tr[ML_FEATURES], tr.actual_target)
    dv = lgb.Dataset(va[ML_FEATURES], va.actual_target)
    bst = lgb.train(params, dt, 300, valid_sets=[dv], callbacks=[lgb.early_stopping(30, verbose=False)])
    acc = float(((bst.predict(va[ML_FEATURES]) > 0.5) == va.actual_target).mean())
    base = float(max(va.actual_target.mean(), 1 - va.actual_target.mean()))
    print(f"دقة التحقق {acc:.3f} مقابل خط الأساس {base:.3f}")
    if acc > base + 0.02:
        bst.save_model(MODEL_PATH)
        print("تم حفظ النموذج:", MODEL_PATH)
    else:
        print("النموذج لا يتفوق على خط الأساس — لم يُحفظ.")


def main():
    init_db()
    if len(sys.argv) > 1 and sys.argv[1] == "train":
        train_model()
        return
    dump_1m_once()
    log.info("تشغيل صَفاء التحليل SPX v2.2")
    tg("🚀 تم تشغيل صَفاء التحليل SPX v2.2\nتوقع الساعات + خطر الهبوط الحاد\n"
       "لا تنفيذ تلقائي — تحليل احتمالي فقط\n" + SIGNATURE)
    while True:
        try:
            tick()
        except Exception as e:
            log.exception("خطأ تشغيلي: %s", e)
        time.sleep(20)


if __name__ == "__main__":
    main()
