# -*- coding: utf-8 -*-
"""
صَفاء التحليل SPX — نسخة المصادر المجانية v1
تصميم أبو سحاب
تحليل احتمالي فقط — لا تنفيذ تلقائي — Fail Closed
"""
import os, re, json, gzip, math, time, sqlite3, logging
from datetime import datetime, timedelta, timezone, time as dtime
from zoneinfo import ZoneInfo

import requests
import numpy as np
import pandas as pd
import yfinance as yf

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
MAX_STALE_MIN = int(env("MAX_STALE_MIN", "35"))      # البيانات المجانية متأخرة ~15 دقيقة
UPDATE_MINUTES = int(env("UPDATE_MINUTES", "60"))
_DEF_W = "NVDA:7.5,AAPL:6.5,MSFT:6.5,AMZN:4,GOOGL:4,META:3,AVGO:2.5,TSLA:2,JPM:1.5,LLY:1.3"
LEADER_W = {k.strip().upper(): float(v) for k, v in
            (p.split(":") for p in env("LEADER_WEIGHTS", _DEF_W).split(",") if ":" in p)}
LEADERS = list(LEADER_W.keys())
SECTORS = env("SECTORS", "XLK,XLF,XLE,XLV,XLY,XLP,XLI,XLU").split(",")
SIGNATURE = "⚡ تصميم أبو سحاب"
CRITICAL = ["SPX", "ES", "VIX", "CHAIN"]   # تعطل أي منها = توقف الإصدار

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
            data_quality REAL, agreement REAL, payload TEXT)""")
        c.execute("CREATE TABLE IF NOT EXISTS realized (day TEXT PRIMARY KEY, high REAL, low REAL, close REAL)")
        c.execute("CREATE TABLE IF NOT EXISTS data_quality (ts TEXT, source TEXT, ok INTEGER, detail TEXT)")
        c.execute("""CREATE TABLE IF NOT EXISTS gex_levels (
            ts TEXT, spot REAL, flip REAL, call_wall REAL, put_wall REAL, net_gex REAL, regime TEXT)""")
        c.execute("CREATE TABLE IF NOT EXISTS options_snapshots (ts TEXT, blob BLOB)")


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
    with sqlite3.connect(DB_PATH) as c:
        c.execute("""INSERT INTO predictions (ts,day,kind,anchor,hi,lo,hi_near,hi_far,lo_near,lo_far,
                     p_up,p_down,p_neutral,score,data_quality,agreement,payload)
                     VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                  (now.isoformat(timespec="seconds"), now.date().isoformat(), kind, fc["anchor"], fc["hi"], fc["lo"],
                   fc["hi_near"], fc["hi_far"], fc["lo_near"], fc["lo_far"],
                   fc["probs"]["up"], fc["probs"]["down"], fc["probs"]["neutral"], fc["score"],
                   fc["data_quality"], fc["agreement"], json.dumps(fc, default=float, ensure_ascii=False)))
        g = fc["gex"]
        c.execute("INSERT INTO gex_levels VALUES (?,?,?,?,?,?,?)",
                  (now.isoformat(timespec="seconds"), g["spot"], g["flip"], g["call_wall"], g["put_wall"],
                   g["net"], g["regime"]))


def save_snapshot(now, chain):
    """لقطة مضغوطة للسلسلة — هذه بيانات التدريب المستقبلية (لا يوجد تاريخ خيارات مجاني)."""
    try:
        df = chain["df"]
        spot = chain["spot"]
        today = now.date()
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
                          json={"chat_id": TG_CHAT, "text": text[:4000], "disable_web_page_preview": True}, timeout=15)
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
    today = now.date()
    days = sorted({d for d in df.index.date if d < today})
    if not days:
        raise SourceError(f"{sym}: لا يوجد إغلاق سابق")
    pdv = days[-1]
    pc = session_close(df, pdv)
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


# ---- خيارات SPX من ملفات Cboe المتأخرة (مجانية، رابط غير رسمي) ----
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
    x["gex"] = x.gamma * x.oi * 100 * spot * spot * 0.01 * sign   # دولار لكل 1% حركة
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


# ---- Alpaca المجاني (IEX) ----
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


def get_spy_vwap(now):
    if not is_rth(now):
        raise SourceError("خارج الجلسة")
    s = datetime.combine(now.date(), dtime(9, 30), tzinfo=NY).astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    r = requests.get(f"{DATA_URL}/v2/stocks/SPY/bars", headers=ah(),
                     params={"timeframe": "1Min", "start": s, "limit": 1000, "feed": STOCK_FEED}, timeout=15)
    bars = (r.json().get("bars") or []) if r.status_code == 200 else []
    if len(bars) < 10:
        raise SourceError("VWAP: شموع غير كافية")
    df = pd.DataFrame(bars)
    vwap = (((df.h + df.l + df.c) / 3) * df.v).sum() / df.v.sum()
    return dict(dist_pct=float((df.c.iloc[-1] / vwap - 1) * 100))


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
        out.update(dxy=float(dx.Close.iloc[-1]), dxy_chg=(float(dx.Close.iloc[-1] / dx.Close.iloc[-2]) - 1) * 100)
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
    """الشركات الكبرى موزونة، وتعمل قبل الافتتاح وأثناء الجلسة (Yahoo، متأخر)."""
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
    """مؤشر ناسداك 100 (^NDX) — حي أثناء الجلسة فقط، وقبل الافتتاح نستخدم NQ."""
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


SOURCES = [
    ("SPX", get_spx), ("ES", lambda n: get_future("ES=F", n)), ("NQ", lambda n: get_future("NQ=F", n)),
    ("VIX", get_vix), ("CHAIN", get_chain), ("LEADERS", get_leaders), ("NASDAQ", get_nasdaq),
    ("BREADTH", get_breadth), ("VWAP", get_spy_vwap), ("MACRO", get_macro), ("NEWS", get_news),
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
    # فحص تطابق Cboe مع SPX: يكشف البيانات القديمة
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
    ld, nd = d.get("LEADERS"), d.get("NASDAQ")
    gex = ch["gex"]
    rth = is_rth(now)
    anchor = spx["spot"] if rth else spx["prev_close"] * es["last"] / es["prev_close"]

    comps = {}
    f, w = [math.tanh(es["ret_pct"] / 0.5)], [0.7]
    if nq:
        f.append(math.tanh(nq["ret_pct"] / 0.7))
        w.append(0.3)
    comps["futures"] = (float(np.average(f, weights=w)), 0.27)
    comps["vix"] = (-math.tanh(vix["chg_pct"] / 6.0), 0.13)
    g = 0.0
    if gex["call_wall"] and 0 <= (gex["call_wall"] - anchor) / anchor < 0.0015:
        g -= 0.4
    if gex["put_wall"] and 0 <= (anchor - gex["put_wall"]) / anchor < 0.0015:
        g += 0.4
    comps["gex"] = (g, 0.13)
    cl = spx["closes"] if rth and spx["closes"] is not None else es["closes"]
    e5, e13 = cl.ewm(span=5, adjust=False).mean().iloc[-1], cl.ewm(span=13, adjust=False).mean().iloc[-1]
    tech = math.tanh((e5 - e13) / cl.iloc[-1] * 1500)
    if vw:
        tech = 0.7 * tech + 0.3 * math.tanh(vw["dist_pct"] / 0.15)
    comps["technical"] = (tech, 0.12)
    if ld:
        comps["leaders"] = (0.5 * math.tanh(ld["wavg"] / 0.5) + 0.5 * math.tanh((ld["wup"] - 0.5) * 4), 0.15)
    if br and br.get("sectors_up") is not None:
        comps["breadth"] = (math.tanh((br["sectors_up"] - 0.5) * 4), 0.05)
    if mc:
        parts = []
        if "dxy_chg" in mc:
            parts.append(-math.tanh(mc["dxy_chg"] / 0.4))
        if "y10_bps" in mc:
            parts.append(-math.tanh(mc["y10_bps"] / 6.0))
        if parts:
            comps["macro"] = (float(np.mean(parts)), 0.08)

    # ناسداك: أثناء الجلسة NDX الفعلي، وقبل الافتتاح الفرق بين NQ وES
    rel, rel_src = None, ""
    if nd and nd["live"]:
        spx_ret = (spx["spot"] / spx["prev_close"] - 1) * 100
        rel, rel_src = nd["ret_pct"] - spx_ret, "NDX"
        comps["nasdaq"] = (math.tanh(nd["ret_pct"] / 0.7), 0.07)
    elif nq:
        rel, rel_src = nq["ret_pct"] - es["ret_pct"], "NQ"

    tw = sum(wt for _, wt in comps.values())
    S = sum(sc * wt for sc, wt in comps.values()) / tw
    quality = tw
    sig_n = [sc for sc, _ in comps.values() if abs(sc) > 0.1]
    agreement = (sum(1 for sc in sig_n if np.sign(sc) == np.sign(S)) / len(sig_n)) if sig_n else 0.0
    conflicts = sum(1 for sc, _ in comps.values() if abs(sc) > 0.15 and np.sign(sc) != np.sign(S))
    near_wall = (gex["call_wall"] and S > 0 and 0 <= (gex["call_wall"] - anchor) / anchor < 0.002) or \
                (gex["put_wall"] and S < 0 and 0 <= (anchor - gex["put_wall"]) / anchor < 0.002)
    rev_score = conflicts + (1 if near_wall else 0)
    reversal = "مرتفع" if rev_score >= 3 else ("متوسط" if rev_score >= 1 else "منخفض")

    neutral = 0.4 * math.exp(-(S / 0.2) ** 2)
    sg = 1 / (1 + math.exp(-4 * S))
    probs = dict(up=(1 - neutral) * sg, down=(1 - neutral) * (1 - sg), neutral=neutral)
    label = max(probs, key=probs.get)

    iv = vix["vix"] if not vix["vix9d"] else 0.5 * vix["vix"] + 0.5 * vix["vix9d"]
    sig_day = 0.6 * (anchor * iv / 100 / math.sqrt(252)) + 0.4 * (spx["atr"] / 1.596)
    sig_day *= 0.9 if gex["regime"] == "POSITIVE" else 1.12
    m = now.hour * 60 + now.minute
    frac = 1.0 if m < 570 else max(0.05, (960 - m) / 390.0)
    sig = sig_day * math.sqrt(frac)
    c = anchor + S * 0.15 * sig
    hi, lo = c + 0.798 * sig, c - 0.798 * sig
    hi_near, hi_far = c + 0.126 * sig, c + 1.645 * sig
    lo_near, lo_far = c - 0.126 * sig, c - 1.645 * sig
    if rth:
        dh, dl = spx["day_high"], spx["day_low"]
        hi, hi_near, hi_far = max(hi, dh), max(hi_near, dh), max(hi_far, dh)
        lo, lo_near, lo_far = min(lo, dl), min(lo_near, dl), min(lo_far, dl)

    L = [f"ES: {lab(es['ret_pct'], 0.15, 'داعم للصعود', 'داعم للهبوط', 'محايد')} ({es['ret_pct']:+.2f}%)"]
    L.append(f"NQ: {lab(nq['ret_pct'], 0.2, 'داعم للصعود', 'داعم للهبوط', 'محايد')} ({nq['ret_pct']:+.2f}%)"
             if nq else "NQ: غير متاح")
    if rel is not None:
        tag = lab(rel, 0.15, "قيادة تقنية", "تأخر تقنية", "متوافق مع SPX")
        base = f"NDX {nd['ret_pct']:+.2f}% | " if rel_src == "NDX" else ""
        L.append(f"ناسداك: {base}الفرق عن SPX {rel:+.2f}% ({tag})")
    v9 = f" | VIX9D {vix['vix9d']:.1f}" if vix["vix9d"] else ""
    L.append(f"VIX: {vix['vix']:.1f} ({vix['chg_pct']:+.1f}%) "
             f"{lab(vix['chg_pct'], 3, 'ارتفاع التقلب', 'انخفاض التقلب', 'مستقر')}{v9}")
    L.append(f"GEX: جاما {'إيجابية' if gex['regime']=='POSITIVE' else 'سلبية'} | Flip {n0(gex['flip'])} | "
             f"مقاومة {n0(gex['call_wall'])} | دعم {n0(gex['put_wall'])}")
    zr = gex["zero_dte_cp_vol"]
    L.append(f"0DTE: {gex['zero_dte_share']*100:.0f}% من الجاما" + (f" | حجم Call/Put {zr:.2f}" if zr else ""))
    if ld:
        L.append(f"الشركات الكبرى (موزونة): {ld['wavg']:+.2f}% | {ld['wup']*100:.0f}% من الوزن صاعد")
        L.append("الأكبر وزناً: " + " | ".join(f"{i['sym']} {i['ret']:+.1f}%" for i in ld["top"]))
    else:
        L.append("الشركات الكبرى: غير متاح")
    if br and br.get("sectors_up") is not None:
        L.append(f"Breadth (قطاعات): {br['sectors_up']*100:.0f}% صاعدة")
    else:
        L.append("Breadth (قطاعات): " + ("غير مؤكد قبل الافتتاح" if not rth else "غير متاح"))
    if mc:
        bits = []
        if "y10" in mc:
            bits.append(f"10Y {mc['y10']:.2f}% ({mc.get('y10_bps', 0):+.0f}bp)")
        if "dxy" in mc:
            bits.append(f"DXY {mc['dxy']:.1f} ({mc['dxy_chg']:+.2f}%)")
        L.append("Macro: " + " | ".join(bits))
    if d.get("NEWS"):
        L.append("أخبار: " + d["NEWS"][0])

    levels = dict(r1=gex["call_wall"] if gex["call_wall"] and gex["call_wall"] > anchor else None, r2=hi_far,
                  s1=gex["put_wall"] if gex["put_wall"] and gex["put_wall"] < anchor else None, s2=lo_far)
    return dict(anchor=anchor, spot=spx["spot"], hi=hi, lo=lo, hi_near=hi_near, hi_far=hi_far,
                lo_near=lo_near, lo_far=lo_far, probs=probs, label=label, score=S, reversal=reversal,
                data_quality=quality, agreement=agreement, gex=gex_public(gex), lines=L, levels=levels,
                comps={k: round(v[0], 3) for k, v in comps.items()})


def gex_public(g):
    return {k: g[k] for k in ("spot", "flip", "call_wall", "put_wall", "net", "regime")}


# ============================ التقارير ============================
def format_report(fc, title):
    p, lv = fc["probs"], fc["levels"]
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
    fc = build_forecast(d, now)
    save_prediction(now, kind, fc)
    return fc, d, now


def job_pre_update():
    return bool(run_forecast("premarket_update"))


def job_pre_final():
    r = run_forecast("premarket")
    if not r:
        return False
    fc, d, now = r
    tg(format_report(fc, "تقرير توقعات S&P 500 — قبل الافتتاح"))
    save_snapshot(now, d["CHAIN"])
    return True


def job_rth(now):
    r = run_forecast("intraday")
    if not r:
        return
    fc, d, now = r
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


def job_eval():
    now = now_ny()
    try:
        df = yf_hist("^GSPC", "5d", "5m")
    except SourceError:
        return False
    m = df.index.hour * 60 + df.index.minute
    t = df[(df.index.date == now.date()) & (m < 960)]
    if t.empty:
        return False
    hi, lo, cl = float(t.High.max()), float(t.Low.min()), float(t.Close.iloc[-1])
    with sqlite3.connect(DB_PATH) as c:
        c.execute("INSERT OR REPLACE INTO realized VALUES (?,?,?,?)", (now.date().isoformat(), hi, lo, cl))
        row = c.execute("SELECT hi,lo FROM predictions WHERE day=? AND kind='premarket'",
                        (now.date().isoformat(),)).fetchone()
    today_line = (f"توقع القمة {n0(row[0])} مقابل الفعلي {n0(hi)} | توقع القاع {n0(row[1])} مقابل الفعلي {n0(lo)}"
                  if row else "لا يوجد توقع قبل الافتتاح اليوم")
    tg("\n".join(["📊 حفظ النتائج والتقييم", f"إغلاق SPX: {cl:,.2f}", today_line, eval_summary(), SIGNATURE]))
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
    once("eval", now, 16 * 60 + 15, 90, job_eval)


def main():
    init_db()
    log.info("تشغيل صَفاء التحليل SPX")
    tg("🚀 تم تشغيل صَفاء التحليل SPX (مصادر مجانية)\nلا تنفيذ تلقائي — تحليل احتمالي فقط\n" + SIGNATURE)
    while True:
        try:
            tick()
        except Exception as e:
            log.exception("خطأ تشغيلي: %s", e)
        time.sleep(20)


if __name__ == "__main__":
    main()