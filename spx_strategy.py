# -*- coding: utf-8 -*-
"""صَفاء التحليل v26.0 — recommendations only; no order execution."""
import os, time, json, math, sqlite3, logging, traceback, re
from datetime import datetime, timedelta, date, time as dtime
from zoneinfo import ZoneInfo
from pathlib import Path
from typing import Optional

import requests
import numpy as np
import pandas as pd
import yfinance as yf
from PIL import Image, ImageDraw, ImageFont
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, brier_score_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

APP_NAME = "صَفاء التحليل"
VERSION = "v26.0"
ET = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("safaa")

# Railway environment variables. Never hardcode credentials.
ALPACA_KEY = os.getenv("APCA_API_KEY_ID", "").strip()
ALPACA_SECRET = os.getenv("APCA_API_SECRET_KEY", "").strip()
ALPACA_DATA_URL = os.getenv("ALPACA_DATA_URL", "https://data.alpaca.markets").rstrip("/")
STOCK_FEED = os.getenv("ALPACA_STOCK_FEED", "iex").strip().lower()
OPTIONS_FEED = os.getenv("ALPACA_OPTIONS_FEED", "indicative").strip().lower()
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()
DB_PATH = os.getenv("DB_PATH", "./data/safaa_v26.db")
ARTIFACT_DIR = Path(os.getenv("ARTIFACT_DIR", "./data/artifacts"))
POLL_SECONDS = max(60, int(os.getenv("POLL_SECONDS", "120")))
OPTION_MAX_UNDERLYINGS = max(5, int(os.getenv("OPTION_MAX_UNDERLYINGS", "35")))
MAX_OPTION_CHAIN_SCANS = max(1, int(os.getenv("MAX_OPTION_CHAIN_SCANS", "6")))
DYNAMIC_SCREENER_TOP = min(100, max(10, int(os.getenv("DYNAMIC_SCREENER_TOP", "40"))))
MAX_OPTION_SPREAD_PCT = float(os.getenv("MAX_OPTION_SPREAD_PCT", "12"))
MAX_OPTION_QUOTE_AGE_SECONDS = int(os.getenv("MAX_OPTION_QUOTE_AGE_SECONDS", "180"))
MAX_STOCK_BAR_AGE_SECONDS = int(os.getenv("MAX_STOCK_BAR_AGE_SECONDS", "1200"))
MIN_OPTION_BID = float(os.getenv("MIN_OPTION_BID", "0.05"))
MIN_OPTION_BID_SIZE = int(os.getenv("MIN_OPTION_BID_SIZE", "1"))
MIN_OPTION_ASK_SIZE = int(os.getenv("MIN_OPTION_ASK_SIZE", "1"))
MIN_SPX_MODEL_ROWS = int(os.getenv("MIN_SPX_MODEL_ROWS", "300"))
SPX_LOOKBACK_DAYS = int(os.getenv("SPX_LOOKBACK_DAYS", "60"))
SPX_HORIZON_BARS = int(os.getenv("SPX_HORIZON_BARS", "6"))
SPX_MIN_AUC = float(os.getenv("SPX_MIN_AUC", "0.53"))
SPX_MIN_ACCURACY = float(os.getenv("SPX_MIN_ACCURACY", "0.52"))
ALERT_COOLDOWN_MINUTES = int(os.getenv("ALERT_COOLDOWN_MINUTES", "45"))
MAX_ALERTS_PER_DAY = int(os.getenv("MAX_ALERTS_PER_DAY", "5"))
UNIVERSE_ENV = os.getenv("STOCK_UNIVERSE", "SPY,QQQ,NVDA,AAPL,MSFT,AMZN,META,TSLA,GOOGL,AMD,AVGO,NFLX,PLTR,COIN,SMCI,ORCL,MU,INTC,CRM,UBER,SHOP,ARM,BA, JPM,LLY, XOM,COST,WMT,DIS,SOFI,IWM,DIA").replace(" ", "")
UNIVERSE = list(dict.fromkeys(x.upper() for x in UNIVERSE_ENV.split(",") if x))[:OPTION_MAX_UNDERLYINGS]

SESSION_OPEN = dtime(9, 30)
SESSION_CLOSE = dtime(16, 0)
PREMARKET_SEND_AFTER = dtime(7, 0)
PREMARKET_SEND_BEFORE = dtime(9, 25)

session = requests.Session()
session.headers.update({"APCA-API-KEY-ID": ALPACA_KEY, "APCA-API-SECRET-KEY": ALPACA_SECRET, "accept": "application/json"})


def now_et() -> datetime:
    return datetime.now(ET)


def iso_utc(dt: datetime) -> str:
    return dt.astimezone(UTC).isoformat().replace("+00:00", "Z")


def parse_dt(value) -> Optional[datetime]:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        return dt.astimezone(UTC)
    except Exception:
        return None


def init_db():
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as con:
        con.execute("""CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT NOT NULL, kind TEXT NOT NULL,
            symbol TEXT, option_symbol TEXT, direction TEXT, underlying_price REAL,
            entry_ask REAL, entry_bid REAL, target REAL, confidence REAL, payload TEXT,
            max_bid REAL, min_bid REAL, status TEXT DEFAULT 'OPEN', last_mark REAL,
            last_update TEXT, result TEXT)""")
        con.execute("CREATE INDEX IF NOT EXISTS idx_pred_date ON predictions(created_at)")
        con.execute("""CREATE TABLE IF NOT EXISTS daily_reports (
            report_date TEXT NOT NULL, report_type TEXT NOT NULL, sent_at TEXT NOT NULL,
            PRIMARY KEY(report_date, report_type))""")


def alpaca_get(path, params=None, timeout=20):
    if not ALPACA_KEY or not ALPACA_SECRET:
        raise RuntimeError("Missing APCA_API_KEY_ID / APCA_API_SECRET_KEY environment variables")
    url = ALPACA_DATA_URL + path
    response = session.get(url, params=params or {}, timeout=timeout)
    if response.status_code == 429:
        raise RuntimeError("Alpaca rate limit reached (429)")
    response.raise_for_status()
    return response.json()


def tg_request(method, data=None, files=None, timeout=30):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        log.warning("Telegram not configured; set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID")
        return False
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/{method}"
    response = requests.post(url, data=data or {}, files=files, timeout=timeout)
    response.raise_for_status()
    result = response.json()
    if not result.get("ok"):
        raise RuntimeError(f"Telegram API error: {result}")
    return True


def tg_text(text: str):
    # Split long messages to stay under Telegram's message limit.
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        log.info("Telegram text skipped (not configured): %s", text[:160])
        return False
    chunks = [text[i:i+3900] for i in range(0, len(text), 3900)] or [" "]
    for chunk in chunks:
        tg_request("sendMessage", {"chat_id": TELEGRAM_CHAT_ID, "text": chunk, "disable_web_page_preview": True})
    return True


def tg_photo(path: Path, caption: str = ""):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        log.info("Telegram photo skipped (not configured): %s", path)
        return False
    with open(path, "rb") as f:
        tg_request("sendPhoto", {"chat_id": TELEGRAM_CHAT_ID, "caption": caption[:1000]}, {"photo": f})
    return True


def arabic_font(size=26, bold=False):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            try: return ImageFont.truetype(candidate, size=size)
            except Exception: pass
    return ImageFont.load_default()


def rtl(text):
    try:
        import arabic_reshaper
        from bidi.algorithm import get_display
        return get_display(arabic_reshaper.reshape(str(text)))
    except Exception:
        return str(text)


def create_card(title, subtitle, rows, filename, accent=(91, 220, 175), footer="تحليل احتمالي — لا يوجد ضمان للربح") -> Path:
    W, H = 1200, max(760, 330 + 66 * len(rows))
    img = Image.new("RGB", (W, H), (12, 22, 37))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle((28, 28, W-28, H-28), radius=30, fill=(17, 31, 49), outline=(47, 67, 88), width=2)
    d.rounded_rectangle((56, 54, 1144, 178), radius=24, fill=(26, 45, 66))
    d.text((W-82, 76), rtl(title), font=arabic_font(42, True), fill=(244, 213, 141), anchor="ra", direction="rtl")
    d.text((W-82, 132), rtl(subtitle), font=arabic_font(24), fill=(196, 210, 225), anchor="ra", direction="rtl")
    y = 215
    for label, value, color in rows:
        d.rounded_rectangle((58, y, 1142, y+54), radius=14, fill=(26, 45, 66))
        d.text((W-82, y+13), rtl(str(label)), font=arabic_font(23, True), fill=(187, 204, 222), anchor="ra", direction="rtl")
        d.text((82, y+13), str(value), font=arabic_font(23, True), fill=color, anchor="la")
        y += 66
    d.line((65, H-95, W-65, H-95), fill=(52, 72, 94), width=2)
    d.text((W-82, H-75), rtl(footer), font=arabic_font(19), fill=(166, 183, 201), anchor="ra", direction="rtl")
    out = ARTIFACT_DIR / filename
    img.save(out, quality=92)
    return out


def fetch_spx_history():
    """Yahoo Finance fallback for actual ^GSPC history; never substitute SPY*10 for SPX."""
    ticker = yf.Ticker("^GSPC")
    df = ticker.history(period=f"{max(7, min(SPX_LOOKBACK_DAYS, 60))}d", interval="5m", auto_adjust=False, prepost=False)
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.rename(columns={c: c.lower() for c in df.columns})
    df.index = pd.to_datetime(df.index, utc=True)
    cols = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
    df = df[cols].dropna(subset=["close"]).sort_index()
    return df


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
    x["vchg"] = x["volume"].pct_change().replace([np.inf, -np.inf], np.nan).clip(-5, 5) if "volume" in x else 0.0
    idx = x.index.tz_convert(ET)
    mins = np.array([i.hour * 60 + i.minute for i in idx])
    x["tod_sin"] = np.sin(2 * np.pi * mins / 1440)
    x["tod_cos"] = np.cos(2 * np.pi * mins / 1440)
    return x.replace([np.inf, -np.inf], np.nan)

FEATURES = ["r1", "r3", "r6", "r12", "vol12", "ema_gap", "range", "vchg", "tod_sin", "tod_cos"]


def train_and_validate_spx(df: pd.DataFrame):
    """Chronological holdout only; no shuffle. AUC/accuracy are descriptive, not guarantees."""
    if df.empty or len(df) < MIN_SPX_MODEL_ROWS + SPX_HORIZON_BARS + 20:
        return None, {"ok": False, "reason": "INSUFFICIENT_HISTORY", "rows": len(df)}
    x = feature_frame(df)
    future_return = x["close"].shift(-SPX_HORIZON_BARS) / x["close"] - 1
    x["target"] = (future_return > 0).astype(int)
    clean = x[FEATURES + ["target"]].dropna()
    if len(clean) < MIN_SPX_MODEL_ROWS:
        return None, {"ok": False, "reason": "INSUFFICIENT_CLEAN_ROWS", "rows": len(clean)}
    split = int(len(clean) * 0.75)
    if split < 200 or len(clean) - split < 60:
        return None, {"ok": False, "reason": "INSUFFICIENT_HOLDOUT", "rows": len(clean)}
    model = Pipeline([("scale", StandardScaler()), ("logit", LogisticRegression(max_iter=1000, class_weight="balanced"))])
    train, test = clean.iloc[:split], clean.iloc[split:]
    model.fit(train[FEATURES], train["target"])
    probs = model.predict_proba(test[FEATURES])[:, 1]
    pred = (probs >= 0.5).astype(int)
    try: auc = float(roc_auc_score(test["target"], probs))
    except Exception: auc = 0.5
    acc = float(accuracy_score(test["target"], pred))
    brier = float(brier_score_loss(test["target"], probs))
    # Refit on all historical rows for live scoring only after reporting holdout results.
    live_model = Pipeline([("scale", StandardScaler()), ("logit", LogisticRegression(max_iter=1000, class_weight="balanced"))])
    live_model.fit(clean[FEATURES], clean["target"])
    metrics = {"ok": auc >= SPX_MIN_AUC and acc >= SPX_MIN_ACCURACY, "auc": auc, "accuracy": acc, "brier": brier,
               "train_rows": int(len(train)), "holdout_rows": int(len(test)), "all_rows": int(len(clean)),
               "horizon_minutes": SPX_HORIZON_BARS * 5, "test_method": "chronological 75/25 holdout; no shuffle"}
    return live_model, metrics


def latest_spx_snapshot(df: pd.DataFrame, model, metrics):
    if df.empty:
        return {"ok": False, "reason": "NO_SPX_DATA"}
    last_ts = df.index[-1].to_pydatetime()
    age = (datetime.now(UTC) - last_ts.astimezone(UTC)).total_seconds()
    local_now = now_et()
    in_session = local_now.weekday() < 5 and SESSION_OPEN <= local_now.time() <= SESSION_CLOSE
    if in_session and age > MAX_STOCK_BAR_AGE_SECONDS:
        return {"ok": False, "reason": "STALE_SPX_DATA", "age_seconds": int(age), "timestamp": last_ts.isoformat()}
    x = feature_frame(df).dropna(subset=FEATURES)
    if x.empty or model is None:
        return {"ok": False, "reason": "MODEL_UNAVAILABLE", "price": float(df["close"].iloc[-1]), "timestamp": last_ts.isoformat(), "metrics": metrics}
    prob_up = float(model.predict_proba(x[FEATURES].iloc[[-1]])[0, 1])
    recent = df.tail(78)
    high, low = float(recent["high"].max()), float(recent["low"].min())
    return {"ok": bool(metrics.get("ok")), "price": float(df["close"].iloc[-1]), "timestamp": last_ts.isoformat(),
            "age_seconds": int(age), "prob_up": prob_up, "prob_down": 1-prob_up, "high_6h": high, "low_6h": low,
            "metrics": metrics, "reason": "OK" if metrics.get("ok") else "MODEL_VALIDATION_BELOW_GATE"}


def get_stock_bars(symbol: str, limit=100):
    end = datetime.now(UTC)
    start = end - timedelta(days=5)
    data = alpaca_get("/v2/stocks/bars", {"symbols": symbol, "timeframe": "5Min", "start": iso_utc(start), "end": iso_utc(end),
                                             "limit": 1000, "feed": STOCK_FEED, "sort": "asc"})
    bars = data.get("bars", {}).get(symbol, [])
    if not bars:
        return pd.DataFrame()
    df = pd.DataFrame(bars)
    df["timestamp"] = pd.to_datetime(df["t"], utc=True)
    df = df.set_index("timestamp").rename(columns={"o":"open", "h":"high", "l":"low", "c":"close", "v":"volume"})
    return df[["open", "high", "low", "close", "volume"]].tail(limit).astype(float)


def stock_signal(symbol: str):
    df = get_stock_bars(symbol)
    if len(df) < 20:
        return None
    last_time = df.index[-1].to_pydatetime()
    age = (datetime.now(UTC) - last_time).total_seconds()
    if now_et().weekday() < 5 and SESSION_OPEN <= now_et().time() <= SESSION_CLOSE and age > MAX_STOCK_BAR_AGE_SECONDS:
        return None
    close = df["close"]
    r3 = float(close.iloc[-1] / close.iloc[-4] - 1)
    r6 = float(close.iloc[-1] / close.iloc[-7] - 1)
    ema_fast = float(close.ewm(span=8, adjust=False).mean().iloc[-1])
    ema_slow = float(close.ewm(span=21, adjust=False).mean().iloc[-1])
    if r3 > 0.001 and r6 > 0 and ema_fast > ema_slow:
        direction = "call"
        strength = min(1.0, 0.45 + abs(r3) * 45 + abs(r6) * 15)
    elif r3 < -0.001 and r6 < 0 and ema_fast < ema_slow:
        direction = "put"
        strength = min(1.0, 0.45 + abs(r3) * 45 + abs(r6) * 15)
    else:
        return None
    return {"symbol": symbol, "direction": direction, "strength": strength, "price": float(close.iloc[-1]),
            "timestamp": last_time.isoformat(), "age_seconds": int(age), "r3": r3, "r6": r6}


def option_chain(symbol: str):
    today = now_et().date().isoformat()
    path = f"/v1beta1/options/snapshots/{symbol}"
    params = {"expiration_date": today, "feed": OPTIONS_FEED, "limit": 1000}
    all_snaps = {}
    for _ in range(4):
        data = alpaca_get(path, params)
        all_snaps.update(data.get("snapshots", {}))
        token = data.get("next_page_token")
        if not token: break
        params["page_token"] = token
    return all_snaps


def evaluate_options_for_signal(signal):
    symbol = signal["symbol"]
    snaps = option_chain(symbol)
    candidates = []
    now = datetime.now(UTC)
    for opt_symbol, snap in snaps.items():
        quote = snap.get("latestQuote") or {}
        bid, ask = quote.get("bp"), quote.get("ap")
        if not isinstance(bid, (int, float)) or not isinstance(ask, (int, float)):
            continue
        if bid < MIN_OPTION_BID or ask <= 0 or ask < bid:
            continue
        spread_pct = 100 * (ask - bid) / max((ask + bid) / 2, 0.01)
        if spread_pct > MAX_OPTION_SPREAD_PCT:
            continue
        if int(quote.get("bs") or 0) < MIN_OPTION_BID_SIZE or int(quote.get("as") or 0) < MIN_OPTION_ASK_SIZE:
            continue
        quote_time = parse_dt(quote.get("t"))
        if quote_time is None or (now - quote_time).total_seconds() > MAX_OPTION_QUOTE_AGE_SECONDS:
            continue
        # Validate OCC-style option symbol and confirm exact same-day expiration and call/put type.
        expected = "C" if signal["direction"] == "call" else "P"
        match = re.match(r"^([A-Z0-9.]{1,6})(\d{6})([CP])(\d{8})$", opt_symbol)
        if not match or match.group(3) != expected:
            continue
        try:
            symbol_expiry = datetime.strptime(match.group(2), "%y%m%d").date().isoformat()
        except ValueError:
            continue
        if symbol_expiry != now_et().date().isoformat():
            continue
        greeks = snap.get("greeks") or {}
        delta = greeks.get("delta")
        # Prefer reasonably directional delta where available, but do not require greeks on indicative feeds.
        delta_score = 0.0
        if isinstance(delta, (int, float)):
            if expected == "C" and delta <= 0: continue
            if expected == "P" and delta >= 0: continue
            delta_score = max(0.0, 1.0 - abs(abs(delta) - 0.45))
        spread_score = max(0.0, 1.0 - spread_pct / MAX_OPTION_SPREAD_PCT)
        liquidity_score = min(1.0, (int(quote.get("bs") or 0) + int(quote.get("as") or 0)) / 20)
        score = 0.55 * signal["strength"] + 0.25 * spread_score + 0.10 * liquidity_score + 0.10 * delta_score
        trade = snap.get("latestTrade") or {}
        candidates.append({"symbol": symbol, "option_symbol": opt_symbol, "direction": signal["direction"],
                           "underlying_price": signal["price"], "bid": float(bid), "ask": float(ask),
                           "mid": (float(bid)+float(ask))/2, "spread_pct": spread_pct,
                           "cost": float(ask)*100, "score": score, "delta": delta,
                           "quote_time": quote_time.isoformat(), "underlying_time": signal["timestamp"],
                           "underlying_strength": signal["strength"], "r3": signal["r3"], "r6": signal["r6"],
                           "option_trade_price": trade.get("p")})
    return sorted(candidates, key=lambda c: c["score"], reverse=True)


def already_alerted(symbol, option_symbol):
    cutoff = (now_et() - timedelta(minutes=ALERT_COOLDOWN_MINUTES)).astimezone(UTC).isoformat()
    with sqlite3.connect(DB_PATH) as con:
        row = con.execute("SELECT id FROM predictions WHERE kind='OPTION' AND symbol=? AND option_symbol=? AND created_at>=?", (symbol, option_symbol, cutoff)).fetchone()
    return row is not None


def count_today_alerts():
    today = now_et().date().isoformat()
    with sqlite3.connect(DB_PATH) as con:
        return con.execute("SELECT COUNT(*) FROM predictions WHERE kind='OPTION' AND created_at LIKE ?", (today+"%",)).fetchone()[0]


def record_prediction(candidate):
    now = datetime.now(UTC).isoformat()
    payload = json.dumps(candidate, ensure_ascii=False)
    with sqlite3.connect(DB_PATH) as con:
        cur = con.execute("""INSERT INTO predictions(created_at,kind,symbol,option_symbol,direction,underlying_price,entry_ask,entry_bid,target,confidence,payload,max_bid,min_bid,last_mark,last_update)
            VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (now, "OPTION", candidate["symbol"], candidate["option_symbol"], candidate["direction"], candidate["underlying_price"], candidate["ask"], candidate["bid"], None, candidate["score"], payload, candidate["bid"], candidate["bid"], candidate["bid"], now))
        return cur.lastrowid


def send_option_alert(candidate):
    ticker = candidate["symbol"]
    direction_ar = "CALL (توقع صعود)" if candidate["direction"] == "call" else "PUT (توقع هبوط)"
    rows = [
        ("الشركة", ticker, (245, 213, 141)),
        ("الاتجاه", direction_ar, (89, 220, 175) if candidate["direction"] == "call" else (255, 125, 145)),
        ("سعر السهم", f"${candidate['underlying_price']:.2f}", (238, 245, 250)),
        ("سعر شراء العقد التقريبي", f"${candidate['ask']:.2f}", (238, 245, 250)),
        ("تكلفة عقد واحد", f"${candidate['cost']:.2f} قبل الرسوم", (245, 213, 141)),
        ("Bid / Ask", f"${candidate['bid']:.2f} / ${candidate['ask']:.2f}", (238, 245, 250)),
        ("السبريد", f"{candidate['spread_pct']:.1f}%", (238, 245, 250)),
        ("جودة الإشارة الداخلية", f"{candidate['score']*100:.0f}/100 (ليست دقة نجاح)", (89, 220, 175)),
    ]
    card = create_card("صَفاء التحليل | فرصة خيارات 0DTE", f"{ticker} — {direction_ar}", rows,
                       f"option_{ticker}_{int(time.time())}.jpg", accent=(89,220,175) if candidate["direction"]=="call" else (255,125,145))
    caption = f"صَفاء التحليل {VERSION} | توصية افتراضية فقط\n{ticker} — {direction_ar}\nالعقد: {candidate['option_symbol']}\nسعر السهم: ${candidate['underlying_price']:.2f} | Ask: ${candidate['ask']:.2f} | تكلفة العقد: ${candidate['cost']:.2f} قبل الرسوم.\nالسبريد: {candidate['spread_pct']:.1f}% | تقييم ترتيب داخلي: {candidate['score']*100:.0f}/100، وليس احتمال ربح أو دقة مثبتة.\nبيانات الخيارات: {OPTIONS_FEED}. لا تنفيذ أوامر. تحقق من السعر قبل أي قرار."
    tg_photo(card, caption)
    tg_text(caption + "\n\nملاحظة: لا يوجد ضمان للربح؛ تتبع البوت افتراضي ويستخدم أسعارًا متاحة وقت الفحص.")


def update_open_tracking():
    with sqlite3.connect(DB_PATH) as con:
        rows = con.execute("SELECT id,symbol,option_symbol,entry_ask,max_bid,min_bid FROM predictions WHERE kind='OPTION' AND status='OPEN'").fetchall()
    for rid, underlying, option_symbol, entry_ask, max_bid, min_bid in rows:
        try:
            data = alpaca_get("/v1beta1/options/snapshots", {"symbols": option_symbol, "feed": OPTIONS_FEED, "limit": 10})
            snap = (data.get("snapshots") or {}).get(option_symbol, {})
            q = snap.get("latestQuote") or {}
            bid = q.get("bp")
            if not isinstance(bid, (int, float)) or bid < 0: continue
            quote_time = parse_dt(q.get("t"))
            if quote_time is None or (datetime.now(UTC)-quote_time).total_seconds() > MAX_OPTION_QUOTE_AGE_SECONDS: continue
            best = max(float(max_bid or 0), float(bid))
            worst = min(float(min_bid or bid), float(bid))
            with sqlite3.connect(DB_PATH) as con:
                con.execute("UPDATE predictions SET max_bid=?,min_bid=?,last_mark=?,last_update=? WHERE id=?", (best, worst, float(bid), datetime.now(UTC).isoformat(), rid))
        except Exception as e:
            log.warning("Tracking failed for %s: %s", option_symbol, e)


def record_daily_report(report_type):
    day = now_et().date().isoformat()
    with sqlite3.connect(DB_PATH) as con:
        cur = con.execute("INSERT OR IGNORE INTO daily_reports(report_date,report_type,sent_at) VALUES(?,?,?)", (day, report_type, datetime.now(UTC).isoformat()))
        return cur.rowcount == 1


def report_already_sent(report_type):
    day = now_et().date().isoformat()
    with sqlite3.connect(DB_PATH) as con:
        return con.execute("SELECT 1 FROM daily_reports WHERE report_date=? AND report_type=?", (day, report_type)).fetchone() is not None


def send_spx_report(snapshot, metrics, report_type="premarket"):
    if not snapshot.get("price"):
        tg_text(f"{APP_NAME} {VERSION}\nتعذر إنشاء تقرير SPX: {snapshot.get('reason','بيانات غير متاحة')}\nلن نصدر توقعًا عند غياب البيانات الموثوقة.")
        return
    price = snapshot["price"]
    ok = bool(snapshot.get("ok"))
    if ok:
        direction = "ميل صاعد" if snapshot["prob_up"] >= 0.5 else "ميل هابط"
        p = max(snapshot["prob_up"], snapshot["prob_down"])
        prob_line = f"احتمال اتجاه النموذج: {p*100:.1f}% (ليس احتمال لمس هدف ولا ضمان ربح)"
    else:
        direction = "لا يوجد توقع اتجاهي مؤهل"
        prob_line = "تم حجب التوقع الاتجاهي لأن اختبار النموذج لم يجتز شروط الجودة."
    report_label = "قبل الافتتاح" if report_type == "premarket" else ("تحديث أثناء الجلسة" if report_type == "intraday" else "ملخص SPX بعد الإغلاق")
    rows = [
        ("السعر المرجعي الفعلي لـ SPX", f"{price:,.2f}", (245,213,141)),
        ("حالة التقرير", report_label, (238,245,250)),
        ("السيناريو", direction, (89,220,175) if ok and snapshot.get("prob_up",0)>=0.5 else (255,125,145) if ok else (196,210,225)),
        ("أعلى مستوى في البيانات الأخيرة", f"{snapshot.get('high_6h', price):,.2f}", (238,245,250)),
        ("أدنى مستوى في البيانات الأخيرة", f"{snapshot.get('low_6h', price):,.2f}", (238,245,250)),
        ("آخر تحديث للبيانات", str(snapshot.get("timestamp", "غير متاح"))[:19], (238,245,250)),
        ("اختبار خارج العينة AUC", f"{metrics.get('auc', float('nan')):.3f}" if metrics.get("auc") is not None else "غير متاح", (89,220,175) if metrics.get("ok") else (255,190,105)),
        ("دقة الاتجاه في الاختبار", f"{metrics.get('accuracy', float('nan'))*100:.1f}%" if metrics.get("accuracy") is not None else "غير متاح", (238,245,250)),
    ]
    card = create_card("صَفاء التحليل | خريطة SPX", "بيانات فعلية متاحة + اختبار تاريخي", rows,
                       f"spx_{report_type}_{now_et().date().isoformat()}.jpg")
    caption = f"{APP_NAME} {VERSION} | {report_label}\nسعر SPX المرجعي: {price:,.2f}\n{prob_line}\nبيانات ياهو المالية قد تكون متأخرة أو محدودة؛ وقت آخر شمعة: {snapshot.get('timestamp','غير متاح')}\nAUC خارج العينة: {metrics.get('auc','N/A')} | الدقة الاتجاهية: {metrics.get('accuracy','N/A')} | Brier: {metrics.get('brier','N/A')}\nلا تُحوّل SPY إلى SPX بضربه في 10، ولا توجد ضمانات للنتيجة."
    tg_photo(card, caption)
    tg_text(caption)


def maybe_premarket_report():
    now = now_et()
    if now.weekday() >= 5 or not (PREMARKET_SEND_AFTER <= now.time() <= PREMARKET_SEND_BEFORE) or report_already_sent("premarket"):
        return
    try:
        hist = fetch_spx_history()
        model, metrics = train_and_validate_spx(hist)
        snapshot = latest_spx_snapshot(hist, model, metrics)
        send_spx_report(snapshot, metrics, "premarket")
        record_daily_report("premarket")
    except Exception as e:
        log.exception("Premarket report failed: %s", e)


def maybe_close_report():
    now = now_et()
    if now.weekday() >= 5 or now.time() < dtime(16, 10) or report_already_sent("close"):
        return
    try:
        hist = fetch_spx_history()
        model, metrics = train_and_validate_spx(hist)
        snapshot = latest_spx_snapshot(hist, model, metrics)
        send_spx_report(snapshot, metrics, "close")
        with sqlite3.connect(DB_PATH) as con:
            today = now.date().isoformat()
            rows = con.execute("SELECT symbol,option_symbol,entry_ask,max_bid,min_bid,status FROM predictions WHERE kind='OPTION' AND created_at LIKE ?", (today+"%",)).fetchall()
        if rows:
            lines = [f"• {s} | {o} | دخول Ask ${float(e or 0):.2f} | أعلى Bid ${float(mx or 0):.2f} | أدنى Bid ${float(mn or 0):.2f}" for s,o,e,mx,mn,st in rows]
            tg_text(f"{APP_NAME} | تقرير تتبع افتراضي اليوم\n" + "\n".join(lines) + "\n\nالأرقام لا تمثل تنفيذًا أو أرباحًا محققة؛ الدخول الافتراضي على Ask والخروج التقريبي على Bid.")
        record_daily_report("close")
    except Exception as e:
        log.exception("Close report failed: %s", e)


def discover_universe():
    """Combine configured core names with Alpaca's real-time most-active/mover screeners.
    Screener data is subscription-dependent; fall back to configured names if unavailable.
    """
    names = list(UNIVERSE)
    if not ALPACA_KEY or not ALPACA_SECRET:
        return names[:OPTION_MAX_UNDERLYINGS]
    try:
        active = alpaca_get("/v1beta1/screener/stocks/most-actives", {"by": "volume", "top": DYNAMIC_SCREENER_TOP})
        items = active.get("most_actives", active.get("mostActives", []))
        for item in items if isinstance(items, list) else []:
            ticker = item.get("symbol") if isinstance(item, dict) else None
            if ticker and ticker not in names:
                names.append(ticker)
    except Exception as e:
        log.warning("Most-active screener unavailable; using configured universe: %s", e)
    try:
        movers = alpaca_get("/v1beta1/screener/stocks/movers", {"top": min(20, DYNAMIC_SCREENER_TOP)})
        for group in (movers.get("gainers", []), movers.get("losers", [])):
            for item in group if isinstance(group, list) else []:
                ticker = item.get("symbol") if isinstance(item, dict) else None
                if ticker and ticker not in names:
                    names.append(ticker)
    except Exception as e:
        log.warning("Mover screener unavailable; continuing with current universe: %s", e)
    return names[:OPTION_MAX_UNDERLYINGS]


def run_option_scan():
    if not ALPACA_KEY or not ALPACA_SECRET:
        log.error("Missing Alpaca credentials; option scan skipped")
        return
    if now_et().weekday() >= 5 or not (SESSION_OPEN <= now_et().time() < SESSION_CLOSE):
        return
    if count_today_alerts() >= MAX_ALERTS_PER_DAY:
        log.info("Daily alert cap reached")
        return
    ranked_signals = []
    # Broad scan: configured core plus most-active/mover screeners. Only top directional
    # underlyings proceed to option-chain requests, helping avoid rate limits.
    symbols = discover_universe()
    for symbol in symbols:
        try:
            signal = stock_signal(symbol)
            if signal:
                ranked_signals.append(signal)
        except Exception as e:
            log.warning("Underlying scan failed for %s: %s", symbol, e)
    ranked_signals.sort(key=lambda x: x["strength"], reverse=True)
    ranked = []
    for signal in ranked_signals[:MAX_OPTION_CHAIN_SCANS]:
        try:
            for candidate in evaluate_options_for_signal(signal)[:2]:
                if candidate["score"] >= 0.70:
                    ranked.append(candidate)
        except Exception as e:
            log.warning("Option chain scan failed for %s: %s", signal["symbol"], e)
    if not ranked:
        log.info("Scan complete: no option passed all filters (symbols checked=%d)", len(symbols))
        return
    ranked.sort(key=lambda c: c["score"], reverse=True)
    chosen = ranked[0]
    if already_alerted(chosen["symbol"], chosen["option_symbol"]):
        log.info("Best candidate already alerted in cooldown: %s", chosen["option_symbol"])
        return
    record_prediction(chosen)
    send_option_alert(chosen)
    log.info("Virtual recommendation sent: %s %s score=%.3f", chosen["symbol"], chosen["option_symbol"], chosen["score"])


def main():
    init_db()
    log.info("%s %s started | universe=%d | feed=%s | options=%s | recommendation-only", APP_NAME, VERSION, len(UNIVERSE), STOCK_FEED, OPTIONS_FEED)
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        log.warning("Telegram env vars are missing; messages will only be logged")
    while True:
        try:
            maybe_premarket_report()
            maybe_close_report()
            # SPX intraday update: only send when a model passes holdout gates and market is open.
            if now_et().weekday() < 5 and SESSION_OPEN <= now_et().time() < SESSION_CLOSE:
                try:
                    hist = fetch_spx_history()
                    model, metrics = train_and_validate_spx(hist)
                    snapshot = latest_spx_snapshot(hist, model, metrics)
                    if snapshot.get("ok") and metrics.get("ok"):
                        # One update per market hour; no repeated messages every polling cycle.
                        report_key = f"intraday_{now_et().hour:02d}"
                        if not report_already_sent(report_key):
                            send_spx_report(snapshot, metrics, "intraday")
                            record_daily_report(report_key)
                    elif snapshot.get("reason") in ("STALE_SPX_DATA", "NO_SPX_DATA"):
                        log.warning("SPX update suppressed: %s", snapshot.get("reason"))
                except Exception as e:
                    log.warning("SPX intraday analysis unavailable: %s", e)
                try:
                    run_option_scan()
                    update_open_tracking()
                except Exception as e:
                    log.exception("Option scan/tracking failed: %s", e)
        except Exception as e:
            log.error("Main loop error: %s\n%s", e, traceback.format_exc())
        time.sleep(POLL_SECONDS)

if __name__ == "__main__":
    main()
# ============================================================
# دالة جلب إشارة SPX الحقيقية المتوافقة مع تحليلاتك في نفس الملف
# ============================================================
def get_spx_signal():
    try:
        hist = fetch_spx_history()
        if hist.empty:
            return {"ok": False, "reason": "NO_HIST_DATA"}
            
        model, metrics = train_and_validate_spx(hist)
        snapshot = latest_spx_snapshot(hist, model, metrics)
        
        if not snapshot.get("ok") or not snapshot.get("price"):
            return {"ok": False, "reason": snapshot.get("reason", "INVALID_SNAPSHOT")}
            
        prob_up = snapshot.get("prob_up", 0.5)
        signal_dir = "صعود 🚀" if prob_up >= 0.5 else "نزول 📉"
        probability = round(max(prob_up, 1 - prob_up) * 100, 1)
        
        return {
            "ok": True,
            "signal": signal_dir,
            "probability": probability,
            "current_price": round(snapshot["price"], 2),
            "timestamp": snapshot.get("timestamp", datetime.now(UTC).isoformat())
        }
    except Exception as e:
        log.exception(f"Error in get_spx_signal: {e}")
        return {"ok": False, "reason": str(e)}

