# ============================================================
# SPX & 10 STOCKS 0DTE AI ADVISOR / SCANNER v16.1 (STABLE)
# ============================================================

import os
import time
import warnings
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

VERSION = "v16.1-STABLE"
NY = ZoneInfo("America/New_York")

TIMEFRAME = "5Min"
HISTORY_DAYS = 60
MIN_TRAIN_ROWS = 150
HORIZON = 6
ATR_TARGET = 0.50
MIN_PROBABILITY = 0.62
MIN_OPTION_SCORE = 75
MAX_SPREAD_PERCENT = 0.15
MIN_OPTION_PREMIUM = 0.20
MAX_STRIKE_DISTANCE = 40
SIGNAL_COOLDOWN_MINUTES = 20
POLL_SECONDS = 30
MAX_TELEGRAM_OPPORTUNITIES = 3
NO_TRADE_FIRST_MIN = 10
NO_TRADE_LAST_MIN = 30

STOCKS = [
    "NVDA", "AAPL", "MSFT", "TSLA", "AMZN",
    "META", "GOOGL", "AMD", "AVGO", "NFLX"
]
ALL_SYMBOLS = ["SPY", "QQQ"] + STOCKS

ALPACA_API_KEY = os.getenv("ALPACA_API_KEY") or os.getenv("APCA_API_KEY_ID") or ""
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY") or os.getenv("APCA_SECRET_KEY_ID") or ""
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

ALPACA_TRADING_URL = os.getenv("ALPACA_TRADING_URL", "https://api.alpaca.markets")
ALPACA_DATA_URL = "https://data.alpaca.markets"
DATA_FEED = os.getenv("ALPACA_DATA_FEED", "iex")

STATE = {"last_sent": {}}
session = requests.Session()
HEADERS = {
    "APCA-API-KEY-ID": ALPACA_API_KEY,
    "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY,
}

def log(message):
    now = datetime.now(NY).strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] {message}", flush=True)

def telegram_send(message):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return False
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "HTML", "disable_web_page_preview": True}
    try:
        r = session.post(url, json=payload, timeout=15)
        return r.ok
    except Exception:
        return False

def now_ny():
    return datetime.now(NY)

def iso_utc(dt):
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=NY)
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")

def market_open_now():
    now = now_ny()
    mins = now.hour * 60 + now.minute
    open_min, close_min = 9 * 60 + 30, 16 * 60
    if mins < open_min: return False, "السوق لم يفتح بعد"
    if mins >= close_min: return False, "السوق مغلق"
    if mins - open_min < NO_TRADE_FIRST_MIN: return False, "أول دقائق السوق"
    if close_min - mins <= NO_TRADE_LAST_MIN: return False, "آخر دقائق السوق"
    return True, "السوق مفتوح"

def alpaca_get(url, params=None, timeout=30):
    try:
        r = session.get(url, headers=HEADERS, params=params, timeout=timeout)
        if r.status_code != 200:
            return None
        return r.json()
    except Exception:
        return None

def fetch_all_bars(symbols):
    end_dt = datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(days=HISTORY_DAYS)
    result = {s: [] for s in symbols}
    page_token = None

    while True:
        params = {
            "symbols": ",".join(symbols), "timeframe": TIMEFRAME,
            "start": iso_utc(start_dt), "end": iso_utc(end_dt),
            "limit": 10000, "feed": DATA_FEED, "sort": "asc",
        }
        if page_token: params["page_token"] = page_token
        data = alpaca_get(f"{ALPACA_DATA_URL}/v2/stocks/bars", params=params, timeout=45)
        if not data: break

        bars = data.get("bars", {})
        for symbol in symbols:
            for b in bars.get(symbol, []):
                result[symbol].append({
                    "timestamp": b.get("t"), "open": float(b.get("o", 0)),
                    "high": float(b.get("h", 0)), "low": float(b.get("l", 0)),
                    "close": float(b.get("c", 0)), "volume": float(b.get("v", 0)),
                })
        page_token = data.get("next_page_token")
        if not page_token: break

    frames = {}
    for symbol, rows in result.items():
        if not rows:
            frames[symbol] = pd.DataFrame()
            continue
        df = pd.DataFrame(rows)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        frames[symbol] = df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    return frames

def rsi(series, period=14):
    delta = series.diff()
    gain, loss = delta.clip(lower=0), -delta.clip(upper=0)
    rs = gain.rolling(period).mean() / loss.rolling(period).mean().replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def atr(df, period=14):
    prev = df["close"].shift(1)
    tr = pd.concat([df["high"] - df["low"], (df["high"] - prev).abs(), (df["low"] - prev).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def make_features(df, market_df=None, qqq_df=None):
    x = df.copy()
    close = x["close"]
    x["ret_1"], x["ret_3"], x["ret_6"], x["ret_12"] = close.pct_change(1), close.pct_change(3), close.pct_change(6), close.pct_change(12)
    x["rsi"], x["atr"] = rsi(close, 14), atr(x, 14)
    x["atr_pct"], x["range_pct"], x["volatility"] = x["atr"] / close, (x["high"] - x["low"]) / close, x["ret_1"].rolling(20).std()
    x["ma_9"], x["ma_20"], x["ma_50"] = close.rolling(9).mean(), close.rolling(20).mean(), close.rolling(50).mean()
    x["ma9_dist"], x["ma20_dist"], x["ma50_dist"] = close / x["ma_9"] - 1, close / x["ma_20"] - 1, close / x["ma_50"] - 1
    x["momentum"], x["acceleration"] = close / close.shift(12) - 1, x["ret_3"] - x["ret_3"].shift(3)
    vol_m, vol_s = x["volume"].rolling(30).mean(), x["volume"].rolling(30).std()
    x["volume_z"] = (x["volume"] - vol_m) / vol_s.replace(0, np.nan)
    
    if market_df is not None and not market_df.empty:
        m = market_df[["timestamp", "close"]].rename(columns={"close": "market_close"})
        x = pd.merge_asof(x.sort_values("timestamp"), m.sort_values("timestamp"), on="timestamp", direction="backward")
        x["market_ret_3"], x["market_ret_12"], x["relative_strength"] = x["market_close"].pct_change(3), x["market_close"].pct_change(12), x["ret_3"] - x["market_close"].pct_change(3)
    else:
        x["market_ret_3"], x["market_ret_12"], x["relative_strength"] = 0, 0, 0

    if qqq_df is not None and not qqq_df.empty:
        q = qqq_df[["timestamp", "close"]].rename(columns={"close": "qqq_close"})
        x = pd.merge_asof(x.sort_values("timestamp"), q.sort_values("timestamp"), on="timestamp", direction="backward")
        x["qqq_ret_3"], x["qqq_ret_12"] = q["qqq_close"].pct_change(3), q["qqq_close"].pct_change(12)
    else:
        x["qqq_ret_3"], x["qqq_ret_12"] = 0, 0

    local_time = x["timestamp"].dt.tz_convert(NY)
    mins = local_time.dt.hour * 60 + local_time.dt.minute
    x["time_sin"], x["time_cos"] = np.sin(2 * np.pi * mins / 1440), np.cos(2 * np.pi * mins / 1440)
    
    future_high = pd.concat([x["high"].shift(-i) for i in range(1, HORIZON + 1)], axis=1).max(axis=1)
    future_low = pd.concat([x["low"].shift(-i) for i in range(1, HORIZON + 1)], axis=1).min(axis=1)
    x["target"] = np.where((future_high >= close + x["atr"] * ATR_TARGET) & ~(future_low <= close - x["atr"] * ATR_TARGET), 1,
                           np.where((future_low <= close - x["atr"] * ATR_TARGET) & ~(future_high >= close + x["atr"] * ATR_TARGET), 0, np.nan))
    return x

FEATURE_COLUMNS = [
    "ret_1", "ret_3", "ret_6", "ret_12", "rsi", "atr_pct", "range_pct", "volatility",
    "ma9_dist", "ma20_dist", "ma50_dist", "momentum", "acceleration", "volume_z",
    "market_ret_3", "market_ret_12", "relative_strength", "qqq_ret_3", "qqq_ret_12", "time_sin", "time_cos"
]

def train_model(feature_df):
    if feature_df.empty: return None
    clean = feature_df.dropna(subset=FEATURE_COLUMNS + ["target"]).copy()
    if len(clean) < MIN_TRAIN_ROWS: return None
    clean["target"] = clean["target"].astype(int)
    n = len(clean)
    train, test = clean.iloc[:int(n * 0.60)], clean.iloc[int(n * 0.80):]
    if train["target"].nunique() < 2 or test["target"].nunique() < 2: return None
    
    models = []
    for seed in [17, 41, 83]:
        m = HistGradientBoostingClassifier(max_iter=250, learning_rate=0.045, max_leaf_nodes=15, min_samples_leaf=25, l2_regularization=1.0, random_state=seed)
        m.fit(train[FEATURE_COLUMNS], train["target"])
        models.append(m)
    probs = [m.predict_proba(test[FEATURE_COLUMNS])[:, 1] for m in models]
    auc = roc_auc_score(test["target"], np.mean(probs, axis=0))
    return {"models": models, "auc": float(np.clip(auc, 0.50, 0.999)), "rows": len(clean)}

def predict(model_info, feature_df):
    if not model_info: return None
    clean = feature_df.dropna(subset=FEATURE_COLUMNS)
    if clean.empty: return None
    latest = clean.iloc[-1]
    X = pd.DataFrame([latest[FEATURE_COLUMNS].values], columns=FEATURE_COLUMNS)
    probs = [m.predict_proba(X)[0][1] for m in model_info["models"]]
    p_up = float(np.mean(probs))
    p_down = 1.0 - p_up
    signal = "CALL" if p_up >= MIN_PROBABILITY else "PUT" if p_down >= MIN_PROBABILITY else "WAIT"
    vol = float(latest["volatility"])
    regime = "HIGH VOL" if vol >= 0.0030 else "LOW VOL" if vol < 0.0015 else "NORMAL"
    return {
        "signal": signal, "p_up": p_up, "p_down": p_down, "confidence": max(p_up, p_down),
        "price": float(latest["close"]), "atr": float(latest["atr"]), "rsi": float(latest["rsi"]),
        "momentum": float(latest["momentum"]), "volatility": vol, "volume_z": float(latest["volume_z"]),
        "relative_strength": float(latest["relative_strength"]), "auc": model_info["auc"], "regime": regime
    }

def main():
    log(f"SPX & STOCKS AI ADVISOR {VERSION} STARTING")
    if not ALPACA_API_KEY or not TELEGRAM_BOT_TOKEN:
        log("Missing API keys or tokens.")
        return
    telegram_send(f"🤖 <b>AI ADVISOR {VERSION}</b>\n\n🚀 بدأ التشغيل واستقرار الخادم بنجاح.")
    
    while True:
        try:
            frames = fetch_all_bars(ALL_SYMBOLS)
            spy = frames.get("SPY")
            if spy is not None and not spy.empty:
                log("Data fetched successfully. Loop running...")
        except Exception as e:
            log(f"Error: {e}")
        time.sleep(POLL_SECONDS)

if __name__ == "__main__":
    main()
