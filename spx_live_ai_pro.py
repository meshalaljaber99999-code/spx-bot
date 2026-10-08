# ============================================================
# SPX & 10 STOCKS 0DTE AI ADVISOR / SCANNER v16.2 (FULL PRO)
# ============================================================
# Recommendation Only - NO ORDER EXECUTION
# ============================================================

import os
import time
import math
import warnings
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

VERSION = "v16.2-FULL-PRO"
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

OPTIONS_CONTRACTS_URL = f"{ALPACA_TRADING_URL}/v2/options/contracts"
OPTIONS_LATEST_QUOTES_URL = f"{ALPACA_DATA_URL}/v1beta1/options/quotes/latest"

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

def get_option_contracts(underlying, option_type, expiration):
    params = {"underlying_symbols": underlying, "status": "active", "expiration_date": expiration, "type": option_type.lower(), "limit": 10000}
    data = alpaca_get(OPTIONS_CONTRACTS_URL, params=params, timeout=30)
    if not data: return []
    return data.get("option_contracts") or data.get("contracts") or []

def get_option_quotes(symbols):
    if not symbols: return {}
    result = {}
    for i in range(0, len(symbols), 100):
        batch = symbols[i:i + 100]
        params = {"symbols": ",".join(batch), "feed": "indicative"}
        data = alpaca_get(OPTIONS_LATEST_QUOTES_URL, params=params, timeout=30)
        if not data: continue
        for symbol, q in data.get("quotes", {}).items():
            bid, ask = q.get("bp"), q.get("ap")
            if bid is None or ask is None: continue
            try:
                bid, ask = float(bid), float(ask)
            except Exception:
                continue
            if bid <= 0 or ask <= 0: continue
            mid = (bid + ask) / 2
            spread_pct = (ask - bid) / mid if mid > 0 else 999
            result[symbol] = {"bid": bid, "ask": ask, "mid": mid, "spread_pct": spread_pct}
    return result

def score_option(pred, contract, quote):
    if not pred or not contract or not quote: return 0
    score = 0.0
    score += min(30, max(0, (pred["confidence"] - 0.50) * 100) * 0.75)
    score += min(15, max(0, (pred["auc"] - 0.50) * 100) * 0.75)
    sp_pct = quote["spread_pct"]
    if sp_pct <= 0.05: score += 20
    elif sp_pct <= 0.08: score += 16
    elif sp_pct <= 0.12: score += 10
    elif sp_pct <= MAX_SPREAD_PERCENT: score += 4
    else: return 0
    if quote["mid"] < MIN_OPTION_PREMIUM: return 0
    score += 8 if quote["mid"] >= 1 else 5 if quote["mid"] >= 0.50 else 2
    score += 8 if abs(pred["momentum"]) >= 0.004 else 5 if abs(pred["momentum"]) >= 0.002 else 1
    score += 8 if pred["volume_z"] >= 2 else 5 if pred["volume_z"] >= 1 else 2
    return int(max(0, min(100, round(score))))

def choose_best_contract(symbol, signal, underlying_price):
    today = now_ny().date().isoformat()
    option_type = "call" if signal == "CALL" else "put"
    contracts = get_option_contracts(symbol, option_type, today)
    if not contracts: return None
    candidates = []
    for c in contracts:
        try:
            strike = float(c.get("strike_price"))
        except Exception:
            continue
        if not c.get("tradable", True): continue
        if symbol == "SPX":
            root = str(c.get("root_symbol", "")).upper()
            c_sym = str(c.get("symbol", "")).upper()
            if "SPXW" not in root and "SPXW" not in c_sym: continue
        distance = abs(strike - underlying_price)
        if distance > MAX_STRIKE_DISTANCE: continue
        candidates.append((distance, c))
    if not candidates: return None
    candidates.sort(key=lambda x: x[0])
    selected = [c for _, c in candidates[:25]]
    quotes = get_option_quotes([c.get("symbol") for c in selected if c.get("symbol")])
    best, best_score = None, -999
    for c in selected:
        sym = c.get("symbol")
        if sym not in quotes: continue
        q = quotes[sym]
        if q["mid"] < MIN_OPTION_PREMIUM or q["spread_pct"] > MAX_SPREAD_PERCENT: continue
        distance = abs(float(c["strike_price"]) - underlying_price)
        candidate_score = (20 - min(20, q["spread_pct"] * 100)) - (distance / max(underlying_price, 1) * 100)
        if candidate_score > best_score:
            best_score = candidate_score
            best = {"contract": c, "quote": q}
    return best

def build_opportunity(symbol, display_symbol, pred):
    if not pred or pred["signal"] not in ("CALL", "PUT"): return None
    contract_data = choose_best_contract(symbol, pred["signal"], pred["price"])
    if not contract_data: return None
    contract, quote = contract_data["contract"], contract_data["quote"]
    score = score_option(pred, contract, quote)
    if score < MIN_OPTION_SCORE: return None
    entry = quote["mid"]
    return {
        "symbol": display_symbol, "signal": pred["signal"], "confidence": pred["confidence"],
        "price": pred["price"], "auc": pred["auc"], "regime": pred["regime"], "score": score,
        "contract_symbol": contract.get("symbol", "UNKNOWN"), "strike": float(contract.get("strike_price", 0)),
        "expiration": contract.get("expiration_date", ""), "entry": entry, "target": entry * 1.40, "stop": entry * 0.70,
        "bid": quote["bid"], "ask": quote["ask"], "spread_pct": quote["spread_pct"]
    }

def market_consensus(predictions):
    valid = [p for p in predictions if p is not None]
    if not valid: return {"bias": "NEUTRAL", "strength": 0}
    call_c = sum(1 for p in valid if p["signal"] == "CALL")
    put_c = sum(1 for p in valid if p["signal"] == "PUT")
    call_p = np.mean([p["p_up"] for p in valid])
    put_p = np.mean([p["p_down"] for p in valid])
    if call_c > put_c: return {"bias": "BULLISH", "strength": float(call_p)}
    if put_c > call_c: return {"bias": "BEARISH", "strength": float(put_p)}
    return {"bias": "NEUTRAL", "strength": float(max(call_p, put_p))}

def format_opportunity(o, rank):
    emoji = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉"
    sig_emoji = "🟢" if o["signal"] == "CALL" else "🔴"
    return (
        f"{emoji} <b>{o['symbol']} {sig_emoji} {o['signal']}</b>\n"
        f"━━━━━━━━━━━━━━━━━━\n"
        f"🔥 القوة: <b>{o['score']}/100</b> | الثقة: <b>{o['confidence']*100:.1f}%</b>\n"
        f"📊 AUC: {o['auc']:.2f} | السعر: <b>${o['price']:.2f}</b>\n"
        f"🎯 السترايك: <b>{o['strike']:.2f}</b>\n"
        f"📜 العقد: <code>{o['contract_symbol']}</code>\n"
        f"💰 دخول: <b>${o['entry']:.2f}</b> | الهدف: <b>${o['target']:.2f}</b> | الوقف: <b>${o['stop']:.2f}</b>\n"
        f"↔️ السبريد: {o['spread_pct']*100:.1f}% | 🌡️ النظام: {o['regime']}\n"
    )

def is_duplicate(o):
    key = f"{o['contract_symbol']}_{o['signal']}"
    now = time.time()
    if key in STATE["last_sent"] and (now - STATE["last_sent"][key]) / 60 < SIGNAL_COOLDOWN_MINUTES:
        return True
    STATE["last_sent"][key] = now
    return False

def main():
    log(f"SPX & STOCKS AI ADVISOR {VERSION} STARTING")
    if not ALPACA_API_KEY or not TELEGRAM_BOT_TOKEN:
        log("Missing API keys or tokens.")
        return
    telegram_send(f"🤖 <b>AI ADVISOR {VERSION}</b>\n\n🚀 بدأ التشغيل الكامل وفحص الأوبشن بنجاح.")
    
    cycle = 0
    while True:
        cycle += 1
        log(f"\n========== SCAN #{cycle} ==========")
        try:
            market_ok, reason = market_open_now()
            if not market_ok:
                log(f"[MARKET] WAIT | {reason}")
                time.sleep(POLL_SECONDS)
                continue

            frames = fetch_all_bars(ALL_SYMBOLS)
            spy = frames.get("SPY")
            qqq = frames.get("QQQ")
            if spy is None or spy.empty:
                time.sleep(POLL_SECONDS)
                continue

            models = {}
            spx = spy.copy()
            for col in ["open", "high", "low", "close"]:
                spx[col] *= 10
            spx_feat = make_features(spx, market_df=spy, qqq_df=qqq)
            spx_model = train_model(spx_feat)
            if spx_model:
                models["SPX"] = {"features": spx_feat, "model": spx_model}

            for symbol in STOCKS:
                df = frames.get(symbol)
                if df is not None and not df.empty:
                    feat = make_features(df, market_df=spy, qqq_df=qqq)
                    model = train_model(feat)
                    if model:
                        models[symbol] = {"features": feat, "model": model}

            if not models:
                time.sleep(POLL_SECONDS)
                continue

            predictions, pred_map = [], {}
            for sym, info in models.items():
                pred = predict(info["model"], info["features"])
                if pred:
                    pred_map[sym] = pred
                    predictions.append(pred)

            consensus = market_consensus(predictions)
            opportunities = []

            for sym, pred in pred_map.items():
                if pred["signal"] in ("CALL", "PUT"):
                    underlying = "SPX" if sym == "SPX" else sym
                    opp = build_opportunity(underlying, sym, pred)
                    if opp and not is_duplicate(opp):
                        opportunities.append(opp)

            opportunities.sort(key=lambda x: (x["score"], x["confidence"]), reverse=True)
            opportunities = opportunities[:MAX_TELEGRAM_OPPORTUNITIES]

            if opportunities:
                msg_lines = [
                    "🚨 <b>AI OPTIONS & SPX ALERT</b>",
                    f"🕒 {now_ny().strftime('%H:%M:%S')} NY | الاتجاه: <b>{consensus['bias']}</b> ({consensus['strength']*100:.1f}%)\n",
                ]
                for i, o in enumerate(opportunities, 1):
                    msg_lines.append(format_opportunity(o, i))
                telegram_send("\n".join(msg_lines))
                log(f"[TELEGRAM] Sent {len(opportunities)} opportunities.")
            else:
                log("No high-quality option opportunity. Telegram silent.")

        except Exception as e:
            log(f"[MAIN ERROR] {e}")
        time.sleep(POLL_SECONDS)

if __name__ == "__main__":
    main()
