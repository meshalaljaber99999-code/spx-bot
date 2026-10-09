
# ============================================================
# SPX 0DTE AI ADVISOR v18.1
# REAL ^GSPC DATA | SPXW OPTIONS | TELEGRAM
# RECOMMENDATIONS ONLY — NO AUTOMATIC ORDER EXECUTION
# ============================================================

import os
import time
import logging
import warnings
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
import yfinance as yf

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer

warnings.filterwarnings("ignore")

# ========================= CONFIG ============================

NY = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")

ALPACA_KEY = (
    os.getenv("ALPACA_API_KEY")
    or os.getenv("APCA_API_KEY_ID")
    or ""
)
ALPACA_SECRET = (
    os.getenv("ALPACA_SECRET_KEY")
    or os.getenv("APCA_API_SECRET_KEY")
    or ""
)

ALPACA_DATA_URL = os.getenv(
    "ALPACA_DATA_URL",
    "https://data.alpaca.markets"
).rstrip("/")

ALPACA_TRADE_URL = os.getenv(
    "ALPACA_TRADE_URL",
    "https://paper-api.alpaca.markets"
).rstrip("/")

STOCK_FEED = os.getenv("ALPACA_DATA_FEED", "iex")
OPTIONS_FEED = os.getenv("ALPACA_OPTIONS_FEED", "indicative")

TG_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TG_CHAT = os.getenv("TELEGRAM_CHAT_ID", "")

POLL_SECONDS = int(os.getenv("POLL_SECONDS", "60"))
MAX_SPX_AGE_MIN = float(os.getenv("MAX_SPX_AGE_MIN", "12"))
MAX_STOCK_AGE_MIN = float(os.getenv("MAX_STOCK_AGE_MIN", "15"))

MIN_MODEL_AUC = float(os.getenv("MIN_MODEL_AUC", "0.52"))
MIN_MODEL_CONFIDENCE = float(
    os.getenv("MIN_MODEL_CONFIDENCE", "0.56")
)

MIN_CONFIRMATIONS = int(os.getenv("MIN_CONFIRMATIONS", "2"))
MIN_RR = float(os.getenv("MIN_RR", "1.25"))

# عدد الشموع المستقبلية المستخدمة في تعريف الهدف التدريبي.
LABEL_HORIZON = 3

STOCKS = [
    "SPY", "QQQ", "NVDA", "AAPL", "MSFT",
    "AMD", "AMZN", "META", "GOOGL", "TSLA"
]

FEATURES = [
    "ret_1", "ret_3", "ret_6",
    "rsi", "ema_spread", "ema_long_spread",
    "atr_pct", "range_pct", "body_ratio",
    "volatility", "momentum_accel"
]

# ========================= LOGGING ===========================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s EDT | %(message)s"
)
log = logging.getLogger("SPX_ADVISOR")


def logmsg(message):
    log.info(message)


# ========================= TIME ==============================

def now_ny():
    return datetime.now(NY)


def today_ny():
    return now_ny().date().isoformat()


def normalize_time(df, column="timestamp"):
    """
    توحيد التوقيت إلى datetime64[ns, UTC].
    هذا يعالج اختلاف datetime64[s, UTC] و datetime64[us, UTC].
    """
    if df is None or df.empty or column not in df.columns:
        return df

    out = df.copy()
    out[column] = pd.to_datetime(
        out[column], utc=True, errors="coerce"
    )

    out = out.dropna(subset=[column])

    # تثبيت الدقة لتفادي MergeError بين مصادر البيانات.
    out[column] = out[column].astype("datetime64[ns, UTC]")

    return out.sort_values(column).reset_index(drop=True)


def market_is_open():
    now = now_ny()

    if now.weekday() >= 5:
        return False

    open_time = now.replace(
        hour=9, minute=30, second=0, microsecond=0
    )
    close_time = now.replace(
        hour=16, minute=0, second=0, microsecond=0
    )

    return open_time <= now <= close_time


# ========================= TELEGRAM ==========================

def send_telegram(message):
    if not TG_TOKEN or not TG_CHAT:
        logmsg("Telegram not configured; message printed locally.")
        logmsg(message.replace("\n", " | "))
        return False

    url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"

    try:
        response = requests.post(
            url,
            json={
                "chat_id": TG_CHAT,
                "text": message,
                "disable_web_page_preview": True
            },
            timeout=15
        )

        if response.status_code != 200:
            logmsg(
                f"Telegram error {response.status_code}: "
                f"{response.text[:300]}"
            )
            return False

        return True

    except Exception as exc:
        logmsg(f"Telegram exception: {exc}")
        return False


# ========================= ALPACA REQUESTS ===================

def alpaca_headers():
    return {
        "APCA-API-KEY-ID": ALPACA_KEY,
        "APCA-API-SECRET-KEY": ALPACA_SECRET
    }


def alpaca_get(url, params=None, timeout=20):
    if not ALPACA_KEY or not ALPACA_SECRET:
        raise RuntimeError(
            "Missing Alpaca keys. Set ALPACA_API_KEY and "
            "ALPACA_SECRET_KEY in your deployment environment."
        )

    response = requests.get(
        url,
        headers=alpaca_headers(),
        params=params,
        timeout=timeout
    )

    if response.status_code >= 400:
        raise RuntimeError(
            f"Alpaca HTTP {response.status_code}: "
            f"{response.text[:500]}"
        )

    return response.json()


# ========================= REAL SPX DATA ====================

def fetch_real_spx():
    """
    يستخدم ^GSPC من Yahoo Finance.
    هذه بيانات المؤشر، وليست أسعار عقود SPXW.
    قد تتأخر البيانات أو تتوقف حسب توفر المصدر.
    """
    ticker = yf.Ticker("^GSPC")

    raw = ticker.history(
        period="60d",
        interval="5m",
        auto_adjust=False,
        prepost=False
    )

    if raw is None or raw.empty:
        raise RuntimeError("Yahoo returned no ^GSPC bars.")

    raw = raw.reset_index()

    time_col = "Datetime" if "Datetime" in raw.columns else "Date"

    raw = raw.rename(columns={
        time_col: "timestamp",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume"
    })

    raw["timestamp"] = pd.to_datetime(
        raw["timestamp"], utc=True, errors="coerce"
    )

    for col in ["open", "high", "low", "close", "volume"]:
        raw[col] = pd.to_numeric(raw[col], errors="coerce")

    raw = raw.dropna(
        subset=["timestamp", "open", "high", "low", "close"]
    )

    raw = normalize_time(raw)

    if raw.empty:
        raise RuntimeError("No valid ^GSPC rows after normalization.")

    latest = raw["timestamp"].iloc[-1]
    age = (
        pd.Timestamp.now(tz="UTC") - latest
    ).total_seconds() / 60.0

    logmsg(
        f"[SPX DATA] ^GSPC bars={len(raw)} | "
        f"last={latest} | age={age:.1f}m"
    )

    if age > MAX_SPX_AGE_MIN:
        raise RuntimeError(
            f"SPX data stale: age={age:.1f} minutes"
        )

    return raw


# ========================= STOCK DATA ========================

def fetch_stock_bars(symbol, limit=1000):
    url = f"{ALPACA_DATA_URL}/v2/stocks/{symbol}/bars"

    end = datetime.now(UTC)
    start = end - timedelta(days=10)

    payload = alpaca_get(
        url,
        params={
            "timeframe": "5Min",
            "start": start.isoformat(),
            "end": end.isoformat(),
            "limit": limit,
            "adjustment": "raw",
            "feed": STOCK_FEED,
            "sort": "asc"
        }
    )

    rows = payload.get("bars", [])

    if not rows:
        raise RuntimeError(
            f"No stock bars for {symbol}. "
            f"Check Alpaca data-feed permissions."
        )

    df = pd.DataFrame(rows)

    df = df.rename(columns={
        "t": "timestamp",
        "o": "open",
        "h": "high",
        "l": "low",
        "c": "close",
        "v": "volume"
    })

    df = normalize_time(df)

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df.dropna(
        subset=["open", "high", "low", "close"]
    ).reset_index(drop=True)


# ========================= FEATURES ==========================

def add_indicators(df):
    """
    حساب المؤشرات على بيانات مصدر واحد.
    جميع العمليات الزمنية تستخدم timestamp موحدًا.
    """
    df = normalize_time(df)

    if df is None or len(df) < 80:
        raise RuntimeError("Insufficient bars to calculate indicators.")

    out = df.copy()

    close = out["close"].astype(float)
    high = out["high"].astype(float)
    low = out["low"].astype(float)
    open_ = out["open"].astype(float)

    out["ret_1"] = close.pct_change(1)
    out["ret_3"] = close.pct_change(3)
    out["ret_6"] = close.pct_change(6)

    delta = close.diff()
    gain = delta.clip(lower=0).ewm(
        alpha=1 / 14, adjust=False
    ).mean()
    loss = (-delta.clip(upper=0)).ewm(
        alpha=1 / 14, adjust=False
    ).mean()

    rs = gain / loss.replace(0, np.nan)
    out["rsi"] = 100 - (100 / (1 + rs))

    ema9 = close.ewm(span=9, adjust=False).mean()
    ema21 = close.ewm(span=21, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()

    out["ema_spread"] = (ema9 - ema21) / close
    out["ema_long_spread"] = (ema21 - ema50) / close

    previous_close = close.shift(1)

    tr = pd.concat([
        high - low,
        (high - previous_close).abs(),
        (low - previous_close).abs()
    ], axis=1).max(axis=1)

    atr = tr.rolling(14).mean()
    out["atr_pct"] = atr / close
    out["range_pct"] = (high - low) / close

    candle_range = (high - low).replace(0, np.nan)
    out["body_ratio"] = (close - open_).abs() / candle_range

    out["volatility"] = out["ret_1"].rolling(15).std()
    out["momentum_accel"] = out["ret_3"].diff(3)

    return out.replace([np.inf, -np.inf], np.nan)


# ========================= FIXED MERGE =======================

def make_features(spx_df, spy_df, qqq_df):
    """
    يدمج SPX مع SPY وQQQ باستخدام merge_asof.
    قبل الدمج:
      1) تحويل التوقيت إلى UTC.
      2) توحيد الدقة إلى ns.
      3) حذف التوقيتات المفقودة.
      4) ترتيب البيانات تصاعديًا.
    """
    spx = add_indicators(spx_df)
    spy = add_indicators(spy_df)
    qqq = add_indicators(qqq_df)

    spx = normalize_time(spx)
    spy = normalize_time(spy)
    qqq = normalize_time(qqq)

    # اختيار أعمدة السوق التي نحتاجها فقط لتجنب تضارب الأسماء.
    spy_cols = spy[[
        "timestamp", "ret_1", "ret_3",
        "ema_spread", "rsi"
    ]].rename(columns={
        "ret_1": "spy_ret_1",
        "ret_3": "spy_ret_3",
        "ema_spread": "spy_ema_spread",
        "rsi": "spy_rsi"
    })

    qqq_cols = qqq[[
        "timestamp", "ret_1", "ret_3",
        "ema_spread", "rsi"
    ]].rename(columns={
        "ret_1": "qqq_ret_1",
        "ret_3": "qqq_ret_3",
        "ema_spread": "qqq_ema_spread",
        "rsi": "qqq_rsi"
    })

    # تثبيت النوع قبل كل عملية merge_asof.
    for frame in (spx, spy_cols, qqq_cols):
        frame["timestamp"] = pd.to_datetime(
            frame["timestamp"], utc=True
        ).astype("datetime64[ns, UTC]")
        frame.sort_values("timestamp", inplace=True)
        frame.reset_index(drop=True, inplace=True)

    merged = pd.merge_asof(
        spx,
        spy_cols,
        on="timestamp",
        direction="backward",
        tolerance=pd.Timedelta("15min")
    )

    merged["timestamp"] = pd.to_datetime(
        merged["timestamp"], utc=True
    ).astype("datetime64[ns, UTC]")

    qqq_cols["timestamp"] = pd.to_datetime(
        qqq_cols["timestamp"], utc=True
    ).astype("datetime64[ns, UTC]")

    merged = pd.merge_asof(
        merged.sort_values("timestamp"),
        qqq_cols.sort_values("timestamp"),
        on="timestamp",
        direction="backward",
        tolerance=pd.Timedelta("15min")
    )

    merged = merged.replace([np.inf, -np.inf], np.nan)
    merged = merged.sort_values("timestamp").reset_index(drop=True)

    return merged


# ========================= ML TRAINING =======================

def train_model(features_df):
    """
    نموذج تصنيف اتجاهي بسيط.
    لا يمثل ضمانًا للربح، ولا يستخدم تقسيمًا عشوائيًا للزمن.
    """
    df = features_df.copy()

    # الهدف: هل الإغلاق بعد LABEL_HORIZON شموع أعلى من الحالي؟
    future_close = df["close"].shift(-LABEL_HORIZON)
    df["target"] = np.where(
        future_close.notna(),
        (future_close > df["close"]).astype(int),
        np.nan
    )

    # إزالة آخر الصفوف التي لا تملك هدفًا مستقبليًا.
    df = df.iloc[:-LABEL_HORIZON].copy()

    feature_cols = FEATURES

    df = df.dropna(subset=feature_cols + ["target"])

    if len(df) < 250:
        raise RuntimeError(
            f"Not enough clean training rows: {len(df)}"
        )

    X = df[feature_cols].astype(float)
    y = df["target"].astype(int)

    split = int(len(df) * 0.80)

    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    if y_train.nunique() < 2 or y_test.nunique() < 2:
        raise RuntimeError(
            "Training/test labels contain only one class."
        )

    model = make_pipeline(
        SimpleImputer(strategy="median"),
        HistGradientBoostingClassifier(
            max_iter=120,
            learning_rate=0.06,
            max_leaf_nodes=15,
            l2_regularization=1.0,
            random_state=42
        )
    )

    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    accuracy = accuracy_score(y_test, predictions)
    auc = roc_auc_score(y_test, probabilities)

    logmsg(
        f"[MODEL] rows={len(df)} | "
        f"test_accuracy={accuracy:.3f} | test_auc={auc:.3f}"
    )

    return model, feature_cols, float(auc), float(accuracy)


# ========================= MARKET CONFIRMATION ===============

def get_confirmation(spx_row, spy_row, qqq_row):
    votes = []

    def vote(row, label):
        rsi = float(row.get("rsi", 50))
        ema = float(row.get("ema_spread", 0))
        ret = float(row.get("ret_3", 0))

        if ema > 0 and ret > 0 and rsi >= 50:
            votes.append((label, "BULLISH"))
        elif ema < 0 and ret < 0 and rsi <= 50:
            votes.append((label, "BEARISH"))
        else:
            votes.append((label, "NEUTRAL"))

    vote(spx_row, "SPX")
    vote(spy_row, "SPY")
    vote(qqq_row, "QQQ")

    bullish = sum(v == "BULLISH" for _, v in votes)
    bearish = sum(v == "BEARISH" for _, v in votes)

    return votes, bullish, bearish


# ========================= SPXW CONTRACTS ====================

def get_spxw_contracts(option_type):
    """
    يحاول جلب عقود SPXW من واجهة Alpaca.
    قد لا تكون عقود SPX/SPXW متاحة لحسابك أو عبر هذا المزود.
    لا يستخدم SPY كبديل لعقد SPXW.
    """
    url = f"{ALPACA_TRADE_URL}/v2/options/contracts"

    params = {
        "underlying_symbols": "SPX",
        "expiration_date": today_ny(),
        "type": option_type,
        "status": "active",
        "limit": 1000
    }

    payload = alpaca_get(url, params=params)
    contracts = payload.get("option_contracts", [])

    results = []

    for contract in contracts:
        symbol = str(contract.get("symbol", "")).upper()
        root = str(contract.get("root_symbol", "")).upper()
        underlying = str(
            contract.get("underlying_symbol", "SPX")
        ).upper()

        if underlying != "SPX":
            continue

        # نريد SPXW فقط، لا عقود SPX القياسية.
        if root == "SPXW" or symbol.startswith("SPXW"):
            results.append(contract)

    return results


def get_option_quote(symbol):
    url = f"{ALPACA_DATA_URL}/v1beta1/options/quotes/latest"

    payload = alpaca_get(
        url,
        params={
            "symbols": symbol,
            "feed": OPTIONS_FEED
        }
    )

    quotes = payload.get("quotes", {})
    quote = quotes.get(symbol)

    if quote is None and quotes:
        quote = next(iter(quotes.values()))

    if not quote:
        return None

    bid = float(quote.get("bp", 0) or 0)
    ask = float(quote.get("ap", 0) or 0)
    quote_time = quote.get("t")

    if bid <= 0 or ask <= 0 or ask < bid:
        return None

    if quote_time:
        qtime = pd.to_datetime(quote_time, utc=True)
        age = (
            pd.Timestamp.now(tz="UTC") - qtime
        ).total_seconds()

        if age > 180:
            logmsg(
                f"[OPTIONS] stale quote {symbol}: age={age:.0f}s"
            )
            return None

    return {
        "symbol": symbol,
        "bid": bid,
        "ask": ask,
        "mid": (bid + ask) / 2.0,
        "spread_pct": (ask - bid) / max((ask + bid) / 2.0, 0.01),
        "timestamp": quote_time
    }


def choose_contract(direction):
    option_type = "call" if direction == "CALL" else "put"

    try:
        contracts = get_spxw_contracts(option_type)
    except Exception as exc:
        logmsg(f"[OPTIONS] Contract lookup failed: {exc}")
        return None

    if not contracts:
        logmsg(
            "[OPTIONS] No SPXW contracts returned. "
            "Check account/API support and contract endpoint filters."
        )
        return None

    # اختيار عقد قريب من السعر الحالي قدر الإمكان.
    # إذا لم توفر البيانات strike واضحًا، لا نخترع قيمة.
    candidates = []

    for contract in contracts:
        symbol = contract.get("symbol")
        if not symbol:
            continue

        try:
            strike = float(contract.get("strike_price"))
        except (TypeError, ValueError):
            continue

        try:
            quote = get_option_quote(symbol)
        except Exception as exc:
            logmsg(f"[OPTIONS] Quote failed for {symbol}: {exc}")
            continue

        if not quote:
            continue

        # نرفض الفارق السعري الكبير.
        if quote["spread_pct"] > 0.20:
            continue

        candidates.append((strike, quote))

    if not candidates:
        logmsg("[OPTIONS] No suitable quoted SPXW contracts.")
        return None

    # لا يوجد هنا سعر SPX حالي لفرض strike معيّن؛
    # نرتب وفق ضيق السبريد والسيولة السعرية المتاحة فقط.
    candidates.sort(
        key=lambda item: (
            item[1]["spread_pct"],
            item[1]["mid"]
        )
    )

    return candidates[0][1]


# ========================= SIGNAL ENGINE =====================

def calculate_signal(model, feature_cols, auc, features_df,
                     spy_df, qqq_df):

    if auc < MIN_MODEL_AUC:
        return {
            "direction": "WAIT",
            "reason": f"Model AUC below threshold ({auc:.3f})"
        }

    latest = features_df.iloc[-1]

    if latest[feature_cols].isna().any():
        return {
            "direction": "WAIT",
            "reason": "Latest feature row contains missing values"
        }

    X_latest = latest[feature_cols].astype(float).to_frame().T
    prob_up = float(model.predict_proba(X_latest)[0, 1])
    prob_down = 1.0 - prob_up

    spy = add_indicators(spy_df).iloc[-1]
    qqq = add_indicators(qqq_df).iloc[-1]

    votes, bullish, bearish = get_confirmation(
        latest, spy, qqq
    )

    logmsg(
        f"[MODEL] prob_up={prob_up:.3f} | "
        f"prob_down={prob_down:.3f} | AUC={auc:.3f}"
    )

    logmsg(
        "[CONFIRMATION] " +
        " | ".join(f"{name}:{value}" for name, value in votes)
    )

    if (
        prob_up >= MIN_MODEL_CONFIDENCE
        and bullish >= MIN_CONFIRMATIONS
    ):
        return {
            "direction": "CALL",
            "probability": prob_up,
            "auc": auc,
            "votes": votes,
            "reason": "Model and market confirmation bullish"
        }

    if (
        prob_down >= MIN_MODEL_CONFIDENCE
        and bearish >= MIN_CONFIRMATIONS
    ):
        return {
            "direction": "PUT",
            "probability": prob_down,
            "auc": auc,
            "votes": votes,
            "reason": "Model and market confirmation bearish"
        }

    return {
        "direction": "WAIT",
        "probability": max(prob_up, prob_down),
        "auc": auc,
        "votes": votes,
        "reason": "No aligned model + confirmation signal"
    }


# ========================= MAIN SCAN ========================

def run_scan():
    logmsg("[SCAN] Starting scan.")

    if not market_is_open():
        logmsg("[MARKET] WAIT | السوق خارج ساعات التداول المعتادة.")
        return

    spx = fetch_real_spx()

    stock_data = {}

    for symbol in ["SPY", "QQQ"]:
        try:
            df = fetch_stock_bars(symbol)
            latest_time = df["timestamp"].iloc[-1]
            age = (
                pd.Timestamp.now(tz="UTC") - latest_time
            ).total_seconds() / 60.0

            logmsg(
                f"[DATA] {symbol}: {len(df)} bars | "
                f"last={latest_time} | age={age:.1f}m"
            )

            if age > MAX_STOCK_AGE_MIN:
                raise RuntimeError(
                    f"{symbol} bars stale: {age:.1f} minutes"
                )

            stock_data[symbol] = df

        except Exception as exc:
            logmsg(f"[DATA ERROR] {symbol}: {exc}")
            logmsg("[SCAN] WAIT | Market confirmation data unavailable.")
            return

    try:
        merged = make_features(
            spx,
            stock_data["SPY"],
            stock_data["QQQ"]
        )

        model, feature_cols, auc, accuracy = train_model(merged)

        signal = calculate_signal(
            model,
            feature_cols,
            auc,
            merged,
            stock_data["SPY"],
            stock_data["QQQ"]
        )

    except Exception as exc:
        logmsg(f"[ANALYSIS ERROR] {type(exc).__name__}: {exc}")
        return

    direction = signal.get("direction", "WAIT")

    if direction == "WAIT":
        logmsg(f"[SIGNAL] WAIT | {signal.get('reason', '')}")
        return

    contract = choose_contract(direction)

    if not contract:
        logmsg(
            f"[SIGNAL] {direction} detected, but no verified "
            "SPXW quote was available. No contract recommendation sent."
        )
        return

    probability = signal.get("probability", 0.0)

    message = (
        f"📊 SPX 0DTE AI ADVISOR v18.1\n\n"
        f"الاتجاه: {direction}\n"
        f"العقد: {contract['symbol']}\n"
        f"Bid: {contract['bid']:.2f}\n"
        f"Ask: {contract['ask']:.2f}\n"
        f"Mid: {contract['mid']:.2f}\n"
        f"Spread: {contract['spread_pct'] * 100:.1f}%\n\n"
        f"Model probability: {probability * 100:.1f}%\n"
        f"Test AUC: {signal.get('auc', 0):.3f}\n"
        f"Reason: {signal.get('reason', '')}\n\n"
        f"⚠️ توصية آلية تجريبية، وليست ضمانًا للربح.\n"
        f"لا يتم تنفيذ أي أمر شراء أو بيع تلقائيًا."
    )

    logmsg(
        f"[SIGNAL] {direction} | contract={contract['symbol']} | "
        f"mid={contract['mid']:.2f}"
    )

    send_telegram(message)


# ========================= STARTUP ===========================

def main():
    logmsg("=" * 60)
    logmsg("SPX 0DTE AI ADVISOR v18.1 STARTING")
    logmsg("Real ^GSPC source | SPXW contract attempt")
    logmsg("Recommendation only | No order execution")
    logmsg(f"Stock feed={STOCK_FEED} | Options feed={OPTIONS_FEED}")
    logmsg("=" * 60)

    if not ALPACA_KEY or not ALPACA_SECRET:
        raise RuntimeError(
            "Missing Alpaca keys. Add ALPACA_API_KEY and "
            "ALPACA_SECRET_KEY to the deployment environment."
        )

    while True:
        try:
            run_scan()
        except KeyboardInterrupt:
            logmsg("Stopped by user.")
            break
        except Exception as exc:
            logmsg(
                f"[MAIN ERROR] {type(exc).__name__}: {exc}"
            )

        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
