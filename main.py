
# ============================================================
# SPX 0DTE AI ADVISOR v18.2 ARABIC
# توصيات خيارات SPX فقط — لا ينفذ أوامر شراء أو بيع
# جميع رسائل تيليجرام باللغة العربية
# ============================================================

import os
import time
import logging
import warnings
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
import yfinance as yf

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, roc_auc_score

warnings.filterwarnings("ignore")

# ======================== الإعدادات =========================

NY = ZoneInfo("America/New_York")
UTC = timezone.utc

API_KEY = (
    os.getenv("ALPACA_API_KEY")
    or os.getenv("APCA_API_KEY_ID")
    or ""
).strip()

API_SECRET = (
    os.getenv("ALPACA_SECRET_KEY")
    or os.getenv("APCA_API_SECRET_KEY")
    or ""
).strip()

# الوضع الافتراضي: حساب Alpaca التجريبي
TRADE_URL = os.getenv(
    "ALPACA_TRADE_URL",
    "https://paper-api.alpaca.markets"
).rstrip("/")

DATA_URL = os.getenv(
    "ALPACA_DATA_URL",
    "https://data.alpaca.markets"
).rstrip("/")

DATA_FEED = os.getenv("ALPACA_DATA_FEED", "iex").lower()
OPTIONS_FEED = os.getenv("ALPACA_OPTIONS_FEED", "indicative").lower()

TG_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TG_CHAT = os.getenv("TELEGRAM_CHAT_ID", "").strip()

MIN_AUC = float(os.getenv("MIN_MODEL_AUC", "0.52"))
MIN_CONFIDENCE = float(os.getenv("MIN_MODEL_CONFIDENCE", "0.56"))
MAX_SPREAD_PCT = float(os.getenv("MAX_OPTION_SPREAD_PCT", "0.20"))

SCAN_SECONDS = max(20, int(os.getenv("SCAN_SECONDS", "60")))

LABEL_HORIZON = 3
MAX_BARS = 5000
MAX_CONTRACTS_TO_CHECK = 15

FEATURES = [
    "ret_1", "ret_3", "ret_6",
    "rsi", "ema_spread", "ema_long_spread",
    "atr_pct", "range_pct", "body_ratio",
    "volatility", "momentum_accel",
    "spy_ret_1", "spy_ret_3", "spy_ema_spread", "spy_rsi",
    "qqq_ret_1", "qqq_ret_3", "qqq_ema_spread", "qqq_rsi",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

HTTP = requests.Session()


def log(message, level="info"):
    getattr(logging, level, logging.info)(message)


def utc_now():
    return datetime.now(UTC)


def ny_now():
    return datetime.now(NY)


def alpaca_headers():
    if not API_KEY or not API_SECRET:
        raise RuntimeError(
            "مفاتيح Alpaca غير موجودة. "
            "أضف ALPACA_API_KEY وALPACA_SECRET_KEY "
            "في متغيرات Railway."
        )

    return {
        "APCA-API-KEY-ID": API_KEY,
        "APCA-API-SECRET-KEY": API_SECRET,
    }


def normalize_bars(df):
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(
            df["timestamp"], utc=True, errors="coerce"
        )
        df = df.dropna(subset=["timestamp"])
        df = df.set_index("timestamp")
    else:
        df.index = pd.to_datetime(
            df.index, utc=True, errors="coerce"
        )
        df = df.loc[~df.index.isna()]

    df.index = pd.DatetimeIndex(
        pd.to_datetime(df.index, utc=True)
    )
    df.index.name = "timestamp"
    df = df[~df.index.duplicated(keep="last")].sort_index()

    for col in ["open", "high", "low", "close", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    required = [
        col for col in ["open", "high", "low", "close"]
        if col in df.columns
    ]

    if required:
        df = df.dropna(subset=required)

    return df


# ======================== بيانات السوق ======================

def get_spx_bars():
    """جلب بيانات مؤشر SPX من Yahoo Finance."""
    try:
        raw = yf.download(
            "^GSPC",
            period="60d",
            interval="5m",
            auto_adjust=False,
            progress=False,
            threads=False,
        )

        if raw is None or raw.empty:
            raise RuntimeError(
                "لم تصل بيانات مؤشر SPX من Yahoo Finance"
            )

        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)

        raw = raw.rename(columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        })

        raw = normalize_bars(raw)
        raw = raw[["open", "high", "low", "close"]].tail(MAX_BARS)

        if len(raw) < 300:
            raise RuntimeError(
                f"عدد شموع SPX غير كافٍ: {len(raw)}"
            )

        age = (
            utc_now() - raw.index[-1].to_pydatetime()
        ).total_seconds() / 60

        log(
            f"[بيانات SPX] عدد الشموع={len(raw)} "
            f"| آخر شمعة={raw.index[-1]} "
            f"| عمر البيانات={age:.1f} دقيقة"
        )

        return raw

    except Exception as exc:
        log(f"[خطأ بيانات SPX] {exc}", "error")
        return pd.DataFrame()


def get_stock_bars(symbol):
    """جلب شموع SPY أو QQQ من Alpaca."""
    try:
        end = utc_now()
        start = end - timedelta(days=10)

        params = {
            "timeframe": "5Min",
            "start": start.isoformat().replace("+00:00", "Z"),
            "end": end.isoformat().replace("+00:00", "Z"),
            "limit": 10000,
            "adjustment": "raw",
            "feed": DATA_FEED,
            "sort": "asc",
        }

        response = HTTP.get(
            f"{DATA_URL}/v2/stocks/{symbol}/bars",
            headers=alpaca_headers(),
            params=params,
            timeout=25,
        )

        if response.status_code >= 400:
            raise RuntimeError(
                f"خطأ Alpaca HTTP {response.status_code}: "
                f"{response.text[:250]}"
            )

        rows = response.json().get("bars", [])

        if not rows:
            raise RuntimeError(
                f"لا توجد بيانات {symbol}؛ مصدر البيانات={DATA_FEED}"
            )

        df = pd.DataFrame(rows).rename(columns={
            "t": "timestamp",
            "o": "open",
            "h": "high",
            "l": "low",
            "c": "close",
            "v": "volume",
        })

        df = normalize_bars(df)
        df = df[["open", "high", "low", "close"]].tail(MAX_BARS)

        if len(df) < 100:
            raise RuntimeError(
                f"عدد شموع {symbol} غير كافٍ: {len(df)}"
            )

        age = (
            utc_now() - df.index[-1].to_pydatetime()
        ).total_seconds() / 60

        log(
            f"[البيانات] {symbol}: عدد الشموع={len(df)} "
            f"| آخر شمعة={df.index[-1]} "
            f"| عمر البيانات={age:.1f} دقيقة"
        )

        return df

    except Exception as exc:
        log(f"[خطأ بيانات {symbol}] {exc}", "error")
        return pd.DataFrame()


# ======================== المؤشرات الفنية ===================

def calc_rsi(close, period=14):
    delta = close.diff()

    gain = delta.clip(lower=0).ewm(
        alpha=1 / period,
        min_periods=period,
        adjust=False,
    ).mean()

    loss = (-delta.clip(upper=0)).ewm(
        alpha=1 / period,
        min_periods=period,
        adjust=False,
    ).mean()

    rs = gain / loss.replace(0, np.nan)

    return 100 - (100 / (1 + rs))


def build_features(bars, prefix=""):
    df = normalize_bars(bars)

    if df.empty:
        return pd.DataFrame()

    o = df["open"]
    h = df["high"]
    low = df["low"]
    c = df["close"]

    result = pd.DataFrame(index=df.index)

    result[f"{prefix}ret_1"] = c.pct_change(1)
    result[f"{prefix}ret_3"] = c.pct_change(3)
    result[f"{prefix}ret_6"] = c.pct_change(6)

    result[f"{prefix}rsi"] = calc_rsi(c)

    ema9 = c.ewm(span=9, adjust=False).mean()
    ema21 = c.ewm(span=21, adjust=False).mean()
    ema50 = c.ewm(span=50, adjust=False).mean()

    result[f"{prefix}ema_spread"] = (
        (ema9 - ema21) / c.replace(0, np.nan)
    )

    result[f"{prefix}ema_long_spread"] = (
        (ema21 - ema50) / c.replace(0, np.nan)
    )

    previous_close = c.shift(1)

    true_range = pd.concat([
        h - low,
        (h - previous_close).abs(),
        (low - previous_close).abs(),
    ], axis=1).max(axis=1)

    atr = true_range.rolling(14).mean()

    result[f"{prefix}atr_pct"] = (
        atr / c.replace(0, np.nan)
    )

    result[f"{prefix}range_pct"] = (
        (h - low) / c.replace(0, np.nan)
    )

    result[f"{prefix}body_ratio"] = (
        (c - o) / (h - low).replace(0, np.nan)
    )

    result[f"{prefix}volatility"] = (
        c.pct_change().rolling(20).std()
    )

    result[f"{prefix}momentum_accel"] = (
        c.pct_change().diff(3)
    )

    return result.replace([np.inf, -np.inf], np.nan)


def make_features(spx, spy, qqq):
    spx = normalize_bars(spx)
    spy = normalize_bars(spy)
    qqq = normalize_bars(qqq)

    if spx.empty or spy.empty or qqq.empty:
        raise RuntimeError(
            "بيانات SPX أو SPY أو QQQ غير متوفرة"
        )

    spx_features = build_features(spx)
    spy_features = build_features(spy, "spy_")
    qqq_features = build_features(qqq, "qqq_")

    spy_features = spy_features[
        [
            "spy_ret_1", "spy_ret_3",
            "spy_ema_spread", "spy_rsi",
        ]
    ]

    qqq_features = qqq_features[
        [
            "qqq_ret_1", "qqq_ret_3",
            "qqq_ema_spread", "qqq_rsi",
        ]
    ]

    for frame in [spx_features, spy_features, qqq_features]:
        frame.index = pd.DatetimeIndex(
            pd.to_datetime(frame.index, utc=True)
        )
        frame.index.name = "timestamp"

    merged = spx_features.join(
        spy_features, how="inner"
    ).join(
        qqq_features, how="inner"
    )

    merged = merged.join(spx[["close"]], how="inner")
    merged = merged.replace([np.inf, -np.inf], np.nan)
    merged = merged.dropna(subset=FEATURES + ["close"])

    if len(merged) < 400:
        raise RuntimeError(
            f"عدد صفوف البيانات المتطابقة غير كافٍ: {len(merged)}"
        )

    return merged.sort_index()


# ======================== التعلم الآلي ======================

def new_model():
    return make_pipeline(
        SimpleImputer(strategy="median"),
        HistGradientBoostingClassifier(
            max_iter=120,
            learning_rate=0.06,
            max_leaf_nodes=15,
            l2_regularization=1.0,
            random_state=42,
        ),
    )


def train_model(df):
    feature_cols = FEATURES.copy()
    data = df.sort_index().copy()

    future_close = data["close"].shift(-LABEL_HORIZON)
    valid = future_close.notna()

    X = data.loc[valid, feature_cols].replace(
        [np.inf, -np.inf], np.nan
    )

    y = (
        future_close.loc[valid] > data.loc[valid, "close"]
    ).astype(int)

    if len(X) < 500:
        raise RuntimeError(
            f"بيانات التدريب غير كافية: {len(X)} صفًا"
        )

    # تقسيم زمني مع استبعاد فترة أفق التنبؤ قبل الاختبار
    split = int(len(X) * 0.80)
    train_end = split - LABEL_HORIZON

    if train_end < 200 or len(X) - split < 100:
        raise RuntimeError("بيانات التدريب أو الاختبار غير كافية")

    X_train = X.iloc[:train_end]
    y_train = y.iloc[:train_end]

    X_test = X.iloc[split:]
    y_test = y.iloc[split:]

    if y_train.nunique() < 2 or y_test.nunique() < 2:
        raise RuntimeError(
            "بيانات التدريب أو الاختبار تحتوي على فئة واحدة فقط"
        )

    evaluator = new_model()
    evaluator.fit(X_train, y_train)

    probabilities = evaluator.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    accuracy = float(
        accuracy_score(y_test, predictions)
    )

    auc = float(
        roc_auc_score(y_test, probabilities)
    )

    baseline_class = int(y_train.mean() >= 0.5)

    baseline_accuracy = float(
        (y_test == baseline_class).mean()
    )

    log(
        f"[النموذج] الصفوف={len(df)} "
        f"| التدريب={len(X_train)} "
        f"| الاختبار={len(X_test)} "
        f"| الدقة={accuracy:.3f} "
        f"| AUC={auc:.3f} "
        f"| خط الأساس={baseline_accuracy:.3f}"
    )

    # تدريب النموذج النهائي على كل الصفوف المتاحة
    live_model = new_model()
    live_model.fit(X, y)

    return live_model, feature_cols, auc, accuracy


def calculate_signal(df, model, feature_cols, auc):
    if auc < MIN_AUC:
        return (
            "WAIT",
            0.0,
            f"تقييم AUC={auc:.3f} أقل من الحد المطلوب {MIN_AUC:.3f}",
        )

    latest = df.iloc[-1]
    live_x = latest[feature_cols].to_frame().T

    probability_up = float(
        model.predict_proba(live_x)[0, 1]
    )

    probability_down = 1.0 - probability_up

    spx_up = (
        latest["ret_1"] > 0
        and latest["ema_spread"] > 0
    )

    spy_up = (
        latest["spy_ret_1"] > 0
        and latest["spy_ema_spread"] > 0
    )

    qqq_up = (
        latest["qqq_ret_1"] > 0
        and latest["qqq_ema_spread"] > 0
    )

    up_votes = sum([
        bool(spx_up),
        bool(spy_up),
        bool(qqq_up),
    ])

    down_votes = 3 - up_votes

    if (
        probability_up >= MIN_CONFIDENCE
        and up_votes >= 2
    ):
        return (
            "CALL",
            probability_up,
            f"احتمال الصعود={probability_up:.1%}؛ "
            f"توافق المؤشرات={up_votes}/3",
        )

    if (
        probability_down >= MIN_CONFIDENCE
        and down_votes >= 2
    ):
        return (
            "PUT",
            probability_down,
            f"احتمال الهبوط={probability_down:.1%}؛ "
            f"توافق المؤشرات={down_votes}/3",
        )

    return (
        "WAIT",
        max(probability_up, probability_down),
        f"لا توجد إشارة متوافقة؛ "
        f"احتمال الصعود={probability_up:.1%}؛ "
        f"احتمال الهبوط={probability_down:.1%}؛ "
        f"أصوات الصعود/الهبوط={up_votes}/{down_votes}",
    )


# ======================== تيليجرام ==========================

def send_telegram(message):
    if not TG_TOKEN or not TG_CHAT:
        log(
            "[تيليجرام] رمز البوت أو معرف المحادثة غير موجود",
            "warning",
        )
        return False

    try:
        response = HTTP.post(
            f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage",
            json={
                "chat_id": TG_CHAT,
                "text": message,
                "disable_web_page_preview": True,
            },
            timeout=20,
        )

        if response.status_code >= 400:
            log(
                f"[خطأ تيليجرام] HTTP {response.status_code}: "
                f"{response.text[:200]}",
                "error",
            )
            return False

        if not response.json().get("ok", False):
            log(
                "[خطأ تيليجرام] لم تؤكد واجهة تيليجرام إرسال الرسالة",
                "error",
            )
            return False

        log("[تيليجرام] تم إرسال الرسالة بنجاح")
        return True

    except Exception as exc:
        log(f"[خطأ تيليجرام] {exc}", "error")
        return False


# ======================== عقود SPXW =========================

def get_spxw_contracts(direction):
    try:
        params = {
            "underlying_symbols": "SPX",
            "expiration_date": ny_now().date().isoformat(),
            "type": "call" if direction == "CALL" else "put",
            "status": "active",
            "limit": 1000,
        }

        response = HTTP.get(
            f"{TRADE_URL}/v2/options/contracts",
            headers=alpaca_headers(),
            params=params,
            timeout=25,
        )

        if response.status_code >= 400:
            raise RuntimeError(
                f"خطأ جلب العقود HTTP {response.status_code}: "
                f"{response.text[:250]}"
            )

        payload = response.json()

        contracts = payload.get(
            "option_contracts",
            payload.get("contracts", []),
        ) or []

        results = []

        for contract in contracts:
            symbol = contract.get("symbol", "")
            root = contract.get("root_symbol", "")

            if not symbol:
                continue

            if root != "SPXW" and not symbol.startswith("SPXW"):
                continue

            try:
                strike = float(contract["strike_price"])
            except (KeyError, TypeError, ValueError):
                continue

            results.append({
                "symbol": symbol,
                "strike": strike,
                "expiration": contract.get("expiration_date", ""),
            })

        return results

    except Exception as exc:
        log(f"[خطأ العقود] {exc}", "error")
        return []


def get_option_quotes(symbols):
    results = {}
    endpoint = f"{DATA_URL}/v1beta1/options/quotes/latest"

    for start in range(0, len(symbols), 100):
        batch = symbols[start:start + 100]

        try:
            response = HTTP.get(
                endpoint,
                headers=alpaca_headers(),
                params={
                    "symbols": ",".join(batch),
                    "feed": OPTIONS_FEED,
                },
                timeout=25,
            )

            if response.status_code >= 400:
                log(
                    f"[خطأ الأسعار] HTTP {response.status_code}: "
                    f"{response.text[:200]}",
                    "warning",
                )
                continue

            payload = response.json()
            quotes = payload.get("quotes", {}) or {}

            if isinstance(quotes, dict):
                results.update(quotes)

        except Exception as exc:
            log(f"[خطأ الأسعار] {exc}", "warning")

    return results


def number_from(quote, *keys):
    for key in keys:
        try:
            value = quote.get(key)

            if value is not None:
                value = float(value)

                if np.isfinite(value):
                    return value

        except (TypeError, ValueError):
            pass

    return None


def choose_contract(direction, spot_price):
    contracts = get_spxw_contracts(direction)

    if not contracts:
        log(
            "[العقود] لم يتم العثور على عقود SPXW",
            "warning",
        )
        return None

    # فحص أقرب 15 سعر تنفيذ من مستوى المؤشر
    contracts.sort(
        key=lambda item: abs(item["strike"] - spot_price)
    )

    candidates = contracts[:MAX_CONTRACTS_TO_CHECK]

    quotes = get_option_quotes(
        [item["symbol"] for item in candidates]
    )

    viable = []

    for contract in candidates:
        quote = quotes.get(contract["symbol"])

        if not isinstance(quote, dict):
            continue

        bid = number_from(quote, "bp", "bid_price", "bid")
        ask = number_from(quote, "ap", "ask_price", "ask")

        if (
            bid is None or ask is None
            or bid <= 0 or ask <= 0 or ask < bid
        ):
            continue

        mid = (bid + ask) / 2.0

        spread_pct = (
            (ask - bid) / mid if mid > 0 else 1.0
        )

        if spread_pct > MAX_SPREAD_PCT:
            continue

        viable.append({
            **contract,
            "bid": bid,
            "ask": ask,
            "mid": mid,
            "spread_pct": spread_pct,
            "distance": abs(contract["strike"] - spot_price),
        })

    if not viable:
        log(
            "[العقود] لا يوجد عقد قريب اجتاز فلاتر الأسعار وفارق السعر",
            "warning",
        )
        return None

    viable.sort(
        key=lambda item: (
            item["distance"],
            item["spread_pct"],
            item["mid"],
        )
    )

    selected = viable[0]

    log(
        f"[العقد المختار] {selected['symbol']} "
        f"| سعر التنفيذ={selected['strike']:.2f} "
        f"| العرض={selected['bid']:.2f} "
        f"| الطلب={selected['ask']:.2f} "
        f"| الفارق={selected['spread_pct']:.1%} "
        f"| SPX={spot_price:.2f}"
    )

    return selected


# ======================== الماسح الرئيسي ====================

def market_is_open():
    now = ny_now()

    if now.weekday() >= 5:
        return False

    minutes = now.hour * 60 + now.minute

    return (
        9 * 60 + 30 <= minutes < 16 * 60
    )


def run_scan():
    log("========== بدء فحص جديد ==========")

    spx = get_spx_bars()
    spy = get_stock_bars("SPY")
    qqq = get_stock_bars("QQQ")

    if spx.empty or spy.empty or qqq.empty:
        log(
            "[الإشارة] انتظار — بيانات السوق غير مكتملة",
            "warning",
        )
        return

    try:
        merged = make_features(spx, spy, qqq)

        model, columns, auc, accuracy = train_model(merged)

        direction, confidence, reason = calculate_signal(
            merged, model, columns, auc
        )

        log(
            f"[الإشارة] {direction} "
            f"| الثقة={confidence:.1%} "
            f"| السبب={reason}"
        )

        # لا ترسل توصية عند عدم وجود إشارة
        if direction == "WAIT":
            return

        spot_price = float(merged.iloc[-1]["close"])

        contract = choose_contract(direction, spot_price)

        signal_ar = (
            "شراء كول (CALL)"
            if direction == "CALL"
            else "شراء بوت (PUT)"
        )

        # تنبيه عربي عند عدم وجود عقد مناسب
        if contract is None:
            message = (
                "⚠️ تنبيه بوت خيارات SPX\n"
                "━━━━━━━━━━━━━━━━━━\n"
                f"📍 الإشارة: {signal_ar}\n"
                f"📊 ثقة النموذج: {confidence:.1%}\n"
                f"💹 مستوى مؤشر SPX: {spot_price:,.2f}\n"
                f"🧠 تقييم النموذج AUC: {auc:.3f}\n\n"
                "لم يتم العثور على عقد مناسب اجتاز "
                "فلاتر السعر وفارق العرض والطلب.\n\n"
                "⛔ لم يتم تنفيذ أي صفقة.\n"
                "ℹ️ هذه توصية فقط."
            )

            send_telegram(message)
            return

        # توضيح سبب الإشارة بالعربية
        if direction == "CALL":
            reason_ar = (
                "النموذج يرجّح الصعود مع توافق اتجاه "
                "المؤشر والأسواق المساندة."
            )
        else:
            reason_ar = (
                "النموذج يرجّح الهبوط مع توافق اتجاه "
                "المؤشر والأسواق المساندة."
            )

        message = (
            "📊 توصية بوت خيارات SPX — الإصدار 18.2\n"
            "━━━━━━━━━━━━━━━━━━\n\n"
            f"📍 نوع الإشارة: {signal_ar}\n"
            f"📊 ثقة النموذج: {confidence:.1%}\n"
            f"💹 مستوى مؤشر SPX: {spot_price:,.2f}\n\n"
            "📑 تفاصيل العقد\n"
            f"🔹 رمز العقد: {contract['symbol']}\n"
            f"🎯 سعر التنفيذ: {contract['strike']:,.2f}\n"
            f"🟢 سعر الطلب (Ask): {contract['ask']:.2f}\n"
            f"🔴 سعر العرض (Bid): {contract['bid']:.2f}\n"
            f"⚖️ متوسط السعر التقريبي: {contract['mid']:.2f}\n"
            f"📉 فارق العرض والطلب: {contract['spread_pct']:.1%}\n\n"
            "🧠 نتائج النموذج\n"
            f"📈 تقييم النموذج AUC: {auc:.3f}\n"
            f"🧪 دقة الاختبار التاريخي: {accuracy:.1%}\n"
            f"📝 سبب الإشارة: {reason_ar}\n\n"
            "━━━━━━━━━━━━━━━━━━\n"
            "⚠️ تنبيه المخاطر:\n"
            "هذه توصية وليست أمر شراء أو بيع.\n"
            "البوت لا ينفذ الصفقات تلقائيًا.\n"
            "خيارات يوم الانتهاء (0DTE) عالية المخاطر "
            "وقد تفقد قيمتها بسرعة."
        )

        send_telegram(message)

    except Exception as exc:
        log(
            f"[خطأ الفحص] {type(exc).__name__}: {exc}",
            "error",
        )


# ======================== التشغيل ===========================

def main():
    log("بدء تشغيل بوت خيارات SPX — الإصدار 18.2")
    log("الوضع: توصيات فقط — لا يوجد تنفيذ تلقائي للصفقات")

    log(
        f"مصدر البيانات={DATA_FEED} "
        f"| مصدر أسعار الخيارات={OPTIONS_FEED} "
        f"| الحد الأدنى AUC={MIN_AUC:.3f} "
        f"| الحد الأدنى للثقة={MIN_CONFIDENCE:.2f}"
    )

    if not API_KEY or not API_SECRET:
        log(
            "مفاتيح Alpaca غير موجودة في متغيرات Railway",
            "error",
        )

    if not TG_TOKEN or not TG_CHAT:
        log(
            "إعدادات تيليجرام غير مكتملة؛ لن يتم إرسال التنبيهات",
            "warning",
        )

    while True:
        try:
            if market_is_open():
                run_scan()
            else:
                log(
                    "[السوق] مغلق | توقيت نيويورك: "
                    + ny_now().strftime("%Y-%m-%d %H:%M:%S %Z")
                )

        except KeyboardInterrupt:
            log("تم إيقاف البوت")
            break

        except Exception as exc:
            log(
                f"[خطأ رئيسي] {type(exc).__name__}: {exc}",
                "error",
            )

        time.sleep(SCAN_SECONDS)


if __name__ == "__main__":
    main()
