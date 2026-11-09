import yfinance as yf, pandas as pd
stamp = pd.Timestamp.now().strftime("%Y%m%d")
for sym, name in [("^GSPC", "spx"), ("ES=F", "es"), ("NQ=F", "nq"), ("^VIX", "vix")]:
    df = yf.Ticker(sym).history(period="7d", interval="1m", prepost=True)
    df.to_csv(f"{name}_1m_{stamp}.csv")
    print(name, len(df))
