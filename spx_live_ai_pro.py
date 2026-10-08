def prepare(raw):
    df = raw.copy()
    df["date"] = df["timestamp"].dt.date
    c = df["spx_close"]
    
    for n in (1, 3, 6, 12):
        df[f"spx_ret_{n}"] = c.pct_change(n)
        df[f"spy_ret_{n}"] = df["spy_close"].pct_change(n)
    
    df["rsi"] = rsi(c)
    df["spx_atr"] = atr(df, "spx_high", "spx_low", "spx_close")
    df["atr_pct"] = df["spx_atr"] / c
    
    df["realized_vol"] = c.pct_change().rolling(12).std() * sqrt(252 * 78) * 100
    df["vol_spread"] = df["vix"] - df["realized_vol"]

    # 🚨 التصحيح الدقيق لـ Previous Day High/Low بدون Lookahead Leakage
    daily_high = df.groupby("date")["spx_high"].transform("max")
    daily_low = df.groupby("date")["spx_low"].transform("min")

    prev_high = daily_high.groupby(df["date"]).first().shift(1)
    prev_low = daily_low.groupby(df["date"]).first().shift(1)

    df["prev_day_high"] = df["date"].map(prev_high)
    df["prev_day_low"] = df["date"].map(prev_low)
    
    # تعويض القيم المفقودة لأول يوم تداول في العينة
    df["prev_day_high"] = df["prev_day_high"].fillna(df["spx_high"].iloc[0])
    df["prev_day_low"] = df["prev_day_low"].fillna(df["spx_low"].iloc[0])

    df["dist_prev_high"] = (c - df["prev_day_high"]) / c
    df["dist_prev_low"] = (c - df["prev_day_low"]) / c

    df["mins_open"] = df["timestamp"].dt.hour * 60 + df["timestamp"].dt.minute - MARKET_OPEN_MIN

    # Causal Opening Range (بدون Lookahead Leakage)
    is_first_30 = (df["mins_open"] >= 0) & (df["mins_open"] <= 30)
    or_highs = df[is_first_30].groupby("date")["spx_high"].max().to_dict()
    or_lows = df[is_first_30].groupby("date")["spx_low"].min().to_dict()

    df["orh"] = df["date"].map(or_highs)
    df["orl"] = df["date"].map(or_lows)
    df.loc[df["mins_open"] <= 30, ["orh", "orl"]] = np.nan

    df["dist_from_orh"] = np.where(df["orh"].notna(), (c - df["orh"]) / c, 0.0)
    df["dist_from_orl"] = np.where(df["orl"].notna(), (c - df["orl"]) / c, 0.0)

    vol_mean = df["spy_volume"].rolling(30, min_periods=5).mean()
    vol_std = df["spy_volume"].rolling(30, min_periods=5).std().replace(0, 1)
    df["volume_zscore"] = (df["spy_volume"] - vol_mean) / vol_std

    # Triple Barrier Method للتسميد
    targets = []
    highs = df["spx_high"].values
    lows = df["spx_low"].values
    closes = c.values
    atrs = df["spx_atr"].values
    dates = df["date"].values

    for i in range(len(df)):
        if i + HORIZON >= len(df) or dates[i] != dates[i + HORIZON]:
            targets.append(np.nan)
            continue
        
        entry_price = closes[i]
        atr_val = atrs[i]
        upper_barrier = entry_price + (0.6 * atr_val)
        lower_barrier = entry_price - (0.4 * atr_val)

        hit_upper = False
        hit_lower = False

        for h in range(1, HORIZON + 1):
            future_idx = i + h
            bar_high = highs[future_idx]
            bar_low = lows[future_idx]

            if not hit_upper and not hit_lower:
                touched_up = bar_high >= upper_barrier
                touched_dn = bar_low <= lower_barrier
                if touched_up and not touched_dn:
                    hit_upper = True
                    break
                elif touched_dn and not touched_up:
                    hit_lower = True
                    break
                elif touched_up and touched_dn:
                    hit_lower = True
                    break

        if hit_upper:
            targets.append(1.0)
        elif hit_lower:
            targets.append(0.0)
        else:
            targets.append(np.nan)

    df["target"] = targets
    return df
