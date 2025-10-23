import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

def play():
    print("Code generated and updated for single-position trading over 10 years with equity curve and average summary (GPT-5)")

    data_dir = "sp500_stock_data"
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    # --- Indicator helper functions ---
    def EMA(series, span): return series.ewm(span=span, adjust=False).mean()
    def SMA(series, window): return series.rolling(window).mean()

    def HMA(series, window):
        def WMA(s, n):
            if n <= 0: return pd.Series(np.nan, index=s.index)
            weights = np.arange(1, n + 1)
            return s.rolling(n).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)
        half, sqrt = int(window/2), int(np.sqrt(window))
        return WMA(2 * WMA(series, half) - WMA(series, window), sqrt)

    def RSI(series, period=14):
        delta = series.diff()
        gain, loss = delta.clip(lower=0), -delta.clip(upper=0)
        avg_gain, avg_loss = gain.rolling(period).mean(), loss.rolling(period).mean()
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def MACD(series, fast=12, slow=26, signal=9):
        ema_fast, ema_slow = EMA(series, fast), EMA(series, slow)
        macd_line = ema_fast - ema_slow
        signal_line = EMA(macd_line, signal)
        return macd_line, signal_line, macd_line - signal_line

    def Stochastic(df, k_period=14, d_period=3):
        low_min = df['Low'].rolling(k_period).min()
        high_max = df['High'].rolling(k_period).max()
        k = 100 * (df['Adj Close'] - low_min) / (high_max - low_min)
        d = k.rolling(d_period).mean()
        return k, d

    def ATR(df, period=14):
        hl = df['High'] - df['Low']
        hc = (df['High'] - df['Adj Close'].shift()).abs()
        lc = (df['Low'] - df['Adj Close'].shift()).abs()
        tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        return tr.rolling(period).mean()

    def Bollinger_Bands(series, window=20, num_std=2):
        sma, std = SMA(series, window), series.rolling(window).std()
        return sma + num_std * std, sma - num_std * std

    # --- Load all stock data ---
    all_data = {}
    all_dates = []
    for filename in os.listdir(data_dir):
        if not filename.endswith(".csv"):
            continue
        df = pd.read_csv(os.path.join(data_dir, filename), parse_dates=['Date'])
        df.sort_values('Date', inplace=True)
        ticker = df['Ticker'].iloc[0] if 'Ticker' in df.columns else os.path.splitext(filename)[0]
        df['EMA_100'] = EMA(df['Adj Close'], 100)
        df['HMA_100'] = HMA(df['Adj Close'], 100)
        df['RSI_14'] = RSI(df['Adj Close'], 14)
        _, _, macd_hist = MACD(df['Adj Close'])
        df['MACD_hist'] = macd_hist
        k, d = Stochastic(df)
        df['Stoch_%K'], df['Stoch_%D'] = k, d
        df['ATR_14'] = ATR(df, 14)
        df['BB_upper'], df['BB_lower'] = Bollinger_Bands(df['Adj Close'])
        df['Avg_Vol_20'] = df['Volume'].rolling(20).mean()
        all_data[ticker] = df
        all_dates += list(df['Date'])

    if not all_data:
        print("No data loaded. Check your folder path.")
        return

    max_date = max(all_dates)
    last_year = max_date.year
    first_year = last_year - 9
    print(f"Running simulation for {first_year}–{last_year}...")

    all_trades_all_years = []

    for theYear in range(first_year, last_year + 1):
        print(f"\n=== YEAR {theYear} ===")
        yearly_data = {}
        for t, df in all_data.items():
            sub = df[df['Date'].dt.year == theYear].reset_index(drop=True)
            if sub.empty: continue
            sub['Prev_MACD_hist'] = sub['MACD_hist'].shift(1)
            sub['Prev_Stoch_%K'] = sub['Stoch_%K'].shift(1)
            sub['Prev_Stoch_%D'] = sub['Stoch_%D'].shift(1)
            yearly_data[t] = sub
        if not yearly_data:
            print(f"No data for {theYear}, skipping.")
            continue

        all_dates_y = sorted(set(pd.concat([df['Date'] for df in yearly_data.values()])))
        global_position = None
        all_trades = []

        for current_date in all_dates_y:
            if global_position is not None:
                ticker = global_position['Ticker']
                df = yearly_data.get(ticker)
                if df is None: continue
                row = df[df['Date'] == current_date]
                if row.empty: continue
                row = row.iloc[0]
                adj_close = row['Adj Close']
                rsi = row['RSI_14']
                macd_hist, prev_macd_hist = row['MACD_hist'], row['Prev_MACD_hist']
                ema_100 = row['EMA_100']
                max_price = global_position['Max Price']

                if adj_close > max_price:
                    global_position['Max Price'] = adj_close
                    max_price = adj_close

                cond_sell = (
                    (adj_close > ema_100 if pd.notna(ema_100) else False)
                    or adj_close < 0.97 * max_price
                    or adj_close < global_position['Stop Loss']
                    or (pd.notna(rsi) and rsi > 65)
                    or ((macd_hist < 0) and (prev_macd_hist >= 0))
                    or (current_date - global_position['Buy Date']).days > 60
                    or current_date == all_dates_y[-1]
                )

                if cond_sell:
                    pct_gain = (adj_close - global_position['Buy Price']) / global_position['Buy Price'] * 100
                    all_trades.append({
                        'Year': theYear,
                        'Ticker': ticker,
                        'Buy Date': global_position['Buy Date'],
                        'Buy Price': global_position['Buy Price'],
                        'Sell Date': current_date,
                        'Sell Price': adj_close,
                        'Pct Gain (%)': pct_gain
                    })
                    global_position = None

            elif global_position is None:
                for ticker, df in yearly_data.items():
                    row = df[df['Date'] == current_date]
                    if row.empty: continue
                    row = row.iloc[0]
                    adj_close = row['Adj Close']
                    hma_100 = row['HMA_100']
                    rsi = row['RSI_14']
                    stoch_k, stoch_d = row['Stoch_%K'], row['Stoch_%D']
                    prev_k, prev_d = row['Prev_Stoch_%K'], row['Prev_Stoch_%D']
                    atr = row['ATR_14']
                    bb_upper, bb_lower = row['BB_upper'], row['BB_lower']
                    volume, avg_vol = row['Volume'], row['Avg_Vol_20']
                    if any(pd.isna([hma_100, rsi, stoch_k, stoch_d, prev_k, prev_d, atr, bb_upper, bb_lower, avg_vol])):
                        continue
                    price_discount = (hma_100 - adj_close) / hma_100 if hma_100 else 0
                    conds = [
                        price_discount >= 0.12,
                        rsi < 30,
                        (prev_k < prev_d) and (stoch_k > stoch_d) and (stoch_k < 40),
                        volume > 1.1 * avg_vol,
                        (atr / adj_close < 0.2) and (bb_lower < adj_close < bb_upper)
                    ]
                    if all(conds):
                        global_position = {
                            'Ticker': ticker,
                            'Buy Date': current_date,
                            'Buy Price': adj_close,
                            'Max Price': adj_close,
                            'Stop Loss': adj_close - 2 * atr
                        }
                        break

        if all_trades:
            all_trades_all_years.append(pd.DataFrame(all_trades))
            print(f"Executed {len(all_trades)} trades for {theYear}")
        else:
            print(f"No trades for {theYear}")

    # === COMBINE AND SUMMARIZE ===
    if all_trades_all_years:
        combined = pd.concat(all_trades_all_years, ignore_index=True)
        yearly = combined.groupby('Year')['Pct Gain (%)'].sum().reset_index()
        yearly['Cumulative Return (%)'] = yearly['Pct Gain (%)'].cumsum()

        # Calculate equity curve
        equity = [100]
        for pct in yearly['Pct Gain (%)']:
            equity.append(equity[-1] * (1 + pct / 100))
        yearly['Equity ($)'] = equity[1:]

        # Compute summary metrics
        avg_annual_gain = yearly['Pct Gain (%)'].mean()
        total_gain = (yearly['Equity ($)'].iloc[-1] / 100 - 1) * 100

        # Add total row
        summary_row = pd.DataFrame({
            'Year': ['TOTAL (10 Years)'],
            'Pct Gain (%)': [yearly['Pct Gain (%)'].sum()],
            'Cumulative Return (%)': [total_gain],
            'Equity ($)': [yearly['Equity ($)'].iloc[-1]],
            'Average Annual Gain (%)': [avg_annual_gain]
        })
        summary_all = pd.concat([yearly, summary_row], ignore_index=True)

        # Save full summary
        summary_path = os.path.join(results_dir, "summary_all_years.csv")
        summary_all.to_csv(summary_path, index=False)

        # Plot equity curve
        plt.figure(figsize=(10, 6))
        plt.plot(yearly['Year'], yearly['Equity ($)'], marker='o', linewidth=2)
        plt.title("Equity Curve (10-Year Compounded Return)")
        plt.xlabel("Year")
        plt.ylabel("Portfolio Value ($)")
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, "equity_curve_10yr.png"))
        plt.close()

        print("\n✅ Saved:")
        print(f"  • {summary_path}")
        print(f"  • results/equity_curve_10yr.png")
        print(f"\n📊 Average Annual Gain: {avg_annual_gain:.2f}%")
        print(f"📈 Total Compounded Gain: {total_gain:.2f}% over 10 years")
    else:
        print("No trades across any of the 10 years.")

# Run simulation
play()
