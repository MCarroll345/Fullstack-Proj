import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

def play():
    print("Code generated using GPT-4.1-mini")

    theYear = 2020 # Year to analyze
    data_dir = "sp500_stock_data"
    results_dir = "results"
    plots_dir = "plots"
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    # --- Indicator functions ---
    def EMA(series, span):
        return series.ewm(span=span, adjust=False).mean()

    def SMA(series, window):
        return series.rolling(window).mean()

    def HMA(series, window):
        # Hull Moving Average
        def WMA(s, n):
            weights = np.arange(1, n + 1)
            return s.rolling(n).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)
        half = int(window / 2)
        sqrt = int(np.sqrt(window))
        wma1 = WMA(series, half)
        wma2 = WMA(series, window)
        diff = 2 * wma1 - wma2
        hma = WMA(diff, sqrt)
        return hma

    def RSI(series, period=14):
        delta = series.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def MACD(series, fast=12, slow=26, signal=9):
        ema_fast = EMA(series, fast)
        ema_slow = EMA(series, slow)
        macd_line = ema_fast - ema_slow
        signal_line = EMA(macd_line, signal)
        hist = macd_line - signal_line
        return macd_line, signal_line, hist

    def Stochastic(df, k_period=14, d_period=3):
        low_min = df['Low'].rolling(k_period).min()
        high_max = df['High'].rolling(k_period).max()
        k = 100 * (df['Close'] - low_min) / (high_max - low_min)
        d = k.rolling(d_period).mean()
        return k, d

    def ATR(df, period=14):
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()
        return atr

    def Bollinger_Bands(series, window=20, num_std=2):
        sma = SMA(series, window)
        std = series.rolling(window).std()
        upper = sma + num_std * std
        lower = sma - num_std * std
        return upper, lower

    all_trades = []

    # Process each stock file
    for filename in os.listdir(data_dir):
        if not filename.endswith(".csv"):
            continue
        filepath = os.path.join(data_dir, filename)
        df = pd.read_csv(filepath, parse_dates=['Date'])
        df.sort_values('Date', inplace=True)
        ticker = df['Ticker'].iloc[0]

        # Calculate indicators on full dataset
        df['EMA_100'] = EMA(df['Adj Close'], 100)
        df['HMA_100'] = HMA(df['Adj Close'], 100)
        df['RSI_14'] = RSI(df['Adj Close'], 14)
        macd_line, signal_line, macd_hist = MACD(df['Adj Close'])
        df['MACD_hist'] = macd_hist
        k, d = Stochastic(df)
        df['Stoch_%K'] = k
        df['Stoch_%D'] = d
        df['ATR_14'] = ATR(df, 14)
        df['BB_upper'], df['BB_lower'] = Bollinger_Bands(df['Adj Close'], 20, 2)
        df['Avg_Vol_20'] = df['Volume'].rolling(20).mean()

        # Filter for theYear
        year_data = df[df['Date'].dt.year == theYear].reset_index(drop=True)
        if year_data.empty:
            continue

        position = None
        trades = []

        # Pre-shift columns for previous values to simplify logic
        year_data['Prev_MACD_hist'] = year_data['MACD_hist'].shift(1)
        year_data['Prev_Stoch_%K'] = year_data['Stoch_%K'].shift(1)
        year_data['Prev_Stoch_%D'] = year_data['Stoch_%D'].shift(1)

        for i, row in year_data.iterrows():
            date = row['Date']
            adj_close = row['Adj Close']
            volume = row['Volume']
            avg_vol = row['Avg_Vol_20']
            ema_100 = row['EMA_100']
            hma_100 = row['HMA_100']
            rsi = row['RSI_14']
            macd_hist = row['MACD_hist']
            prev_macd_hist = row['Prev_MACD_hist']
            stoch_k = row['Stoch_%K']
            stoch_d = row['Stoch_%D']
            prev_k = row['Prev_Stoch_%K']
            prev_d = row['Prev_Stoch_%D']
            atr = row['ATR_14']
            bb_upper = row['BB_upper']
            bb_lower = row['BB_lower']
            month = date.month

            # Skip if indicators missing
            if pd.isna([ema_100, hma_100, rsi, macd_hist, stoch_k, stoch_d, prev_macd_hist,
                        prev_k, prev_d, atr, avg_vol, bb_upper, bb_lower]).any():
                continue

            if position is None:
                # Buy Conditions:
                price_discount = (hma_100 - adj_close) / hma_100 if hma_100 != 0 else 0
                cond_price = price_discount >= 0.12
                cond_rsi = rsi < 30
                cond_stoch = (prev_k < prev_d) and (stoch_k > stoch_d) and (stoch_k < 40)
                cond_vol = volume > 1.1 * avg_vol
                cond_volatility = (atr / adj_close < 0.2) and (bb_lower < adj_close < bb_upper)

                buy_signal = all([cond_price,cond_volatility,cond_vol,cond_stoch,cond_rsi])

                if buy_signal:
                    position = {
                        'Buy Date': date,
                        'Buy Price': adj_close,
                        'Buy HMA_100': hma_100,
                        'Buy Index': i,
                        'Max Price': adj_close,
                        'Stop Loss': adj_close - 2 * atr
                    }

            else:
                holding_days = (date - position['Buy Date']).days
                max_price = position['Max Price']

                if adj_close > max_price:
                    position['Max Price'] = adj_close
                    max_price = adj_close

                cond_price_sell = adj_close > ema_100
                cond_trailing_stop = adj_close < 0.97 * max_price
                cond_atr_stop = adj_close < position['Stop Loss']
                cond_rsi_sell = rsi > 65
                cond_macd_sell = (macd_hist < 0) and (prev_macd_hist >= 0)
                cond_max_days = holding_days > 60
                cond_eoy = (i == len(year_data) - 1)

                sell_signal = any([
                    cond_price_sell,
                    cond_trailing_stop,
                    cond_atr_stop,
                    cond_rsi_sell,
                    cond_macd_sell,
                    cond_max_days,
                    cond_eoy
                ])

                if sell_signal:
                    buy_price = position['Buy Price']
                    pct_below_ma = (position['Buy HMA_100'] - buy_price) / position['Buy HMA_100'] * 100
                    pct_gain = (adj_close - buy_price) / buy_price * 100
                    trades.append({
                        'Ticker': ticker,
                        'Buy Date': position['Buy Date'],
                        'Buy Price': buy_price,
                        'Pct Below 100MA at Buy (%)': pct_below_ma,
                        'Sell Date': date,
                        'Sell Price': adj_close,
                        'Pct Gain (%)': pct_gain,
                        'Holding Days': holding_days
                    })
                    position = None

        if trades:
            all_trades.extend(trades)

            # Plotting
            plot_data = df[df['Date'].dt.year == theYear]
            plt.figure(figsize=(14,7))
            plt.plot(plot_data['Date'], plot_data['Adj Close'], label='Adj Close')
            plt.plot(plot_data['Date'], plot_data['HMA_100'], label='HMA 100')
            plt.plot(plot_data['Date'], plot_data['EMA_100'], label='EMA 100')

            # Add buy/sell vertical lines
            for trade in trades:
                buy_date = trade['Buy Date']
                sell_date = trade['Sell Date']
                plt.axvline(buy_date, color='green', linestyle='--', alpha=0.7, label='Buy Signal')
                plt.axvline(sell_date, color='red', linestyle='--', alpha=0.7, label='Sell Signal')

            plt.title(f"{ticker} {theYear} Trades")
            plt.xlabel("Date")
            plt.ylabel("Price")
            plt.legend(loc='upper left')

            # Avoid duplicate labels in legend
            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            plt.legend(by_label.values(), by_label.keys())

            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, f"{ticker}_{theYear}.png"))
            plt.close()

    # Compile all trades into DataFrame
    trades_df = pd.DataFrame(all_trades)
    if not trades_df.empty:
        graphs_dir = os.path.join(results_dir, "graphs")
        os.makedirs(graphs_dir, exist_ok=True)
        summ_dir = os.path.join(results_dir, "summaries")
        os.makedirs(summ_dir, exist_ok=True)
        perf_dir = os.path.join(results_dir, "performance")
        os.makedirs(perf_dir, exist_ok=True)
        # Calculate total profit in % (sum of individual trade % gains)
        total_profit_pct = trades_df['Pct Gain (%)'].sum()

        # You can also calculate total number of trades and average gain per trade:
        total_trades = len(trades_df)
        avg_gain_pct = trades_df['Pct Gain (%)'].mean()

        # Create a summary DataFrame with one row
        summary_df = pd.DataFrame({
            'Ticker': ['TOTAL'],
            'Buy Date': ['-'],
            'Buy Price': ['-'],
            'Pct Below 100MA at Buy (%)': ['-'],
            'Sell Date': ['-'],
            'Sell Price': ['-'],
            'Pct Gain (%)': [total_profit_pct],
            'Holding Days': ['-'],
            'Total Trades': [total_trades],
            'Average Gain (%)': [avg_gain_pct]
        })

        # Append the summary row to the original DataFrame
        output_df = pd.concat([trades_df, summary_df], ignore_index=True)

        output_df.to_csv(os.path.join(perf_dir, f"{theYear}_perf.csv"), index=False)
        print(f"Saved results with summary to {os.path.join(perf_dir, f'{theYear}_perf.csv')}")

        # --- New summary for wins/losses ---
        wins = trades_df[trades_df['Pct Gain (%)'] > 0]
        losses = trades_df[trades_df['Pct Gain (%)'] <= 0]

        total_wins = len(wins)
        total_losses = len(losses)
        total_money_won = wins['Pct Gain (%)'].sum()  # sum of % gains on wins
        total_money_lost = losses['Pct Gain (%)'].sum()  # sum of % gains on losses (negative)
        net_profit_loss = total_money_won + total_money_lost

        summary_simple = pd.DataFrame({
            'Total Wins': [total_wins],
            'Total Losses': [total_losses],
            'Total Money Won (%)': [total_money_won],
            'Total Money Lost (%)': [total_money_lost],
            'Net Profit/Loss (%)': [net_profit_loss]
        })

        summary_simple.to_csv(os.path.join(summ_dir, f"{theYear}_wins_losses_summary.csv"), index=False)
        print(f"Saved simplified win/loss summary to {os.path.join(summ_dir, f'{theYear}_wins_losses_summary.csv')}")

        # --- Plot profits and losses ---
        plt.figure(figsize=(12, 6))
        profits = trades_df['Pct Gain (%)']

        plt.plot(trades_df.index, profits, marker='o', linestyle='-', color='blue')
        plt.axhline(0, color='black', linewidth=0.8)
        plt.title(f'Trade Profits and Losses in {theYear}')
        plt.xlabel('Trade Number')
        plt.ylabel('Profit / Loss (%)')
        plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter())

        plt.tight_layout()
        plt.savefig(os.path.join(graphs_dir, f"{theYear}_profits_losses_line.png"))
        plt.close()

        print(f"Saved profit/loss line graph to {os.path.join(graphs_dir, f'{theYear}_profits_losses_line.png')}")
    else:
        print("No trades found for the given year and criteria.")

# To run the strategy:
play()
