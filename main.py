import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def play():
    print("Code generated using GPT-4.1-mini")

    theYear = 2024
    data_dir = "sp500_stock_data"
    plots_dir = "plots"
    os.makedirs(plots_dir, exist_ok=True)

    # --- Indicator functions ---
    def EMA(series, span):
        return series.ewm(span=span, adjust=False).mean()

    def SMA(series, window):
        return series.rolling(window).mean()

    def HMA(series, window):
        # Hull Moving Average: WMA(2*WMA(n/2) - WMA(n)), sqrt(n))
        # Approximate WMA using numpy.convolve weights
        def WMA(s, n):
            weights = np.arange(1, n + 1)
            return s.rolling(n).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)
        half_length = int(window / 2)
        wma1 = WMA(series, half_length)
        wma2 = WMA(series, window)
        diff = 2 * wma1 - wma2
        hma = WMA(diff, int(np.sqrt(window)))
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
        upper_band = sma + num_std * std
        lower_band = sma - num_std * std
        return upper_band, lower_band

    # Hardcoded strong months for calendar filter (e.g., April, November)
    strong_months = {4, 11}

    all_trades = []

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
        df['MACD'] = macd_line
        df['MACD_signal'] = signal_line
        df['MACD_hist'] = macd_hist
        k, d = Stochastic(df)
        df['Stoch_%K'] = k
        df['Stoch_%D'] = d
        df['ATR_14'] = ATR(df, 14)
        df['BB_upper'], df['BB_lower'] = Bollinger_Bands(df['Adj Close'], 20, 2)
        df['Avg_Vol_20'] = df['Volume'].rolling(20).mean()

        # Filter data to theYear only
        year_data = df[df['Date'].dt.year == theYear].reset_index(drop=True)
        if year_data.empty:
            continue

        position = None  # track active buy position
        trades = []

        for i, row in year_data.iterrows():
            date = row['Date']
            adj_close = row['Adj Close']
            volume = row['Volume']
            avg_vol = row['Avg_Vol_20']
            ema_100 = row['EMA_100']
            hma_100 = row['HMA_100']
            rsi = row['RSI_14']
            macd_hist = row['MACD_hist']
            stoch_k = row['Stoch_%K']
            stoch_d = row['Stoch_%D']
            atr = row['ATR_14']
            bb_upper = row['BB_upper']
            bb_lower = row['BB_lower']
            month = date.month

            # Skip if any indicators not ready (NaN)
            if pd.isna([ema_100, hma_100, rsi, macd_hist, stoch_k, stoch_d, atr, avg_vol]).any():
                continue

            # --- Buy conditions ---
            if position is None:
                # Price 30% below HMA 100
                price_discount = (hma_100 - adj_close) / hma_100 if hma_100 != 0 else 0
                condition_price = price_discount >= 0.3

                # Momentum: RSI < 30, MACD histogram positive or just turned positive, stochastic %K crossing above %D below 20
                cond_rsi = rsi < 30
                # MACD hist turning positive (current hist > 0 and previous <= 0)
                prev_macd_hist = year_data.at[i-1, 'MACD_hist'] if i > 0 else None
                cond_macd = (macd_hist > 0) and (prev_macd_hist is not None) and (prev_macd_hist <= 0)

                # Stochastic crossover below 20
                prev_k = year_data.at[i-1, 'Stoch_%K'] if i > 0 else None
                prev_d = year_data.at[i-1, 'Stoch_%D'] if i > 0 else None
                cond_stoch = False
                if prev_k is not None and prev_d is not None:
                    cond_stoch = (prev_k < prev_d) and (stoch_k > stoch_d) and (stoch_k < 20)

                # Volume spike > 1.5 * avg volume
                cond_vol = volume > 1.5 * avg_vol

                # Volatility moderate: ATR < 3% of price, price inside Bollinger Bands
                cond_volatility = (atr / adj_close < 0.03) and (adj_close > bb_lower) and (adj_close < bb_upper)

                # Calendar filter: month in strong_months
                cond_month = month in strong_months

                # Combine buy conditions
                buy_signal = (condition_price and cond_rsi and cond_macd and cond_stoch and cond_vol and cond_volatility and cond_month)

                if buy_signal:
                    position = {
                        'Buy Date': date,
                        'Buy Price': adj_close,
                        'Buy HMA_100': hma_100,
                        'Buy Index': year_data.index[i],
                        'Max Price': adj_close,  # For trailing stop tracking
                        'Stop Loss': adj_close - 2 * atr  # ATR based stop loss
                    }

            else:
                # --- Sell conditions ---
                holding_days = (date - position['Buy Date']).days
                max_price = position['Max Price']

                # Update max price for trailing stop
                if adj_close > max_price:
                    position['Max Price'] = adj_close
                    max_price = adj_close

                # Price > EMA 100 (could swap for HMA or SMA)
                cond_price_sell = adj_close > ema_100

                # Trailing stop: price drops 5% below max price since buy
                cond_trailing_stop = adj_close < 0.95 * max_price

                # ATR stop loss hit
                cond_atr_stop = adj_close < position['Stop Loss']

                # Momentum weakening: RSI > 70 or MACD histogram turned negative
                cond_rsi_sell = rsi > 70
                prev_macd_hist = year_data.at[i-1, 'MACD_hist'] if i > 0 else None
                cond_macd_sell = (macd_hist < 0) and (prev_macd_hist is not None) and (prev_macd_hist >= 0)

                # Max holding days exceeded
                cond_max_days = holding_days > 40

                # End of year sell
                cond_eoy = (i == len(year_data) - 1)

                # Combine sell conditions
                sell_signal = (cond_price_sell or cond_trailing_stop or cond_atr_stop or cond_rsi_sell or cond_macd_sell or cond_max_days or cond_eoy)

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

            #

play()