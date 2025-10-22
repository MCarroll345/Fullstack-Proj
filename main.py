"""
sp500_trading_backtest.py

Backtest script that predicts buy/sell times without looking into the future.

Features:
- Loads CSV files from a folder `sp500_stock_data/` (expects files like `AAPL.csv`, `MSFT.csv`)
- Two strategies included:
    1) Moving Average Crossover (fast/slow MA) -- deterministic, no ML.
    2) Walk-forward ML classifier (optional, commented and experimental)
- Produces a spreadsheet of trades for the chosen year with columns:
    Ticker, Buy Date, Buy Price, Sell Date, Sell Price, Percentage Profit
- Produces charts for each ticker traded with buy/sell markers saved to `plots/`
- Adjustable variables to ensure a finite number of trades (max trades per ticker, min days between trades)

Usage:
    python sp500_trading_backtest.py --folder sp500_stock_data --year 2020

Notes:
- The script never uses future information when making a prediction or deciding a trade.
- The ML walk-forward section trains only on historical data up to the prediction day.
- Requires: pandas, numpy, matplotlib, tqdm, scikit-learn (if you use the ML section)

"""

import os
import argparse
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# ----------------------
# Parameters you can tune
# ----------------------
FOLDER = 'sp500_stock_data'
OUTPUT_TRADES_CSV = 'trades_{year}.csv'
PLOTS_DIR = 'plots'

# Strategy parameters (MA crossover) - adjust to change trade count
FAST_MA = 10          # shorter moving average window (days)
SLOW_MA = 30          # longer moving average window (days)
MIN_DAYS_BETWEEN_TRADES = 10  # to avoid too many trades
MAX_TRADES_PER_TICKER = 10    # finite limit per ticker
TAKE_PROFIT = 0.10    # 10% take profit (optional exit)
STOP_LOSS = -0.05     # -5% stop loss (optional exit)

# Trading rules
MIN_VOLUME = 1000     # ignore very illiquid rows (optional)
PRICE_COL = 'Adj Close'  # column to use as execution price
DATE_COL = 'Date'

# ----------------------
# Helper functions
# ----------------------

def load_csv_for_ticker(path):
    df = pd.read_csv(path, parse_dates=[DATE_COL])
    # normalize column names
    df.columns = [c.strip() for c in df.columns]
    if DATE_COL not in df.columns:
        # try lowercase date
        if 'date' in df.columns:
            df.rename(columns={'date': DATE_COL}, inplace=True)
    df = df.sort_values(DATE_COL).reset_index(drop=True)
    return df


def compute_indicators(df):
    # compute returns and moving averages
    df = df.copy()
    df['return_1d'] = df[PRICE_COL].pct_change()
    df['ma_fast'] = df[PRICE_COL].rolling(FAST_MA, min_periods=1).mean()
    df['ma_slow'] = df[PRICE_COL].rolling(SLOW_MA, min_periods=1).mean()
    df['ma_diff'] = df['ma_fast'] - df['ma_slow']
    return df


# Simple backtest using MA crossover
# No lookahead: signal on day t is computed using data up to t (rolling windows)
# and trade is executed at the next day's open (we use next day's price as execution if available)

def backtest_ma_crossover(df, year, max_trades_per_ticker=MAX_TRADES_PER_TICKER,
                          min_days_between=MIN_DAYS_BETWEEN_TRADES):
    trades = []
    df = compute_indicators(df)
    df['year'] = df[DATE_COL].dt.year
    df_year = df[df['year'] == int(year)].reset_index(drop=True)
    if df_year.empty:
        return pd.DataFrame(trades)

    position = None
    last_trade_day = None
    trades_done = 0

    # Iterate day by day
    for i in range(len(df_year)-1):
        today = df_year.loc[i]
        tomorrow = df_year.loc[i+1]  # execution at next day's price (no lookahead beyond that)

        # skip illiquid days
        if 'Volume' in df_year.columns and (pd.isna(today['Volume']) or today['Volume'] < MIN_VOLUME):
            continue

        # apply constraints
        if trades_done >= max_trades_per_ticker:
            break
        if last_trade_day is not None and (today[DATE_COL] - last_trade_day).days < min_days_between:
            continue

        # generate buy signal: fast MA crosses above slow MA (ma_diff turns positive)
        # We compute the sign change using today's ma_diff and previous day's ma_diff computed on historical data
        if i == 0:
            prev_ma_diff = np.nan
        else:
            prev_ma_diff = df_year.loc[i-1]['ma_diff']

        ma_diff_today = today['ma_diff']

        buy_signal = False
        sell_signal = False

        if position is None:
            # buy when ma_diff crosses from <=0 to >0
            if (not pd.isna(prev_ma_diff)) and prev_ma_diff <= 0 and ma_diff_today > 0:
                buy_signal = True
        else:
            # sell when ma_diff crosses from >=0 to <0 OR stop loss / take profit reached
            if (not pd.isna(prev_ma_diff)) and prev_ma_diff >= 0 and ma_diff_today < 0:
                sell_signal = True
            else:
                # check stop loss / take profit against today's close (we'll execute sell next day)
                current_ret = today[PRICE_COL] / position['buy_price'] - 1
                if current_ret >= TAKE_PROFIT or current_ret <= STOP_LOSS:
                    sell_signal = True

        # Execute buy at tomorrow's price
        if buy_signal and position is None:
            buy_price = tomorrow[PRICE_COL]
            buy_date = tomorrow[DATE_COL]
            position = {'buy_date': buy_date, 'buy_price': buy_price, 'ticker': today.get('Ticker', None)}
            last_trade_day = buy_date
            trades_done += 1

        # Execute sell at tomorrow's price
        if sell_signal and position is not None:
            sell_price = tomorrow[PRICE_COL]
            sell_date = tomorrow[DATE_COL]
            pct = (sell_price / position['buy_price'] - 1) * 100
            trades.append({
                'Ticker': position.get('ticker', ''),
                'Buy Date': position['buy_date'].strftime('%Y-%m-%d'),
                'Buy Price': round(position['buy_price'], 4),
                'Sell Date': sell_date.strftime('%Y-%m-%d'),
                'Sell Price': round(sell_price, 4),
                'Percentage Profit': round(pct, 4)
            })
            position = None
            last_trade_day = sell_date

    # If still holding at the end of year, close at last available price
    if position is not None:
        last_row = df_year.iloc[-1]
        sell_price = last_row[PRICE_COL]
        sell_date = last_row[DATE_COL]
        pct = (sell_price / position['buy_price'] - 1) * 100
        trades.append({
            'Ticker': position.get('ticker', ''),
            'Buy Date': position['buy_date'].strftime('%Y-%m-%d'),
            'Buy Price': round(position['buy_price'], 4),
            'Sell Date': sell_date.strftime('%Y-%m-%d'),
            'Sell Price': round(sell_price, 4),
            'Percentage Profit': round(pct, 4)
        })

    trades_df = pd.DataFrame(trades)
    return trades_df


# Plot function with buy/sell markers

def plot_trades(df, trades_df, ticker, year, outdir=PLOTS_DIR):
    os.makedirs(outdir, exist_ok=True)
    df = df.copy()
    df['year'] = df[DATE_COL].dt.year
    df_year = df[df['year'] == int(year)].reset_index(drop=True)
    if df_year.empty:
        return None

    fig, ax = plt.subplots(figsize=(12,6))
    ax.plot(df_year[DATE_COL], df_year[PRICE_COL], label=f'{ticker} {PRICE_COL}')
    # plot moving averages
    if 'ma_fast' in df_year.columns:
        ax.plot(df_year[DATE_COL], df_year['ma_fast'], linestyle='--', linewidth=1, label=f'MA{FAST_MA}')
    if 'ma_slow' in df_year.columns:
        ax.plot(df_year[DATE_COL], df_year['ma_slow'], linestyle='-', linewidth=1, label=f'MA{SLOW_MA}')

    # overlay trades
    my_trades = trades_df[trades_df['Ticker'] == ticker]
    for _, row in my_trades.iterrows():
        buy_date = pd.to_datetime(row['Buy Date'])
        sell_date = pd.to_datetime(row['Sell Date'])
        # find price points if they exist in df_year, otherwise skip marker
        b_price = row['Buy Price']
        s_price = row['Sell Price']
        ax.scatter([buy_date], [b_price], marker='^', s=100, label='Buy')
        ax.scatter([sell_date], [s_price], marker='v', s=100, label='Sell')
        ax.plot([buy_date, sell_date], [b_price, s_price], linestyle=':', linewidth=1)

    ax.set_title(f'{ticker} trades in {year}')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.legend()
    fname = os.path.join(outdir, f'{ticker}_{year}.png')
    plt.tight_layout()
    fig.savefig(fname)
    plt.close(fig)
    return fname


# Main orchestration

def main(folder, year, strategy='ma'):
    all_trades = []
    files = [f for f in os.listdir(folder) if f.lower().endswith('.csv')]
    if not files:
        print('No CSV files found in', folder)
        return

    for f in tqdm(files, desc='Tickers'):
        ticker = os.path.splitext(f)[0]
        path = os.path.join(folder, f)
        try:
            df = load_csv_for_ticker(path)
        except Exception as e:
            print('Failed to load', f, 'error:', e)
            continue

        # ensure price column exists
        if PRICE_COL not in df.columns:
            # try common variants
            for alt in ['Adj Close', 'Adj_Close', 'Close']:
                if alt in df.columns:
                    df[PRICE_COL] = df[alt]
                    break
        if PRICE_COL not in df.columns:
            print('No suitable price column for', ticker)
            continue

        # add Ticker column for later use
        df['Ticker'] = ticker

        if strategy == 'ma':
            trades_df = backtest_ma_crossover(df, year)
        else:
            # placeholder for other strategies
            trades_df = backtest_ma_crossover(df, year)

        if not trades_df.empty:
            # fill ticker column if empty
            trades_df['Ticker'] = trades_df['Ticker'].replace('', ticker)
            all_trades.append(trades_df)
            # plot
            plot_trades(df, trades_df, ticker, year)

    if all_trades:
        result = pd.concat(all_trades, ignore_index=True)
    else:
        result = pd.DataFrame(columns=['Ticker','Buy Date','Buy Price','Sell Date','Sell Price','Percentage Profit'])

    outcsv = OUTPUT_TRADES_CSV.format(year=year)
    result.to_csv(outcsv, index=False)
    print(f'Saved trades to {outcsv}')
    print(f'Saved plots to {PLOTS_DIR}')

    # ---- Summary statistics ----
    if not result.empty:
        avg_gain = result['Percentage Profit'].mean()
        total_gain = result['Percentage Profit'].sum()
        median_gain = result['Percentage Profit'].median()
        win_rate = (result['Percentage Profit'] > 0).mean() * 100

        print("\n=== Trade Summary ===")
        print(f"Total trades: {len(result)}")
        print(f"Average % profit per trade: {avg_gain:.2f}%")
        print(f"Median % profit per trade: {median_gain:.2f}%")
        print(f"Total % gain (sum of trades): {total_gain:.2f}%")
        print(f"Win rate: {win_rate:.1f}%")

        # Optional: summary by ticker
        summary = (
            result.groupby('Ticker')['Percentage Profit']
            .agg(['count', 'mean', 'sum'])
            .rename(columns={'count': 'Num Trades', 'mean': 'Avg % Profit', 'sum': 'Total % Profit'})
            .sort_values('Total % Profit', ascending=False)
        )
        summary_csv = f'summary_{year}.csv'
        summary.to_csv(summary_csv)
        print(f"\nSaved per-ticker summary to {summary_csv}")
    else:
        print("\nNo trades were executed for this year.")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', default=FOLDER, help='Folder containing ticker CSVs')
    parser.add_argument('--year', required=True, help='Year to test (e.g. 2020)')
    parser.add_argument('--strategy', default='ma', choices=['ma'], help='Which strategy to run')
    args = parser.parse_args()
    main(args.folder, args.year, strategy=args.strategy)


# ----------------------
# Optional: Walk-forward ML (advanced)
# ----------------------
# The following outlines a walk-forward approach you can implement if you want a predictive
# model that "learns" from past data without looking into the future. It is intentionally
# left commented because it requires scikit-learn and careful testing. If you'd like, I can
# enable and tune this section for you.
#
# Basic idea:
# 1. Build features for each date (returns, MAs, RSI, volume features, etc.) using only past data
# 2. Define a label using forward return over N days (e.g. 5 days) > threshold (this label is only
#    available for training; when predicting at day t you must only use features up to t)
# 3. For each prediction day t in the test year, train the model on all historical data up to t-1
#    and predict for day t. If model predicts positive, buy at t+1 open and hold until opposite signal.
#
# This walk-forward is slower but respects the no-lookahead rule.
# ----------------------
