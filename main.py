"""
Create a stock forecaster
"""
import yfinance as yf
import datetime
import pandas as pd
import ast

import configparser

pd.options.plotting.backend = "plotly"

config = configparser.ConfigParser()
config.read('config.ini')

num_days = int(config['DEFAULT']['num_days'])
symbol = config['DEFAULT']['symbol']
interval = config['DEFAULT']['interval']
indicators = ast.literal_eval(config['DEFAULT']['indicators'])

start_date = datetime.date.today() - datetime.timedelta(num_days)
end_date = datetime.datetime.today()

df = yf.download(symbol, start=start_date, end=end_date, interval=interval)

for col in df.columns:
    df.rename(columns={col: col.lower()}, inplace=True, errors='raise')

print(df.head())

fig = df['close'].plot()
#fig.show()


def exponential_smooth(df: pd.DataFrame, alpha: float) -> pd.DataFrame:
    """

    Args:
        df:
        alpha:

    Returns:

    """
    return df.ewm(alpha=alpha).mean()


def get_indicator_data(df: pd.DataFrame, indicators: list) -> pd.DataFrame:
    """

    Args:
        df:

    Returns:

    """
    for indicator in indicators:
        ind_df = eval('TA.' + indicator + '(df)')
        if not isinstance(ind_df, pd.DataFrame):
            ind_df = ind_df.to_frame()
        df = df.merge(ind_df, left_index=True, right_index=True)
    df = df.rename(columns={"14 period EMV.": "14 period EMV"})

    # Calculate moving averages for features
    df['ema50'] = df['close'] / df['close'].ewm(50).mean()
    df['ema21'] = df['close'] / df['close'].ewm(21).mean()
    df['ema15'] = df['close'] / df['close'].ewm(14).mean()
    df['ema5'] = df['close'] / df['close'].ewm(5).mean()

    # Normalise volume value with moving average
    df['norm_vol'] = df['volume'] / df['volume'].ewm(5).mean()

    cols_to_del = ['open', 'high', 'low', 'volume', 'adj close']
    for col in cols_to_del:
        try:
            del (df[col])
        except:
            print(f'Unable to delete {col} as it does not exist')

    return df


def produce_prediction(df: pd.DataFrame, window: int=15) -> pd.DataFrame:
    """

    Args:
        df:
        window:

    Returns:

    """
    pred = (df.shift(-window)['close'] >= df['close'])
    df['pred'] = pred.iloc[:-window].astype(int)
    return df


df = exponential_smooth(df, 0.7)
fig = df['close'].plot()
fig.show()

df = get_indicator_data(df, indicators)
print(df.columns)

df = produce_prediction(df, 15)

df= df.dropna()

print(df.tail())
