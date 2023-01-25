"""
Create a stock forecaster
"""
import yfinance as yf
import datetime
import pandas as pd
import ast

import configparser

import numpy as np
from finta import TA
import matplotlib.pyplot as plt

from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
from sklearn import metrics

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


def train_random_forest(X_train: pd.DataFrame,
                        y_train: pd.Series,
                        X_test: pd.DataFrame,
                        y_test: pd.Series,
                        print_pred: bool=False
                        ) -> RandomForestClassifier:
    """

    Args:
        X_train:
        y_train:
        X_test:
        y_test:
        print_pred:

    Returns:

    """
    rf = RandomForestClassifier()
    params_rf = {'n_estimators': [110, 130, 140, 150, 160, 180, 200]}
    rf_gs = GridSearchCV(rf, params_rf, cv=5)
    rf_gs.fit(X_train, y_train)
    rf_best = rf_gs.best_estimator_

    if print_pred:
        pred = rf_best.predict(X_test)
        print(classification_report(y_test, pred))
        print(confusion_matrix(y_test, pred))

    return rf_best


def train_KNN(X_train: pd.DataFrame,
              y_train: pd.Series,
              X_test: pd.DataFrame,
              y_test: pd.Series,
              print_pred: bool=False
              ) -> KNeighborsClassifier:
    knn = KNeighborsClassifier()

    params_knn = {'n_neighbors': np.arange(1, 25)}
    knn_gs = GridSearchCV(knn, params_knn, cv=5)
    knn_gs.fit(X_train, y_train)
    knn_best = knn_gs.best_estimator_

    if print_pred:
        print(knn_gs.best_params_)
        prediction = knn_best.predict(X_test)
        print(classification_report(y_test, prediction))
        print(confusion_matrix(y_test, prediction))

    return knn_best


def train_ensemble(rf_model: RandomForestClassifier,
                   knn_model: KNeighborsClassifier,
                   X_train: pd.DataFrame,
                   y_train: pd.Series,
                   X_test: pd.DataFrame,
                   y_test: pd.Series,
                   print_score: bool=False
                   ) -> VotingClassifier:
    """

    Args:
        rf_model:
        knn_model:
        X_train:
        y_train:
        X_test:
        y_test:

    Returns:

    """
    estimators = [
        ('rf', rf_model),
        ('knn', knn_model)
    ]
    ensemble = VotingClassifier(estimators, voting='hard')
    ensemble.fit(X_train, y_train)

    if print_score:
        pred = ensemble.predict(X_test)
        print(classification_report(y_test, pred))
        print(confusion_matrix(y_test, pred))

    return ensemble


# Feature Engineering
df = exponential_smooth(df, 0.7)
fig = df['close'].plot()
# fig.show()

df = get_indicator_data(df, indicators)
print(df.columns)

df = produce_prediction(df, 15)

df = df.dropna()

print(df.tail())

# Create Ensemble model
y = df['pred']
features = [x for x in df.columns if x not in ['pred']]
X = df[features]

(X_train,
 X_test,
 y_train,
 y_test) = train_test_split(X, y, train_size=(7 * len(X) // 10), shuffle=False)

rf_model = train_random_forest(X_train, y_train, X_test, y_test, True)

knn_model = train_KNN(X_train, y_train, X_test, y_test, True)

ensemble_model = train_ensemble(rf_model,
                                knn_model,
                                X_train,
                                y_train,
                                X_test,
                                y_test,
                                True)
