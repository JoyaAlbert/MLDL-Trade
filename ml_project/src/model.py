"""Entrenamiento y predicción de modelo simple."""
from typing import Tuple
import numpy as np
import pandas as pd


def prepare_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['target'] = (df['Adj Close'].shift(-1) > df['Adj Close']).astype(int)
    return df.dropna()


def train_model(df: pd.DataFrame):
    """Entrena un LogisticRegression sencillo y devuelve (model, scaler)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    df2 = prepare_labels(df)
    features = ['ret', 'sma_5', 'sma_10', 'sma_20', 'vol_10']
    X = df2[features].values
    y = df2['target'].values

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    clf = LogisticRegression(max_iter=200)
    clf.fit(Xs, y)
    return clf, scaler


def predict_next(clf, scaler, df: pd.DataFrame) -> Tuple[int, float]:
    """Predice la probabilidad de subida para el siguiente día y devuelve (pred, prob_up).

    pred: 1 si se predice subida, 0 si bajada.
    prob_up: probabilidad estimada de subida.
    """
    features = ['ret', 'sma_5', 'sma_10', 'sma_20', 'vol_10']
    last = df[features].iloc[-1:].fillna(0)
    Xs = scaler.transform(last.values)
    prob = clf.predict_proba(Xs)[0][1]
    pred = int(prob > 0.5)
    return pred, float(prob)
