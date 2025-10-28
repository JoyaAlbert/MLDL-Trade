"""Script runner: descarga, crea features, entrena y predice para una lista de tickers.

Ejecutar directamente: `python3 ml_project/run.py`
"""
import os
import sys
from pprint import pprint

# Asegurar que `src` esté en sys.path
ROOT = os.path.dirname(__file__)
SRC = os.path.join(ROOT, 'src')
sys.path.insert(0, SRC)

import data as data_module
import features as features_module
import model as model_module


def recommend_from_prob(prob: float) -> str:
    if prob >= 0.55:
        return 'BUY'
    if prob <= 0.45:
        return 'SELL'
    return 'HOLD'


def run_for_ticker(ticker: str):
    print(f"\n==== {ticker} ====")
    try:
        df = data_module.history_until_previous_day(ticker, period='400d')
    except Exception as e:
        print(f"Error descargando datos para {ticker}: {e}")
        return

    df_feat = features_module.add_features(df)

    # Entrenar con todo el histórico disponible (demostración)
    try:
        clf, scaler = model_module.train_model(df_feat)
    except Exception as e:
        print(f"Error entrenando modelo para {ticker}: {e}")
        return

    pred, prob = model_module.predict_next(clf, scaler, df_feat)
    action = recommend_from_prob(prob)

    print(f"Predicción subida (prob): {prob:.3f} -> {action}")


def main():
    tickers = ['BTC-USD', 'TSLA']
    print("MLDL-Trade: predicciones simples para tickers")
    for t in tickers:
        run_for_ticker(t)


if __name__ == '__main__':
    main()
