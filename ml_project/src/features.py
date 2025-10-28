"""Cálculo de features sencillas para series de precios."""
import pandas as pd


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega features básicos: retornos, medias móviles y volatilidad.

    Requiere columna 'Adj Close'. Devuelve DataFrame con filas sin NaNs.
    """
    df = df.copy()
    # Aceptar varias posibles columnas de precio ajustado/close que pueda devolver yfinance
    price_col = None
    candidates = ['Adj Close', 'Adj_Close', 'AdjClose', 'Close', 'close']
    # búsqueda directa por nombres habituales
    for c in candidates:
        if c in df.columns:
            price_col = c
            break

    # Si no encontramos una coincidencia exacta, buscar columnas que contengan 'adj' y 'close'
    if price_col is None:
        lower_cols = [str(c).lower() for c in df.columns]
        # preferir columnas que contengan both 'adj' and 'close'
        for i, lc in enumerate(lower_cols):
            if 'adj' in lc and 'close' in lc:
                price_col = df.columns[i]
                break

    # si aún no, tomar la primera columna que contenga 'close'
    if price_col is None:
        for i, lc in enumerate(lower_cols):
            if 'close' in lc:
                price_col = df.columns[i]
                break

    # finalmente, si no hay ninguna columna con 'close', intentar columnas con 'price' o 'adj'
    if price_col is None:
        for i, lc in enumerate(lower_cols):
            if 'price' in lc or 'adj' in lc:
                price_col = df.columns[i]
                break

    if price_col is None:
        raise ValueError("DataFrame no contiene columna 'Adj Close' ni 'Close' ni otra columna de precio reconocible")

    # Normalizar a 'Adj Close' para el resto del pipeline
    if str(price_col) != 'Adj Close':
        df['Adj Close'] = df[price_col]

    df['ret'] = df['Adj Close'].pct_change()
    for w in (5, 10, 20):
        df[f'sma_{w}'] = df['Adj Close'].rolling(window=w).mean()

    df['vol_10'] = df['ret'].rolling(window=10).std()

    df = df.dropna()
    return df
