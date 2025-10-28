"""Módulo para descargar datos de mercado usando yfinance."""
from datetime import datetime
import pandas as pd
import yfinance as yf


def download_history(ticker: str, period: str = "365d") -> pd.DataFrame:
    """Descarga histórico de precios para `ticker` usando yfinance.

    Devuelve un DataFrame con índice de fecha y columnas [Open, High, Low, Close, Adj Close, Volume].
    """
    # Usar auto_adjust=True para obtener precios ajustados y evitar FutureWarning
    df = yf.download(ticker, period=period, progress=False, auto_adjust=True)
    if df is None or df.empty:
        raise RuntimeError(f"No se pudieron descargar datos para {ticker}")
    df.index = pd.to_datetime(df.index)
    # Si yfinance devuelve columnas MultiIndex (e.g., when tickers are included),
    # normalizamos a columnas simples. Para un único ticker, preferimos usar
    # la primera parte del nombre de columna (e.g., ('Close','BTC-USD') -> 'Close').
    if hasattr(df.columns, 'nlevels') and df.columns.nlevels > 1:
        new_cols = []
        for col in df.columns:
            try:
                # col suele ser una tupla (level0, level1)
                level0 = col[0]
                level1 = col[1]
            except Exception:
                # si por alguna razón no es indexado, convertir a str
                new_cols.append(str(col))
                continue

            if level1 == ticker or level1 == '' or pd.isna(level1):
                new_cols.append(level0)
            else:
                # si hay otros tickers en el MultiIndex, concatenar para evitar colisiones
                new_cols.append(f"{level0}_{level1}")

        df.columns = new_cols
    return df


def get_previous_trading_day(df: pd.DataFrame) -> pd.Timestamp:
    """Devuelve el índice (Timestamp) del último día de trading anterior a hoy dentro del DataFrame.

    Para evitar errores al comparar índices timezone-aware con objetos timezone-naive, comparamos
    a nivel de fecha (datetime.date) en lugar de Timestamps completos.
    """
    # fecha UTC actual (date)
    today_date = pd.Timestamp.utcnow().date()
    # obtener array de fechas (datetime.date) desde el índice — evita problemas de tz
    idx_dates = df.index.date
    mask = idx_dates < today_date
    if not mask.any():
        return None
    # devolver el último Timestamp del índice que cumple la condición
    return df.index[mask].max()


def history_until_previous_day(ticker: str, period: str = "365d") -> pd.DataFrame:
    """Descarga histórico y devuelve datos hasta el último día de trading (inclusive)."""
    df = download_history(ticker, period=period)
    prev = get_previous_trading_day(df)
    if prev is None:
        raise RuntimeError("No se encontró día de trading anterior en los datos descargados")
    return df.loc[:prev].copy()
