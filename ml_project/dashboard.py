"""Streamlit dashboard para visualizar datos descargados y la predicción simple.

Ejecutar:
    streamlit run ml_project/dashboard.py
"""
import os
import sys
from datetime import timedelta

ROOT = os.path.dirname(__file__)
SRC = os.path.join(ROOT, 'src')
sys.path.insert(0, SRC)

import streamlit as st
import pandas as pd
import plotly.express as px

import data as data_module
import features as features_module
import model as model_module


st.set_page_config(page_title='MLDL-Trade Dashboard', layout='wide')

st.title('MLDL-Trade — Dashboard')

with st.sidebar:
    st.header('Configuración')
    # Lista de tickers disponibles y seleccionados en session_state
    if 'available_tickers' not in st.session_state:
        st.session_state.available_tickers = [
            'BTC-USD', 'TSLA', 'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'ETH-USD'
        ]
    if 'selected_tickers' not in st.session_state:
        st.session_state.selected_tickers = ['BTC-USD', 'TSLA']

    # Multiselect para elegir tickers a mostrar (por defecto BTC y TSLA)
    st.multiselect('Tickers (selecciona uno o más)',
                   options=st.session_state.available_tickers,
                   default=st.session_state.selected_tickers,
                   key='selected_tickers')

    # Permitir al usuario añadir cualquier ticker que soporte yfinance
    new_ticker = st.text_input('Agregar ticker (ej: AAPL o BTC-USD)', key='new_ticker')
    if st.button('Agregar ticker'):
        nt = new_ticker.strip().upper()
        if nt:
            if nt not in st.session_state.available_tickers:
                st.session_state.available_tickers.append(nt)
            # Añadir a seleccionados automáticamente
            if nt not in st.session_state.selected_tickers:
                st.session_state.selected_tickers.append(nt)

    # Opción de periodo y botón de refresh
    period = st.selectbox('Periodo histórico', ['90d','180d','400d','800d'], index=2)
    refresh = st.button('Actualizar / Descargar')

    # Mostrar una lista corta de sugerencias (no exhaustiva) con tickers comunes
    st.markdown('**Sugerencias:** AAPL, MSFT, GOOGL, AMZN, ETH-USD')

# Lectura de tickers seleccionados para el resto del script
tickers = st.session_state.get('selected_tickers', ['BTC-USD','TSLA'])


# Explicación / Glosario para usuarios principiantes
with st.expander('¿Qué significan los términos usados? (Glosario)', expanded=False):
    st.markdown(
        """
        - **Adj Close (Precio ajustado):** precio de cierre ajustado por divisiones (splits) y dividendos. Es el precio más usado para análisis históricos.
        - **Close / Precio de cierre:** precio al final de la sesión de trading. Si no hay `Adj Close`, usamos `Close`.
        - **SMA (Simple Moving Average / Media móvil simple):** promedio del precio durante las últimas N barras (por ejemplo SMA 5 = media de 5 días). Se usa para identificar tendencias.
        - **Returns (Retornos):** cambio porcentual diario del precio: (P_t / P_{t-1}) - 1. Indica si el precio subió o bajó y cuánto.
        - **Volatilidad (std):** medida de dispersión de los retornos (ej. `vol_10` = desviación estándar de retornos en 10 días). Más volatilidad = movimientos más grandes.
        - **Probabilidad de subida:** valor entre 0 y 1 que indica la probabilidad estimada de que el precio suba el día siguiente según el modelo.
        - **Recomendación (BUY/HOLD/SELL):** traducción simple de la probabilidad: si la probabilidad es alta -> BUY; baja -> SELL; intermedio -> HOLD.
        """
    )


def plot_price(df: pd.DataFrame, ticker: str):
    fig = px.line(df, x=df.index, y='Adj Close', title=f'{ticker} — Adj Close')
    # medias
    for w in (5,10,20):
        if f'sma_{w}' in df.columns:
            fig.add_scatter(x=df.index, y=df[f'sma_{w}'], mode='lines', name=f'SMA {w}')
    st.plotly_chart(fig, use_container_width=True)


def plot_returns(df: pd.DataFrame, ticker: str):
    fig = px.line(df, x=df.index, y='ret', title=f'{ticker} — Returns')
    st.plotly_chart(fig, use_container_width=True)


def process_and_show(ticker: str, period: str):
    st.subheader(ticker)
    try:
        df = data_module.history_until_previous_day(ticker, period=period)
    except Exception as e:
        st.error(f'Error descargando datos para {ticker}: {e}')
        return

    df_feat = features_module.add_features(df)

    # Mostrar tabla con últimas filas
    st.write('Últimas filas (features):')
    st.dataframe(df_feat.tail(10))

    # Gráficos
    cols = st.columns(2)
    with cols[0]:
        plot_price(df_feat, ticker)
    with cols[1]:
        plot_returns(df_feat, ticker)

    # Entrenar modelo y predecir
    try:
        clf, scaler = model_module.train_model(df_feat)
        pred, prob = model_module.predict_next(clf, scaler, df_feat)
        action = 'BUY' if prob >= 0.55 else 'SELL' if prob <= 0.45 else 'HOLD'

        # Mostrar recomendación con visual más llamativa
        color = '#f59e0b'  # amber por defecto (HOLD)
        if action == 'BUY':
            color = '#16a34a'  # verde

        elif action == 'SELL':
            color = '#e11d48'  # rojo

        box_html = f"""
        <div style='border-radius:8px; padding:16px; background:{color}; color:white; text-align:center; margin-top:8px;'>
          <div style='font-size:32px; font-weight:800; margin-top:6px;'>{action}</div>
          <div style='font-size:14px; opacity:0.95; margin-top:6px;'>Probabilidad de subida: {prob:.3f}</div>
        </div>
        """
        st.markdown(box_html, unsafe_allow_html=True)
    except Exception as e:
        st.warning(f'No se pudo entrenar/predicción: {e}')


def main():
    st.write('Selecciona ticker(s) y pulsa **Actualizar / Descargar**')
    if not tickers:
        st.info('Selecciona al menos un ticker en la barra lateral.')
        return

    for t in tickers:
        process_and_show(t, period)

    st.sidebar.markdown('---')
    st.sidebar.write('Conexión: yfinance (Internet requerida)')


if __name__ == '__main__':
    main()
