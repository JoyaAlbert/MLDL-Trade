# ML Trading Mini-project

Proyecto pequeño que descarga datos históricos de trading (ej. `BTC-USD` y `TSLA`), crea características simples, entrena un modelo básico y predice la tendencia del día siguiente para dar una recomendación (BUY/HOLD/SELL).

Requisitos:
- Python 3.8+
- Ver `requirements.txt` para dependencias

Uso rápido:

1) Crear un entorno e instalar dependencias:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r ml_project/requirements.txt
```

2) Ejecutar el runner (descarga datos y predice para BTC y TSLA):

```bash
python3 ml_project/run.py
```

3) Ejecutar el dashboard interactivo (Streamlit):

```bash
# activar entorno virtual si procede
source .venv/bin/activate
pip install -r ml_project/requirements.txt
streamlit run ml_project/dashboard.py
```

En el dashboard puedes seleccionar tickers y periodo histórico. El dashboard descargará datos (yfinance), mostrará gráficos y la probabilidad de subida para el siguiente día.

Notas:
- El script usa `yfinance` para descargar datos históricos. Necesita conexión a Internet.
- El modelo es intencionalmente simple (logistic regression) y sirve como demostración. No uses esto para trading real.
