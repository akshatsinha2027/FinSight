import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["KMP_WARNINGS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # suppress TF logs

import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
from tensorflow.keras.models import load_model

# ───────────────────────────────────────────────
# Streamlit Page Config
st.set_page_config(page_title="FinSight - Stock Forecasting App", layout="wide")
st.title("📈 FinSight: Stock Forecasting with GRU")
st.write("Enter a stock ticker symbol to forecast future prices using GRU deep learning model.")

# ───────────────────────────────────────────────
# 1️⃣ Import Stock Data
def import_data(ticker: str):
    yf.utils.set_proxy(None)
    df = yf.download(ticker, period='max', auto_adjust=False)
    df = df.reset_index()
    df.columns = ['Date', 'Adj Close', 'Close', 'High', 'Low', 'Open', 'Volume']
    df = df[['Date', 'Open', 'Close', 'Adj Close', 'High', 'Low', 'Volume']]
    return df

# ───────────────────────────────────────────────
# 2️⃣ Feature Engineering
def feature_engineering(df: pd.DataFrame):
    from ta.trend import MACD
    from ta.momentum import RSIIndicator

    macd = MACD(close=df['Adj Close'])
    df['MACD_Line'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    df['MACD_Histogram'] = macd.macd_diff()

    rsi = RSIIndicator(close=df['Adj Close'])
    df['RSI_14'] = rsi.rsi()

    df['Daily_Return_%'] = 100 * (df['Close'] - df['Close'].shift(1)) / df['Close'].shift(1)
    
    for w in [10, 20, 50, 100, 200]:
        df[f'SMA_{w}'] = df['Adj Close'].rolling(window=w).mean()
        df[f'EMA_{w}'] = df['Adj Close'].ewm(span=w, adjust=False).mean()

    for lag in [1, 2, 3, 5, 7, 14]:
        df[f'Close_LAG{lag}'] = df['Close'].shift(lag)

    df = df.dropna()
    return df

# ───────────────────────────────────────────────
# 3️⃣ GRU Model Training
def model_train(df, n_steps=100, k=1):
    from sklearn.preprocessing import MinMaxScaler
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import GRU, Dense

    data = df[['Adj Close']].values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    x, y = [], []
    for i in range(n_steps, len(scaled_data) - k + 1):
        x.append(scaled_data[i - n_steps:i])
        y.append(scaled_data[i:i + k])
    x, y = np.array(x), np.array(y)

    split = int(0.8 * len(x))
    x_train, x_test = x[:split], x[split:]
    y_train, y_test = y[:split], y[split:]

    model = Sequential([
        GRU(80, input_shape=(x_train.shape[1], 1), return_sequences=True),
        GRU(60),
        Dense(y_train.shape[1])
    ])
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(x_train, y_train, epochs=20, batch_size=32, verbose=0)
    model.save("models/finsight_gru.h5")

    return model, scaler, x_test, y_test

# ───────────────────────────────────────────────
# 4️⃣ Prediction + Evaluation
def predict(model, scaler, x_test, y_test):
    y_pred = model.predict(x_test)
    y_pred_inv = scaler.inverse_transform(y_pred)
    y_test_inv = scaler.inverse_transform(y_test)

    import matplotlib.pyplot as plt
    import io

    fig, ax = plt.subplots()
    ax.plot(y_test_inv, label="Actual", alpha=0.8)
    ax.plot(y_pred_inv, label="Predicted", alpha=0.8)
    ax.legend()
    ax.set_title("Predicted vs Actual Prices")

    from sklearn.metrics import mean_absolute_percentage_error
    mape = mean_absolute_percentage_error(y_test_inv, y_pred_inv) * 100

    return fig, mape

# ───────────────────────────────────────────────
# Streamlit UI
ticker = st.text_input("Enter Stock Ticker (e.g. AAPL, TSLA, MSFT):", "AAPL")
n_steps = st.slider("Lookback Window (Days)", 50, 200, 100)
k = st.slider("Days to Forecast Ahead", 1, 10, 1)

if st.button("Run Forecast"):
    with st.spinner("Fetching and processing data..."):
        df = import_data(ticker)
        df = feature_engineering(df)
        st.success(f"✅ Data fetched successfully! {len(df)} rows ready.")

    with st.spinner("Training GRU model... (approx. 1–2 mins)"):
        model, scaler, x_test, y_test = model_train(df, n_steps, k)

    with st.spinner("Generating predictions..."):
        fig, mape = predict(model, scaler, x_test, y_test)
        st.pyplot(fig)
        st.metric(label="MAPE (Mean Absolute % Error)", value=f"{mape:.2f}%")
        st.success("✅ Forecast complete!")
