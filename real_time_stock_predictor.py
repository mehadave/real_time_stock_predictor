import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from datetime import datetime
from streamlit_autorefresh import st_autorefresh

st.set_page_config(page_title="📈 Real-Time Stock Price Predictor", layout="centered")
st.title("📊 Real-Time Stock Price Predictor")

# 🔁 Optional: Auto-refresh every 60 seconds
st_autorefresh(interval=60000, key="auto_refresh")

# Sidebar
st.sidebar.header("Stock Settings")
ticker = st.sidebar.text_input("Enter Ticker Symbol (e.g., AAPL)", "AAPL")
interval = st.sidebar.selectbox("Select Interval", ["1m", "5m", "15m", "30m"], index=0)
period = st.sidebar.selectbox("Select Period", ["1d", "5d", "7d"], index=0)

# Download intraday data
@st.cache_data(ttl=60)  # Refresh every minute
def load_intraday_data(ticker, interval, period):
    df = yf.download(ticker, interval=interval, period=period)
    return df

data = load_intraday_data(ticker, interval, period)

if data.empty:
    st.warning("No data available. Please check the ticker or interval.")
    st.stop()

# Show data
st.write(f"### Live Intraday Data for {ticker}")
st.dataframe(data.tail())

# Feature engineering
data['Target'] = data['Close'].shift(-1)
data['MA_5'] = data['Close'].rolling(window=5).mean()
data.dropna(inplace=True)

X = data[['Close', 'MA_5']]
y = data['Target']

# Simple split (90% train, 10% test)
split_index = int(len(X) * 0.9)
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

st.markdown("### 🔍 Model Evaluation")
st.write(f"**Mean Squared Error:** {mse:.4f}")
st.write(f"**R² Score:** {r2:.4f}")

# Plot predictions
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(y_test.values[-50:], label="Actual", marker='o')
ax.plot(y_pred[-50:], label="Predicted", marker='x')
ax.set_title(f"{ticker} Price Prediction ({interval} interval)")
ax.set_ylabel("Price ($)")
ax.set_xlabel("Time Steps")
ax.legend()
ax.grid(True)
st.pyplot(fig)

# Predict the next value
latest = X.iloc[-1].values.reshape(1, -2)
next_price = model.predict(latest)[0]

st.markdown("### 🔮 Real-Time Prediction")
st.write(f"**Predicted next {interval} close for {ticker}:** ${next_price:.2f}")
st.caption(f"Last updated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
