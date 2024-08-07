import os
import numpy as np
import pandas as pd
import yfinance as yf
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt

# Suppress oneDNN informational messages
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Function to get stock data
def get_stock_data(stock_symbol, start_date, end_date):
    stock_data = yf.download(stock_symbol, start=start_date, end=end_date)
    return stock_data

# Input from the user
stock_symbol = input("Enter the stock symbol (e.g., AAPL for Apple): ").upper()
start_date = input("Enter the start date for data (YYYY-MM-DD): ")
end_date = input("Enter the end date for data (YYYY-MM-DD): ")
future_days = int(input("Enter the number of future days to predict: "))

# Fetch the stock data
data = get_stock_data(stock_symbol, start_date, end_date)

if data.empty:
    print(f"No data found for {stock_symbol}. Please check the stock symbol and date range.")
else:
    # Preprocess the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

    # Create training datasets
    prediction_days = 60
    X_train, y_train = [], []
    for x in range(prediction_days, len(scaled_data)):
        X_train.append(scaled_data[x - prediction_days:x, 0])
        y_train.append(scaled_data[x, 0])

    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    # Build the LSTM model
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))  # Prediction of the next closing value

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=25, batch_size=32)

    # Predict future prices
    test_input = scaled_data[-prediction_days:]  # last known data
    future_predictions = []

    for _ in range(future_days):
        test_input = test_input.reshape(1, prediction_days, 1)
        prediction = model.predict(test_input)
        future_predictions.append(prediction[0, 0])
        test_input = np.append(test_input[:, 1:, :], prediction).reshape(-1, 1)

    future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

    # Generate future dates
    future_dates = pd.date_range(data.index[-1], periods=future_days + 1)[1:]

    # Plot the results
    plt.figure(figsize=(14, 5))
    plt.plot(data.index[-len(test_input):], scaler.inverse_transform(scaled_data[-len(test_input):]), color='blue', label='Actual Stock Price')
    plt.plot(future_dates, future_predictions, color='red', label='Predicted Future Prices')
    plt.title(f'{stock_symbol} Future Stock Price Prediction')
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()

