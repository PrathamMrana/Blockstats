import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.model_selection import train_test_split

name = "BTCUSDT"
# Load your dataset
df = pd.read_csv(f'CryptoPredictor/{name}_price_data.csv')

# Check if Date column exists
if 'Date' in df.columns:
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
else:
    raise ValueError("Dataset must have a 'Date' column.")

# Check if Price column exists
if 'Price' not in df.columns:
    raise ValueError("Dataset must have a 'Price' column.")

# Display first few rows to ensure data is loaded correctly
print(df.head())

# Ensure there is no missing data
df = df.dropna()

# Normalize the dataset using MinMaxScaler to bring values between 0 and 1
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df['Price'].values.reshape(-1, 1))

# Create a function to prepare the data for LSTM model
def create_dataset(data, time_step=60):
    X, y = [], []
    for i in range(time_step, len(data)):
        X.append(data[i-time_step:i, 0])  # Previous 60 values
        y.append(data[i, 0])  # The next value (price)
    return np.array(X), np.array(y)

# Create the dataset
time_step = 60
X, y = create_dataset(scaled_data, time_step)

# Reshape the input data to be 3D [samples, time steps, features] for LSTM
X = X.reshape(X.shape[0], X.shape[1], 1)

# Split the data into training and testing sets
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Build the LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32)

# Make predictions
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# Inverse transform the predictions to original scale
train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)
y_train_actual = scaler.inverse_transform(y_train.reshape(-1, 1))
y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

# Plot the results
plt.figure(figsize=(12, 6))

# Plot training and testing data
plt.plot(df.index[:len(y_train_actual)], y_train_actual, label="Actual Prices (Training)", color='blue')
plt.plot(df.index[len(y_train_actual):len(y_train_actual)+len(y_test_actual)], y_test_actual, label="Actual Prices (Testing)", color='green')

# Plot predicted prices
plt.plot(df.index[:len(train_predict)], train_predict, label="Predicted Prices (Training)", color='red', linestyle='--')
plt.plot(df.index[len(train_predict):len(train_predict)+len(test_predict)], test_predict, label="Predicted Prices (Testing)", color='orange', linestyle='--')

# Predict the next 30 days
last_60_days = scaled_data[-60:].reshape(1, -1)
last_60_days = last_60_days.reshape((1, last_60_days.shape[1], 1))

predicted_prices = []
for _ in range(30):
    pred_price = model.predict(last_60_days)
    predicted_prices.append(pred_price[0][0])
    
    # Append the predicted value to the last_60_days and reshape it for the next prediction
    last_60_days = np.append(last_60_days[:,1:,:], pred_price.reshape(1, 1, 1), axis=1)

# Inverse transform the predicted prices
predicted_prices = scaler.inverse_transform(np.array(predicted_prices).reshape(-1, 1))

# Generate future dates for next 30 days
future_dates = pd.date_range(df.index[-1], periods=31, freq='D')[1:]  # Next 30 days

# Plot actual data, predicted prices and next 30 days
plt.plot(df.index, df['Price'], label="Actual Prices", color='blue')
plt.plot(future_dates, predicted_prices, label="Predicted Prices (Next 30 days)", color='red', linestyle='--')

plt.title(f'{name} Cryptocurrency Price Prediction for Next 30 Days')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.xticks(rotation=45)

# Ensure the 'static' directory exists
if not os.path.exists('static'):
    os.makedirs('static')

# Save the plot as a .jpg image
image_path = os.path.join('CryptoPredictor/static', f'{name}_price_prediction_with_future.jpg')
plt.savefig(image_path)

# Verify if the file is saved
if os.path.exists(image_path):
    print(f"✅ Image successfully saved at: {image_path}")
else:
    print("❌ Image saving failed!")

plt.close()

# Calculate RMSE for performance evaluation
train_rmse = np.sqrt(mean_squared_error(y_train_actual, train_predict))
test_rmse = np.sqrt(mean_squared_error(y_test_actual, test_predict))

print(f'Training RMSE: {train_rmse}')
print(f'Testing RMSE: {test_rmse}')
