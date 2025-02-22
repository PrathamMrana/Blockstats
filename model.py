import requests
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
name1 ="bitcoin"
def fetch_data(name):
    url = f'https://api.coingecko.com/api/v3/coins/{name}/market_chart?vs_currency=usd&days=365'
    response = requests.get(url)
    data = response.json()
    prices = [entry[1] for entry in data['prices']]  # Extract the prices
    dates = [entry[0] for entry in data['prices']]   # Extract timestamps
    df = pd.DataFrame({'Date': pd.to_datetime(dates, unit='ms'), 'Price': prices})
    df.set_index('Date', inplace=True)
    return df

df = fetch_data(name1)
df['Days'] = (df.index - df.index.min()).days

X = df[['Days']]
y = df['Price']

scaler = MinMaxScaler(feature_range=(0, 1))
X_scaled = scaler.fit_transform(X)
y_scaled = scaler.fit_transform(np.array(y).reshape(-1, 1))

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(LSTM(units=50, return_sequences=True))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=64, verbose=1)

model.save("bitcoin_model.h5")