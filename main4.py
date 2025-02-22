from flask import Flask, render_template, jsonify, send_file
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
import os

# Use the Agg backend for matplotlib
import matplotlib
matplotlib.use('Agg')

app = Flask(__name__)

# Load the model once when the app starts
MODEL_PATH = 'bitcoin_model.h5'
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file '{MODEL_PATH}' not found!")

model = load_model(MODEL_PATH)

def fetch_data(name):
    """Fetches historical price data from CoinGecko API."""
    url = f'https://api.coingecko.com/api/v3/coins/{name}/market_chart?vs_currency=usd&days=365'
    response = requests.get(url)

    if response.status_code != 200:
        raise Exception(f"Failed to fetch data: {response.json()}")

    data = response.json()
    prices = [entry[1] for entry in data['prices']]  # Extract prices
    dates = [entry[0] for entry in data['prices']]   # Extract timestamps

    df = pd.DataFrame({'Date': pd.to_datetime(dates, unit='ms'), 'Price': prices})
    df.set_index('Date', inplace=True)
    return df

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict/<crypto>')
def predict(crypto):
    try:
        print(f"Fetching data for {crypto}")
        df = fetch_data(crypto)
        df['Days'] = (df.index - df.index.min()).days

        # Scaling the features (Days) and target (Price)
        scaler_price = MinMaxScaler(feature_range=(0, 1))
        scaler_days = MinMaxScaler(feature_range=(0, 1))

        df['Scaled_Price'] = scaler_price.fit_transform(df[['Price']])
        df['Scaled_Days'] = scaler_days.fit_transform(df[['Days']])

        # Prepare input data for LSTM
        sequence_length = 60  # Adjust based on your model training setup
        X = np.array([df['Scaled_Days'].values[i-sequence_length:i] for i in range(sequence_length, len(df))])

        # Ensure shape compatibility for LSTM
        X = X.reshape(X.shape[0], X.shape[1], 1)

        print("Predicting prices")
        predictions = model.predict(X)
        predictions = scaler_price.inverse_transform(predictions)

        # Align prediction dates with actual data
        df_pred = df.iloc[sequence_length:].copy()
        df_pred['Predicted_Price'] = predictions

        print("Generating plot")
        plt.figure(figsize=(10, 6))
        plt.plot(df.index, df['Price'], label='Actual Prices', color='blue')
        plt.plot(df_pred.index, df_pred['Predicted_Price'], label='Predicted Prices', color='red')
        plt.xlabel('Date')
        plt.ylabel('Price in USD')
        plt.title(f'{crypto.capitalize()} Price Prediction using LSTM')
        plt.legend()

        # Save the plot in the static folder
        image_path = f'static/{crypto}_prediction.png'
        plt.savefig(image_path)
        plt.close()

        print(f"Plot saved to {image_path}")
        return send_file(image_path, mimetype='image/png')
    
    except Exception as e:
        print(f"Error: {e}")
        return jsonify(error=str(e)), 500

if __name__ == '__main__':
    app.run(debug=True)
