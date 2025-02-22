from flask import Flask, render_template, jsonify
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import load_model
import io
import base64

app = Flask(__name__)

# Load the model once when the application starts
model = load_model('bitcoin_model.h5')

def fetch_data(name):
    url = f'https://api.coingecko.com/api/v3/coins/{name}/market_chart?vs_currency=usd&days=365'
    response = requests.get(url)
    data = response.json()
    prices = [entry[1] for entry in data['prices']]  # Extract the prices
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
        df = fetch_data(crypto)
        df['Days'] = (df.index - df.index.min()).days

        X = df[['Days']]
        y = df['Price']

        scaler = MinMaxScaler(feature_range=(0, 1))
        df2 = scaler.fit_transform(np.array(df['Price']).reshape(-1, 1))

        X = scaler.transform(X)
        X = X.reshape(X.shape[0], X.shape[1], 1)

        predictions = model.predict(X)
        predictions = scaler.inverse_transform(predictions)

        plt.figure(figsize=(10, 6))
        plt.plot(df.index, df['Price'], label='Actual Prices')
        plt.plot(df.index, predictions, label='Predicted Prices', color='red')
        plt.xlabel('Date')
        plt.ylabel('Price in USD')
        plt.title(f'{crypto.capitalize()} Price Prediction using LSTM')
        plt.legend()

        # Save the plot as an image file
        image_path = f'static/{crypto}_prediction.png'
        plt.savefig(image_path)
        plt.close()  # Close the plot to free memory

        return send_file(image_path, mimetype='image/png')
    except Exception as e:
        print(f"Error: {e}")
        return jsonify(error=str(e)), 500

if __name__ == '__main__':
    app.run(debug=True)