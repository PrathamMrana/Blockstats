import requests
import pandas as pd
import os

# Fetch data using Binance API
def fetch_data(symbol="BTCUSDT", interval="1d", limit=730):
    url = f'https://api.binance.com/api/v3/klines'
    params = {
        'symbol': symbol,    # e.g., BTCUSDT for Bitcoin to USDT
        'interval': interval,  # 1d (daily) data
        'limit': limit        # Number of data points (730 days here)
    }
    response = requests.get(url, params=params)
    
    if response.status_code == 200:
        data = response.json()
        # Extract timestamps (open time) and closing prices (Close is at index 4)
        prices = [float(entry[4]) for entry in data]
        dates = [entry[0] for entry in data]
        df = pd.DataFrame({'Date': pd.to_datetime(dates, unit='ms'), 'Price': prices})
        df.set_index('Date', inplace=True)
        df.to_csv(f'CryptoPredictor/{symbol}_price_data.csv')
        return df
    else:
        print("Error fetching data:", response.status_code)
        return pd.DataFrame()

# List of cryptocurrencies to fetch
cryptocurrencies = [
    "BTCUSDT",  # Bitcoin
    "MATICUSDT", # Polygon
    "LINKUSDT",  # Chainlink
    "LTCUSDT",   # Litecoin
    "XLMUSDT",   # Stellar
    "ETHUSDT",   # Ethereum
    "USDTUSDT",  # Tether
    "BNBUSDT"    # Binance Coin
]

# Loop through each cryptocurrency and fetch data
for crypto in cryptocurrencies:
    print(f"Fetching data for {crypto}...")
    df = fetch_data(symbol=crypto)
    if not df.empty:
        print(f"Data for {crypto} saved successfully.")
    else:
        print(f"Failed to fetch data for {crypto}.")
