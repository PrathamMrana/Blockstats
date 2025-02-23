from flask import Flask, jsonify, render_template
import requests

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/current_price')
def current_price():
    # Replace 'BTCUSDT' with the desired trading pair
    symbol = 'BTCUSDT'
    url = f'https://api.binance.com/api/v3/ticker/price?symbol={symbol}'
    
    response = requests.get(url)
    data = response.json()
    
    price = data.get('price')
    
    if price:
        return jsonify(price=float(price))
    else:
        return jsonify(error="Failed to fetch price"), 500

if __name__ == '__main__':
    app.run(debug=True)