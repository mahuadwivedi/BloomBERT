from flask import Flask, render_template, request, jsonify
from datetime import datetime, timedelta
import requests
import random

app = Flask(__name__, template_folder='templates', static_folder='static')

@app.route('/')
def index():
    return render_template('converter.html')

@app.route('/convert')
def convert():
    amount = float(request.args.get('amount'))
    from_currency = request.args.get('from')
    to_currency = request.args.get('to')

    try:
        url = f"https://api.frankfurter.app/latest?amount={amount}&from={from_currency}"
        response = requests.get(url)
        data = response.json()

        converted_amount = data['rates'].get(to_currency)
        if converted_amount is None:
            return jsonify({"error": f"No rate found for {to_currency}"}), 400

        # Generate fake historical data (for graph)
        today = datetime.today()
        dates = [(today - timedelta(days=i)).strftime('%Y-%m-%d') for i in range(6, -1, -1)]
        base_rate = converted_amount / amount
        history = [round(base_rate + random.uniform(-0.02, 0.02), 4) for _ in dates]

        return jsonify({
            "converted": converted_amount,
            "history": {
                "dates": dates,
                "rates": history
            }
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500



if __name__ == '__main__':
    app.run(debug=True)
