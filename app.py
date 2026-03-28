from flask import Flask, request, jsonify
from src.predict import HousePricePredictor

app = Flask(__name__)

# Load model once (important for performance)
predictor = HousePricePredictor()


@app.route("/")
def home():
    return "House Price Prediction API is running!"


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        prediction = predictor.predict(data)

        return jsonify({
            "predicted_price": prediction
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)