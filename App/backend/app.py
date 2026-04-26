from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import joblib

app = Flask(__name__)
CORS(app)

# ---------------- LOAD EVERYTHING ----------------
model = tf.keras.models.load_model("../Model/churn_model.keras")
scaler = joblib.load("../Model/scaler.pkl")
encoder = joblib.load("../Model/encoder.pkl")

# feature order (VERY IMPORTANT)
numerical_cols = [
    'CreditScore', 'Age', 'Tenure', 'Balance',
    'NumOfProducts', 'HasCrCard',
    'IsActiveMember', 'EstimatedSalary'
]

categorical_cols = ['Gender', 'Geography']


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        # ---------------- NUMERICAL ----------------
        num = [[
            data['creditScore'],
            data['age'],
            data['tenure'],
            data['balance'],
            data['numProducts'],
            data['hasCrCard'],
            data['isActiveMember'],
            data['salary']
        ]]

        # ---------------- CATEGORICAL ----------------
        cat = [[
            data['gender'],
            data['geography']
        ]]

        # ---------------- PIPELINE ----------------
        cat_encoded = encoder.transform(cat)
        final_input = np.hstack((num, cat_encoded))
        final_scaled = scaler.transform(final_input)

        # ---------------- PREDICTION ----------------
        prediction_probs = model.predict(final_scaled)[0]
        prediction_prob = float(prediction_probs[1])
        prediction = int(np.argmax(prediction_probs))

        return jsonify({
            "prediction": prediction,
            "probability": float(prediction_prob)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == '__main__':
    app.run(debug=True, port=5000)