from flask import Flask, request, jsonify
import joblib
import numpy as np

# Load model
model = joblib.load('fraud_detection_model.pkl')
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    features = np.array(data['features']).reshape(1, -1)

    # Kiểm tra số lượng feature có đúng không
    expected_features = model.n_features_in_
    if features.shape[1] != expected_features:
        return jsonify({"error": f"Expected {expected_features} features, but got {features.shape[1]}"}), 400

    # Dự đoán xác suất nếu mô hình hỗ trợ predict_proba
    if hasattr(model, "predict_proba"):
        probability = model.predict_proba(features)[:, 1][0]  # Xác suất gian lận
    else:
        probability = model.predict(features)[0]  # Nếu không có predict_proba, dùng predict

    return jsonify({"fraud_probability": round(float(probability), 4)})

if __name__ == '__main__':
    app.run(debug=True)
