from flask import Flask, request, jsonify
import joblib
import numpy as np
import requests
import threading
import time

# Load model
model = joblib.load('fraud_detection_model.pkl')
print("Model loaded successfully.")  # Debug print
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    print("Received data:", data)  # Debug print
    features = np.array(data['features']).reshape(1, -1)

    # Check feature count
    expected_features = model.n_features_in_
    if features.shape[1] != expected_features:
        return jsonify({"error": f"Expected {expected_features} features, but got {features.shape[1]}"}), 400

    # Predict probability if model supports predict_proba
    if hasattr(model, "predict_proba"):
        probability = model.predict_proba(features)[:, 1][0]
        print("Probability (predict_proba):", probability)  # Debug print
    else:
        probability = model.predict(features)[0]
        print("Probability (predict):", probability)  # Debug print

    # Constructing the output message in Vietnamese with just the feature values
    feature_values = ", ".join([str(value) for value in features[0]])
    response_message = f"Vậy tỉ lệ gian lận từ 8 features ({feature_values}) là: {round(probability, 4)}"

    return jsonify({"message": response_message})

def send_test_request():
    # Allow some time for the server to start
    time.sleep(5)  
    print("Sending test request...")  # Debug print
    api_url = "http://127.0.0.1:5000/predict"
    headers = {"Content-Type": "application/json"}
    data = {
        "features": [0.1, -1.2, -1.5, 10.3, 0.2, -0.5, 7.0, 1.2]  # Example input with 8 features
    }

    try:
        response = requests.post(api_url, json=data, headers=headers)
        print("Response Status Code:", response.status_code)  # Check status code
        print("Test Request Response:", response.json())  # Debug output
    except Exception as e:
        print("Request failed:", e)

if __name__ == '__main__':
    # Start Flask server in a separate thread to allow testing while running the server
    flask_thread = threading.Thread(target=lambda: app.run(debug=True, use_reloader=False))
    flask_thread.daemon = True
    flask_thread.start()

    # Send a test request in a separate thread after Flask starts
    threading.Thread(target=send_test_request, daemon=True).start()

    # Keep the main thread alive
    while True:
        time.sleep(1)
