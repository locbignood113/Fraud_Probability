import requests

api_url = "http://127.0.0.1:5000/predict"
headers = {"Content-Type": "application/json"}

data = {
    "features": [0.1, -1.2, -1.5, 10.3, 0.2, -0.5, 7.0, 1.2]  # 8 features
}

response = requests.post(api_url, json=data, headers=headers)
print(response.json())  # Kết quả dự đoán
