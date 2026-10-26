import requests

url = "http://127.0.0.1:5000/recommend"

response = requests.post(url, json={"movie": "batman"})
print(response.json())