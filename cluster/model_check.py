import requests
import json
host = "http://192.168.123.171:8000"
target = "http://192.168.123.171:8000/v2/repository/index"

response = requests.post(target)
print(response.json())
MODEL_NAME = "yolov11_ver1_3"

MODEL_VERSION=1
msg = f"v2/models/{MODEL_NAME}/versions/{MODEL_VERSION}/config"
target = f"{host}/{msg}"
print(target)
response = requests.get(target)
print(response.json())