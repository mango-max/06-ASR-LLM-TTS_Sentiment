import requests
import json

url = "http://127.0.0.1:8877/api/v1/emotion"
data = {
    "text": "明天到底吃什么啊啊啊啊啊啊"
}

response = requests.post(url, json=data)
result = response.json()

print(f"文本: {data['text']}")
print(f"情感: {result['emotion']}")
print(f"推理耗时: {result['time']}")