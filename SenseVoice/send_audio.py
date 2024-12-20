import requests

def send_audio_files(files="./data/output.wav"):
    url = "http://127.0.0.1:8866/api/v1/asr"
    
    # Headers
    headers = {
        "accept": "application/json",
    }

    # 上传文件
    files = [
        ("files", (files, open(files, "rb"), "audio/wav")),    # ("files", ("zero_shot_0.wav", open("zero_shot_0.wav", "rb"), "audio/wav"))
    ]
    data = {
        "keys": "string",
        "lang": "zh"
    }

    # 请求数据
    response = requests.post(url, headers=headers, files=files, data=data)

    # 输出返回的原始数据
    print("原始响应内容:", response.json())

   # 处理返回数据
    if response.status_code == 200:
        # 返回解析到的数据
        text = response.json().get('result')[0].get('text')
        # print("处理后的文本内容:", text)
        return text
    else:
        print(f"请求失败，状态码: {response.status_code}")
        print("响应内容:", response.text)
        return response.status_code, response.text
    
if __name__ == "__main__":
    print(send_audio_files())