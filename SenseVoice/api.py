# Set the device with environment, default is cuda:0
# export SENSEVOICE_DEVICE=cuda:1

import os, re
from fastapi import FastAPI, File, Form
from fastapi.responses import HTMLResponse
from typing_extensions import Annotated
from typing import List
from enum import Enum
import torchaudio
from model import SenseVoiceSmall
from funasr.utils.postprocess_utils import rich_transcription_postprocess
from io import BytesIO
import torch
import requests


class Language(str, Enum):
    auto = "auto"
    zh = "zh"
    en = "en"
    yue = "yue"
    ja = "ja"
    ko = "ko"
    nospeech = "nospeech"

def get_available_device():
    # if torch.cuda.is_available():
    if 0:
        device = os.getenv("SENSEVOICE_DEVICE", "cuda:0")
        gpu_name = torch.cuda.get_device_name(0)
        print(f"使用 GPU 推理: {gpu_name} ({device})")
    else:
        device = "cpu"
        print("语音识别使用 CPU 推理")
    return device

model_dir = "./SenseVoiceSmall"
device = get_available_device()
m, kwargs = SenseVoiceSmall.from_pretrained(model=model_dir, device=device)
m.eval()

regex = r"<\|.*\|>"

app = FastAPI()


@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <!DOCTYPE html>
    <html>
        <head>
            <meta charset=utf-8>
            <title>Api information</title>
        </head>
        <body>
            <a href='./docs'>Documents of API</a>
        </body>
    </html>
    """

@app.post("/api/v1/asr")
async def turn_audio_to_text(files: Annotated[List[bytes], File(description="wav or mp3 audios in 16KHz")], keys: Annotated[str, Form(description="name of each audio joined with comma")], lang: Annotated[Language, Form(description="language of audio content")] = "auto"):
    audios = []
    audio_fs = 0
    for file in files:
        file_io = BytesIO(file)
        data_or_path_or_list, audio_fs = torchaudio.load(file_io)
        data_or_path_or_list = data_or_path_or_list.mean(0)
        audios.append(data_or_path_or_list)
        file_io.close()
    if lang == "":
        lang = "auto"
    if keys == "":
        key = ["wav_file_tmp_name"]
    else:
        key = keys.split(",")
    res = m.inference(
        data_in=audios,
        language=lang, # "zh", "en", "yue", "ja", "ko", "nospeech"
        use_itn=False,
        ban_emo_unk=True,
        key=key,
        fs=audio_fs,
        **kwargs,
    )
    if len(res) == 0:
        return {"clean_text": "", "emotion": "NEUTRAL"}
    
    # 定义所有可能的情感类型
    EMOTIONS = ["HAPPY", "SAD", "ANGRY", "NEUTRAL", "FEARFUL", "DISGUSTED", "SURPRISED"]
    
    # 只处理第一个结果
    it = res[0][0]
    it["raw_text"] = it["text"]
    it["clean_text"] = re.sub(regex, "", it["text"], 0, re.MULTILINE)
    it["text"] = rich_transcription_postprocess(it["text"])
    
    # 提取情感信息
    emotion_match = re.findall(r"<\|(.*?)\|>", it["raw_text"])
    emotion = "NEUTRAL"  # 默认情��
    if emotion_match:
        for e in emotion_match:
            if e in EMOTIONS:
                emotion = e
                break
    
    # 准备ASR结果
    asr_result = {
        "clean_text": it["clean_text"],
        "emotion": emotion
    }
    
    # 发送到情感分析API
    try:
        emotion_api_url = "http://127.0.0.1:8877/api/v1/emotion"
        emotion_response = requests.post(
            emotion_api_url,
            json={"text": it["clean_text"]},
            timeout=5
        )
        emotion_response.raise_for_status()
        emotion_result = emotion_response.json()
        
        # 合并结果
        final_result = {
            "clean_text": asr_result["clean_text"],
            "emotion": asr_result["emotion"],
            "emotion_2": emotion_result["emotion"]
        }
        return final_result
        
    except Exception as e:
        print(f"情感分析API调用失败: {e}")
        # 如果情感分析失败，返回原始结果
        return asr_result

@app.get("/api/v1/device_info")
async def get_device_info():
    """返回当前使用的推理设备信息"""
    device = os.getenv("SENSEVOICE_DEVICE", "cuda:0")
    device_info = {
        "device": device,
        "gpu_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        device_info["gpu_name"] = torch.cuda.get_device_name(0)
        device_info["gpu_memory"] = {
            "total": torch.cuda.get_device_properties(0).total_memory / (1024**2),  # MB
            "allocated": torch.cuda.memory_allocated(0) / (1024**2),  # MB
            "cached": torch.cuda.memory_reserved(0) / (1024**2)  # MB
        }
    return device_info
