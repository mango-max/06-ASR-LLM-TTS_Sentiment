import os
from fastapi import FastAPI
from pydantic import BaseModel
import torch
import numpy as np
import onnxruntime
from transformers import BertTokenizer
import time

class TextRequest(BaseModel):
    text: str

def get_available_device():
    if torch.cuda.is_available():
        device = os.getenv("EMOTION_DEVICE", "cuda:0")
        gpu_name = torch.cuda.get_device_name(0)
        print(f"使用 GPU 推理: {gpu_name} ({device})")
    else:
        device = "cpu"
        print("情感分析使用 CPU 推理")
    return device

class EmotionPredictor:
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('./best_emotion_model')
        self.device = torch.device(get_available_device())
        
        # 配置ONNX运行时
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        provider_options = [
            {
                'device_id': 0,
                'gpu_mem_limit': 4 * 1024 * 1024 * 1024,
            },
            {}
        ]
        self.session = onnxruntime.InferenceSession(
            'emotion_model.onnx',
            providers=providers,
            provider_options=provider_options
        )
        
        self.emotion_map = {
            0: "喜欢",
            1: "快乐",
            2: "悲伤",
            3: "愤怒",
            4: "厌恶"
        }

    def preprocess(self, text):
        # 处理单条文本
        encodings = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=128,
            return_tensors='pt',
            return_token_type_ids=False
        )
        return {
            'input_ids': encodings['input_ids'].numpy(),
            'attention_mask': encodings['attention_mask'].numpy()
        }

    def predict(self, text):
        # 记录开始时间
        start_time = time.time()
        
        # 预处理输入
        inputs = self.preprocess(text)
        
        # 模型推理
        outputs = self.session.run(None, inputs)
        prediction = np.argmax(outputs[0], axis=1)[0]
        
        # 计算总耗时（毫秒）
        inference_time = round((time.time() - start_time) * 1000, 2)
        
        # 返回情感标签和耗时
        return {
            "emotion": self.emotion_map[prediction],
            "time": f"{inference_time}ms"
        }

# 初始化FastAPI应用和情感分析器
app = FastAPI()
predictor = EmotionPredictor()

@app.get("/")
async def root():
    return {
        "message": "情感分析API服务",
        "endpoints": {
            "/api/v1/emotion": "情感分析接口",
            "/api/v1/device_info": "获取设备信息"
        }
    }

@app.post("/api/v1/emotion")
async def analyze_emotion(request: TextRequest):
    """
    分析单条文本的情感
    
    请求体格式:
    {
        "text": "要分析的文本"
    }
    
    返回格式:
    {
        "emotion": "情感标签"
    }
    """
    result = predictor.predict(request.text)
    # 只返回情感标签，不返回时间信息
    return {"emotion": result["emotion"]}

@app.get("/api/v1/device_info")
async def get_device_info():
    """返回当前使用的推理设备信息"""
    device = os.getenv("EMOTION_DEVICE", "cuda:0")
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