import torch
from transformers import BertTokenizer
import time
import numpy as np
import onnxruntime
import gc

class EmotionPredictor:
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('./best_emotion_model')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
        if torch.cuda.is_available():
            print(f"GPU型号: {torch.cuda.get_device_name(0)}")
            print(f"当前GPU显存使用: {torch.cuda.memory_allocated()/1024**2:.1f}MB")
        self.emotion_map = {
            0: "喜欢",
            1: "快乐",
            2: "悲伤",
            3: "愤怒",
            4: "厌恶"
        }
        
    def preprocess(self, text):
        # 使用与训练时相同的tokenizer设置
        encodings = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=128,
            return_tensors='pt',
            return_token_type_ids=False
        )
        return {k: v.to(self.device) for k, v in encodings.items()}

    def predict_pytorch(self, text):
        from transformers import BertForSequenceClassification
        
        # 计算加载时间
        load_start = time.time()
        model = BertForSequenceClassification.from_pretrained('./best_emotion_model').to(self.device)
        model.eval()
        load_time = (time.time() - load_start) * 1000  # 转换为毫秒
        
        # 预处理
        inputs = self.preprocess(text)
        
        # 推理
        infer_start = time.time()
        with torch.no_grad():
            outputs = model(**inputs)
            prediction = torch.argmax(outputs.logits, dim=1).item()
        inference_time = (time.time() - infer_start) * 1000  # 转换为毫秒
        
        # 清理内存
        del model
        torch.cuda.empty_cache()
        gc.collect()
        
        return self.emotion_map[prediction], load_time, inference_time

    def predict_torchscript(self, text):
        # 计算加载时间
        load_start = time.time()
        traced_model = torch.jit.load('emotion_model_traced.pt').to(self.device)
        load_time = (time.time() - load_start) * 1000  # 转换为毫秒
        
        # 预处理
        inputs = self.preprocess(text)
        
        # 推理
        infer_start = time.time()
        with torch.no_grad():
            outputs = traced_model(
                inputs['input_ids'],
                inputs['attention_mask']
            )
            prediction = torch.argmax(outputs, dim=1).item()
        inference_time = (time.time() - infer_start) * 1000  # 转换为毫秒
        
        # 清理内存
        del traced_model
        torch.cuda.empty_cache()
        gc.collect()
        
        return self.emotion_map[prediction], load_time, inference_time

    def predict_onnx(self, text):
        # 计算加载时间
        load_start = time.time()
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        provider_options = [
            {
                'device_id': 0,
                'gpu_mem_limit': 4 * 1024 * 1024 * 1024,
            },
            {}
        ]
        session = onnxruntime.InferenceSession(
            'emotion_model.onnx',
            providers=providers,
            provider_options=provider_options
        )
        load_time = (time.time() - load_start) * 1000  # 转换为毫秒
        
        # 预处理
        inputs = self.preprocess(text)
        onnx_inputs = {
            'input_ids': inputs['input_ids'].cpu().numpy(),
            'attention_mask': inputs['attention_mask'].cpu().numpy()
        }
        
        # 推理
        infer_start = time.time()
        outputs = session.run(None, onnx_inputs)
        prediction = np.argmax(outputs[0], axis=1)[0]
        inference_time = (time.time() - infer_start) * 1000  # 转换为毫秒
        
        # 清理内存
        del session
        gc.collect()
        
        return self.emotion_map[prediction], load_time, inference_time

def main():
    predictor = EmotionPredictor()
    print("情感分析模型已加载，当前使用GPU:", torch.cuda.get_device_name(0))
    print("\n请输入要分析的文本（输入'q'退出）：")
    
    while True:
        text = input("\n> ")
        if text.lower() == 'q':
            break
            
        print("\n开始分析...")
        
        # PyTorch模型
        emotion, load_time, infer_time = predictor.predict_pytorch(text)
        print(f"\nPyTorch模型结果:")
        print(f"情感: {emotion}")
        print(f"模型加载耗时: {load_time:.2f}ms")
        print(f"推理耗时: {infer_time:.2f}ms")
        print(f"总耗时: {(load_time + infer_time):.2f}ms")
        
        # TorchScript模型
        emotion, load_time, infer_time = predictor.predict_torchscript(text)
        print(f"\nTorchScript模型结果:")
        print(f"情感: {emotion}")
        print(f"模型加载耗时: {load_time:.2f}ms")
        print(f"推理耗时: {infer_time:.2f}ms")
        print(f"总耗时: {(load_time + infer_time):.2f}ms")
        
        # ONNX模型
        emotion, load_time, infer_time = predictor.predict_onnx(text)
        print(f"\nONNX模型结果:")
        print(f"情感: {emotion}")
        print(f"模型加载耗时: {load_time:.2f}ms")
        print(f"推理耗时: {infer_time:.2f}ms")
        print(f"总耗时: {(load_time + infer_time):.2f}ms")

if __name__ == '__main__':
    main() 