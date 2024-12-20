import torch
from transformers import BertForSequenceClassification, BertTokenizer
import time
import pandas as pd
import numpy as np
import onnxruntime
import gc
import warnings

warnings.filterwarnings("ignore", message=".*Torch was not compiled with flash attention.*")

def load_test_data(test_file, tokenizer, max_length=128):
    df = pd.read_csv(test_file)
    texts = df['text'].tolist()
    labels = df['label'].tolist()
    
    # 使用与训练时相同的tokenizer设置
    encodings = tokenizer(
        texts,
        truncation=True,
        padding='max_length',
        max_length=max_length,
        return_tensors=None,
        return_token_type_ids=False  # 不返回token_type_ids
    )
    return encodings, labels

def process_batch(encodings, labels, batch_size=32):
    total_samples = len(labels)
    for i in range(0, total_samples, batch_size):
        batch_end = min(i + batch_size, total_samples)
        # 只处理input_ids和attention_mask
        batch_encodings = {
            'input_ids': torch.tensor(encodings['input_ids'][i:batch_end]).cuda(),
            'attention_mask': torch.tensor(encodings['attention_mask'][i:batch_end]).cuda()
        }
        batch_labels = torch.tensor(labels[i:batch_end]).cuda()
        
        yield batch_encodings, batch_labels

def test_pytorch_model(model_path, encodings, labels, batch_size=32):
    model = BertForSequenceClassification.from_pretrained(model_path).cuda()
    model.eval()
    
    total_time = 0
    total_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for batch_encodings, batch_labels in process_batch(encodings, labels, batch_size):
            start_time = time.time()
            outputs = model(**batch_encodings)
            predictions = torch.argmax(outputs.logits, dim=1)
            end_time = time.time()
            
            total_time += (end_time - start_time)
            total_correct += (predictions == batch_labels).sum().item()
            total_samples += len(batch_labels)
            
            del outputs, predictions
            torch.cuda.empty_cache()
    
    accuracy = total_correct / total_samples
    avg_time = total_time / total_samples
    
    del model
    torch.cuda.empty_cache()
    gc.collect()
    
    return accuracy, avg_time

def test_traced_model(model_path, encodings, labels, batch_size=32):
    traced_model = torch.jit.load(model_path).cuda()
    
    total_time = 0
    total_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for batch_encodings, batch_labels in process_batch(encodings, labels, batch_size):
            start_time = time.time()
            outputs = traced_model(
                batch_encodings['input_ids'],
                batch_encodings['attention_mask']
            )
            predictions = torch.argmax(outputs, dim=1)
            end_time = time.time()
            
            total_time += (end_time - start_time)
            total_correct += (predictions == batch_labels).sum().item()
            total_samples += len(batch_labels)
            
            del outputs, predictions
            torch.cuda.empty_cache()
    
    accuracy = total_correct / total_samples
    avg_time = total_time / total_samples
    
    del traced_model
    torch.cuda.empty_cache()
    gc.collect()
    
    return accuracy, avg_time

def test_onnx_model(onnx_path, encodings, labels, batch_size=32):
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    provider_options = [
        {
            'device_id': 0,
            'gpu_mem_limit': 4 * 1024 * 1024 * 1024,  # 4GB GPU内存限制
        },
        {}
    ]
    
    session = onnxruntime.InferenceSession(
        onnx_path,
        providers=providers,
        provider_options=provider_options
    )
    
    # 验证是否使用GPU
    if 'CUDAExecutionProvider' not in session.get_providers():
        raise RuntimeError("ONNX模型未能使用GPU，请检查CUDA支持")
    print("ONNX模型成功使用GPU运行")
    
    # 获取模型输入名称
    # input_names = [input.name for input in session.get_inputs()]
    # print(f"ONNX模型输入名称: {input_names}")
    
    total_time = 0
    total_correct = 0
    total_samples = 0
    
    for batch_encodings, batch_labels in process_batch(encodings, labels, batch_size):
        # 根据模型实际需要的输入准备数据
        onnx_inputs = {
            'input_ids': batch_encodings['input_ids'].cpu().numpy(),
            'attention_mask': batch_encodings['attention_mask'].cpu().numpy()
        }
        
        start_time = time.time()
        outputs = session.run(None, onnx_inputs)
        predictions = torch.tensor(np.argmax(outputs[0], axis=1)).cuda()
        end_time = time.time()
        
        total_time += (end_time - start_time)
        total_correct += (predictions == batch_labels).sum().item()
        total_samples += len(batch_labels)
        
        del outputs, predictions
        torch.cuda.empty_cache()
    
    accuracy = total_correct / total_samples
    avg_time = total_time / total_samples
    
    del session
    gc.collect()
    
    return accuracy, avg_time

def inspect_onnx_model(onnx_path):
    session = onnxruntime.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    
    print("\nONNX模型信息:")
    print("输入:")
    for input in session.get_inputs():
        print(f"- 名称: {input.name}")
        print(f"  形状: {input.shape}")
        print(f"  类型: {input.type}")
    
    print("\n输出:")
    for output in session.get_outputs():
        print(f"- 名称: {output.name}")
        print(f"  形状: {output.shape}")
        print(f"  类型: {output.type}")

def main():
    print("正在使用GPU:", torch.cuda.get_device_name(0))
    
    # 首先检查ONNX模型
    # print("\n检查ONNX模型结构...")
    # inspect_onnx_model('emotion_model.onnx')
    
    # 加载tokenizer和测试数据
    tokenizer = BertTokenizer.from_pretrained('./best_emotion_model')
    encodings, labels = load_test_data('test.csv', tokenizer)
    
    batch_size = 32
    results = []
    
    try:
        # 分别测试每个模型
        print("\n开始测试 PyTorch 模型...")
        pt_accuracy, pt_time = test_pytorch_model('./best_emotion_model', encodings, labels, batch_size)
        results.append(('PyTorch', pt_accuracy, pt_time))
        
        print("\n开始测试 TorchScript 模型...")
        traced_accuracy, traced_time = test_traced_model('emotion_model_traced.pt', encodings, labels, batch_size)
        results.append(('TorchScript', traced_accuracy, traced_time))
        
        print("\n开始测试 ONNX 模型...")
        onnx_accuracy, onnx_time = test_onnx_model('emotion_model.onnx', encodings, labels, batch_size)
        results.append(('ONNX', onnx_accuracy, onnx_time))
        
    except RuntimeError as e:
        if "out of memory" in str(e):
            print(f"\n警告：GPU内存不足，请尝试减小batch_size（当前值：{batch_size}）")
            return
        raise e
    
    # 打印结果
    print("\n性能比较结果：")
    print(f"{'模型类型':<15} {'准确率':<10} {'平均推理时间(ms)':<15}")
    print("-" * 40)
    for name, accuracy, inf_time in results:
        print(f"{name:<15} {accuracy:.4f}    {inf_time*1000:.2f}")

if __name__ == '__main__':
    main()
