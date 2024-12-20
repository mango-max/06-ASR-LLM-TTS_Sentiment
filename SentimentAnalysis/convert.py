import torch
import time
import numpy as np
from transformers import AutoTokenizer, AutoConfig, BertForSequenceClassification
import onnx
import onnxruntime as ort
from typing import Dict, List, Tuple
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EmotionModelWrapper(torch.nn.Module):
    def __init__(self, model_path: str):
        super().__init__()
        self.model = BertForSequenceClassification.from_pretrained(model_path)
        self.model.eval()
    
    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits

class ModelConverter:
    def __init__(
        self, 
        model_path: str,
        max_length: int = 128,
        batch_size: int = 1,
        device: str = None
    ):
        self.model_path = model_path
        self.max_length = max_length
        self.batch_size = batch_size
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        logger.info(f"使用设备: {self.device}")
        
        # 加载模型和分词器
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = EmotionModelWrapper(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        # 生成示例输入
        self.dummy_input = self._prepare_dummy_input()
    
    def _prepare_dummy_input(self) -> Dict[str, torch.Tensor]:
        text = "这是一个示例文本"
        inputs = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {k: v.to(self.device) for k, v in inputs.items()}
    
    def convert_to_torchscript(self) -> None:
        try:
            logger.info("开始转换为TorchScript...")
            
            traced_model = torch.jit.trace(
                self.model,
                (
                    self.dummy_input['input_ids'],
                    self.dummy_input['attention_mask']
                ),
                strict=False
            )
            
            # 进行优化
            traced_model = torch.jit.optimize_for_inference(traced_model)
            
            # 保存模型
            traced_model.save("emotion_model_traced.pt")
            
            # 验证模型
            loaded_model = torch.jit.load("emotion_model_traced.pt")
            loaded_model.to(self.device)
            test_output = loaded_model(
                self.dummy_input['input_ids'],
                self.dummy_input['attention_mask']
            )
            
            logger.info("TorchScript转换完成，模型已保存为emotion_model_traced.pt")
            return loaded_model
            
        except Exception as e:
            logger.error(f"TorchScript转换失败: {str(e)}")
            raise
    
    def convert_to_onnx(self) -> None:
        try:
            logger.info("开始转换为ONNX...")
            
            # 确保模型在评估模式
            self.model.eval()
            
            # 设置更高的opset版本
            torch.onnx.export(
                self.model,
                (
                    self.dummy_input['input_ids'],
                    self.dummy_input['attention_mask']
                ),
                "emotion_model.onnx",
                input_names=['input_ids', 'attention_mask'],
                output_names=['output'],
                dynamic_axes={
                    'input_ids': {0: 'batch_size'},
                    'attention_mask': {0: 'batch_size'},
                    'output': {0: 'batch_size'}
                },
                opset_version=14,  # 更新到版本14
                do_constant_folding=True,
                verbose=False
            )
            
            # 验证ONNX模型
            onnx_model = onnx.load("emotion_model.onnx")
            onnx.checker.check_model(onnx_model)
            
            # 检查ONNX Runtime支持的provider
            available_providers = ort.get_available_providers()
            logger.info(f"可用的ONNX Runtime providers: {available_providers}")
            
            # 验证推理
            session = ort.InferenceSession(
                "emotion_model.onnx",
                providers=available_providers
            )
            
            # 进行测试推理
            test_inputs = {
                'input_ids': self.dummy_input['input_ids'].cpu().numpy(),
                'attention_mask': self.dummy_input['attention_mask'].cpu().numpy()
            }
            _ = session.run(None, test_inputs)
            
            logger.info("ONNX转换完成，模型已保存为emotion_model.onnx")
            
        except Exception as e:
            logger.error(f"ONNX转换失败: {str(e)}")
            raise
    
    def verify_predictions(self, test_texts: List[str]) -> None:
        logger.info("开始验证预测结果...")
        
        # 准备输入数据
        inputs = self.tokenizer(
            test_texts,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # PyTorch模型预测
        with torch.no_grad():
            torch_outputs = self.model(
                inputs['input_ids'],
                inputs['attention_mask']
            )
            torch_preds = torch.argmax(torch_outputs, dim=1).cpu().numpy()
        
        # TorchScript模型预测
        ts_model = torch.jit.load("emotion_model_traced.pt")
        ts_model.to(self.device)
        with torch.no_grad():
            ts_outputs = ts_model(
                inputs['input_ids'],
                inputs['attention_mask']
            )
            ts_preds = torch.argmax(ts_outputs, dim=1).cpu().numpy()
        
        # ONNX模型预测
        ort_session = ort.InferenceSession(
            "emotion_model.onnx",
            providers=ort.get_available_providers()
        )
        onnx_inputs = {
            'input_ids': inputs['input_ids'].cpu().numpy(),
            'attention_mask': inputs['attention_mask'].cpu().numpy()
        }
        onnx_outputs = ort_session.run(None, onnx_inputs)
        onnx_preds = np.argmax(onnx_outputs[0], axis=1)
        
        # 比较结果
        torch_match_ts = np.array_equal(torch_preds, ts_preds)
        torch_match_onnx = np.array_equal(torch_preds, onnx_preds)
        
        # 输出验证结果
        logger.info("\n预测结果验证:")
        for i, text in enumerate(test_texts):
            logger.info(f"\n文本: {text}")
            logger.info(f"PyTorch预测: {torch_preds[i]}")
            logger.info(f"TorchScript预测: {ts_preds[i]}")
            logger.info(f"ONNX预测: {onnx_preds[i]}")
        
        logger.info(f"\n模型一致性检查:")
        logger.info(f"PyTorch和TorchScript预测一致: {torch_match_ts}")
        logger.info(f"PyTorch和ONNX预测一致: {torch_match_onnx}")

def main():
    # 模型路径
    model_path = "best_emotion_model"
    
    # 创建转换器实例
    converter = ModelConverter(model_path)
    
    # 转换模型
    converter.convert_to_torchscript()
    converter.convert_to_onnx()
    
    # 验证预测结果
    test_texts = [
        "这个产品很好用，我很喜欢",
        "服务态度太差了，非常不满意",
        "还可以，一般般吧",
        "快递很快，商品质量不错"
    ]
    converter.verify_predictions(test_texts)

if __name__ == "__main__":
    main()