import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertForSequenceClassification, AutoTokenizer, AutoConfig
from transformers import get_linear_schedule_with_warmup
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
from tqdm import tqdm

class EmotionDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text = self.data.iloc[idx]['text']
        label = self.data.iloc[idx]['label']
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def train():
    # 加载数据
    print("正在加载数据...")
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
    
    # 加载预训练模型和分词器
    print("正在加载模型...")
    model_name = "IDEA-CCNL/Erlangshen-Roberta-110M-Sentiment"
    
    # 创建配置
    config = AutoConfig.from_pretrained(model_name)
    config.num_labels = 5  # 设置标签数量
    
    # 加载模型
    model = BertForSequenceClassification.from_pretrained(
        model_name,
        config=config,
        ignore_mismatched_sizes=True  # 忽略尺寸不匹配
    )
    
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # 创建数据集
    train_dataset = EmotionDataset(train_df, tokenizer)
    test_dataset = EmotionDataset(test_df, tokenizer)
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=16,
        shuffle=True,
        num_workers=2
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=32,
        num_workers=2
    )
    
    # 检查CUDA是否可用并打印设备信息
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        print(f"当前使用的GPU: {torch.cuda.get_device_name(0)}")
        print(f"可用的GPU数量: {torch.cuda.device_count()}")
    
    # 确保模型使用GPU
    model = model.to(device)
    print(f"模型已移至设备: {next(model.parameters()).device}")
    
    # 优化器和学习率调度器
    optimizer = torch.optim.AdamW(  # 使用PyTorch的AdamW
        model.parameters(), 
        lr=2e-5, 
        weight_decay=0.01
    )
    
    # 添加学习率调度器
    num_epochs = 5
    total_steps = len(train_loader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=total_steps * 0.1,
        num_training_steps=total_steps
    )
    
    # 训练
    best_accuracy = 0
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            total_loss += loss.item()
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch+1}, Average Loss: {avg_loss:.4f}')
        
        # 评估
        model.eval()
        predictions = []
        actual_labels = []
        
        print("正在评估...")
        with torch.no_grad():
            for batch in tqdm(test_loader):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                preds = torch.argmax(outputs.logits, dim=1)
                predictions.extend(preds.cpu().numpy())
                actual_labels.extend(labels.cpu().numpy())
        
        accuracy = accuracy_score(actual_labels, predictions)
        print(f'Epoch {epoch+1}, Test Accuracy: {accuracy:.4f}')
        
        # 打印详细的分类报告
        print("\n分类报告:")
        print(classification_report(actual_labels, predictions))
        
        # 保存最佳模型
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            print(f"保存最佳模型，准确率: {accuracy:.4f}")
            model.save_pretrained('best_emotion_model')
            tokenizer.save_pretrained('best_emotion_model')

if __name__ == '__main__':
    train()