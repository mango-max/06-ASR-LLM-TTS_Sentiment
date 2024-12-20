import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter
import re
import emoji
from imblearn.over_sampling import RandomOverSampler

def clean_text(text):
    """清理文本数据"""
    if not isinstance(text, str):
        return ""
    
    # 移除emoji
    text = emoji.replace_emoji(text, '')
    
    # 移除URL
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # 移除HTML标签
    text = re.sub(r'<.*?>', '', text)
    
    # 移除中括号内容 [xxx]
    text = re.sub(r'\[.*?\]', '', text)
    
    # 移除@用户
    text = re.sub(r'@\w+', '', text)
    
    # 移除所有类型的引号（包括中英文引号）
    text = re.sub(r'["""\'''′″「」『』''""❝❞〝〞〟＂]', '', text)  # 移除各种引号
    
    # 移除特殊符号
    text = re.sub(r'[&;〉〈《》【】〔〕←→]', '', text)  # 移除特殊符号
    text = re.sub(r'[_\-+=<>|♀♂℃\(\)\[\]\/\\\~@]', '', text)  # 移除其他特殊字符
    
    # 移除多余空格
    text = re.sub(r'\s+', ' ', text)
    
    # 保留单个标点，去除重复标点
    text = re.sub(r'([。，！？~、])\1+', r'\1', text)
    
    # 移除数字和英文
    # text = re.sub(r'[a-zA-Z0-9]', '', text)
    
    # 移除首尾空格
    text = text.strip()
    
    # 移除空文本
    if len(text) < 2:  # 如果清洗后文本太短，直接返回空
        return ""
        
    return text

def analyze_text_length(texts):
    """分析文本长度分布"""
    lengths = [len(text) for text in texts]
    print(f"\n文本长度统计:")
    print(f"最小长度: {min(lengths)}")
    print(f"最大长度: {max(lengths)}")
    print(f"平均长度: {np.mean(lengths):.2f}")
    print(f"中位数长度: {np.median(lengths):.2f}")
    print(f"90%分位数长度: {np.percentile(lengths, 90):.2f}")
    
    return lengths

def main():
    # 读取数据集
    print("正在读取数据集...")
    df = pd.read_csv('OCEMOTION.csv', 
                     header=None,
                     names=['index', 'text', 'label'],
                     sep='\t',  # 使用制表符作为分隔符
                     quoting=3,  # QUOTE_NONE
                     encoding='utf-8'  # 指定编码
                     )
    
    # 清理文本
    print("正在清理文本...")
    df['text'] = df['text'].apply(clean_text)
    
    # 分析清理前后的文本长度
    original_lengths = analyze_text_length(df['text'])
    
    # 设置文本长度阈值
    MIN_LENGTH = 4  # 最小长度阈值
    MAX_LENGTH = 128  # 最大长度阈值
    
    # 过滤掉过长或过短的文本
    df = df[(df['text'].str.len() >= MIN_LENGTH) & 
            (df['text'].str.len() <= MAX_LENGTH)]
    
    # 标签映射
    label_map = {
        'like': 0,
        'happiness': 1, 
        'sadness': 2,
        'anger': 3,
        'disgust': 4
    }
    
    # 过滤出需要的5个类别
    target_labels = list(label_map.keys())
    df = df[df['label'].isin(target_labels)]
    
    # 将文本标签转换为数字标签
    df['label'] = df['label'].map(label_map)
    
    # 打印类别分布
    print("\n原始数据集中各类别样本数量:")
    print(df['label'].value_counts())
    
    # 处理类别不平衡
    print("\n使用过采样平衡数据集...")
    ros = RandomOverSampler(random_state=42)
    texts = df['text'].values.reshape(-1, 1)
    labels = df['label'].values
    texts_resampled, labels_resampled = ros.fit_resample(texts, labels)
    
    # 转换回DataFrame
    df_balanced = pd.DataFrame({
        'text': texts_resampled.flatten(),
        'label': labels_resampled
    })
    
    print("\n平衡后各类别样本数量:")
    print(df_balanced['label'].value_counts())
    
    # 划分训练集和测试集
    train_df, test_df = train_test_split(
        df_balanced, 
        test_size=0.2, 
        random_state=42, 
        stratify=df_balanced['label']
    )
    
    # 保存处理后的数据
    train_df.to_csv('train.csv', index=False)
    test_df.to_csv('test.csv', index=False)
    
    # 打印示例数据
    print("\n清洗后的示例数据:")
    for i in range(5):
        print(f"文本: {df_balanced['text'].iloc[i]}")
        print(f"标签: {df_balanced['label'].iloc[i]}")
        print(f"长度: {len(df_balanced['text'].iloc[i])}\n")

if __name__ == "__main__":
    main() 