import pandas as pd
import torch
from torch.utils.data import Dataset
import os
import json

# ======================标签映射======================
def get_label_map(data_df=None, save_path="./label_map.json"):
    """
    自动生成标签映射：
    1. 如果文件存在 → 直接加载
    2. 如果不存在 → 从数据自动提取 → 保存到json
    """
    if os.path.exists(save_path):
        with open(save_path, "r", encoding="utf-8") as f:
            label_map = json.load(f)
        return label_map

    # 从数据中自动提取所有唯一标签
    labels = sorted(data_df["label_str"].unique())
    label_map = {label: idx for idx, label in enumerate(labels)}

    # 保存到文件
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(label_map, f, ensure_ascii=False, indent=4)

    print(f"✅ 自动生成标签映射，已保存到 {save_path}")
    return label_map

# ======================数据读取======================
def load_txt_data(txt_path):
    data = []
    if not os.path.exists(txt_path):
        print(f"文件不存在：{txt_path}")
        return pd.DataFrame()

    with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip() #去掉空白字符
            if not line:
                continue
            # 按_!_分割字段！
            parts = line.split("_!_")
            if len(parts) < 4:
                continue

            # 字段顺序：0 新闻ID 1 分类code 2 分类名称(英文) 3 新闻标题 4 关键词
            category_name_en = parts[2]
            title = parts[3]

            if category_name_en in label_map:
                data.append({
                    "text": title,
                    "label": label_map[category_name_en] #键值对对应
                })
    df = pd.DataFrame(data)
    print(f"加载 {txt_path} 完成：{len(df)} 条有效数据")
    return df

# ====================== Dataset ======================
class NewsDataset(Dataset):
    def __init__(self, df, tokenizer, max_len):
        self.df = df
        self.tokenizer = tokenizer # 分词器
        self.max_len = max_len

    # 多少条
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = str(self.df.iloc[idx]["text"])  #iloc取第idx行
        label = self.df.iloc[idx]["label"]

        # tokenizer Hugging Face 分词器 自带的功能  生成：input_ids + attention_mask
        enc = self.tokenizer(
            text,
            truncation=True,  # 超长截断
            padding="max_length",  # 0到max_len
            max_length=self.max_len,
            return_tensors="pt"
        )

        return {
            "input_ids": enc["input_ids"].flatten(),
            "attention_mask": enc["attention_mask"].flatten(),
            "label": torch.tensor(label, dtype=torch.long) # 长整型
        }