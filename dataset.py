import pandas as pd
import torch
from torch.utils.data import Dataset
import os

from transformers import BertTokenizer

from utils.config_utils import get_label2id

class NewsDataset(Dataset):
    def __init__(self, cfg, txt_path):
        self.cfg = cfg
        self.model_name = cfg["model_name"]
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name) # 分词器
        self.max_len = cfg["max_len"]
        self.label_map_path = cfg["label_map_path"]

        self.label2id = get_label2id(
            txt_path=cfg["train_path"],
            save_path=self.label_map_path
        )
        data = []
        if not os.path.exists(txt_path):
            raise FileNotFoundError(f"数据集文件不存在，请检查路径：{txt_path}")

        with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()  # 去掉空白字符
                if not line:
                    continue  # 如果not false
                # 按_!_分割字段！
                parts = line.split("_!_")
                if len(parts) < 4:
                    continue

                # 字段顺序：0 新闻ID 1 分类code 2 分类名称(英文) 3 新闻标题 4 关键词
                category_name_en = parts[2]
                title = parts[3]

                if category_name_en in self.label2id:
                    data.append({
                        "text": title,
                        "label": self.label2id[category_name_en]  # 键值对对应
                    })
        self.df = pd.DataFrame(data)
        print(f"加载 {txt_path} 完成：{len(self.df)} 条有效数据")
    # 多少条
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = str(self.df.iloc[idx]["text"])  #iloc取第idx行text列
        label = self.df.iloc[idx]["label"]
        return {"text": text, "label": label}

    def collate_fn(self, batch):
        #直接取self的tokenizer/max_len
        texts = [item["text"] for item in batch]
        labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)

        enc = self.tokenizer(
            texts,
            truncation=True,
            padding="longest",
            max_length=self.max_len,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "label": labels
        }
