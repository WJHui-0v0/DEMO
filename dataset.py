import torch
from torch.utils.data import Dataset, DataLoader
import os
from transformers import BertTokenizer
from utils.config_utils import get_label2id

class NewsDataset(Dataset):
    def __init__(self, cfg, mode):
        self.cfg = cfg
        self.model_name = cfg["model_name"]
        self.max_len = cfg["max_len"]
        self.label_map_path = cfg["label_map_path"]
        self.tokenizer = cfg["tokenizer"]
        self.mode = mode
        # 对应文件路径
        path_map = {
            "train": cfg["train_path"],
            "dev": cfg["dev_path"],
            "test": cfg["test_path"]
        }
        if mode not in path_map:
            raise ValueError("mode只能填train/dev/test")
        self.txt_path = path_map[mode]

        self.label2id = get_label2id(cfg)
        if not os.path.exists(self.txt_path):
            raise FileNotFoundError(f"数据集文件不存在，请检查路径：{self.txt_path}")
        self.data = None

    def load_data(self):
        data = []
        with open(self.txt_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.read().splitlines()
        for line in lines:
            line = line.strip()  # 去掉空白字符
            if not line:
                continue  # 如果not false
            # 按_!_分割字段！
            parts = line.split("_!_")
            if len(parts) <  4:
                continue
            # 字段顺序：0 新闻ID 1 分类code 2 分类名称(英文) 3 新闻标题 4 关键词
            category_name_en = parts[2]
            title = parts[3]
            if category_name_en in self.label2id:
                data.append({
                    "text": title,
                    "label": self.label2id[category_name_en]  # 键值对对应
                })
        print(f"加载 {self.txt_path} 完成：{len(data)} 条有效数据")
        return data
    # 多少条
    def __len__(self):
        if self.data is None:
            self.data = self.load_data()
        return len(self.data)

    def __getitem__(self, idx):
        if self.data is None:
            self.data = self.load_data()
        return self.data[idx]


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
