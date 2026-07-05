import pandas as pd
from torch.utils.data import Dataset
import os

# 数据读取
def load_txt_data(txt_path,label_map):
    data = []
    if not os.path.exists(txt_path):
        print(f"文件不存在：{txt_path}")
        return pd.DataFrame()

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
            '''valid_labels = set(label_map.keys())
            if category_name_en in valid_labels:'''
            if category_name_en in label_map:
                data.append({
                    "text": title,
                    "label": label_map[category_name_en] #键值对对应
                })
    df = pd.DataFrame(data)
    print(f"加载 {txt_path} 完成：{len(df)} 条有效数据")
    return df

#Dataset
class NewsDataset(Dataset):
    def __init__(self, df, tokenizer, max_len):
        self.df = df
        self.tokenizer = tokenizer # 分词器
        self.max_len = max_len
    # 多少条
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = str(self.df.iloc[idx]["text"])  #iloc取第idx行text列
        label = self.df.iloc[idx]["label"]
        return {"text": text, "label": label}
