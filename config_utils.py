import json
import os

def load_config(config_path="exp_bert-base-chinese_lr2e-5_bs16_len128_0704.json"):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在：{config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    return cfg

def get_label2id(txt_path, save_path):
    # 读取
    if os.path.exists(save_path):
        with open(save_path, "r", encoding="utf-8") as f:
            label2id = json.load(f)
        return label2id

    if not os.path.exists(txt_path):
        raise FileNotFoundError(f"数据集文件不存在：{txt_path}")

    # 遍历文件收集所有原始标签 集合
    label_set = set()
    with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("_!_")
            if len(parts) < 4:
                continue
            category_en = parts[2]
            label_set.add(category_en)
    # enumerate 生成映射
    sorted_labels = sorted(label_set)    # 固定标签排序
    label2id = {}
    for idx, label in enumerate(sorted_labels):
        label2id[label] = idx

    # 保存
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(label2id, f, ensure_ascii=False, indent=4)
    print(f"标签映射生成完成，共 {len(label2id)} 类，保存至: {save_path}")

    return label2id