# ===================== 一、第三方库导入 =====================
import argparse
import random
import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import BertTokenizer, get_linear_schedule_with_warmup
from sklearn.metrics import classification_report
from tqdm import tqdm
import swanlab
import os
# 模块导入
from config_utils import load_config, get_label2id
from dataset import NewsDataset
from model import BertClassifier
from train import train_epoch, eval_epoch
def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

def main():
    parser = argparse.ArgumentParser(description="00demo1")
    parser.add_argument(
        "--config_path",
        type=str,
        default="exp_bert-base-chinese_lr2e-5_bs16_len128_0704.json",
        help="配置json路径，默认config.json"
    )
    args = parser.parse_args()
    # 加载全局配置
    cfg = load_config(args.config_path)

    swanlab.init(
        project=cfg["project"],
        experiment_name=cfg["experiment_name"],
        config=cfg
    )

    seed_num = cfg.get("seed", 520)

    set_seed(seed_num)
    # 自动选择GPU/CPU设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("训练使用设备:",device,"随机种子:",seed_num)

    tokenizer = BertTokenizer.from_pretrained(cfg["model_name"])  # BERT 模型专用的分词器

    label2id = get_label2id(txt_path=cfg["train_path"], save_path=cfg["label_map_path"])

    train_dataset = NewsDataset.load_txt_data(txt_path=cfg["dev_path"],label_map=label2id, tokenizer=tokenizer, max_len=cfg["max_len"])
    dev_dataset = NewsDataset.load_txt_data(txt_path=cfg["dev_path"], label_map=label2id, tokenizer=tokenizer, max_len=cfg["max_len"])
    test_dataset = NewsDataset.load_txt_data(txt_path=cfg["dev_path"],label_map=label2id, tokenizer=tokenizer, max_len=cfg["max_len"])


    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["batch_size"],
        shuffle=True,
        collate_fn=train_dataset.collate_fn
        # collate_fn 只能接收一个参数 batch，用 lambda 做一层参数包装、转发。
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=cfg["batch_size"],
        shuffle=False,
        collate_fn=dev_dataset.collate_fn
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg["batch_size"],
        shuffle=False,
        collate_fn=test_dataset.collate_fn
    )

    # 初始化模型
    model = BertClassifier(
        model_name=cfg["model_name"],
        num_classes=cfg["num_classes"],
        dropout=cfg["dropout"]
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=cfg["lr"])  # 更新参数
    total_steps = len(train_loader) * cfg["epochs"] # 一个 epoch 有多少个 batch * 训练轮数  =  总更新步数

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        int(0.1 * total_steps),
        total_steps
    )
    save_path = cfg.get("save_path", "best_model_checkpoint.pth")
    best_acc_dev = 0
    for epoch in range(cfg["epochs"]):
        print(f"\n===== Epoch {epoch+1}/{cfg['epochs']} =====")
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, scheduler, criterion, device)
        dev_loss, dev_acc, preds, trues = eval_epoch(model, dev_loader, criterion, device)

        if dev_acc > best_acc_dev:
            best_acc_dev = dev_acc

            checkpoint_dict = {
                "current_epoch": epoch,  # 当前训练到第几轮
                "model_state_dict": model.state_dict(),  # 模型权重
                "optimizer_state_dict": optimizer.state_dict(),  # 优化器参数
                "scheduler_state_dict": scheduler.state_dict(),  # 学习率调度器状态
                "best_validation_acc": best_acc_dev  # 之前最优验证集精度
            }

            torch.save(checkpoint_dict, save_path)
            print(f"最优模型已保存：{save_path}")

        swanlab.log({
            "train/loss": train_loss,
            "train/acc": train_acc,
            "dev/loss": dev_loss,
            "dev/acc": dev_acc,
            "best/acc": best_acc_dev
        })

        print(f"train loss: {train_loss:.4f} acc: {train_acc:.4f}")
        print(f"dev loss: {dev_loss:.4f} acc: {dev_acc:.4f} best: {best_acc_dev:.4f}")

    print("\n===== 加载最优权重进行最终测试 =====")
    model.load_state_dict(torch.load(save_path)["model_state_dict"])
    model.to(device)

    test_loss, test_acc, preds, trues = eval_epoch(model, test_loader, device)
    swanlab.log({
        "test/loss": test_loss,
        "test/acc": test_acc
    })
    print("\n===== 最终分类报告 =====")
    print(classification_report(trues, preds, target_names=list(label2id.keys()), digits=4))

    swanlab.finish()

if __name__ == "__main__":
    main()
