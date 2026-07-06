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

def my_classification_report(y_true, y_pred, target_names, digits=4):
    n_classes = len(target_names)  #总数量
    # 1. 初始化 TP, FP, FN ,第i个类别的真实样本总数,全0数组
    tp = np.zeros(n_classes, dtype=int)
    fp = np.zeros(n_classes, dtype=int)
    fn = np.zeros(n_classes, dtype=int)
    support = np.zeros(n_classes, dtype=int)
    # 2. 遍历所有样本统计
    for true_label_id, pred_label_id  in zip(y_true, y_pred):
        support[true_label_id] += 1
        if true_label_id == pred_label_id:
            tp[true_label_id] += 1
        else:
            fp[pred_label_id] += 1  # 预测成了pred_label_id，这个类假正例+1
            fn[true_label_id] += 1  # 真实是true_label_id，没预测出来，假负例+1
    # 3. 逐类计算 P/R/F1
    precision = np.zeros(n_classes, dtype=float)
    recall = np.zeros(n_classes, dtype=float)
    f1 = np.zeros(n_classes, dtype=float)
    eps = 1e-8  # 防止除0

    for i in range(n_classes):
        p = tp[i] / (tp[i] + fp[i] + eps)
        r = tp[i] / (tp[i] + fn[i] + eps)
        f = 2 * p * r / (p + r + eps)
        precision[i] = p
        recall[i] = r
        f1[i] = f

    # 4. 计算全局指标
    total_correct = sum(tp)     #直接求和列表里所有数字
    total_samples = sum(support)
    accuracy = total_correct / total_samples

    # macro avg 无权重平均
    macro_p = np.mean(precision)
    macro_r = np.mean(recall)
    macro_f1 = np.mean(f1)

    # weighted avg：按 support 加权
    weight = support / (total_samples + eps)
    weighted_p = np.sum(precision * weight)
    weighted_r = np.sum(recall * weight)
    weighted_f1 = np.sum(f1 * weight)

    # 5. 格式化表格
    headers = ["precision", "recall", "f1-score", "support"]
    max_name_len = max(len(name) for name in target_names)
    col_width = max(max_name_len, len("weighted avg"))

    # 表头
    fmt_head = "{:>{w}} " + " {:>10}" * 4   # > 为 右对齐
    report_lines = [fmt_head.format("", *headers, w=col_width)]  # * 序列解包
    report_lines.append("")

    # 每一类行
    row_fmt = "{:>{w}} " + f"{{:>10.{digits}f}}" * 3 + " {:>10d}"
    #f-string 会把单层 {} 识别为变量占位，想输出一个普通的大括号字符，写双层
    for cls_name, p, r, f, s in zip(target_names, precision, recall, f1, support):
        report_lines.append(row_fmt.format(cls_name, p, r, f, s, w=col_width))
    report_lines.append("")  # 添加空行

    # accuracy 行
    acc_fmt = "{:>{w}} " + " {:>10} {:>10}" + f"{{:>10.{digits}f}}" + " {:>10d}"
    report_lines.append(acc_fmt.format("accuracy", "", "", accuracy, total_samples, w=col_width))

    # macro avg
    avg_fmt = "{:>{w}} " + f" {{:>10.{digits}f}}" * 3 + " {:>10d}"
    report_lines.append(avg_fmt.format("macro avg", macro_p, macro_r, macro_f1, total_samples, w=col_width))

    # weighted avg
    report_lines.append(avg_fmt.format("weighted avg", weighted_p, weighted_r, weighted_f1, total_samples, w=col_width))

    return "\n".join(report_lines)


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

    train_dataset = NewsDataset.load_txt_data(txt_path=cfg["train_path"],label_map=label2id, tokenizer=tokenizer, max_len=cfg["max_len"])
    dev_dataset = NewsDataset.load_txt_data(txt_path=cfg["dev_path"], label_map=label2id, tokenizer=tokenizer, max_len=cfg["max_len"])
    test_dataset = NewsDataset.load_txt_data(txt_path=cfg["test_path"],label_map=label2id, tokenizer=tokenizer, max_len=cfg["max_len"])


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
    report = my_classification_report(
        y_true=trues,
        y_pred=preds,
        target_names=list(label2id.keys()),
        digits=4
    )
    print(report)
    swanlab.finish()

if __name__ == "__main__":
    main()
