# 导入库
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import BertTokenizer, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm
import swanlab

# 导入自定义模块
from config import BertConfig
from dataset import load_txt_data, NewsDataset
from model import BertClassifier

# 实例化参数配置
cfg = BertConfig()



# ====================== 训练/评估 ======================
def train_epoch(model, loader, opt, scheduler, device):  # 优化器，学习率调度器
    model.train()
    total_loss = 0
    preds, trues = [], []   # 收集所有批次的预测值和真实标签
    for batch in tqdm(loader, desc="Train"):
        input_ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        label = batch["label"].to(device)

        opt.zero_grad()
        logits = model(input_ids, mask)
        loss = nn.CrossEntropyLoss()(logits, label)
        loss.backward()
        opt.step()
        scheduler.step()

        total_loss += loss.item()
        preds.extend(torch.argmax(logits, dim=1).cpu().numpy())    # argmax(logits, dim=1) 按行找最大值的索引（位置）转为numpy数组，存入列表
        trues.extend(label.cpu().numpy())

    return total_loss / len(loader), accuracy_score(trues, preds)

def eval_epoch(model, loader, device):
    model.eval()  # Dropout 关闭，固定参数
    total_loss = 0
    preds, trues = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Eval"):
            input_ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            label = batch["label"].to(device)

            logits = model(input_ids, mask)
            loss = nn.CrossEntropyLoss()(logits, label)
            total_loss += loss.item()

            preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            trues.extend(label.cpu().numpy())
    return total_loss / len(loader), accuracy_score(trues, preds), preds, trues

# ====================== 主程序 ======================
if __name__ == "__main__":

    # ====================== SwanLab 初始化 ======================
    swanlab.init(
        project=cfg.project,
        experiment_name=cfg.experiment_name,
        config=cfg
    )
    config = swanlab.config


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("使用设备:", device)

    tokenizer = BertTokenizer.from_pretrained(cfg.model_name)  # BERT 模型专用的分词器

    # 读取数据
    # ====================== 读取数据（自动生成/加载标签映射） ======================
    # 生成标签映射字典label_map 第一次运行自动生成，后续直接读取
    # train_df 处理好的训练数据  label_map 分类名称 → 数字标签 的映射表
    train_df, label_map = load_txt_data(cfg.train_path, cfg.label_map_path)

    # 加载测试集：使用和训练集**完全相同**的标签映射，保证分类编号一致
    # 第二个返回值是 label_map， _ 表示不需要
    test_df, _ = load_txt_data(cfg.test_path, cfg.label_map_path)

    if len(train_df) == 0 or len(test_df) == 0:
        print("检查数据集是否存在")
        exit()

    train_dataset = NewsDataset(train_df, tokenizer, cfg.max_len)
    test_dataset = NewsDataset(test_df, tokenizer, cfg.max_len)

    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)  # 训练打乱
    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False)

    # 初始化模型
    model = BertClassifier(
        model_name=cfg.model_name,
        num_classes=cfg.num_classes,
        dropout=cfg.dropout
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=cfg.lr)   # 更新参数
    total_steps = len(train_loader) * cfg.epochs   # 一个 epoch 有多少个 batch * 训练轮数  =  总更新步数
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.1 * total_steps), total_steps)

    best_acc = 0
    for epoch in range(cfg.epochs):
        print(f"\n===== Epoch {epoch+1}/{cfg.epochs} =====")
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, scheduler, device)
        test_loss, test_acc, preds, trues = eval_epoch(model, test_loader, device)

        # 保存最优模型
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), cfg.save_path)
            print(f"最优模型已保存：{cfg.save_path}")

        swanlab.log({
            "train_loss": train_loss,
            "train_acc": train_acc,
            "test_loss": test_loss,
            "test_acc": test_acc,
            "best_acc": best_acc
        })

        print(f"Train loss: {train_loss:.4f} acc: {train_acc:.4f}")
        print(f"Test  loss: {test_loss:.4f} acc: {test_acc:.4f} best: {best_acc:.4f}")

    # ====================== 加载最优权重做最终评估 ======================
    print("\n===== 加载最优权重进行最终测试 =====")
    model.load_state_dict(torch.load(cfg.save_path))
    model.to(device)
    final_test_loss, final_test_acc, final_preds, final_trues = eval_epoch(model, test_loader, device)

    print("\n===== 最终分类报告 =====")
    print(classification_report(final_trues, final_preds, target_names=list(label_map.keys()), digits=4))

    swanlab.finish()