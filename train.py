
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from tqdm import tqdm
def train_epoch(model, loader, opt, scheduler,criterion, device):  # 优化器，学习率调度器
    model.train()
    total_loss = 0
    preds, trues = [], []   # 收集所有批次的预测值和真实标签

    for batch in tqdm(loader, desc="Train"):

        input_ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        label = batch["label"].to(device)

        opt.zero_grad()
        logits = model(input_ids, mask)
        loss = criterion(logits, label)
        loss.backward()
        opt.step()
        scheduler.step()

        total_loss += loss.item()
        preds.extend(torch.argmax(logits, dim=1).cpu().numpy())    # argmax(logits, dim=1) 按行找最大值的索引（位置）转为numpy数组，存入列表（分数最高的类别）
        trues.extend(label.cpu().numpy())

    avg_loss = total_loss / len(loader)
    acc = accuracy_score(trues, preds)
    return avg_loss, acc

def eval_epoch(model, loader, criterion,device):
    model.eval()  # Dropout 关闭，固定参数
    total_loss = 0
    preds, trues = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Eval"):
            input_ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            label = batch["label"].to(device)

            logits = model(input_ids, mask)
            loss = criterion(logits, label)
            total_loss += loss.item()

            preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            trues.extend(label.cpu().numpy())
    return total_loss / len(loader), accuracy_score(trues, preds), preds, trues

