import os

import torch
from sklearn.metrics import accuracy_score
from tqdm import tqdm


class Trainer:
    def __init__(self,model,opt,scheduler,criterion,device,save_path):
        self.model = model
        self.opt = opt
        self.scheduler = scheduler
        self.criterion = criterion
        self.device = device
        self.best_dev_acc = 0.0
        self.save_path = save_path

    def train_epoch(self, loader):  
        self.model.train()
        total_loss = 0
        preds, trues = [], []   
        
        for batch in tqdm(loader, desc="Train"):
            input_ids = batch["input_ids"].to(self.device)
            mask = batch["attention_mask"].to(self.device)
            label = batch["label"].to(self.device)

            self.opt.zero_grad()
            logits = self.model(input_ids, mask)
            loss = self.criterion(logits, label)
            loss.backward()
            self.opt.step()
            self.scheduler.step()

            total_loss += loss.item()
            preds.extend(torch.argmax(logits, dim=1).cpu().numpy())    
            trues.extend(label.cpu().numpy())
            
        avg_loss = total_loss / len(loader)
        acc = accuracy_score(trues, preds)

        return avg_loss, acc

    
    def eval_epoch(self, loader):
        self.model.eval()  
        total_loss = 0
        preds, trues = [], []
        with torch.no_grad():
            for batch in tqdm(loader, desc="Eval"):
                input_ids = batch["input_ids"].to(self.device)
                mask = batch["attention_mask"].to(self.device)
                label = batch["label"].to(self.device)

                logits = self.model(input_ids, mask)
                loss = self.criterion(logits, label)
                total_loss += loss.item()

                batch_preds = torch.argmax(logits, dim=1).cpu().numpy()
                batch_trues = label.cpu().numpy()
                preds.extend(batch_preds)
                trues.extend(batch_trues)

        avg_loss = total_loss / len(loader)
        acc = accuracy_score(trues, preds)
        if acc > self.best_dev_acc:
            self.best_dev_acc = acc
            checkpoint = {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.opt.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "best_acc": self.best_dev_acc
            }

            save_dir = os.path.dirname(self.save_path)
            if save_dir and not os.path.exists(save_dir): # 路径不为空且文件夹不存在就创建
                os.makedirs(save_dir, exist_ok=True)      # exist_ok=True存在就跳过

            torch.save(checkpoint, self.save_path)
            print(f"更新，最优模型已保存至 {self.save_path}")

        return avg_loss, acc, preds, trues

    def test_epoch(self, loader):
        self.model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for batch in tqdm(loader, desc="Final Test"):
                input_ids = batch["input_ids"].to(self.device)
                mask = batch["attention_mask"].to(self.device)
                label = batch["label"].to(self.device)

                logits = self.model(input_ids, mask)

                batch_preds = torch.argmax(logits, dim=1).cpu().numpy()
                batch_trues = label.cpu().numpy()
                preds.extend(batch_preds)
                trues.extend(batch_trues)

        acc = accuracy_score(trues, preds)

        return acc, preds, trues
            
