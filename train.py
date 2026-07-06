import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from tqdm import tqdm


class Trainer:
    def __init__(self,model,opt,sheduler,criterion,device):
        self.model = model
        self.opt = opt
        self.sheduler = sheduler
        self.criterion = criterion
        self.device = device
        
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
            self.sheduler.step()

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
        return avg_loss, acc, preds, trues
            
