import torch.nn as nn
from transformers import BertModel

class BertClassifier(nn.Module):
    def __init__(self, model_name, num_classes, dropout):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)      # Dropout层
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)     # 全连接层 输入维度：BERT的隐藏层维度 768

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids, attention_mask)  # 输入到bert模型中 last_hidden_state、pooler_output 句子级别的语义表征能力
        feat = self.dropout(out.pooler_output)
        logits = self.classifier(feat)   # 得到模型对每个类别的预测得分
        return logits  # 未归一化，后续CrossEntropyLoss