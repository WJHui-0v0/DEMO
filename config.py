import json
import os

class BertConfig:
    def __init__(self, config_path="./config.json"):
        """
        通用配置类：自动加载 JSON，自动绑定所有参数
        无需手动写 self.xxx = xxx
        """
        # 读取配置
        with open(config_path, "r", encoding="utf-8") as f:
            config_dict = json.load(f)

        # 自动把所有 key 变成类属性！！！
        self.__dict__.update(config_dict)

# 测试
if __name__ == "__main__":
    cfg = BertConfig()
    print(cfg.train_path)
    print(cfg.batch_size)
    print(cfg.lr)