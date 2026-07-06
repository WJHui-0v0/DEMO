# Demo1 BERT中文新闻多分类
基于 bert-base-chinese 实现今日头条15类别新闻标题文本分类，使用PyTorch + SwanLab 完成训练、评估、实验可视化全流程。

## 一、项目简介
1. 15分类中文新闻标题自动分类
2. 预训练模型：bert-base-chinese
3. 数据集：今日头条新闻分类数据集，详见[网盘数据集](https://pan.baidu.com/s/1UzTzXXvc02K-rxY720j3fw?pwd=1111)demo1文本分类，[数据来源参考](https://github.com/aceimnorstuvwxz/toutiao-text-classfication-dataset)。
   - 训练集：3000条
   - 验证集：1000条
   - 测试集：1064条
4. 实验工具
   - 训练框架：PyTorch
   - 预训练库：Hugging Face工具
   - 实验可视化：SwanLab [在线链接](https://swanlab.cn/@displan0v0/bert-toutiao-classify/runs/exn5cj8s3ybg2b15ibf5m/chart)
5. 核心功能
   - 自定义Dataset实现文本分词、编码、动态批处理
   - JSON 配置文件统一管理超参数，自动生成标签-数字映射文件
   - 训练循环、验证评估、最优模型自动保存完整 checkpoint
   - 完整分类指标输出（Acc、Precision、Recall、F1）
   - 训练损失/精度曲线SwanLab可视化记录

## 二、环境依赖
### 1. Python版本要求
Python >= 3.8
### 2. 一键安装命令
```
pip install -r requirements.txt
```
## 三、项目目录结构

```
DEMO/
├── data/                          # 数据集目录
│   ├── label_map.json             # 自动生成的标签-数字映射文件
│   ├── train_3k.txt               # 训练集，共3000条
│   ├── dev_1k.txt                 # 验证集，1000条，用于训练、保存最优模型
│   └── test_1k.txt                # 测试集，1064条
├── configs/                       # 配置json文件目录
│   └──exp_bert-base-chinese_lr2e-5_bs16_len128_0704.json    # 超参数配置文件，存储模型路径、批次、学习率、训练轮数、数据集路径等全部可调参数；
├── utils                          # 工具目录
│   ├── config_utils.py            # 提供配置加载与标签编码生成两个工具函数
│   └── utils_metrics.py           # 随机种子+分类报告
├── dataset.py                     # 自定义数据集Dataset类
├── main.py                        # 执行数据加载→训练→最优模型保存→测试评估全流程
├── train.py                       # 含单轮训练、单轮验证两个函数的类
├── requirements.txt               # 项目环境依赖清单，包含torch、transformers、swanlab、sklearn等第三方库
├── model.py                       # 基于预训练中文 BERT 搭建分类网络 BertClassifier
```

## 四、exp_bert-base-chinese_lr2e-5_bs16_len128_0704.json 配置说明

```
{
   "project": "bert-toutiao-classify",           / SwanLab项目名称
    "experiment_name": "bert-base-chinese0704",  / 本次实验名称，用于区分多组调参实验
    "model_name": "bert-base-chinese",           / 使用的预训练中文BERT模型名称，自动从HuggingFace下载
    "num_classes": 15,                           / 数据集总分类类别数量
    "max_len": 128,                              / 文本输入token最大截断长度
    "batch_size": 16,                            / 每批次样本数量
    "lr": 2e-5,                                  / 模型基础学习率
    "epochs": 5,                                 / 总轮次
    "dropout": 0.1,                              / 分类头dropout概率，用于缓解模型过拟合
    "train_path": "./data/train_3k.txt",         / 训练集路径
    "test_path": "./data/test_1k.txt",           / 测试集路径，用于最终泛化效果评估
    "dev_path": "./data/dev_1k.txt",             / 验证集路径，训练中实时评估、保存最优权重
    "save_path": "./best_bert.pth",              / 验证集最优模型权重保存路径
    "seed": 111,                                 / 全局随机种子，固定后保证实验结果可复现
    "label_map_path": "./label_map.json"         / 标签转数字ID映射文件存储路径 
} 
```
## 五、实验结果总结
- 最优验证准确率： 82.80%
- 测试准确率达到： 83.83%
- 最终分类报告
```
                    precision    recall  f1-score   support

  news_agriculture     0.8246    0.8868    0.8545        53
          news_car     0.9208    0.9394    0.9300        99
      news_culture     0.9710    0.8701    0.9178        77
          news_edu     0.8750    0.9459    0.9091        74
news_entertainment     0.8879    0.8796    0.8837       108
      news_finance     0.7969    0.6892    0.7391        74
         news_game     0.7647    0.8125    0.7879        80
        news_house     0.8889    0.8163    0.8511        49
     news_military     0.7534    0.7971    0.7746        69
       news_sports     0.9565    0.8544    0.9026       103
        news_story     0.8667    0.7647    0.8125        17
         news_tech     0.7293    0.8509    0.7854       114
       news_travel     0.8226    0.8644    0.8430        59
        news_world     0.7123    0.7027    0.7075        74
             stock     1.0000    0.5714    0.7273        14

          accuracy                         0.8383      1064
         macro avg     0.8514    0.8164    0.8284      1064
      weighted avg     0.8436    0.8383    0.8385      1064
```
- Loss & Acc 曲线如下 [在线链接](https://swanlab.cn/@displan0v0/bert-toutiao-classify/runs/exn5cj8s3ybg2b15ibf5m/chart)

![img.png](data/img.png)

## 五、实验分析
- 本实验基于 BERT 模型完成多类别新闻文本分类任务，共设置 5 轮训练，通过验证集精度保存最优模型。
训练过程中模型训练损失持续下降、训练准确率逐步提升至 96.47%，验证集准确率在第 4 轮达到峰值 82.80%，
第 5 轮出现轻微过拟合。在包含 1064 条样本的测试集上，模型整体分类准确率为 83.83%，
加权平均 F1 值 0.8385；体育、汽车、文化、教育等样本充足的大类识别效果优异，
而股票、故事等少样本类别及财经、国际、军事等语义相近类别存在识别短板，
召回与 F1 指标偏低。分析表明模型存在轻度过拟合、相似领域文本区分困难等问题，