## 🧪 Demo 1：文本分类任务

### ✅ 任务目标

基于 [bert-base-chinese](https://huggingface.co/google-bert/bert-base-chinese)模型，完成文本分类任务的训练与评估，掌握完整的模型构建与调参流程。

### 🔧 实现要求

1. 使用 PyTorch 和 Hugging Face 生态，**独立实现如下模块**：

   * 数据预处理流程
   * 训练与测试全过程
2. 使用 [SwanLab](https://www.swanlab.cn/) 等工具进行训练过程与结果的可视化。
3. 使用**今日头条文本分类**数据集见，详见网盘数据集中的demo1文本分类，[数据来源参考](https://github.com/aceimnorstuvwxz/toutiao-text-classfication-dataset)。

### 📊 实验汇报要求

* 参考指标：acc：83%
* 分析超参数（如学习率、batch size、dropout 等）对结果的影响
* 结合可视化图表，对实验结果进行深入解读

<img width="1533" height="674" alt="image" src="https://github.com/user-attachments/assets/2f70e311-8a72-4979-9a16-bda320c0cf0b" />

训练损失曲线从第一轮开始持续平稳下降，最终收敛至 0.1666，训练准确率则同步稳步上升，最终达到 96.83%
测试准确率从初始的 80.5% 左右逐步提升至最终的 83.92%
