
import numpy as np

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

