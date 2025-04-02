import os
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 定义 IOU 阈值和对应的文件夹
iou_thresholds = [0.2, 0.4, 0.45, 0.5, 0.6, 0.8]
folder_prefix = "../eval@"

data = {}  # 存储算法在各个 IOU 阈值下的准确率

for iou in iou_thresholds:
    folder = f"{folder_prefix}{iou}"
    if not os.path.exists(folder):
        print(f"文件夹 {folder} 不存在，跳过")
        continue

    for file in os.listdir(folder):
        if file.endswith(".xlsx"):
            algorithm_name = file.split("_")[0]
            file_path = os.path.join(folder, file)
            try:
                df = pd.read_excel(file_path)
            except Exception as e:
                print(f"读取 {file_path} 失败：{e}")
                continue

            # 计算平均准确率（Precision = TP / (TP + FP)）
            avg_precision = df['precision'].mean()  # 假设 Excel 文件有 'precision' 列

            if algorithm_name not in data:
                data[algorithm_name] = {}
            data[algorithm_name][iou] = avg_precision  # 存储每个算法在不同 IOU 阈值下的准确率

# 绘制 IOU 阈值和准确率的曲线，添加固定点 (0, 1) 和 (1, 0)
order = ['cellprofiler', 'MEDIAR', 'cellpose', 'cellpose3', 'sam', 'stardist', 'deepcell', 'hovernet']
colors = {
    'cellprofiler': '#ff7f0e',
    'MEDIAR': '#d62728',
    'cellpose': '#1f77b4',
    'cellpose3': '#2ca02c',
    'sam': '#8c564b',
    'stardist': '#9467bd',
    'deepcell': '#17becf',
    'hovernet': '#e377c2',
}

plt.figure(figsize=(8, 6))
for algo in order:
    if algo in data:
        sorted_data = sorted(data[algo].items(), key=lambda x: x[0])
        iou_values = [0] + [x[0] for x in sorted_data] + [1]
        precisions = [1] + [x[1] for x in sorted_data] + [0]
        plt.plot(iou_values, precisions, marker='o', label=algo, color=colors[algo])

plt.xlabel("IoU matching threshold")
plt.ylabel("Average precision")
plt.title("ssDNA")
plt.legend()
plt.grid(True)
plt.savefig("ssDNA.svg", format="svg", dpi=600)
plt.show()