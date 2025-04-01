import os 
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import OrderedDict

"""

Contrast invert and uninvert colors

"""
# 设置反色前和反色后的文件路径
paths = {
    'invert': r"D:\benchmark_plot\dataset\DAPI\eval",
    'uninvert': r"D:\benchmark_plot\dataset\HE\eval_uninvert"
}

# 定义算法顺序
order = [ 
    'MEDIAR','cellpose','deepcell'
]

# 定义颜色
color_uninvert = '#8E8BFE'  # 未反色的颜色
color_invert = '#FEA3A2'    # 反色的颜色

# 初始化字典用于存储各算法数据
data = {alg: {'invert': [], 'uninvert': []} for alg in order}

# 遍历两个目录读取数据
for key, path in paths.items():
    for file_name in os.listdir(path):
        if file_name.endswith('.xlsx'):
            prefix = file_name.split('_')[0]  # 假设算法名是文件名的第一个部分
            if prefix in order:
                file_path = os.path.join(path, file_name)
                df = pd.read_excel(file_path)
                if 'f1' in df.columns:
                    data[prefix][key].append(df['f1'].mean())  # 假设目标列名为 'f1'

# 计算每个算法反色和未反色的平均值
avg_data = {
    alg: {
        'invert': np.mean(values['invert']) if values['invert'] else 0,
        'uninvert': np.mean(values['uninvert']) if values['uninvert'] else 0
    }
    for alg, values in data.items()
}

# 对数据按指定顺序排序
sorted_avg_data = OrderedDict((alg, avg_data[alg]) for alg in order if alg in avg_data)

# 设置字体
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 50

# 开始绘图
fig, ax = plt.subplots(figsize=(20, 10))
x = np.arange(len(sorted_avg_data))
width = 0.4

# 绘制每个算法的未反色和反色的条形图，并在柱子上显示数值
for i, (alg, values) in enumerate(sorted_avg_data.items()):
    rects1 = ax.bar(x[i] - width / 2, values['uninvert'], width, color=color_uninvert, label='HE' if i == 0 else "")  # 未反色
    rects2 = ax.bar(x[i] + width / 2, values['invert'], width, color=color_invert, label='DAPI' if i == 0 else "")  # 反色
    
    # 在每个柱子上显示数值
    ax.text(x[i] - width / 2, values['uninvert'] + 0.02, f'{values["uninvert"]:.2f}', ha='center', va='bottom', fontsize=20)
    ax.text(x[i] + width / 2, values['invert'] + 0.02, f'{values["invert"]:.2f}', ha='center', va='bottom', fontsize=20)

# 设置图表的标签和标题
ax.set_ylabel('f1 Value')
ax.set_title('HE vs DAPI')
ax.set_xticks(x)
ax.set_xticklabels(sorted_avg_data.keys(), rotation=45, ha='right')  # 旋转横坐标并调整位置
ax.set_ylim(0, 1)

# 调整图形布局，确保横坐标显示完整
plt.tight_layout()

# 删除上面和右边的边框
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# 添加图例
handles, labels = ax.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax.legend(by_label.values(), by_label.keys(), loc='upper left', fontsize=20)

# 保存图像
output_dir = 'D:\\benchmark_plot\\comparison_results'
os.makedirs(output_dir, exist_ok=True)
plt.savefig(os.path.join(output_dir, 'algorithms_comparison.svg'), dpi=600)
plt.show()
plt.close()

print(f'算法比较图已保存至: {output_dir}')
