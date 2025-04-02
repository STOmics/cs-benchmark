import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import OrderedDict

class BenchmarkPlot:
    def __init__(self, input_dir, output_path, order, colors):
        self.input_dir = input_dir
        self.output_path = output_path
        self.order = order
        self.colors = colors
        self.index = ('precision', 'recall', 'f1','dice')
        self.data_dict = {alg: {metric: [] for metric in self.index} for alg in order}
        self.avg_dict = {}

    def read_data(self):
        for file_name in os.listdir(self.input_dir):
            if file_name.endswith('.xlsx'):
                alg_name = file_name.split('_')[0]
                if alg_name in self.order:
                    file_path = os.path.join(self.input_dir, file_name)
                    df = pd.read_excel(file_path)
                    for metric in self.index:
                        if metric in df.columns:
                            mean_val = df[metric].mean()
                            self.data_dict[alg_name][metric].append(mean_val)

    def calculate_averages(self):
        self.avg_dict = {
            alg: {metric: np.mean(vals) if vals else 0 for metric, vals in metrics.items()}
            for alg, metrics in self.data_dict.items()
        }

    def plot(self):
        order_means = OrderedDict((key, self.avg_dict[key]) for key in self.order if key in self.avg_dict)
        plt.rcParams['font.family'] = 'Arial'
        plt.rcParams['font.size'] = 70
        fig, axs = plt.subplots(figsize=(24, 12))
        x = np.arange(len(self.index))
        width = 0.13
        multiplier = 0

        for attribute, measurement in order_means.items():
            offset = width * multiplier
            rects = axs.bar(
                x + offset,
                [round(val, 2) for val in measurement.values()],
                width,
                label=attribute,
                color=self.colors.get(attribute, '#000000'),
                alpha=0.62
            )
            #axs.bar_label(rects, padding=3,fontsize=15)  
            multiplier += 1
        axs.set_xticks(x + width * len(self.order) / 2, self.index)
        axs.set_ylabel('Evaluation Index')
        axs.set_title(f'staining - {self.input_dir.split("/")[-2]}')
        axs.set_ylim(0, 1)
        axs.spines['top'].set_visible(False)
        axs.spines['right'].set_visible(False)
        #axs.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=30)  
        plt.savefig(os.path.join(self.output_path, f'{self.input_dir.split("/")[-1]}_benchmark.svg'), dpi=600, bbox_inches='tight')
        plt.close()

    def execute(self):
        self.read_data()
        self.calculate_averages()
        self.plot()

# Example usage
input_dir = ''
output_path = ''
order = ['cellprofiler','MEDIAR','cellpose', 'cellpose3','sam', 'stardist','deepcell']
colors = {
    'cellprofiler': '#ff7f0e',
    'MEDIAR': '#d62728',
    'cellpose': '#1f77b4',
    'cellpose3': '#2ca02c',
    'sam': '#8c564b',
    'stardist': '#9467bd',
    'deepcell': '#17becf',
}
benchmark_plot = BenchmarkPlot(input_dir, output_path, order, colors)
benchmark_plot.execute()
