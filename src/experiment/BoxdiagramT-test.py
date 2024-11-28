import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams
from scipy.stats import ttest_ind

class BoxDiagramTTest:
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.data_list = []
        self.algorithms = []

    @staticmethod
    def calculate_bin_averages(data, metrics, column):
        bin_averages = {}
        data[column + '_bin'], bins = pd.qcut(data[column], 3, retbins=True, labels=['Low', 'Medium', 'High'])
        for label in ['Low', 'Medium', 'High']:
            bin_index = data[data[column + '_bin'] == label]
            if len(bin_index) < 5:
                continue
            averages = bin_index[metrics].mean()
            bin_averages[label] = averages
        return bin_averages

    @staticmethod
    def perform_t_tests(data, metrics, column):
        t_test_results = {}
        for metric in metrics:
            t_test_results[metric] = {}
            low_group = data[data[column + '_bin'] == 'Low'][metric]
            medium_group = data[data[column + '_bin'] == 'Medium'][metric]
            high_group = data[data[column + '_bin'] == 'High'][metric]

            t_stat_low_med, p_value_low_med = ttest_ind(low_group, medium_group, equal_var=False)
            t_stat_med_high, p_value_med_high = ttest_ind(medium_group, high_group, equal_var=False)

            t_test_results[metric]['Low vs Medium'] = (t_stat_low_med, p_value_low_med)
            t_test_results[metric]['Medium vs High'] = (t_stat_med_high, p_value_med_high)

        return t_test_results

    @staticmethod
    def get_significance_stars(p_value):
        if p_value < 0.0001:
            return '****'
        elif p_value < 0.001:
            return '***'
        elif p_value < 0.01:
            return '**'
        elif p_value < 0.05:
            return '*'
        else:
            return ''

    def read_data(self):
        for file_name in os.listdir(self.folder_path):
            if file_name.endswith('.csv'):
                file_path = os.path.join(self.folder_path, file_name)
                data = pd.read_csv(file_path, encoding='ISO-8859-1')
                data = data[(data != 0).all(axis=1)]
                numeric_data = data.select_dtypes(include=['number'])
                self.data_list.append(numeric_data)
                self.algorithms.append(file_name.split('_')[0])

    def plot_metrics_separately(self, column, save_path):
        metrics = ['f1']
        comparisons = ['Low vs Medium', 'Medium vs High']
        custom_colors = ['#F8766D', '#00BA38', '#619CFF']
        plt.rcParams['font.family'] = 'Arial'
        plt.rcParams['font.size'] = 30

        for metric in metrics:
            fig, axes = plt.subplots(1, len(self.algorithms), figsize=(15, 6), sharey=True)

            for col_idx, (data, algo) in enumerate(zip(self.data_list, self.algorithms)):
                bin_averages = self.calculate_bin_averages(data, metrics, column)
                t_test_results = self.perform_t_tests(data, metrics, column)

                ax = axes[col_idx]
                sns.boxplot(x=data[column + '_bin'], y=data[metric], hue=data[column + '_bin'],
                            palette=custom_colors, ax=ax, dodge=False, showfliers=False)
                ax.set_title(f'{algo}')
                ax.set_xlabel('')
                ax.set_ylabel(metric.capitalize())
                ax.set_ylim(0, 1.2)
                ax.set_xticklabels([])

                y_max = 1
                y_offsets = [y_max * 0.05 * (i + 1) for i in range(len(comparisons))]

                for j, group in enumerate(comparisons):
                    if group in t_test_results[metric]:
                        t_stat, p_value = t_test_results[metric][group]
                        stars = self.get_significance_stars(p_value)
                        if stars:
                            x1 = data[column + '_bin'].cat.categories.get_loc(group.split(' vs ')[0])
                            x2 = data[column + '_bin'].cat.categories.get_loc(group.split(' vs ')[1])
                            line_y_start = y_max + y_offsets[j] - 0.05
                            line_y_end = line_y_start + 0.03
                            ax.plot([x1, x1, x2, x2], [line_y_start, line_y_end, line_y_end, line_y_start], lw=1, c='black')
                            ax.text((x1 + x2) * .5, line_y_end - 0.02, stars, ha='center', va='bottom', color='black', fontsize=20)

            if column == 'average_distance':
                column = 'average distance'
            fig.suptitle(f"{column}", fontsize=25)
            metric_save_path = os.path.join(save_path, f'{metric}_comparison_{column}.pdf')
            plt.savefig(metric_save_path, bbox_inches='tight')
            plt.close()

    def execute(self):
        self.read_data()
        save_path = os.path.join(self.folder_path, "boxplots")
        os.makedirs(save_path, exist_ok=True)
        for col in ['average_distance', 'fov_area', 'y_pred_pixel_sum']:
            self.plot_metrics_separately(col, save_path)

# Example usage
folder_path = r''
box_diagram_t_test = BoxDiagramTTest(folder_path)
box_diagram_t_test.execute()
