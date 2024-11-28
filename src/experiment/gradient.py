import os
import numpy as np
import cv2
from skimage.measure import label, regionprops
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from scipy.stats import mannwhitneyu
import seaborn as sns
import tifffile
from sklearn.metrics import f1_score

# Set the directories for input images, ground truth (GT) masks, and prediction masks
input_dir = 'D:/dataset/HE/img_gray_invert/'
gt_dir = 'D:/dataset/HE/gt/'
pred_dirs = ['D:/dataset/HE/output/MEDIAR/', 'D:/dataset/HE/output/stardist/', 
             'D:/dataset/HE/output/cellpose/', 'D:/dataset/HE/output/cellpose3/', 
             'D:/dataset/HE/output/sam/', 'D:/dataset/HE/output/cellprofiler', 
             'D:/dataset/HE/output/deepcell']  
output_dir = 'D:/dataset/HE'

# Get all image filenames in the directory
image_files = [f for f in os.listdir(input_dir) if f.endswith('.tif')]

algorithm_results = {}

# Loop through each algorithm directory
for pred_dir in pred_dirs:
    algorithm_name = os.path.basename(pred_dir.strip('/'))
    all_gradients = []
    all_f1_scores = []

    # Loop through each image
    for image_file in image_files:
        # Read input image, GT mask, and predicted mask
        cell_image = cv2.imread(os.path.join(input_dir, image_file), cv2.IMREAD_GRAYSCALE)
        gt_mask = cv2.imread(os.path.join(gt_dir, image_file.replace('-img.tif', '-mask.tif')), cv2.IMREAD_GRAYSCALE)
        pred_mask = tifffile.imread(os.path.join(pred_dir, image_file))
        # Label the GT and predicted masks
        gt_labels = label(gt_mask)
        pred_labels = label(pred_mask)

        # Extract centroids of GT and predicted regions
        gt_props = regionprops(gt_labels)
        pred_props = regionprops(pred_labels)

        gt_centroids = np.array([prop.centroid for prop in gt_props])
        pred_centroids = np.array([prop.centroid for prop in pred_props])
        if len(gt_centroids) == 0 or len(pred_centroids) == 0:
            continue
        
        # Compute the distance matrix between GT and predicted centroids
        cost_matrix = np.linalg.norm(gt_centroids[:, np.newaxis] - pred_centroids, axis=2)

        # Use Hungarian algorithm to match labels
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        # Create a map of matched labels
        matched_pred_labels = np.zeros_like(pred_labels)
        for i, j in zip(row_ind, col_ind):
            matched_pred_labels[pred_labels == pred_props[j].label] = gt_props[i].label

        # Calculate the gradient and F1 score for each cell
        cell_gradients = []
        f1_scores = []

        for region in gt_props:
            minr, minc, maxr, maxc = region.bbox
            cell_region = cell_image[minr:maxr, minc:maxc]
            gt_region = gt_labels[minr:maxr, minc:maxc]
            pred_region = matched_pred_labels[minr:maxr, minc:maxc]

            # Compute the gradient (Sobel operator)
            sobel_x = cv2.Sobel(cell_region, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(cell_region, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)  # Calculate gradient magnitude
            gradient = np.mean(gradient_magnitude[gt_region == region.label])  # Compute the average gradient
            cell_gradients.append(gradient)

            # Calculate the F1 score
            y_true = (gt_region == region.label).flatten()
            y_pred = (pred_region == region.label).flatten()
            if np.sum(y_true) == 0:
                f1 = 0
            else:
                f1 = f1_score(y_true, y_pred, average='binary')
            f1_scores.append(f1)

        
        all_gradients.extend(cell_gradients)
        all_f1_scores.extend(f1_scores)

    # Store the results for each algorithm
    algorithm_results[algorithm_name] = (all_gradients, all_f1_scores)


sns.set_theme(style="whitegrid")
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 30
colors = sns.color_palette(['#F8766D', '#00BA38', '#619CFF'])
groups = ['Low', 'Mid', 'High']  # 修改为3个组别

num_algorithms = len(algorithm_results)
fig, axes = plt.subplots(nrows=1, ncols=num_algorithms, figsize=(20, 8), sharey=True)
fig.suptitle('F1 Score vs Cell Gradient Groups for Different Algorithms')

for ax in axes:
    ax.spines['top'].set_color('black')
    ax.spines['right'].set_color('black')
    ax.spines['left'].set_color('black')
    ax.spines['bottom'].set_color('black')
    ax.spines['top'].set_linewidth(1.25)    
    ax.spines['right'].set_linewidth(1.25)  
    ax.spines['left'].set_linewidth(1.25)   
    ax.spines['bottom'].set_linewidth(1.25) 

for i, (algorithm, (all_gradients, all_f1_scores)) in enumerate(algorithm_results.items()):
    ax = axes[i]
    ax.tick_params(axis='y', labelsize=20)  

    thresholds = np.percentile(all_gradients, [33, 66])

    # Divide the gradients into low, mid, and high groups
    low_group = [f1 for grad, f1 in zip(all_gradients, all_f1_scores) if grad <= thresholds[0]]
    mid_group = [f1 for grad, f1 in zip(all_gradients, all_f1_scores) if thresholds[0] < grad <= thresholds[1]]
    high_group = [f1 for grad, f1 in zip(all_gradients, all_f1_scores) if grad > thresholds[1]]

    # Calculate significance between groups
    _, p_low_mid = mannwhitneyu(low_group, mid_group)
    _, p_mid_high = mannwhitneyu(mid_group, high_group)

    means = [np.mean(low_group), np.mean(mid_group), np.mean(high_group)]

    sns.barplot(x=groups, y=means, palette=colors, ax=ax)
    ax.set_title(algorithm, fontsize=20)

    ax.set_xticklabels([])  

    if i == 0:
        ax.set_ylabel('F1 Score', fontsize=20) 
    else:
        ax.set_ylabel('')  

    def add_significance_bar(ax, x1, x2, y, p_value, offset=0.05):
        bar_height = y + offset
        ax.plot([x1, x1, x2, x2], [y, bar_height, bar_height, y], color='black')
        if p_value < 0.0001:
            significance = '****'
        elif p_value < 0.001:
            significance = '***'
        elif p_value < 0.01:
            significance = '**'
        elif p_value < 0.05:
            significance = '*'
        else:
            significance = 'ns'
        ax.text((x1 + x2) / 2, bar_height, significance, ha='center', va='bottom',fontsize=12)

    y_max = max(means)
    add_significance_bar(ax, 0, 1, y_max, p_low_mid)
    add_significance_bar(ax, 1, 2, y_max + 0.1, p_mid_high)

plt.tight_layout(rect=[0, 0.03, 1, 0.95]) 
plt.savefig(os.path.join(output_dir, 'HE_gradient.pdf'), dpi=600)
plt.show()
