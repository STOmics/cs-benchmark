import cv2
import numpy as np
import pandas as pd
import os
import logging
from skimage.measure import label, regionprops
from skimage.segmentation import find_boundaries
from scipy.spatial.distance import cdist
from tqdm import tqdm
import argparse
import sys
from cellmorphology import CellImageAnalyzer  # Assuming CellImageAnalyzer is in the cellmorphology module

# Set up logging
models_logger = logging.getLogger(__name__)

def search_files(file_path, exts):
    """
    Search for files with specific extensions within a given directory.

    Args:
        file_path (str): The path of the directory to search.
        exts (list of str): A list of file extensions to search for (e.g., ['.tif', '.png']).

    Returns:
        list of str: A list of file paths that match the given extensions.
    """
    files_ = []
    for root, dirs, files in os.walk(file_path):
        for f in files:
            fn, ext = os.path.splitext(f)
            if ext in exts:
                files_.append(os.path.join(root, f))
    return files_

def main(gt_path, output_path):
    """
    Main function to analyze cell morphology from images and save the results.

    Args:
        gt_path (str): The path to the ground truth images.
        output_path (str): The path to save the output CSV file.

    This function:
        - Searches for image files in the given path.
        - Initializes CellImageAnalyzer for each file.
        - Computes contours and morphological statistics.
        - Generates a DataFrame with the results and saves it to CSV.
    """
    # Search for image files with specific extensions in the given path
    files = search_files(gt_path, ['.tif', '.png', '.jpg', '.tiff'])
    all_file = pd.DataFrame()  # Initialize an empty DataFrame

    # Iterate through the files and analyze each
    for i, file in enumerate(tqdm(files, desc="mask analysis")):
        analyzer = CellImageAnalyzer(file)  # Initialize CellImageAnalyzer
        analyzer.compute_contours_and_stats()  # Compute contours and stats for the image
        analyzer.calculate_morphology_stats()  # Calculate morphology statistics
        df = analyzer.generate_dataframe()  # Generate a DataFrame with the statistics
        mean_values = df.mean()  # Calculate mean values for the statistics

        # Add filename to the DataFrame
        mean_values['filename'] = os.path.basename(file)
        all_file = all_file._append(mean_values, ignore_index=True)  # Append to the main DataFrame
        
    # Replace infinite values with NaN and set the filename as index
    all_file.replace([np.inf, -np.inf], np.nan, inplace=True)
    all_file.set_index('filename', inplace=True)
    # Save the DataFrame to a CSV file
    all_file.to_csv(os.path.join(output_path, 'cell_morphology_data.csv'))

if __name__ == '__main__':
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Add parameters for ground truth path and output path")
    parser.add_argument('-g', "--gt_path", help="The path to the ground truth images")
    parser.add_argument('-o', "--output", help="The output file path")

    args = parser.parse_args()
    gt_path = args.gt_path
    output_path = args.output
    main(gt_path, output_path)
    sys.exit()
