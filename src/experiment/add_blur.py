import os
import cv2
import numpy as np
import pandas as pd

# Set the input and output directories
input_dir = 'D:/dataset/ssDNA/input/'
gaussian_blur_output_dir_1 = 'D:/dataset/ssDNA/gaussian_blur_output_1/'  # weak Gaussian blur
gaussian_blur_output_dir_2 = 'D:/dataset/ssDNA/gaussian_blur_output_2/'  # medium 
gaussian_blur_output_dir_3 = 'D:/dataset/ssDNA/gaussian_blur_output_3/'  # strong
sobel_output_dir = 'D:/dataset/ssDNA/sobel_output/'
csv_file_path = os.path.join('D:/dataset/ssDNA', 'image_gradients.csv')

# Create the output directories
os.makedirs(gaussian_blur_output_dir_1, exist_ok=True)
os.makedirs(gaussian_blur_output_dir_2, exist_ok=True)
os.makedirs(gaussian_blur_output_dir_3, exist_ok=True)
os.makedirs(sobel_output_dir, exist_ok=True)


results = []

# Iterate over all the images in the input directory
for filename in os.listdir(input_dir):
    if filename.endswith('.tif') or filename.endswith('.jpg') or filename.endswith('.png'):
        
        image_path = os.path.join(input_dir, filename)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        if image is None:
            continue

        # Apply Gaussian blur - weak intensity
        blurred_image_1 = cv2.GaussianBlur(image, (5, 5), 1)  
        gaussian_blur_output_path_1 = os.path.join(gaussian_blur_output_dir_1, filename)
        cv2.imwrite(gaussian_blur_output_path_1, blurred_image_1)
        
        # Apply Gaussian blur - medium intensity
        blurred_image_2 = cv2.GaussianBlur(image, (9, 9), 2)  
        gaussian_blur_output_path_2 = os.path.join(gaussian_blur_output_dir_2, filename)
        cv2.imwrite(gaussian_blur_output_path_2, blurred_image_2)
        
        # Apply Gaussian blur - strong intensity
        blurred_image_3 = cv2.GaussianBlur(image, (13, 13), 3)  
        gaussian_blur_output_path_3 = os.path.join(gaussian_blur_output_dir_3, filename)
        cv2.imwrite(gaussian_blur_output_path_3, blurred_image_3)

        # Calculate Sobel gradients
        def calculate_sobel_gradients(img):
            sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
            return np.mean(gradient_magnitude), np.std(gradient_magnitude)
        
        # Calculate the gradient for the original image
        orig_mean_grad, orig_std_grad = calculate_sobel_gradients(image)
        
        # Calculate the gradient for the weak Gaussian blurred image
        blur1_mean_grad, blur1_std_grad = calculate_sobel_gradients(blurred_image_1)
        
        # Calculate the gradient for the medium Gaussian blurred image
        blur2_mean_grad, blur2_std_grad = calculate_sobel_gradients(blurred_image_2)

        # Calculate the gradient for the strong Gaussian blurred image
        blur3_mean_grad, blur3_std_grad = calculate_sobel_gradients(blurred_image_3)

        sobel_combined = np.hstack((cv2.convertScaleAbs(cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)),
                                    cv2.convertScaleAbs(cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3))))
        sobel_output_path = os.path.join(sobel_output_dir, filename.replace('.tif', '_sobel.png'))
        cv2.imwrite(sobel_output_path, cv2.normalize(sobel_combined, None, 0, 255, cv2.NORM_MINMAX))

        results.append({
            'Filename': filename,
            'Orig_Mean_Gradient': orig_mean_grad,
            'Orig_Std_Gradient': orig_std_grad,
            'Blur1_Mean_Gradient': blur1_mean_grad,
            'Blur1_Std_Gradient': blur1_std_grad,
            'Blur2_Mean_Gradient': blur2_mean_grad,
            'Blur2_Std_Gradient': blur2_std_grad,
            'Blur3_Mean_Gradient': blur3_mean_grad,
            'Blur3_Std_Gradient': blur3_std_grad
        })

        print(f"Processed {filename}")

df = pd.DataFrame(results)
df.to_csv(csv_file_path, index=False)

print(f"Processing completed. Results saved to {csv_file_path}")
