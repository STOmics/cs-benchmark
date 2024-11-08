import os 
import time
import argparse
import cv2
import sys
import numpy as np

work_path = os.path.abspath('.')
__py__ = {
    'MEDIAR': '/media/Data/shican/software/miniconda3/envs/MEDIAR/bin/python',
    'cellpose': 'python',
    'cellpose3': 'python',
    'deepcell': 'python',
    'sam': 'python',
    'stardist': 'python',
    'cellprofiler': '/media/Data/shican/software/miniconda3/envs/cp4/bin/python'
}
__methods__ = ['MEDIAR', 'cellpose', 'cellpose3', 'sam', 'stardist', 'deepcell', 'cellprofiler']

__script__ = {
    'MEDIAR': os.path.join(work_path, 'src/methods/MEDIAR/mediar_main.py'),
    'cellpose': os.path.join(work_path, 'src/methods/cellpose_main.py'),
    'cellpose3': os.path.join(work_path, 'src/methods/cellpose3_main.py'),
    'deepcell': os.path.join(work_path, 'src/methods/deepcell_main.py'),
    'sam': os.path.join(work_path, 'src/methods/sam_main.py'),
    'stardist': os.path.join(work_path, 'src/methods/stardist_main.py'),
    'cellprofiler': os.path.join(work_path, 'src/methods/cellprofiler/cellprofiler_main.py')
}

def split_image(image, max_size=2000):
    height, width = image.shape[:2]
    patches = []
    for y in range(0, height, max_size):
        for x in range(0, width, max_size):
            patch = image[y:y+max_size, x:x+max_size]
            patches.append((patch, (y, x)))
    return patches, (height, width)

def stitch_patches(patches, image_size, max_size=2000):
    if len(patches[0][0].shape) == 3:
        full_image = np.zeros((image_size[0], image_size[1], 3), dtype=np.uint8)
    else:
        full_image = np.zeros(image_size, dtype=np.uint8)
        
    for patch, (y, x) in patches:
        full_image[y:y+patch.shape[0], x:x+patch.shape[1]] = patch
    return full_image

def generate_grayscale_negative(image_path, output_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    negative_img = cv2.bitwise_not(img)
    cv2.imwrite(output_path, negative_img)

def generate_negative(image_path, output_path):
    img = cv2.imread(image_path)
    negative_img = cv2.bitwise_not(img)
    cv2.imwrite(output_path, negative_img)

def process_images_in_directory(image_dir, new_dir, img_type):
    for filename in os.listdir(image_dir):
        image_path = os.path.join(image_dir, filename)
        if os.path.isfile(image_path):
            new_image_path = os.path.join(new_dir, filename)
            if img_type == 'he':
                generate_grayscale_negative(image_path, new_image_path)
            elif img_type == 'mif':
                generate_negative(image_path, new_image_path)
    return new_dir

def main():
    parser = argparse.ArgumentParser(description="you should add those parameter")
    parser.add_argument('-i', "--input", help="the input directory path", required=True)
    parser.add_argument('-o', "--output", help="the output directory", required=True)
    parser.add_argument("-m", "--method", nargs='+', help='/'.join(__methods__), required=True)
    parser.add_argument("-t", "--img_type", help="ss/he", required=True)
    parser.add_argument("-g", "--gpu", help="the gpu index", default="-1")

    args = parser.parse_args()
    image_dir = args.input
    output_path = args.output
    methods = args.method
    is_gpu = args.gpu
    img_type = args.img_type.lower()
    processed_dir = image_dir
    assert os.path.isdir(image_dir), "Input path must be a directory"
    print(f'get {len(os.listdir(image_dir))} files')
    for m in methods:
        assert m in __methods__

    if img_type == 'he' or img_type == 'mif':
        new_dir = os.path.join(os.path.dirname(image_dir), 'processed_images')
        os.makedirs(new_dir, exist_ok=True)
        processed_dir = process_images_in_directory(image_dir, new_dir, img_type)

    # Step 1: Split the image into patches
    for filename in os.listdir(image_dir):
        image_path = os.path.join(image_dir, filename)
        image = cv2.imread(image_path)

        if image.shape[0] > 2000 or image.shape[1] > 2000:
            patches, original_size = split_image(image)

            # Save patches temporarily
            patch_output_dir = os.path.join(output_path, 'patches', os.path.splitext(filename)[0])
            os.makedirs(patch_output_dir, exist_ok=True)

            for idx, (patch, (y, x)) in enumerate(patches):
                temp_patch_path = os.path.join(patch_output_dir, f'patch_{y}_{x}.tif')
                cv2.imwrite(temp_patch_path, patch)

            # Step 2: Process patches using methods
            for m in methods:
                if m in ['cellprofier']:
                    cmd = '{} {} -i {} -o {} -g {} -t {}'.format(
                        __py__[m], __script__[m], image_dir, os.path.join(output_path, m), is_gpu, img_type)
                    print(cmd)
                    os.system(cmd)
                    t = time.time() - start
                    print(f'{m} ran for a total of {t} s')
                    continue
                method_output_dir = os.path.join(output_path, m, os.path.splitext(filename)[0])
                os.makedirs(method_output_dir, exist_ok=True)

                # Run the segmentation methods on the patch directory
                start = time.time()
                cmd = '{} {} -i {} -o {} -g {} -t {}'.format(
                    __py__[m], __script__[m], patch_output_dir, method_output_dir, is_gpu, img_type)
                print(cmd)
                os.system(cmd)
                t = time.time() - start
                print(f'{m} ran for a total of {t} s')

                # Step 3: After segmentation, stitch the processed patches back into the full image
                processed_patches = []
                for idx, (patch, (y, x)) in enumerate(patches):
                    processed_patch = cv2.imread(os.path.join(method_output_dir, f'patch_{y}_{x}.tif'))
                    processed_patches.append((processed_patch, (y, x)))

                stitched_result = stitch_patches(processed_patches, original_size)
                final_result_path = os.path.join(output_path, m, filename)
                os.makedirs(os.path.dirname(final_result_path), exist_ok=True)
                cv2.imwrite(final_result_path, stitched_result)
                print(f'{m} result for {filename} saved to {final_result_path}')

        else:
            for m in methods:
                start = time.time()
                if m == 'cellprofiler':
                    cmd = '{} {} -i {} -o {} -g {} -t {}'.format(
                        __py__[m], __script__[m], image_dir, os.path.join(output_path, m), is_gpu, img_type)
                elif m == 'stardist' and img_type == 'he':
                    cmd = '{} {} -i {} -o {} -g {} -t {}'.format(
                        __py__[m], __script__[m], image_dir, os.path.join(output_path, m), is_gpu, img_type)
                elif (m == 'cellpose' or  m == 'cellpose3' or m == 'MEDIAR') and img_type == 'mif':
                    cmd = '{} {} -i {} -o {} -g {} -t {}'.format(
                        __py__[m], __script__[m], image_dir, os.path.join(output_path, m), is_gpu, img_type)
                else:
                    cmd = '{} {} -i {} -o {} -g {} -t {}'.format(
                        __py__[m], __script__[m], processed_dir, os.path.join(output_path, m), is_gpu, img_type)
                print(cmd)
                os.system(cmd)
                t = time.time() - start
                print(f'{m} ran for a total of {t} s')
                print(f'{m} result saved to {os.path.join(output_path, m)}')

if __name__ == '__main__':
    main()
