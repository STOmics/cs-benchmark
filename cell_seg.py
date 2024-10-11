import os
import time
import argparse
import cv2
import sys

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
    if img_type == 'he' or img_type == 'mif':
        new_dir = os.path.join(os.path.dirname(image_dir), 'processed_images')
        os.makedirs(new_dir, exist_ok=True)
        processed_dir = process_images_in_directory(image_dir, new_dir, img_type)

    for m in methods:
        assert m in __methods__

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
        print('{} ran for a total of {} s'.format(m, t))
        print('{} result saved to {}'.format(m, os.path.join(output_path, m)))

if __name__ == '__main__':
    main()
    sys.exit()
