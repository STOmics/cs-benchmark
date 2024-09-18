import os
import time
import subprocess
import argparse

work_path = os.path.abspath('.')
# work_path = '/data/work/benchmark/benchmark'
__py__ = {
    'MEDIAR':'/storeData/USER/data/01.CellBin/00.user/fanjinghong/home/anaconda3/envs/MEDIAR/bin/python',
    'cellpose': '/storeData/USER/data/01.CellBin/00.user/fanjinghong/home/anaconda3/envs/cellpose/bin/python',
    'cellpose3':'/storeData/USER/data/01.CellBin/00.user/fanjinghong/home/anaconda3/envs/cellpose3/bin/python',
    'deepcell': '/storeData/USER/data/01.CellBin/00.user/fanjinghong/home/anaconda3/envs/deepcell/bin/python',
    'sam': '/storeData/USER/data/01.CellBin/00.user/fanjinghong/home/anaconda3/envs/sam/bin/python',
    'lt': '/storeData/USER/data/01.CellBin/00.user/fanjinghong/home/anaconda3/envs/benchmark/bin/python',
    'stardist':'/storeData/USER/data/01.CellBin/00.user/fanjinghong/home/anaconda3/envs/stardist/bin/python',
}
__methods__ = ['MEDIAR','cellpose','cellpose3', 'sam', 'lt','v3','stardist']


__script__ = {
    'MEDIAR':os.path.join(work_path,'/storeData/USER/data/01.CellBin/00.user/fanjinghong/code/benchmark2/src/methods/MEDIAR/MEDIAR/iMEDIAR.py'),
    'cellpose': os.path.join(work_path, '/storeData/USER/data/01.CellBin/00.user/fanjinghong/code/benchmark2/src/methods/cellpose/icellpose.py'),
    'cellpose3':os.path.join(work_path,'/storeData/USER/data/01.CellBin/00.user/fanjinghong/code/benchmark2/src/methods/cellpose3/icellpose3.py'),
    'deepcell': os.path.join(work_path, '/storeData/USER/data/01.CellBin/00.user/fanjinghong/code/benchmark2/src/methods/deepcell/ideepcell2.py'),
    'sam': os.path.join(work_path, '/storeData/USER/data/01.CellBin/00.user/fanjinghong/code/benchmark2/segment/sam_main.py'),
    'lt': os.path.join(work_path, '/storeData/USER/data/01.CellBin/00.user/fanjinghong/code/benchmark2/src/methods/lt.py'),
    'stardist':os.path.join(work_path,'/storeData/USER/data/01.CellBin/00.user/fanjinghong/code/benchmark2/segment/stardist_main.py')
}
v3_model_ss = "/storeData/USER/data/01.CellBin/00.user/fanjinghong/code/benchmark2/segment/models/cellseg_bcdu_SHDI_221008_tf.onnx"
v3_model_he = "/storeData/USER/data/01.CellBin/00.user/fanjinghong/code/benchmark2/segment/models/cellseg_bcdu_H_231221_tf.onnx"

USAGE = 'cell_seg'
PROG_VERSION = 'v0.0.1'
def main():

    parser = argparse.ArgumentParser(description="you should add those parameter")
    parser.add_argument('-i', "--input", help="the input img path")
    parser.add_argument('-o', "--output", help="the output file")
    parser.add_argument("-m", "--method", nargs='+', help='/'.join(__methods__))
    parser.add_argument("-t", "--img_type", help="ss/he")
    parser.add_argument("-g", "--gpu", help="the gpu index", default="-1")


    args = parser.parse_args()
    image_path = args.input
    output_path = args.output
    methods = args.method
    gpu = args.gpu
    img_type = args.img_type
    is_gpu = True
    if os.path.isdir(image_path):
        print(f'get {len(os.listdir(image_path))} files')
    for m in methods: assert m in __methods__
    for m in methods:
        start = time.time()
        if m == 'v3':
            if is_gpu == True:
                gpu = '0'
            if str.lower(img_type) == "he":
                cmd = f'{__py__[m]} {__script__[m]} -i {image_path} -o {os.path.join(output_path, m)} -g {gpu} -p {v3_model_he} -t {img_type}'
            else:
                cmd = f'{__py__[m]} {__script__[m]} -i {image_path} -o {os.path.join(output_path, m)} -g {gpu} -p {v3_model_ss} -t {img_type}'
        else:
            cmd = '{} {} -i {} -o {} -g {} -t {}'.format(__py__[m], __script__[m], 
                                        image_path, os.path.join(output_path, m), is_gpu, img_type)
        os.system(cmd)
        t = time.time() - start
        print('{} ran for a total of {} s'.format(m, t))
        print('{} result saved to {}'.format(m, os.path.join(output_path, m))) 


if __name__ == '__main__':
    import sys

    main()
    sys.exit()
    