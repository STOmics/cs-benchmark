import os
import time
import subprocess
import argparse

work_path = os.path.abspath('.')
# work_path = '/data/work/benchmark/benchmark'
__py__ = {
    'MEDIAR':'anaconda3/envs/MEDIAR/bin/python',
    'cellpose': 'anaconda3/envs/cellpose/bin/python',
    'cellpose3':'anaconda3/envs/cellpose3/bin/python',
    'deepcell': 'anaconda3/envs/deepcell/bin/python',
    'sam': 'anaconda3/envs/sam/bin/python',
    'stardist':'home/anaconda3/envs/stardist/bin/python',
}
__methods__ = ['MEDIAR','cellpose','cellpose3', 'sam','stardist','deepcell']


__script__ = {
    'MEDIAR':os.path.join(work_path,'src/methods/MEDIAR/MEDIAR/iMEDIAR.py'),
    'cellpose': os.path.join(work_path, 'src/methods/cellpose/icellpose.py'),
    'cellpose3':os.path.join(work_path,'src/methods/cellpose3/icellpose3.py'),
    'deepcell': os.path.join(work_path, 'src/methods/deepcell/ideepcell2.py'),
    'sam': os.path.join(work_path, 'src/methods/sam_main.py'),
    'stardist':os.path.join(work_path,'src/methods/stardist_main.py')
}

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
    is_gpu = args.gpu
    img_type = args.img_type
    
    if os.path.isdir(image_path):
        print(f'get {len(os.listdir(image_path))} files')
    for m in methods: assert m in __methods__
    for m in methods:
        start = time.time()
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
    