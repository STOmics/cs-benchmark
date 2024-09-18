import os
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2,40).__str__()
import cv2
from scipy import ndimage as ndi
from skimage import (
    color, feature, filters, measure, segmentation, io
)
import numpy as np
import os

from cellbin.utils import clog
import sys
import argparse
from utils import cell_dataset, auto_make_dir
import tifffile


class Segmentation:

    def __init__(self):
        self.img_path = None

        self.raw_img = None
        self.img = None
        self.mask = None
        self.label = None
        self.is_HE = False
        self.roi = None  # [x0, y0, w, h]
        self.tag = None

    def load(self, img_path, mRNA_path, signal_pbar=None):
        self.mRNA_path = mRNA_path
        self.img_path = img_path
        # self.raw_img = cv2.imread(self.img_path, cv2.IMREAD_GRAYSCALE)
        self.raw_img = cv2.imread(self.img_path, -1) 
        if self.roi: 
            x0, y0, w, h = self.roi
            self.raw_img = self.raw_img[y0: y0 + h, x0: x0 + w]
        # clog.info('Load stained image form {}'.format(self.img_path))

    def pre_process(self,
                    threshold='auto',
                    clipLimit = 5, 
                    tileGridSize = (7, 7)
                    ):
        clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
        # clog.info('Pre process use Adaptive Histogram Euqalization, clipLimit={}, tileGridSize={}.'.format(clipLimit, tileGridSize))
        if self.raw_img.ndim == 3:
            raw_img_ = cv2.cvtColor(self.raw_img.copy(), cv2.COLOR_BGR2GRAY)
        else:
            raw_img_ = self.raw_img.copy()
        if self.is_HE: raw_img_ = 255 - raw_img_
            
        raw_img_ = clahe.apply(raw_img_)
        if threshold == 'auto': threshold, _ = cv2.threshold(raw_img_.copy(), 0, 255, cv2.THRESH_OTSU)
        # clog.info('OTSU segmentation processing completed, Used Threshold: {}'.format(threshold))
            
        if threshold > 0: 
            _, self.img = cv2.threshold(raw_img_.copy(), threshold, 255, cv2.THRESH_TOZERO)
            # clog.info('TOZERO segmentation processing completed.')
        else: self.img = raw_img_.copy()
        # clog.info('End of preprocessing')

    def watershed(self,
                  block_size=41,
                  offset=0.003,
                  min_distance=15,
                  expand_distance=0,
                  verbose=True,
                  signal_pbar=None
                  ):
        img = self.img.copy()
        threshold = filters.threshold_local(img, block_size=block_size, offset=offset)
        distance = ndi.distance_transform_edt(img > threshold)

        local_max_coords = feature.peak_local_max(distance, min_distance=min_distance)
        local_max_mask = np.zeros(distance.shape, dtype=bool)
        local_max_mask[tuple(local_max_coords.T)] = True
        markers = measure.label(local_max_mask)

        self.mask = segmentation.watershed(-distance, markers, mask=img)
        if expand_distance > 0: self.mask = segmentation.expand_labels(self.mask, distance=expand_distance)
        t = self.instance2semantics(self.mask)
        return t

    def save_scGEM(self,
                   save_path,
                   verbose=True,
                   signal_pbar=None,
                   minus_min=True, 
                   ):
        import pandas as pd

        try: data = pd.read_csv(self.mRNA_path, sep='\t', comment="#")
        except: pass

        seg_cell_coor = []
        min_x = data['x'].min() if minus_min else 0
        min_y = data['y'].min() if minus_min else 0
        
        for i in range(self.mask.shape[0]):
            for j in range(self.mask.shape[1]):
                c = self.mask[i, j]
                if c:
                    seg_cell_coor.append([i + min_x, j + min_y, c])
        if signal_pbar:
            signal_pbar.emit(70)
        seg_cell_coor = pd.DataFrame(seg_cell_coor, columns=['x', 'y', 'cell'])
        cell_data = pd.merge(data, seg_cell_coor, how='left', on=['x', 'y'])
        cell_data = cell_data.dropna()
        cell_data['cell'] = cell_data['cell'].astype(int)
        # name = os.path.basename(self.mRNA_path)
        # name = os.path.splitext(name)[0]
        gem_fn = os.path.join(save_path, '{}_scgem.csv.gz'.format(self.tag))
        cell_data.to_csv(gem_fn, index=False, sep='\t', compression="gzip")
        # coor_fn = os.path.join(save_path, f'{name}.ssDNA_coor.csv')
        # seg_cell_coor.to_csv(os.path.join(save_path, f'{args.i}.ssDNA_coor.csv'), index=False)
        # clog.info(f'single-cell GEM save path: {gem_fn}')
            
    @staticmethod
    def get_color(img):
        m1 = int(255 - img[:,:,0].mean())
        m2 = int(255 - img[:,:,1].mean())
        m3 = 0
        return (m1, m2, m3)
            
    @staticmethod
    def instance2semantics(ins):
        h, w = ins.shape[:2]
        tmp0 = ins[1:, 1:] - ins[:h-1, :w-1]
        ind0 = np.where(tmp0 != 0)

        tmp1 = ins[1:, :w-1] - ins[:h-1, 1:]
        ind1 = np.where(tmp1 != 0)
        ins[ind1] = 0
        ins[ind0] = 0
        ins[np.where(ins > 0)] = 1
        return np.array(ins, dtype=np.uint8)

    def __repr__(self):
        t = f"ssDNA Image Segmentation Object\n" \
            f"Raw   Image Path: {self.img_path}\n" \
            f"GEM   Data  Path: {self.mRNA_path}"
        return t


def lt_method(para, args):
    # set up the object
    sobj = Segmentation()
    sobj.is_HE = False
    sobj.roi = None

    bs = 41
    ot =  0.003
    md = 15
    ed = 2

    output_path = para.output
    if os.path.isdir(para.image_path):
        imgs = cell_dataset(para.image_path, ['.tif', '.jpg', '.png'])
    else: imgs = [para.image_path]

    # load the registered ssDNA stained image and the Gem file
    import tqdm
    for i in tqdm.tqdm(imgs, 'Local Threshold'):
        if 'HE' in i: sobj.is_HE = True
        else: sobj.is_HE = False
        sobj.load(img_path=i, mRNA_path=None)
        sobj.pre_process(threshold='auto')
        # Performs watershed segmentation on the image
        mask = sobj.watershed(block_size=bs, offset=ot, min_distance=md, expand_distance=ed)
        
        out_file = auto_make_dir(i, src=para.image_path, output=output_path)
        tifffile.imwrite(out_file, mask, compression='zlib')


USAGE = 'Local Threshold'
PROG_VERSION = 'v0.0.1'


def main():
    # test_stitch_entry('D:\\DATA\\stitchingv2_test\\motic\\result\\demo\\scope_info.json')
    arg_parser = argparse.ArgumentParser(usage=USAGE)
    arg_parser.add_argument("--version", action="version", version=PROG_VERSION)
    arg_parser.add_argument("-o", "--output", action="store", dest="output",
                            type=str, default=None, help="Save path of stitch result files.")
    arg_parser.add_argument("-i", "--image_path", action="store", dest="image_path",
                            type=str, default=None, help="FOV images storage location.")
    arg_parser.add_argument("-g", "--is_gpu", action="store", dest="is_gpu",
                            type=bool, default=False, help="Use GPU or not.")

    arg_parser.set_defaults(func=lt_method)
    (para, args) = arg_parser.parse_known_args()
    print(para, args)
    para.func(para, args)


if __name__ == '__main__':
    return_code = main()
    sys.exit(return_code)