import os
import cv2
import numpy as np
import pandas as pd

from stio.matrix_loader import MatrixLoader  # input is file path
from cellbin.contrib.fast_correct import Fast
from cellbin.contrib.GMM_correct import GMM
from cellbin.modules import CellBinElement
from cellbin.utils import clog


class CellLabelling(CellBinElement):
    def __init__(self, mask, gene_file):
        self.mask = None
        self.fast_mask = None
        self.gem = gene_file
        self.bgef = None
        self.cgef = None
        self.exp_matrix = None
        self.process = 8

        self.set_mask(mask)
        # self.set_gem(gene_file)
        # self.set_matrix()

    def set_process(self, p):
        self.process = p

    def run_fast(self, distance=10):
        f = Fast(self.mask, distance, self.process)
        f.process()
        self.set_fast_mask(f.mask)
        self.set_gem(self.gem)
        self.set_matrix()
        return f.mask, self.exp_matrix

    def run_gmm(self, threshold=20, process=10):
        g = GMM(self.exp_matrix, threshold, process)
        g.cell_correct()
        self.exp_matrix = g.gmm_result
        return None, g.gmm_result

    def set_mask(self, mask):
        try:
            # self.mask = tifi.imread(mask)
            self.mask = mask
            self.mask[self.mask > 0] = 1
            self.mask = self.mask.astype(np.uint8)
        except:
            clog.error(f'unable to load cell mask {mask}!')

    def set_fast_mask(self, mask):
        try:
            self.fast_mask = mask
            self.fast_mask[self.fast_mask > 0] = 1
            self.fast_mask = self.fast_mask.astype(np.uint8)
        except:
            clog.error(f'unable to load cell mask {mask}!')

    def set_gem(self, gene_file, new_file='temp.gem'):
        # if gene_file.lower().endswith(('.bgef', '.gef')):
        #     from gefpy.bgef_reader_cy import BgefR
        #     self.bgef = BgefR(gene_file, 1, 8)
        #     ml = MatrixLoader(gene_file)
        #     ml.bgef2gem(bgef_file=gene_file, gem_file=new_file, binsize=1)
        #     self.gem = pd.read_csv(new_file, sep='\t', skiprows=6)
        # else:
        def row(gem_path):
            if gem_path.endswith('.gz'):
                import gzip
                with gzip.open(gem_path, 'rb') as f:
                    first_line = bytes.decode(f.readline())
                    if '#' in first_line:
                        rows = 6
                    else:
                        rows = 0
            else:
                with open(gem_path, 'rb') as f:
                    first_line = bytes.decode(f.readline())
                    if '#' in first_line:
                        rows = 6
                    else:
                        rows = 0
            return rows

        self.gem = pd.read_csv(gene_file, sep='\t', skiprows=row(gene_file))
        # else:
        #     self.gem = None

    def draw_ori_mask(self, output_path):
        self.mask[self.mask > 0] = 255
        cv2.imwrite(os.path.join(output_path, 'ori_mask.png'), self.mask)

    def draw_mask_comparison(self, corrected_mask, output_path='./'):
        img = np.zeros((*self.mask.shape, 3), dtype=np.uint8)
        self.mask[self.mask > 0] = 255
        corrected_mask[corrected_mask > 0] = 255
        img[:, :, 1] = self.mask
        img[:, :, 0] = corrected_mask
        cv2.imwrite(os.path.join(output_path, 'mask_comparison.png'), img)

    @staticmethod
    def draw_corrected_mask(corrected_mask, output_path='./'):
        corrected_mask[corrected_mask > 0] = 255
        return corrected_mask
        # cv2.imwrite(os.path.join(output_path, 'correct_mask.png'), corrected_mask)

    @staticmethod
    def create_cgef(cgef_file, bgef_file, mask_file, block_sizes=None):
        MatrixLoader.bgef2cgef(cgef_file, bgef_file, mask_file, block_sizes)

    def get_gem(self):
        return self.gem

    def set_matrix(self):
        if self.gem is None:
            clog.error('Gene expression matrix is None!')
        else:
            self.creat_cell_gxp()

    def get_matrix(self):
        return self.exp_matrix

    def creat_cell_gxp(self):
        clog.info("Loading mask file...")
        _, maskImg = cv2.connectedComponents(self.fast_mask, connectivity=8)
        clog.info("Reading data..")
        self.gem['x'] -= self.gem['x'].min()
        self.gem['y'] -= self.gem['y'].min()

        assert "MIDCount" in self.gem.columns
        self.gem['CellID'] = maskImg[self.gem['y'], self.gem['x']]
        self.gem = self.gem[self.gem['CellID'] > 0]
        self.gem['CellID'] = self.gem['CellID'] - 1
        self.exp_matrix = self.gem

    def get_gem_arr(self):
        if self.gem:
            gem_img = np.zeros([self.gem['y'].max() + 1, self.gem['x'].max() + 1], dtype=np.uint8)
            gem_img[self.gem['y'], self.gem['x']] = 255
            return gem_img
        else:
            clog.error('Gene expression matrix does not exist...')

    def get_exp_arr(self):
        if self.exp_matrix:
            exp_img = np.zeros([self.exp_matrix['y'].max() + 1, self.exp_matrix['x'].max() + 1], dtype=np.uint8)
            exp_img[self.exp_matrix['y'], self.exp_matrix['x']] = 255
            return exp_img
        else:
            clog.error('Gene expression matrix does not exist...')

    def compare_cell_size(self):
        _, maskImg_1 = cv2.connectedComponents(self.mask, connectivity=8)
        _, maskImg_2 = cv2.connectedComponents(self.fast_mask, connectivity=8)
        ori_dict = dict(zip(*np.unique(maskImg_1, return_counts=True)))
        new_dict = dict(zip(*np.unique(maskImg_2, return_counts=True)))
        merged = {k: (ori_dict[k], new_dict[k]) for k in ori_dict}
        return merged

    def cell_mid(self):
        gem = self.exp_matrix
        cell_gem = gem[self.mask[gem['y'], gem['x']] > 0]
        fast_gem = gem[self.fast_mask[gem['y'], gem['x']] > 0]
        cell_mid = cell_gem['MIDCount'].sum()
        fast_mid = fast_gem['MIDCount'].sum()
        return round(cell_mid / fast_mid, 3), round(cell_mid / len(gem), 3), round(fast_mid / len(gem), 3)

    def gene_number(self, type='fast', bin=20):
        if type == 'fast':
            mask = self.fast_mask
        else:
            mask = self.mask
        _, maskImg = cv2.connectedComponents(mask, connectivity=8)
        bin_img = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.uint16)
        group = self.exp_matrix.groupby(['x', 'y']).agg(np.sum)
        idx = np.array(group.index.tolist())
        bin_img[idx[:, 1], idx[:, 0]] = group['MIDCount'].tolist()
        # total = bin_img[res['y'],res['x']].sum()
        total = bin_img.sum()
        y, x = np.where(mask > 0)
        bin_size = (y.max() - y.min()) * (x.max() - x.min()) / (bin ** 2)
        return int(total / bin_size), int(total / maskImg.max())

    def gene_type(self, type='fast', bin=20):
        if type == 'fast':
            mask = self.fast_mask
        else:
            mask = self.mask
        _, maskImg = cv2.connectedComponents(mask, connectivity=8)
        bin_img = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.uint16)
        group = self.exp_matrix.groupby(['x', 'y'])['geneID'].unique()
        genid_list = [len(i) for i in group.values.tolist()]
        idx = np.array(group.index.tolist())
        bin_img[idx[:, 1], idx[:, 0]] = genid_list
        total = bin_img.sum()
        y, x = np.where(mask > 0)
        bin_size = (y.max() - y.min()) * (x.max() - x.min()) / (bin ** 2)
        return int(total / bin_size), int(total / maskImg.max())

    def get_rate(self):
        cell_gem = self.exp_matrix[self.mask[self.exp_matrix['y'], self.exp_matrix['x']] > 0]
        fast_gem = self.exp_matrix[self.fast_mask[self.exp_matrix['y'], self.exp_matrix['x']] > 0]
        gem_amount = len(self.exp_matrix)
        return [round(len(cell_gem) / gem_amount, 2), round(len(fast_gem) / gem_amount, 2)]

    def qc(self):
        bin_size = (self.exp_matrix['y'].max() - self.exp_matrix['y'].min()) * (
                    self.exp_matrix['x'].max() - self.exp_matrix['x'].min())
        bin200_size = bin_size / (200 ** 2)
        bin50_size = bin_size / (50 ** 2)
        gene_amount = np.sum(self.exp_matrix.groupby(['x', 'y'])['geneID'].size().tolist())
        gene_type = len(self.exp_matrix['geneID'])

        if (gene_type / bin50_size) < 500 or (gene_amount / bin200_size) < 5000:
            clog.info('---QC failed!---')
        else:
            clog.info('---QC Success!---')

    def get_comparison(self):
        cell_size_dict = self.compare_cell_size()  # dict - {label:(before, after)}
        mid_list = self.cell_mid()  # List - [before/after MID, before/gem, after/gem]
        gene_num_ori = self.gene_number('ori', 20)  # int, int - bin20 gene number, cellbin gene number
        gene_num_fast = self.gene_number('fast', 20)  # int, int - bin20 gene number, cellbin gene number
        gene_type_ori = self.gene_type('ori', 50)  # int, int - bin50 gene type, cellbin gene number
        gene_type_fast = self.gene_type('fast', 50)  # int, int - bin50 gene type, cellbin gene number
        return cell_size_dict, mid_list, [gene_num_ori, gene_num_fast, gene_type_ori, gene_type_fast]


if __name__ == '__main__':
    # a = MatrixLoader(r"D:\git_libs\cellbin\test\C01333A3.gem.gz")
    # a.gem2bgef(gem_file = r"D:\git_libs\cellbin\test\C01333A3.gem.gz", bgef_file=r"D:\git_libs\cellbin\test\C01333A3.bgef", binsize=[1],region=None)
    # a.bgef2gem(bgef_file=r"D:\git_libs\cellbin\test\C01333A3.bgef", gem_file = r"D:\git_libs\cellbin\test\C01333A3_test.gem.gz", binsize=1)
    # from gefpy.bgef_reader_cy import BgefR
    # bgef_reader = BgefR(r"D:\git_libs\cellbin\test\C01333A3.bgef",1,8)
    # explist = bgef_reader.get_expression()
    # gene_data = bgef_reader.get_gene_data()
    # print(explist)
    # print(gene_data)
    # print(len(gene_data[1]))
    import tifffile as tifi

    mask = tifi.imread(r"D:\yiwen_correct_data\SS200000406TL_D1_mask.tif")
    lif = CellLabelling(mask,
                        r"D:\yiwen_correct_data\SS200000406TL_D1.tissue.gem.gz")
    # fast = Fast(lif.mask)
    # fast.process()
    # fast_mask = fast.get_mask_fast()
    fast_mask, matx = lif.run_fast()
    output_path = r'D:\yiwen_correct_data'
    cv2.imwrite(os.path.join(output_path, 'correct_mask.png'), fast_mask)
    # matx = lif.get_matrix()
    matx.to_csv(os.path.join(output_path, 'correct_mask.gem'), sep='\t', index=False)
    lif.get_rate()
    lif.qc()
    # lif.draw_corrected_mask(fast_mask)
    # lif.draw_mask_comparison(fast_mask)
