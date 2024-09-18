# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 11:14:43 2023

@author: wuyiwen
"""
import os
import datetime
import cv2
import numpy as np

from cellbin.contrib.cell_mask import Mask
from cellbin.utils import clog


class Fast(Mask):
    def __init__(self, mask, distance=10, process=8):
        """
        Fast class to generate corrected mask.
        Args:
            mask: cell mask
            distance: correction distance
            process: number of cores to use
        """
        super().__init__(mask.copy())
        self.processes = process
        self.distance = distance

    def getNeighborLabels8(self, label, x, y, width, height):
        lastLabel = None
        for xx in range(max(x - 1, 0), min(height - 1, x + 2), 1):
            for yy in range(max(y - 1, 0), min(width - 1, y + 2), 1):
                if xx == x and yy == y:
                    continue
                l = label[xx, yy]
                if l != 0:
                    if not lastLabel:
                        lastLabel = l
                    elif lastLabel != l:
                        return None
        return lastLabel

    def addNeighboursToQueue8(self, queued, queue, x, y, width, height):
        try:
            if queued[x * width + (y - 1)] == 0 and x >= 0 and x < height and y >= 0 and y < width:
                queued[x * width + (y - 1)] = 1
                queue.append((x, y - 1))
            if queued[(x - 1) * width + y] == 0 and x >= 0 and x < height and y >= 0 and y < width:
                queued[(x - 1) * width + y] = 1
                queue.append((x - 1, y))
            if queued[(x + 1) * width + y] == 0 and x >= 0 and x < height and y >= 0 and y < width:
                queued[(x + 1) * width + y] = 1
                queue.append((x + 1, y))
            if queued[x * width + (y + 1)] == 0 and x >= 0 and x < height and y >= 0 and y < width:
                queued[x * width + (y + 1)] = 1
                queue.append((x, y + 1))
            if queued[(x - 1) * width + (y - 1)] == 0 and x >= 0 and x < height and y >= 0 and y < width:
                queued[(x - 1) * width + (y - 1)] = 1
                queue.append((x - 1, y - 1))
            if queued[(x - 1) * width + (y + 1)] == 0 and x >= 0 and x < height and y >= 0 and y < width:
                queued[(x - 1) * width + (y + 1)] = 1
                queue.append((x - 1, y + 1))
            if queued[(x + 1) * width + (y - 1)] == 0 and x >= 0 and x < height and y >= 0 and y < width:
                queued[(x + 1) * width + (y - 1)] = 1
                queue.append((x + 1, y - 1))
            if queued[(x + 1) * width + (y + 1)] == 0 and x >= 0 and x < height and y >= 0 and y < width:
                queued[(x + 1) * width + (y + 1)] = 1
                queue.append((x + 1, y + 1))
        except:
            pass

    def crop_mask(self, mask):
        x, y = np.where(mask > 0)
        start_x, start_y, end_x, end_y = max(np.min(x) - 100, 0), max(np.min(y) - 100, 0), min(np.max(x) + 100,
                                                                                               mask.shape[0]), min(
            np.max(y) + 100, mask.shape[1])
        start = (start_x, start_y)
        end = (end_x, end_y)
        cropmask = mask[start_x:end_x, start_y:end_y]
        return start, end, cropmask

    def array_to_block(self, a, p, q, step=100):
        '''
        Divides array a into subarrays of size p-by-q
        p: block row size
        q: block column size
        '''
        m = a.shape[0]  # image row size
        n = a.shape[1]  # image column size
        # pad array with NaNs so it can be divided by p row-wise and by q column-wise
        bpr = ((m - 1) // p + 1)  # blocks per row
        bpc = ((n - 1) // q + 1)  # blocks per column
        M = p * bpr
        N = q * bpc
        A = np.nan * np.ones([M, N], dtype=np.half)
        A[:a.shape[0], :a.shape[1]] = a
        block_list = []
        previous_row = 0
        for row_block in range(bpr):
            previous_row = row_block * p
            previous_column = 0
            for column_block in range(bpc):
                previous_column = column_block * q
                if previous_row == 0:
                    if previous_column == 0:
                        block = A[previous_row:previous_row + p, previous_column:previous_column + q]
                    elif previous_column == (bpc - 1) * q:
                        block = A[previous_row:previous_row + p, previous_column - (step * column_block):]
                    else:
                        block = A[previous_row:previous_row + p,
                                previous_column - (step * column_block):previous_column - (step * column_block) + q]
                elif previous_row == (bpr - 1) * p:
                    if previous_column == 0:
                        block = A[previous_row - (step * row_block):, previous_column:previous_column + q]
                    elif previous_column == (bpc - 1) * q:
                        block = A[previous_row - (step * row_block):, previous_column - (step * column_block):]
                    else:
                        block = A[previous_row - (step * row_block):,
                                previous_column - (step * column_block):previous_column - (step * column_block) + q]
                else:
                    if previous_column == 0:
                        block = A[previous_row - (step * row_block):previous_row - (step * row_block) + p,
                                previous_column:previous_column + q]
                    elif previous_column == (bpc - 1) * q:
                        block = A[previous_row - (step * row_block):previous_row - (step * row_block) + p,
                                previous_column - (step * column_block):]
                    else:
                        block = A[previous_row - (step * row_block): previous_row - (step * row_block) + p,
                                previous_column - (step * column_block): previous_column - (step * column_block) + q]
                        # remove nan columns and nan rows
                nan_cols = np.all(np.isnan(block), axis=0)
                block = block[:, ~nan_cols]
                nan_rows = np.all(np.isnan(block), axis=1)
                block = block[~nan_rows, :]
                # append
                if block.size:
                    block_list.append(block.astype(np.uint8))
        return block_list, (bpr, bpc)

    def create_edm_label(self, mask):
        """

        Args:
            mask: cropped cell mask

        Returns:
            edm: distance map of input mask
            maskImg: connected components of input mask

        """
        from scipy import ndimage
        _, maskImg = cv2.connectedComponents(mask, connectivity=8)
        mask[mask > 0] = 255
        mask = cv2.bitwise_not(mask)
        edm = ndimage.distance_transform_edt(mask)
        edm[edm > 255] = 255
        edm = edm.astype(np.uint8)
        return edm, maskImg

    def process_queue(self, queued, queue, label, width, height):
        # print (f'start to iterate queue at {datetime.datetime.now()}', flush = True)
        while queue:
            x, y = queue.popleft()
            l = self.getNeighborLabels8(label, x, y, width, height)
            if not l:
                continue
            label[x, y] = l
            self.addNeighboursToQueue8(queued, queue, x, y, width, height)
        return label

    def correct(self, mask, dis, idx):
        from collections import deque

        edm, label = self.create_edm_label(mask)
        height, width = edm.shape
        queued = [0] * width * height
        queue = deque()
        # point = namedtuple('Points',['x','y','label'])
        for i in range(0, height, 1):
            for j in range(0, width, 1):
                val = edm[i, j].astype(int)
                if val > dis:
                    queued[i * width + j] = 1
                    continue
                l = label[i, j].astype(int)
                if l != 0:
                    queued[i * width + j] = 1
                    continue
                else:
                    if i > 0 and i < height - 1:
                        if label[i - 1, j] != 0 or label[i + 1, j] != 0:
                            queued[i * width + j] = 1
                            queue.append((i, j))
                    if j > 0 and j < width - 1:
                        if label[i, j - 1] != 0 or label[i, j + 1] != 0:
                            queued[i * width + j] = 1
                            queue.append((i, j))
        label = self.process_queue(queued, queue, label, width, height)
        label[label > 0] = 1
        label = label.astype(np.uint8)
        return label, idx

    def handle_error(self, error):
        clog.error(error)

    def merge_by_row(self, arr, loc, step=100):
        r, c = loc
        half_step = step // 2
        full_img = arr[0][:-half_step]
        for rr in range(1, r):
            if rr == r - 1:
                full_img = np.concatenate((full_img, arr[rr][half_step:]), axis=0)
            else:
                full_img = np.concatenate((full_img, arr[rr][half_step:-half_step]), axis=0)
        return full_img

    def merge_by_col(self, final_result, loc, step=100):
        r, c = loc
        row_list = []
        half_step = step // 2
        for rr in range(r):
            row_img = final_result[rr * c][:, :-half_step]
            for cc in range(1, c):
                if cc == c - 1:
                    row_img = np.concatenate((row_img, final_result[rr * c + cc][:, half_step:]), axis=1)
                else:
                    row_img = np.concatenate((row_img, final_result[rr * c + cc][:, half_step:-half_step]), axis=1)
            row_list.append(row_img)
        return row_list

    def process(self):
        clog.info(f'Fast Labeling starts at {datetime.datetime.now()}')
        import multiprocessing as mp
        clog.info(f"Fast Labeling using {self.processes} processes")
        pool = mp.Pool(processes=self.processes)
        start, end, cropmask = self.crop_mask(self.mask)
        masks, loc = self.array_to_block(cropmask, 2000, 2000, step=100)
        final_result = []
        processes = []
        for i, ma in enumerate(masks):
            result = pool.apply_async(self.correct, (ma, self.distance, i,), error_callback=self.handle_error)
            processes.append(result)
        pool.close()
        pool.join()
        for p in processes:
            final_result.append(p.get())
        final_result = sorted(final_result, key=lambda x: x[1])
        final_result = [arr for arr, i in final_result]

        row_list = self.merge_by_col(final_result, loc, step=100)
        final_img = self.merge_by_row(row_list, loc, step=100)

        # mask = np.zeros(shape,dtype=np.uint8)
        self.mask[start[0]:end[0], start[1]:end[1]] = final_img
        clog.info(f'Fast Labeling ends at {datetime.datetime.now()}\n')

    def get_mask_fast(self):
        """

        Returns: corrected mask, dtype is np.ndarray

        """
        return self.mask


if __name__ == '__main__':
    import tifffile as tifi
    mask = tifi.imread(r"D:\Cell_Bin\data\edm\SS200000135TL_D1_regist_8nei.tif")
    a = Fast(mask,10,8)

    a.process()
