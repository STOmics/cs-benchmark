# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 09:30:48 2023

@author: ywu28328
"""
import os
import cv2
import numpy as np


class Mask(object):
    def __init__(self, mask):
        self.mask = None
        self.set_mask(mask)

    def set_mask(self, mask):
        if mask is None: print('mask is None!')
        if isinstance(mask, np.ndarray):
            self.mask = mask
        else:
            print('Input mask is not numpy array!')

    def get_mask(self):
        return self.mask

    def est_para(self):
        """
        Estimate parameters

        Returns
        -------
        None.

        """
        _, maskImg = cv2.connectedComponents(self.mask, connectivity=8)
        cell_avg_area = np.count_nonzero(self.mask) / np.max(maskImg)
        if cell_avg_area >= 350:
            print(f'cell average size is {cell_avg_area}, d recommend 10')
        else:
            radius = int(np.sqrt(400 / np.pi) - np.sqrt(cell_avg_area / np.pi))
            print(f'd recommend at least {radius}')
        print(f'processes recommend set to {int(os.cpu_count() * 0.7)}')







