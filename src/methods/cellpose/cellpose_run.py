import sys

sys.path.append('.cellpose')

from scipy.ndimage import distance_transform_edt
import numpy as np
import os
from math import ceil
import patchify
import cv2
#import matplotlib.pyplot as plt
from cellpose import models, utils


class CellSegmentation:
    def __init__(self, open_path, save_path, photo_size, photo_step, dmin, dmax, step):
        self.open_path = open_path
        self.save_path = save_path
        self.photo_size = photo_size
        self.photo_step = photo_step
        self.dmin = dmin
        self.dmax = dmax
        self.step = step

    def _process_image(self, img_data):
        overlap = self.photo_size - self.photo_step
        if (overlap % 2) == 1:
            overlap = overlap + 1
        act_step = ceil(overlap / 2)
        im = cv2.imread(self.open_path)
        dir_image1 = self.open_path.split('/')[-1].strip('.tif')
        image = np.array(im)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        res_image = np.pad(gray_image, ((act_step, act_step), (act_step, act_step)), 'constant')
        a = res_image.shape[0]
        b = res_image.shape[1]
        res_a = ceil((a - self.photo_size) / self.photo_step) * self.photo_step + self.photo_size
        res_b = ceil((b - self.photo_size) / self.photo_step) * self.photo_step + self.photo_size
        padding_rows = res_a - a
        padding_cols = res_b - b
        regray_image = np.pad(res_image, ((0, padding_rows), (0, padding_cols)), mode='constant')
        patches = patchify.patchify(regray_image, (self.photo_size, self.photo_size), step=self.photo_step)
        wid = patches.shape[0]
        high = patches.shape[1]
        model = models.Cellpose(gpu=True, model_type='cyto2')
        a_patches = np.full((wid, high, (self.photo_step), (self.photo_step)), 255)

        for i in range(wid):
            for j in range(high):
                img_data = patches[i, j, :, :]
                num0min = wid * high * 800000000000000
                for k in range(self.dmin, self.dmax, self.step):

                    masks, flows, styles, diams = model.eval(img_data, diameter=k, channels=[0, 0], flow_threshold=0.9)
                    num0 = np.sum(masks == 0)

                    if num0 < num0min:
                        num0min = num0
                        outlines = utils.masks_to_outlines(masks)
                        outlines = (outlines == True).astype(int) * 255

                        try:
                            a_patches[i, j, :, :] = outlines[act_step:(self.photo_step + act_step),
                                                    act_step:(self.photo_step + act_step)]
                            output = masks.copy()
                        except:
                            a_patches[i, j, :, :] = output[act_step:(self.photo_step + act_step),
                                                    act_step:(self.photo_step + act_step)]

        patch_nor = patchify.unpatchify(a_patches, ((wid) * (self.photo_step), (high) * (self.photo_step)))
        nor_imgdata = np.array(patch_nor)
        cropped_1 = nor_imgdata[0:gray_image.shape[0], 0:gray_image.shape[1]]
        cropped_1 = np.uint8(cropped_1)
        return cropped_1

    def _post_image(self, process_image):
        contour_thickness = 0
        contour_coords = np.argwhere(process_image == 255)
        distance_transform = distance_transform_edt(process_image == 0)
        expanded_image = np.zeros_like(process_image)
        for y, x in contour_coords:
            mask = distance_transform[y, x] <= contour_thickness
            expanded_image[y - contour_thickness:y + contour_thickness + 1,
            x - contour_thickness:x + contour_thickness + 1] = mask * 255
        contours, _ = cv2.findContours(expanded_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        height, width = process_image.shape
        black_background = np.zeros((height, width), dtype=np.uint8)
        for contour in contours:
            area = cv2.contourArea(contour)
            if 20 < area < 10000:
                cv2.drawContours(black_background, [contour], -1, 255, thickness=cv2.FILLED)
        black_background = np.uint8(black_background)
        return black_background, expanded_image

    def _merger_image(self, merger_image1, merger_image2):
        merger_image1[merger_image2 == 255] = 0
        return merger_image1

    def segment_cells(self):
        inverted_image = self._process_image(self.open_path)
        post_image, expanded_image = self._post_image(inverted_image)
        result_image = self._merger_image(post_image, expanded_image)
        cv2.imwrite(self.save_path, result_image)
        # contours, _ = cv2.findContours(inverted_image, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
        # h, w = result_image.shape[:2]
        # outline = np.zeros((h, w, 3), dtype=np.uint8)
        # img = cv2.imread(self.open_path)
        # if img.ndim == 2:
        #     show_r = cv2.cvtColor(img.copy(), cv2.COLOR_GRAY2BGR)
        # else:
        #     show_r = img.copy()
        # cv2.drawContours(show_r, contours=contours, contourIdx=-1, color=(255, 0, 0), thickness=1, lineType=cv2.LINE_AA)
        # plt.figure(figsize=(12, 6))
        # plt.subplot(1, 2, 1)
        # plt.imshow(show_r)
        # plt.title('Cell Segmentation, number of cells:{}'.format(len(contours)))
        # areas = [cv2.contourArea(c) for c in contours]
        # plt.subplot(1, 2, 2)
        # plt.hist(areas, bins=len(areas) // 20, range=(0, len(areas)), color='skyblue', alpha=0.8)
        # plt.title('Cell area')
        # plt.xlabel('Value (pix)')
        # plt.ylabel('Frequency')
        # plt.show()


if __name__ == '__main__':
    ## **Run data**

    Inference = input("Inference for FB or mIf:(please input FB or mIf)\n")

    ## - Scene 1: Inference for FB
    if Inference == 'FB':
        open_path = input("输入图片打开路径：\n")
        save_path = input("输入图片保存路径：\n")

        for foldName, subfolders, filenames in os.walk(open_path):  # 用os.walk方法取得path路径下的文件夹路径，子文件夹名，所有文件名
            total_image = 0
            now = 0
            for filename in filenames:  # 计算图片文件总数
                if filename.count('.tif') and not filename.count('_mask.tif'):
                    total_image += 1
            for filename in filenames:
                if filename.count('.tif') and not filename.count('_mask.tif'):
                    now += 1
                    # open_path = "./339_img.tif"
                    # save_path = "./339_img_mask.tif"
                    image_path = open_path + '/' + filename
                    new_filename = filename[0:-4] + '_mask.tif'  # 拼接文件名
                    image_save_path = save_path + '/' + new_filename
                    print('分割图片中：(' + str(now) + '/' + str(total_image) + ')\n' + image_path)
                    if os.path.exists(image_save_path):
                        print('文件已存在')
                        break
                    # 细胞分割参数
                    photo_size = 512
                    photo_step = 512
                    dmin = 15
                    dmax = 95
                    step = 10
                    cell_segmenter = CellSegmentation(image_path, image_save_path, photo_size, photo_step, dmin, dmax,
                                                      step)
                    cell_segmenter.segment_cells()

    ## Scene 2: Inference for mIf
    elif Inference == 'mIf':
        open_path = input("输入图片打开路径：\n")
        save_path = input("输入图片保存路径：\n")
        for foldName, subfolders, filenames in os.walk(open_path):  # 用os.walk方法取得path路径下的文件夹路径，子文件夹名，所有文件名
            total_image = 0
            now = 0
            for filename in filenames:
                if filename.count('.tif') and not filename.count('_mask.tif'):
                    total_image += 1
            for filename in filenames:
                if filename.count('.tif') and not filename.count('_mask.tif'):
                    now += 1
                    # open_path = "./339_img.tif"
                    # save_path = "./339_img_mask.tif"
                    image_path = open_path + '/' + filename
                    new_filename = filename[0:-4] + '_mask.tif'
                    image_save_path = save_path + '/' + new_filename
                    print('分割图片中：(' + str(now) + '/' + str(total_image) + ')\n' + image_path)
                    if os.path.exists(image_save_path):
                        print('文件已存在')
                        break
                    photo_size = 512
                    photo_step = 512
                    dmin = 30
                    dmax = 40
                    step = 5

                    cell_segmenter = CellSegmentation(image_path, image_save_path, photo_size, photo_step, dmin, dmax,
                                                      step)
                    cell_segmenter.segment_cells()
