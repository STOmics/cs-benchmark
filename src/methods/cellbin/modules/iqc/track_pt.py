import numpy as np
import tqdm
import multiprocessing as mp
from collections import Counter

from cellbin.image import Image
from cellbin.dnn.pdetect.pt_detector import PointDetector
from cellbin.utils import clog


def no_enhance(img_obj):
    return img_obj.image


def divergence(shape):
    rows, cols = shape
    diver_origin = np.zeros((rows, cols, 2))
    rows_arr = np.arange(1, rows + 1)
    cols_arr = np.arange(1, cols + 1)
    for col in range(cols):
        for row in range(rows):
            if row < rows // 2:
                r_value = -rows_arr[:rows // 2 - row].sum()
            else:
                if rows // 2 == 0:
                    r_value = rows_arr[:row - rows // 2].sum()
                else:
                    r_value = rows_arr[:row - (rows - 1) // 2].sum()
            if col < cols // 2:
                c_value = -cols_arr[:cols // 2 - col].sum()
            else:
                if cols // 2 == 0:
                    c_value = cols_arr[:col - cols // 2].sum()
                else:
                    c_value = cols_arr[:col - (cols - 1) // 2].sum()
            diver_origin[row, col] = np.array([r_value, c_value])

    diver_map = np.abs(diver_origin[:, :, 0]) + np.abs(diver_origin[:, :, 1])
    return diver_map


class TrackPointQC(object):
    def __init__(self, th=5, th2=20, good_thresh=5):
        """
        This class is used to do track point detection (using deep learning object detection algo).
        Will also do track point quality control based on the distribution among fovs, counts of
        detections and the confidence score of prediction


        Track Eval:

        is used to evaluate the track points result from object detection algo.
        Considering:
            1. track pts count of each fov
            2. track pts dets confidence score of each fov
            3. fov position. further better


        Args:
            model_path (): detect weight path
            enhance_func (): enhance method used to preprocess image
            detect_channel ():
                the channel you choose to do track detect if input is multichannel
                default is -1 -> just use the original image
            th (): [th, th2) -> track_point_score = 1
            th2 (): [th2, inf) -> track_point_score = 2

        Result:
            self.track_result: track point detection result by deep learning method
                - {'row_col': [[pt_x, pt_y, conf], angle]}
                - no dets: {}

            self.fov_mask: score for each fov
                - numpy 2d array (success)
                - np.array([]) (fail)

            self.fovs_score (dict): score for all fovs in order (fov_score > 0)
                - {'row_col': fov_score}
                - {} if no fov_score is greater than 0

            self.score: track_pts eval score, score interval is [0, 1], higher -> better
                - int
                - Recommended score threshold for dapi is 0.4?

        """
        self.detect_channel = -1
        self.th = th
        self.th2 = th2
        self.good_thresh = good_thresh
        self.ci = PointDetector()
        self.track_result = dict()
        self.score = 0
        self.fov_mask = np.array([])
        self.fovs_order = []
        self.good_fov_count = 0

        self.process = 5  # TODO: change this
        self.most_freq_angle = None

    def set_detect_channel(self, c):
        self.detect_channel = c

    def set_enhance_func(self, f):
        self.ci.set_func(f)

    def set_multi_process(self, n):
        self.process = n

    def set_threshold(self, th, th2, good_thresh):
        self.th = th
        self.th2 = th2
        self.good_thresh = good_thresh

    def img_read(self, img_path, buffer):
        img_obj = Image()
        img_obj.read(img_path, buffer)
        if img_obj.ndim == 3 and self.detect_channel != -1:
            img_obj.get_channel(ch=self.detect_channel)  # 2d array
        return img_obj

    def track_detect(self, img_dict: dict, buffer=None):
        """
        This function will do track detection using object detection (deep learning) algo.
        self.track_result will be empty if no detections

        Args:
            img_dict (dict): {'row_col': img_path}

        """
        self.track_result = dict()
        if self.process <= 1:
            for key, img_path in tqdm.tqdm(img_dict.items(), file=clog.tqdm_out, mininterval=10, desc='Track points detect'):
                img_obj = self.img_read(img_path, buffer)
                cp, angle = self.ci.predict(img_obj)
                if angle is None or len(cp) == 0:
                    continue
                self.track_result[key] = [cp, angle]
        else:
            processes = []
            pool = mp.Pool(processes=self.process)
            clog.info(f"Track point detection using {self.process} processes")
            for key, img_path in tqdm.tqdm(img_dict.items(),  file=clog.tqdm_out, mininterval=10, desc='Track points detect'):
                img_obj = self.img_read(img_path, buffer)
                sub_process = pool.apply_async(self.ci.predict, args=(img_obj,))
                processes.append([key, sub_process])
            pool.close()
            pool.join()
            for key, p in processes:
                cp, angle = p.get()
                if angle is None or len(cp) == 0:
                    continue
                self.track_result[key] = [cp, angle]

    def track_eval(self, ):
        """
        This func will evaluate track cross quality for fovs

        Returns:
            self.score: fov track cross score
            self.fov_mask: 2d array, contain score for each fov
            self.fovs_order: rank all fovs based on score

        """
        clog.info(f"Track eval using threshold: 0 ~ {self.th} = 1, "
                  f"{self.th} ~ {self.th2} = 2, "
                  f"good fov thresh = {self.good_thresh}")
        self.score = 0
        self.fov_mask = np.array([])
        self.fovs_order = []
        self.good_fov_count = 0

        if len(self.track_result) == 0:
            return

        max_row, max_col = -1, -1
        for key in self.track_result.keys():
            splits = key.split('_')
            row, col = int(splits[0]), int(splits[1])
            max_row = max(row, max_row)
            max_col = max(col, max_col)

        pt_score_1 = 1
        pt_score_2 = 2
        # pt_count_mask = np.zeros((max_row + 1, max_col + 1))  # val: count of cp * mean(conf)
        conf_mask = np.zeros((max_row + 1, max_col + 1))  # no dets: 0
        val_pt_mask = np.zeros_like(conf_mask)
        max_pt_mask = np.ones_like(conf_mask) * pt_score_2
        fovs_name = np.empty_like(conf_mask, dtype='object')
        good_fov = 0
        all_angles = []
        for key, val in self.track_result.items():
            splits = key.split('_')
            row, col = int(splits[0]), int(splits[1])
            cps, angle = val
            if len(cps) >= 5:
                good_fov += 1
                all_angles.append(angle)
            cps_arr = np.array(cps)
            cur_counts = len(cps_arr)
            conf_mean = cps_arr.mean(axis=0)[-1]
            # pt_count_mask[row, col] = cur_counts
            if self.th <= cur_counts < self.th2:
                val_pt_mask[row, col] = pt_score_1
            elif cur_counts >= self.th2:
                val_pt_mask[row, col] = pt_score_2
            conf_mask[row, col] = conf_mean
            fovs_name[row, col] = key
        if len(all_angles) != 0:
            occurence_count = Counter(all_angles)
            most_common_one = occurence_count.most_common(1)[0]
            self.most_freq_angle = most_common_one
        diver_map = divergence(val_pt_mask.shape)
        # template_mask = diver_map * max_pt_mask
        val_pt_mask_norm = val_pt_mask / max_pt_mask
        result_mask = diver_map * val_pt_mask_norm * conf_mask
        score = result_mask.sum() / diver_map.sum()

        self.score = score
        self.fov_mask = result_mask
        self.good_fov_count = good_fov

        fovs_score = {}
        for row, col in np.ndindex(result_mask.shape):
            cur_score = result_mask[row, col]
            if cur_score > 0:
                cur_name = fovs_name[row, col]
                fovs_score[cur_name] = cur_score

        if len(fovs_score) != 0:
            self.fovs_order = [k for k, v in sorted(fovs_score.items(), key=lambda item: item[1], reverse=True)]


if __name__ == '__main__':
    import pickle

    result = r"D:\PycharmProjects\scripts\qc_scripts\saved_dictionary.pkl"
    with open(result, 'rb') as f:
        loaded_dict = pickle.load(f)
    track_qc = TrackPointQC(None, "")
    track_qc.track_result = loaded_dict
    track_qc.track_eval()
