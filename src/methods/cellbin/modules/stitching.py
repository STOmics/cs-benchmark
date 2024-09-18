import numpy as np
import os
import time

from cellbin.modules import CellBinElement
from cellbin.utils import clog

from cellbin.image.wsi_stitch import StitchingWSI
from cellbin.image import Image

from cellbin.contrib import FOVAligner
from cellbin.contrib import GlobalLocation
from cellbin.contrib import TemplateReference
from cellbin.contrib import TemplateReferenceV2


class Stitching(CellBinElement):
    """
    拼接模块类 包含拼接坐标计算及模板推导
    """
    def __init__(self, is_stitched=False):
        """
        Args:
            is_stitched: True | False 用于判断是否是拼接大图

        Returns:

        """
        super(Stitching, self).__init__()
        self.fov_location = None
        self.fov_x_jitter = None
        self.fov_y_jitter = None
        self.rows = self.cols = None
        self._fov_height = self._fov_width = self._fov_channel = None
        self._fov_dtype = None
        self.height = self.width = None
        self._overlap = 0.1

        self.__jitter_diff = None
        self.__jitter_x_diff = None
        self.__jitter_y_diff = None
        self.__offset_diff = None
        self.__template_max_value = -1
        self.__template_mean_value = -1
        self.__template_std_value = -1
        self.__template_qc_conf = -1
        self.__template_re_conf = -1
        self.__template_global_diff = -1  # @lizepeng update on 2023/05/17
        self.__image = None
        self._set_location_flag = False
        self._is_stitched = is_stitched
        self._template = None

        # 2023/08/28新增 用于开发对接track线模板FOV信息
        self._fov_index = None
        self._fov_best_point = None

    def _init_parm(self, src_image: dict):
        test_image_path = list(src_image.values())[0]

        if isinstance(test_image_path, str) or \
                (isinstance(test_image_path, np.ndarray) and len(test_image_path) != 4):
            img = Image()
            img.read(test_image_path)

            self._fov_height = img.height
            self._fov_width = img.width
            self._fov_channel = img.channel
            self._fov_dtype = img.dtype
        else:
            self._fov_height = test_image_path[1]
            self._fov_width = test_image_path[3]

    def set_size(self, rows, cols):
        """

        Args:
            rows:行数
            cols:列数

        Returns:

        """
        try:
            rows = int(rows)
        except:
            clog.info("Rows type error(expect int).")

        try:
            cols = int(cols)
        except:
            clog.info("Cols type error(expect int).")

        self.rows = rows
        self.cols = cols

    def set_global_location(self, loc):
        """

        Args:
            loc:拼接坐标信息

        Returns:

        """
        assert type(loc) == np.ndarray, "Location type error."
        self.fov_location = loc
        self._set_location_flag = True

    def set_jitter(self, h_j, v_j):
        """

        Args:
            h_j: 水平方向偏移矩阵
            v_j: 竖直方向偏移矩阵

        Returns:

        """
        if h_j is not None and v_j is not None:
            assert h_j.shape == v_j.shape, "Jitter ndim is diffient"
            self.fov_x_jitter = h_j
            self.fov_y_jitter = v_j

    def _get_jitter(self, src_fovs, fft_channel=0, process=10):
        """
        计算FFT特征偏移量
        Args:
            src_fovs: {'row_col':'image_path'}
            fft_channel: 选择fft特征通道
        """
        jitter_model = FOVAligner(src_fovs, self.rows, self.cols, multi=True, channel=fft_channel)
        jitter_model.set_process(process)
        if self.fov_x_jitter is None or self.fov_y_jitter is None:
            start_time = time.time()
            clog.info('Start jitter mode.')
            jitter_model.create_jitter()
            self.fov_x_jitter = jitter_model.horizontal_jitter
            self.fov_y_jitter = jitter_model.vertical_jitter

            if np.max(self.fov_x_jitter) == np.min(self.fov_x_jitter):
                self.fov_x_jitter = np.zeros_like(self.fov_x_jitter) - 999
                self.fov_x_jitter[:, 1:, 0] = - int(self._fov_width * self._overlap)
                self.fov_x_jitter[:, 1:, 1] = 0
            if np.max(self.fov_y_jitter) == np.min(self.fov_y_jitter):
                self.fov_y_jitter = np.zeros_like(self.fov_y_jitter) - 999
                self.fov_y_jitter[1:, :, 0] = 0
                self.fov_y_jitter[1:, :, 1] = - int(self._fov_height * self._overlap)

            self.__jitter_diff, self.__jitter_x_diff, self.__jitter_y_diff = \
                jitter_model._offset_eval(self._fov_height, self._fov_width, self._overlap)
            end_time = time.time()
            clog.info("Calculate jitter time -- {}s".format(end_time - start_time))
        else:
            jitter_model.horizontal_jitter = self.fov_x_jitter
            jitter_model.vertical_jitter = self.fov_y_jitter
            self.__jitter_diff, self.__jitter_x_diff, self.__jitter_y_diff = \
                jitter_model._offset_eval(self._fov_height, self._fov_width, self._overlap)
            clog.info("Have jitter matrix, skip this mode.")

    def _get_location(self, ):
        """
        Returns: 坐标生成
        """
        if self.fov_location is None:
            start_time = time.time()
            clog.info('Start location mode.')
            location_model = GlobalLocation()
            location_model.set_size(self.rows, self.cols)
            location_model.set_image_shape(self._fov_height, self._fov_width)
            location_model.set_jitter(self.fov_x_jitter, self.fov_y_jitter)
            location_model.create_location()
            self.fov_location = location_model.fov_loc_array
            self.__offset_diff = location_model.offset_diff
            end_time = time.time()
            clog.info("Calculate location time -- {}s".format(end_time - start_time))
        else:
            clog.info("Have location coord, skip this mode.")

    def stitch(self, src_fovs: dict, output_path: str = None, stitch=True, fft_channel=0):
        """
        Args:
            src_fovs: {'row_col':'image_path'}
            output_path:
            stitch: 是否拼接图像
            fft_channel: 选择fft特征通道
        """

        self._init_parm(src_fovs)

        if not self._is_stitched:
            if not self._set_location_flag:
                # 求解偏移矩阵
                self._get_jitter(src_fovs=src_fovs, fft_channel=fft_channel)

                # 求解拼接坐标
                self._get_location()

            # 拼接
            if stitch and output_path is not None:
                start_time = time.time()
                clog.info('Start stitch mode.')
                wsi = StitchingWSI()
                wsi.set_overlap(self._overlap)
                wsi.mosaic(src_fovs, self.fov_location, multi=False)
                # wsi.save(os.path.join(output_path, 'fov_stitched.tif'), compression=True)  # 统一放到外面io @dzh
                end_time = time.time()
                clog.info("Stitch image time -- {}s".format(end_time - start_time))

                self.__image = wsi.buffer
        else:
            clog.info("Image is stitched, skip all stitch operations.")

    def template(self, pts, scale_x, scale_y, rotate, chipno, index, correct_points,
                 pair_thresh, qc_thresh, range_thresh, correct_thresh, cluster_num,
                 scale_range, rotate_range, search_thresh, rotate_fov_min, method_threshold,
                 output_path=None, homo=False):
        """

        Args:
            pts: QC检点的所有点信息
            scale_x: x方向尺度
            scale_y: y方向尺度
            rotate: 角度
            chipno: 芯片标准周期
            index: 模板FOV索引
            correct_points: 修正点
            pair_thresh: 超参
            qc_thresh: 超参
            range_thresh: 超参
            correct_thresh: 超参
            cluster_num: 超参
            method_threshold: 芯片QC阈值
            output_path: 无
            homo: 无

        Returns:

        """
        if index is not None and correct_points is not None:
            tr = TemplateReference()
            tr.set_scale(scale_x, scale_y)
            tr.set_threshold(pair_thresh, qc_thresh, range_thresh, correct_thresh, cluster_num)
            tr.set_rotate(rotate)
            tr.set_chipno(chipno)
            tr.set_fov_location(self.fov_location)
            tr.set_qc_points(index, pts)

            tr.first_template_correct(target_points=correct_points, index=index)
            tr.reference_template(mode='multi')
            clog.info(f"Global template using method V1")
        else:
            tr = TemplateReferenceV2()

            if scale_x is not None and scale_y is not None:
                tr.set_scale(scale_x, scale_y)
            if rotate is not None:
                tr.set_rotate(rotate)

            tr.set_threshold(pair_thresh, qc_thresh, range_thresh, correct_thresh, cluster_num)
            tr.set_threshold_v2(scale_range=scale_range, rotate_range=rotate_range,
                                search_thresh=search_thresh, rotate_fov_min=rotate_fov_min)
            tr.set_chipno(chipno)
            tr.set_fov_location(self.fov_location)
            tr.set_qc_points(pts)
            clog.info(f"Global template using method V2")
            tr.reference_template_v2(method_threshold=method_threshold)

        if output_path is not None: tr.save_template(output_path)
        max_value, mean_value, std_value, qc_conf, re_conf = tr.get_template_eval()
        global_diff = tr.get_global_eval()

        self.__template_global_diff = global_diff
        self.__template_max_value = max_value
        self.__template_mean_value = mean_value
        self.__template_std_value = std_value
        self.__template_qc_conf = qc_conf
        self.__template_re_conf = re_conf
        self._template = tr.template

        # 2023/08/28新增
        try: self._fov_index, self._fov_best_point = tr.get_fov_info()
        except: pass

        clog.info("Reference template -- max value:{:.3f}  mean value:{:.3f}  "
                  "std value:{:.3f}  qc conf:{:.3f}  re conf:{:.3f}".format(
            max_value, mean_value, std_value, qc_conf, re_conf))

        if homo:
            mat, template = tr.homography_image()

        return self._template, [tr.scale_x, tr.scale_y, tr.rotation]

    def get_fov_info(self, ):
        return self._fov_index, self._fov_best_point

    def get_image(self):
        return self.__image

    def get_stitch_eval(self):
        """

        Returns: Dict 各项评估指标

        """
        eval = dict()

        if not self._is_stitched:
            eval['stitch_diff'] = self.__offset_diff
            eval['jitter_diff'] = self.__jitter_diff

            eval['stitch_diff_max'] = np.max(self.__offset_diff)
            eval['jitter_diff_max'] = np.max(self.__jitter_diff)

        eval['template_max'] = self.__template_max_value
        eval['template_mean'] = self.__template_mean_value
        eval['template_std'] = self.__template_std_value
        eval['template_qc_conf'] = self.__template_qc_conf
        eval['template_re_conf'] = self.__template_re_conf

        return eval

    def get_template_global_eval(self):
        """

        Returns: Array 模板评估的矩阵

        """
        return self.__template_global_diff

    def _get_jitter_eval(self):
        """
        Returns: 拼接偏移量热图展示用, 显微镜拼接
        """
        if not self._is_stitched:
            return self.__jitter_x_diff, self.__jitter_y_diff
        return -1, -1

    def _get_stitch_eval(self, ):
        """
        Returns: 拼接偏移量热图展示用，自研拼接
        """
        jitter_x_diff = np.zeros([self.rows, self.cols]) - 1
        jitter_y_diff = np.zeros([self.rows, self.cols]) - 1

        for row in range(self.rows):
            for col in range(self.cols):
                if row > 0 and self.fov_y_jitter[row, col, 0] != 999:
                    _jit = self.fov_location[row, col] - self.fov_location[row - 1, col] - [0, self._fov_height]
                    _dif = _jit - self.fov_y_jitter[row, col]
                    jitter_y_diff[row, col] = (_dif[0] ** 2 + _dif[1] ** 2) ** 0.5

                if col > 0 and self.fov_x_jitter[row, col, 0] != 999:
                    _jit = self.fov_location[row, col] - self.fov_location[row, col - 1] - [self._fov_width, 0]
                    _dif = _jit - self.fov_x_jitter[row, col]
                    jitter_x_diff[row, col] = (_dif[0] ** 2 + _dif[1] ** 2) ** 0.5

        return jitter_x_diff, jitter_y_diff

    def mosaic(self, ):
        pass


if __name__ == "__main__":
    clog.info(os.path.dirname(__file__))
