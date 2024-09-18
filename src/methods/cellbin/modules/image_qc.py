import copy
import os
import numpy as np
from typing import Union
import time

from cellbin.utils.file_manager import rc_key
from cellbin.modules import CellBinElement, StainType
from cellbin.modules.iqc.classify_fov import ClassifyFOV
from cellbin.dnn.tseg.yolo.detector import TissueSegmentationYolo
from cellbin.image.augmentation import pt_enhance_method, line_enhance_method, clarity_enhance_method
from cellbin.modules.iqc.regist_qc import RegistQC
from cellbin.modules.iqc.clarity_qc import ClarityQC
from cellbin.image.wsi_split import SplitWSI
from cellbin.modules.iqc.stitch_qc import StitchQC
from cellbin.image.wsi_stitch import StitchingWSI
from cellbin.contrib.track_roi_picker import TrackROIPicker
from cellbin.image import Image
from cellbin.image.mask import iou
from cellbin.utils import clog
from cellbin.contrib.fov_aligner import FOVAligner
from cellbin.utils.json_config import ConfigReader
from cellbin.utils.json_config import Channels
from cellbin.dnn.weights import auto_download_weights


class ImageQualityControl(CellBinElement):
    def __init__(self, ):
        super(ImageQualityControl, self).__init__()
        self._tissue_detector = TissueSegmentationYolo()
        self._rqc = RegistQC()
        self._cqc = ClarityQC()
        self._cl_fov = ClassifyFOV()
        self.config = ConfigReader()

        self.debug = False
        self.detect_channel = Channels()
        self.weight_names: list = []
        self.zoo_dir: str = ""
        self.stain_weights: list = []

        self.image_root: str = ""
        self.image_map: Union[dict, str] = {}
        self._stain_type: str = ""
        self._stereo_chip: list = []
        self._fov_rows: int = -1
        self._fov_cols: int = -1
        self._is_stitched: bool = False
        self._fov_size: tuple = ()
        self._magnification: int = -1

        self._mosaic: np.ndarray = np.array([])
        self._fovs_loc: np.ndarray = np.array([])

        # flag
        self.tissue_cut_flag: bool = False
        self.pt_qc_flag: bool = False
        self.line_qc_flag: bool = False
        self.regist_qc_flag: bool = False  # pt_qc pass + line_qc pass = regist_qc pass
        self.clarity_flag: bool = False  # clarity pass or failed
        self.global_template_flag: bool = False  # global template eval success or fail
        self.microscope_stitch: bool = False

        # output
        # Image info
        self.total_fov_count: int = -1

        # tissue qc output
        self._box_mosaic: list = [-1, -1, -1, -1]  # bounding box coord on stitched image, [0, 4830, 27924, 24395], list
        self._fov_tissue_mask: np.ndarray = np.array([])  # r x c, 2d array
        self.stitch_roi: tuple = ()  # bounding box coord on fov, (0, 2, 9, 9), tuple
        self._tissue_mask: np.ndarray = np.array([])  # stitched image size, 2d array
        self.tissue_mask_score: float = -1.0  # float
        self.tissue_area: int = -1  # tissue mask area, int

        # regist qc output
        self.track_score: float = -1.0
        self.track_line_score: float = -1.0
        self.track_pts: dict = {}
        self.rotation: float = -1.0
        self.scale_x: float = -1.0
        self.scale_y: float = -1.0
        self.good_fov_count: int = -1

        self.track_pt_fov_mask: np.ndarray = np.array([])
        self.track_line_best_match: list = []
        self._src_fov_info: list = []  # template fov for stitch, [k, template_pts, x_scale, y_scale, rotation]

        # clarity qc output
        self.clarity_score = -1.0  # float，清晰度得分
        self.clarity_preds_arr = -1
        self.clarity_heatmap = np.array([])  # numpy array，清晰度结果呈现在原图上
        self.clarity_cluster = None  # plt plot object，清晰度成团聚类结果

        self.clarity_topk_result = []
        self.clarity_counts = {}

        # stitch qc + global template qc output
        self.stitch_diff = -1  # np ndarray
        self.jitter_diff = -1  # np ndarray
        self.stitch_diff_max = -1.0  # float
        self.jitter_diff_max = -1.0  # float
        self.template_max = -1.0  # float
        self.template_mean = -1.0  # float
        self.template_std = -1.0  # float
        self.template_qc_precision = -1.0  # float
        self.template_qc_recall = -1.0  # float
        self.global_template = -1  # list
        self.jitter = [-1, -1]
        self.stitch_fov_loc = -1
        self.stitch_template_global = -1  # global heatmap, r x c x 1

        # 2023/08/28新增 用于开发对接track线模板FOV信息
        self.stitch_fov_index = -1
        self.stitch_fov_best_point = -1

        # time cost
        self.prepare_time = 0
        self.tissue_seg_time = 0
        self.track_detect_time = 0
        self.track_line_time = 0
        self.stitch_qc_time = 0
        self.clarity_qc_time = 0

        # step control
        self.step_total = {}
        self.step_index = -1

    def auto_load_weights(self, ):
        self.stain_weights = getattr(self.config.model_weights_config, self._stain_type)
        self.weight_names = [
            self.stain_weights.track_pt_detect,
            self.stain_weights.tissue_seg,
            self.stain_weights.clarity_eval
        ]
        auto_download_weights(self.zoo_dir, self.weight_names)

    def initialize(self):
        """
        Initialize dnn model

        """
        # 导入模型权重
        tissue_detector_file = os.path.join(self.zoo_dir, self.stain_weights.tissue_seg)
        pt_detector_file = os.path.join(self.zoo_dir, self.stain_weights.track_pt_detect)
        clarity_file = os.path.join(self.zoo_dir, self.stain_weights.clarity_eval)
        try:
            self._tissue_detector.f_init_model(tissue_detector_file)
            self._rqc.track_pt_qc.ci.load_model(pt_detector_file)
            self._cqc.load_model(clarity_file)
        except Exception as e:
            clog.error(f"{e}")
            clog.error(f"dnn model weights loading failed")
            return 1
        return 0

    def initialize_json(self, json_path: str):
        """
        Initialize config file, will initialize all the thresholds and

        Args:
            json_path (): config json path

        Returns:

        """
        self.config.load_from_json(json_path)
        self.detect_channel = getattr(self.config.channel_config, self._stain_type)

    @staticmethod
    def iou_calculation(clarity_mask, tissue_mask):
        iou_result = iou(clarity_mask, tissue_mask)
        return iou_result

    def _fov_box(self, b):
        w, h = self._fov_size
        y_begin, y_end, x_begin, x_end = b
        row = y_begin // h
        col = x_begin // w
        self._fovs_loc[row, col] = [x_begin, y_begin]
        self.image_map[rc_key(row, col)] = b

    def _get_jitter(self, ):
        if not self._is_stitched:
            r, c = self._fovs_loc.shape[:2]
            clog.info(f"Fov aligner using channel: {self.detect_channel.fft}")
            fa = FOVAligner(
                images_path=self.image_map,
                rows=r,
                cols=c,
                channel=self.detect_channel.fft
            )
            fa.set_process(self.config.running_config.stitch_running_process)
            fa.create_jitter()
            self.jitter = [fa.horizontal_jitter, fa.vertical_jitter]

    def _prepare(self, ):
        """
        This func is mainly used to deal with
        - split large tif to fovs

        Returns:

        """
        if self._is_stitched:
            image_reader = Image()
            img_path = os.path.join(self.image_root, self.image_map)
            image_reader.read(img_path)
            self._mosaic = image_reader.image
            # h_, w_ = image_reader.height, image_reader.width
            w, h = self._fov_size
            # self._fov_rows, self._fov_cols = [(h_ // h) + 1, (w_ // w) + 1]
            self.image_map = dict()
            wsi = SplitWSI(img=self._mosaic, win_shape=(h, w),
                           overlap=0, batch_size=1, need_fun_ret=False, need_combine_ret=False)
            _box_lst, _fun_ret, _dst = wsi.f_split2run()
            self._fov_rows, self._fov_cols = wsi.y_nums, wsi.x_nums
            self._fovs_loc = np.zeros((self._fov_rows, self._fov_cols, 2), dtype=int)
            for b in _box_lst:
                self._fov_box(b)
        else:
            wsi = StitchingWSI()
            self._fov_rows, self._fov_cols = self._fovs_loc.shape[:2]
            wsi.mosaic(src_image=self.image_map, loc=self._fovs_loc, downsample=1)
            self._mosaic = wsi.buffer
        self.total_fov_count = len(self.image_map)  # 大图需要从这里获取total fov count

    def _classify_fov(self, ):
        """
        This func will do:
        - tissue cut based on "stitched" image
        - classify fovs based on tissue cut result
        - classified result will contain
            - tissue fov
            - non tissue fov

        Returns:
            self._fov_tissue_mask: classified fov result mask
            self._tissue_mask : tissue mask
            self._box_mosaic: tissue bounding box on stitched image
            self.stitch_roi: stitched roi. col, row, col, row
            self.tissue_area: sum of tissue mask

        """
        self._cl_fov.set_detector(self._tissue_detector)
        self._cl_fov.classify(
            mosaic=self._mosaic,
            fov_loc=self._fovs_loc,
            fov_size=self._fov_size,
            expand=1,
            ch=self.detect_channel.fft
        )
        self.tissue_cut_flag = self._cl_fov.success
        self._tissue_mask = self._cl_fov.tissue_mask  # stitched image size, 2d array
        if self.tissue_cut_flag:
            self._fov_tissue_mask = self._cl_fov.tissue_fov_map  # r x c, 2d array
            # bounding box coord on stitched image, [0, 4830, 27924, 24395], list
            self._box_mosaic = self._cl_fov.tissue_bbox_in_mosaic()
            self.stitch_roi = self._cl_fov.tissue_fov_roi  # bounding box coord on fov, (0, 2, 9, 9), tuple
            self.tissue_area = self._cl_fov.tissue_detector.mask_num  # tissue cut area, int

    def track_pt_qc(self, ):
        """
        This func will do
        - track pt detect
        - track line detect
        - track line result match

        Returns:

        """
        # set threshold
        self._rqc.set_chip_template(self._stereo_chip)
        self._rqc.set_track_pt_thresh(
            th=self.config.track_pt_config.track_point_first_level_threshold,
            th2=self.config.track_pt_config.track_point_second_level_threshold,
            good_thresh=self.config.track_pt_config.track_point_good_threshold,
        )
        self._rqc.set_topk(self.config.track_line_config.track_line_topk)
        self._rqc.set_track_pt_process(self.config.running_config.pt_running_process)

        # start
        buffer = None
        if self._is_stitched:
            buffer = self._mosaic

        # Track点检测
        self._rqc.run_pt_qc(
            fovs=self.image_map,
            enhance_func=pt_enhance_method.get(self._stain_type, None),
            detect_channel=self.detect_channel.pt_detect,
            buffer=buffer
        )

    def track_line_qc(self):
        # Track线检测
        buffer = None
        if self._is_stitched:
            buffer = self._mosaic
        if self._magnification <= 15:
            line_fovs = None
        else:
            trp = TrackROIPicker(
                images_path=self.image_map, jitter_list=self.jitter,
                tissue_mask=self._fov_tissue_mask,
                track_points=self._rqc.track_pt_qc.track_result()
            )
            line_fovs = trp.getRoiImages()

        self._rqc.run_line_qc(
            line_fovs=line_fovs,
            detect_channel=self.detect_channel.line_detect,
            magnification=self._magnification,
            buffer=buffer,
            enhance_func=line_enhance_method.get(self._stain_type, None),
        )

    def _get_stitch_inputs(self):
        stitch_inputs = {
            "template_pts": None,
            "x_scale": None,
            "y_scale": None,
            "rotation": None,
            "fov_index": None,
        }
        self.track_score = self._rqc.pt_score
        # 1. pt pass -> angle provided
        if self.track_score > self.config.track_pt_config.track_point_score_threshold:
            self.pt_qc_flag = True
            most_angle, count = self._rqc.track_pt_qc.most_freq_angle
            if count / self.total_fov_count > self.config.track_pt_config.track_point_good_threshold:
                stitch_inputs['rotation'] = most_angle
        else:
            self.pt_qc_flag = False
        self.track_pts = self._rqc.pt_result  # 未检测到为空

        # 2. track line pass -> all provided
        self.track_line_score = self._rqc.line_score
        if self.track_line_score > self.config.track_line_config.track_line_score_threshold:
            self.line_qc_flag = True
            self._src_fov_info = self._rqc.best_match
            self.rotation = self._src_fov_info[-1]
            self.scale_x = self._src_fov_info[-3]
            self.scale_y = self._src_fov_info[-2]
            src_fov, template_pts, x_scale, y_scale, rotation = self._src_fov_info
            stitch_inputs['template_pts'] = template_pts
            stitch_inputs['x_scale'] = x_scale
            stitch_inputs['y_scale'] = y_scale
            stitch_inputs['rotation'] = rotation
            stitch_inputs['fov_index'] = src_fov
        else:
            self.line_qc_flag = False

        # Track点 + Track线通过 = regist qc通过
        if self.pt_qc_flag and self.line_qc_flag:
            self.regist_qc_flag = True
        else:
            self.regist_qc_flag = False

        self.track_pt_fov_mask = self._rqc.track_pt_qc.fov_mask
        self.track_line_best_match = self._rqc.best_match
        self.good_fov_count = self._rqc.good_fov_count

        return stitch_inputs

    def _stitch_template(self, template_config, total_cross_pt, stitch_inputs):
        """
        track线检测版本模板推导
        Args:
            template_config:
            total_cross_pt:
            stitch_inputs:

        Returns:

        """
        #TODO 芯片类型临时阈值变量
        method_threshold = 0.01 if self._stain_type == "HE" else 0.1

        sth_qc = StitchQC(
            is_stitched=self._is_stitched,
            src_fovs=self.image_map,
            pts=total_cross_pt,
            scale_x=stitch_inputs["x_scale"],
            scale_y=stitch_inputs["y_scale"],
            rotate=stitch_inputs["rotation"],
            chipno=self._stereo_chip,
            index=stitch_inputs["fov_index"],
            correct_points=stitch_inputs["template_pts"],
            pair_thresh=template_config.template_pair_points_threshold,
            qc_thresh=template_config.template_pair_points_qc_threshold,
            range_thresh=template_config.template_range_image_size_threshold,
            correct_thresh=template_config.template_pair_correct_threshold,
            cluster_num=template_config.template_cluster_num_threshold,
            scale_range=template_config.template_v2_scale_range_threshold,
            rotate_range=template_config.template_v2_rotate_range_threshold,
            search_thresh=template_config.template_v2_search_range_threshold,
            rotate_fov_min=template_config.template_v2_rotate_fov_min_threshold,
            method_threshold=method_threshold,
            fft_channel=self.detect_channel.fft
        )

        # 代入初始的拼接坐标
        if self._fovs_loc is not None:
            sth_qc.set_location(self._fovs_loc)
        sth_qc.set_size(self._fov_rows, self._fov_cols)

        # 下列此处开始推导模板
        dct, template_global = sth_qc.run_qc()
        self.microscope_stitch = True  # 显微镜拼接flag

        first_sth_qc = copy.deepcopy(sth_qc)
        first_dct = copy.deepcopy(dct)
        first_template_global = copy.deepcopy(template_global)

        # 若模板推导结果小于0.1或者直接为-1，则说明要么是track点不太行，要么是拼接坐标有点问题，所以用FFT计算相邻FOV的真实偏差，并重推拼接坐标
        # 以及再来一边模板推导
        if -1 <= dct.get('template_re_conf', -1.0) < 0.1 \
                and not self._is_stitched \
                and self._rqc.line_score != 0:
            clog.info("Microscope coordinates have significant errors, use BGI stitching algorithm")
            self.microscope_stitch = False
            if isinstance(self.jitter[0], int) or isinstance(self.jitter[1], int):
                self._get_jitter()  # FFT计算过相邻FOV偏差矩阵 分别时行向和列向[h_j, v_j] -- 均为 r * c * 2 矩阵
            # self._get_jitter()
            sth_qc.set_jitter(self.jitter)
            sth_qc.fov_location = None
            dct, template_global = sth_qc.run_qc()

        if max(first_dct.get('template_re_conf', -1.0), first_dct.get('template_max', -1.0)) \
                > \
                max(dct.get('template_re_conf', -1.0), dct.get('template_max', -1.0)):
            return first_sth_qc, first_dct, first_template_global

        return sth_qc, dct, template_global

    def _stitching_qc(self, ):
        """
        拼接总体QC模块 主要调用 StitchQC 实现拼接坐标计算以及模板推导计算
        并包含各类评估指标信息
        """
        template_config = self.config.global_template_config
        clog.info(f"Stitch qc using channel: {self.detect_channel.fft}")
        total_cross_pt = dict()

        for k, v in self._rqc.pt_result.items():
            total_cross_pt[k] = v[0]

        # 第一次推导模板使用新算法
        stitch_inputs = self._get_stitch_inputs()
        sth_qc, dct, template_global = self._stitch_template(template_config, total_cross_pt, stitch_inputs)

        first_sth_qc = copy.deepcopy(sth_qc)
        first_dct = copy.deepcopy(dct)
        first_template_global = copy.deepcopy(template_global)

        #新算法不行 老算法来一遍
        if -1 <= dct.get('template_re_conf', -1.0) < 0.1:
            self.track_line_qc()
            stitch_inputs = self._get_stitch_inputs()
            if self.line_qc_flag:
                total_cross_pt[stitch_inputs["fov_index"]] = stitch_inputs["template_pts"]
                sth_qc, dct, template_global = self._stitch_template(template_config, total_cross_pt, stitch_inputs)
            else:
                # 老算法也无能为力(^_^)
                clog.info(f"Cannot find template points.")

        if max(first_dct.get('template_re_conf', -1.0), first_dct.get('template_max', -1.0)) \
                > \
                max(dct.get('template_re_conf', -1.0), dct.get('template_max', -1.0)):
            sth_qc = first_sth_qc
            dct = first_dct
            template_global = first_template_global

        # 如果track线检测开启，则不会在拼接qc模块获取到模板fov信息
        _fov_index, _fov_best_point = sth_qc.get_fov_info()
        if stitch_inputs['fov_index'] is not None and stitch_inputs['template_pts'] is not None:
            self.stitch_fov_index = stitch_inputs['fov_index']
            self.stitch_fov_best_point = stitch_inputs['template_pts']
        elif _fov_index is not None and _fov_best_point is not None:
            self.stitch_fov_index, self.stitch_fov_best_point = _fov_index, _fov_best_point

        # stitch module
        self.stitch_template_global = template_global  # 全局模板偏差，即track点坐标与模板推导坐标的距离标量值，r * c矩阵
        self.stitch_fov_loc = sth_qc.fov_location  # 最终的拼接坐标，显微镜坐标 | 自研拼接坐标
        self.stitch_diff = dct.get('stitch_diff', -1.0)  # 自研拼接误差矩阵（r * c）,在计算了自研拼接坐标时才有值，否则-1
        self.jitter_diff = dct.get('jitter_diff', -1.0)  # 显微镜拼接误差矩阵（r * c）, 在计算了自研拼接坐标时才有值，否则-1
        self.stitch_diff_max = dct.get('stitch_diff_max', -1.0)  # stitch_diff的最大值，float
        self.jitter_diff_max = dct.get('jitter_diff_max', -1.0)  # jitter_diff的最大值，float

        # 异常值填充
        if self.stitch_diff is None:
            self.stitch_diff = self.jitter_diff = -1
            self.stitch_diff_max = self.jitter_diff_max = -1

        # template module
        self.scale_x, self.scale_y, self.rotation = sth_qc.get_scale_and_rotation()  # float类型 scale rotate 基本值
        self.global_template = sth_qc.template  # 模板[x, y, index_x, index_y]
        self.template_max = dct.get('template_max', -1.0)  # 模板区域面积占比 有用
        self.template_mean = dct.get('template_mean', -1.0)  # 无用
        self.template_std = dct.get('template_std', -1.0)  # 无用
        self.template_qc_precision = dct.get('template_qc_conf', -1.0)  # 无用
        self.template_qc_recall = dct.get('template_re_conf', -1.0)  # 模板召回率值 有用
        if self.template_max == -1 or self.template_mean == -1 or self.template_std == -1:
            self.global_template_flag = False
        else:
            self.global_template_flag = True

    def _clarity_qc(self, ):
        """
        This func will do clarity eval on stitched image

        Returns:

        """
        if self.tissue_cut_flag:
            x0, y0, x1, y1 = self._box_mosaic
            clarity_input = self._mosaic[y0: y1, x0: x1]
        else:
            clarity_input = self._mosaic
        clog.info(f"Mosaic size: {self._mosaic.shape}, clarity input: {clarity_input.shape}")
        self._cqc.set_enhance_func(clarity_enhance_method.get(self._stain_type, None))
        self._cqc.run(
            img=clarity_input,
            detect_channel=self.detect_channel.clarity
        )

        self._cqc.cluster()
        self.clarity_cluster = self._cqc.fig  # plt plot object，清晰度成团聚类结果
        self.clarity_topk_result = self._cqc.topk_result

        # clarity qc output
        self._cqc.post_process()
        self.clarity_cut_size = self._cqc.cl_classify.img_size
        self.clarity_overlap = self._cqc.cl_classify.overlap
        self.clarity_score = self._cqc.score  # float，清晰度得分
        self.clarity_heatmap = self._cqc.draw_img  # numpy array，清晰度结果呈现在原图上
        self.clarity_preds_arr = self._cqc.preds  # clarity predictions array

        self.clarity_counts = self._cqc.counts

    def set_is_stitched(self, stitched):
        self._is_stitched = stitched

    def set_fov_loc(self, loc):
        self._fovs_loc = loc

    def set_fov_size(self, s):
        self._fov_size = s

    def set_stereo_chip(self, c):
        self._stereo_chip = c

    def set_stain_type(self, s):
        self._stain_type = s

    def set_magnification(self, m):
        assert m in [10, 20, 40]
        self._magnification = m

    def set_zoo_dir(self, z):
        self.zoo_dir = z

    def set_debug_mode(self, d):
        self.debug = d

    def prepare_steps(self):
        for k, v in vars(self.config.operation).items():
            if v:
                self.step_total[k] = v
        clog.info(f"Total steps to do: {len(self.step_total)}")

    def run(self, image_root: str, image_map):
        """
        image_root:
            - fovs: the directory where microscope config file locate (str)
            - stitched: the directory where the stitched image locate (str)
        image_map:
            - fovs: relative image path of each fov (dict)
            - stitched: relative image path of stitched image (str)
        """
        clog.info("-------------------Start QC-------------------")
        clog.info(f"Is_stitched: {self._is_stitched}, fov size: {self._fov_size}, stain_type: {self._stain_type} \n"
                  f"Magnification: {self._magnification}, debug mode: {self.debug}, chip: {self._stereo_chip}")
        self.prepare_steps()
        start_time = time.time()
        clog.info(f'Image QC start time: {start_time}')

        self.image_root = image_root
        self.image_map = image_map
        clog.info(f"Image root : {self.image_root}")
        self._prepare()
        end_time = time.time()
        self.prepare_time = end_time - start_time
        clog.info(f'Prepare time cost: {self.prepare_time}')
        start_time = end_time

        if self.config.operation.tissue_segment:
            self.step_index += 1
            step_name = list(self.step_total.keys())[self.step_index]
            clog.info(f"Current step index: {self.step_index}, {step_name}")
            self._classify_fov()
            end_time = time.time()
            self.tissue_seg_time = end_time - start_time
            clog.info(f'Classify time cost: {self.tissue_seg_time}')
            start_time = end_time

        if self.config.operation.track_pt_detect:
            self.step_index += 1
            step_name = list(self.step_total.keys())[self.step_index]
            clog.info(f"Current step index: {self.step_index}, {step_name}")
            self.track_pt_qc()
            end_time = time.time()
            self.track_detect_time = end_time - start_time
            clog.info(f'Track_pt time cost: {self.track_detect_time}')
            start_time = end_time

        if self.config.operation.track_line_detect:
            self.step_index += 1
            step_name = list(self.step_total.keys())[self.step_index]
            clog.info(f"Current step index: {self.step_index}, {step_name}")
            self.track_line_qc()
            end_time = time.time()
            self.track_line_time = end_time - start_time
            clog.info(f'Track_line time cost: {self.track_line_time}')
            start_time = end_time

        if self.config.operation.stitch_qc:
            self.step_index += 1
            step_name = list(self.step_total.keys())[self.step_index]
            clog.info(f"Current step index: {self.step_index}, {step_name}")
            self._stitching_qc()
            end_time = time.time()
            self.stitch_qc_time = end_time - start_time
            clog.info(f'Stitching time cost: {self.stitch_qc_time}')
            start_time = end_time

        if self.config.operation.clarity_qc:
            self.step_index += 1
            step_name = list(self.step_total.keys())[self.step_index]
            clog.info(f"Current step index: {self.step_index}, {step_name}")
            self._clarity_qc()
            end_time = time.time()
            self.clarity_qc_time = end_time - start_time
            clog.info(f'Clarity time cost: {self.clarity_qc_time}')
            start_time = end_time

            # self.tissue_mask_score = self.iou_calculation(
            #     clarity_mask=self._cqc.black_img,
            #     tissue_mask=self._tissue_mask
            # )
        clog.info("-------------------End QC-------------------")

    @property
    def src_fov_info(self):
        return self._src_fov_info

    @property
    def box_mosaic(self):
        return self._box_mosaic

    @property
    def fov_tissue_mask(self):
        return self._fov_tissue_mask

    @property
    def tissue_mask(self):
        return self._tissue_mask
