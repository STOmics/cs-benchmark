import numpy as np

from cellbin.modules.stitching import Stitching


class StitchQC(object):
    def __init__(self, is_stitched,
                 src_fovs: dict,
                 pts=None,
                 scale_x=None,
                 scale_y=None,
                 rotate=None,
                 chipno=None,
                 index=None,
                 correct_points=None,
                 pair_thresh=10,
                 qc_thresh=5,
                 range_thresh=5000,
                 correct_thresh=20,
                 cluster_num=10,
                 scale_range=0.7,
                 rotate_range=35,
                 search_thresh=500,
                 rotate_fov_min=7,
                 method_threshold=0.1,
                 fft_channel=0):
        """
        Args:
            is_stitched: 是否拼接大图
            src_fovs: 小图路径
            pts: QC所有点
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
            fft_channel: fft计算通道索引
        """

        self._is_stitched = is_stitched

        self._src_fovs = src_fovs
        self._pts = pts
        self._scale_x = scale_x
        self._scale_y = scale_y
        self._rotate = rotate
        self._chipno = chipno
        self._index = index
        self._correct_points = correct_points

        self._fft_channel = fft_channel
        self._pair_thresh = pair_thresh
        self._qc_thresh = qc_thresh
        self._range_thresh = range_thresh
        self._correct_thresh = correct_thresh
        self._cluster_num = cluster_num

        #模板推导V2参数
        self._scale_range = scale_range
        self._rotate_range = rotate_range
        self._search_thresh = search_thresh
        self._rotate_fov_min = rotate_fov_min
        self._method_threshold = method_threshold

        self.template = -1 #默认返回值

        self._h_jitter = None
        self._v_jitter = None
        self.fov_location = None
        self._rows = None
        self._cols = None

        # 2023/08/28新增 用于开发对接track线模板FOV信息
        self._fov_index = None
        self._fov_best_point = None

    def set_jitter(self, jitter):
        """

        Args:
            jitter: [h_j, v_j] 两个方向的偏移矩阵

        Returns:

        """
        if jitter is not None:
            self._h_jitter, self._v_jitter = jitter

    def set_size(self, rows, cols):
        """

        Args:
            rows: 行
            cols: 列

        Returns:

        """
        self._rows = rows
        self._cols = cols

    def set_location(self, loc):
        """

        Args:
            loc: 拼接坐标

        Returns:

        """
        self.fov_location = loc

    def run_qc(self):
        """

        Returns: 拼接及模板的评估指标

        """
        stitch = Stitching(self._is_stitched)

        stitch.set_size(self._rows, self._cols)
        stitch.set_jitter(self._h_jitter, self._v_jitter)

        if self._is_stitched:
            location = self._set_image_location()
            stitch.set_global_location(location)

        if self.fov_location is None:
            stitch.stitch(src_fovs=self._src_fovs, stitch=False, fft_channel=self._fft_channel)
            self.fov_location = stitch.fov_location
        else: stitch.set_global_location(self.fov_location)

        if self._pts is not None:
            self.template, parm = stitch.template(pts=self._pts,
                                                  scale_x=self._scale_x,
                                                  scale_y=self._scale_y,
                                                  rotate=self._rotate,
                                                  chipno=self._chipno,
                                                  index=self._index,
                                                  correct_points=self._correct_points,
                                                  pair_thresh=self._pair_thresh,
                                                  qc_thresh=self._qc_thresh,
                                                  range_thresh=self._range_thresh,
                                                  correct_thresh=self._correct_thresh,
                                                  cluster_num=self._cluster_num,
                                                  scale_range=self._scale_range,
                                                  rotate_range=self._rotate_range,
                                                  search_thresh=self._search_thresh,
                                                  rotate_fov_min=self._rotate_fov_min,
                                                  method_threshold=self._method_threshold
                                                  )

            self._scale_x, self._scale_y, self._rotate = parm
            self._fov_index, self._fov_best_point = stitch.get_fov_info()

        eval = stitch.get_stitch_eval()
        template_global_eval = stitch.get_template_global_eval()

        return eval, template_global_eval

    def get_fov_info(self, ):
        return self._fov_index, self._fov_best_point

    def get_scale_and_rotation(self):
        """
        :return: scale x & scale y & rotate
        """
        if self._scale_x is None:
            return -1, -1, -1
        return self._scale_x, self._scale_y, self._rotate

    def _set_image_location(self):
        """大图坐标生成函数"""
        location = np.zeros((self._rows, self._cols, 2), dtype=int)
        for k, v in self._src_fovs.items():
            row, col = [int(i) for i in k.split('_')]
            location[row, col] = [v[2], v[0]]

        return location


if __name__ == "__main__":
    from cellbin.utils.file_manager import imagespath2dict
    import h5py

    ipr_path = r"D:\AllData\Big\QC\D01666C1D3\D01666C1D3_20230306_105435_1.1.ipr"
    pts = {}
    with h5py.File(ipr_path) as conf:
        qc_pts = conf['QCInfo/CrossPoints/']
        for i in qc_pts.keys():
            pts[i] = conf['QCInfo/CrossPoints/' + i][:]
        scalex = conf['Register'].attrs['ScaleX']
        scaley = conf['Register'].attrs['ScaleY']
        rotate = conf['Register'].attrs['Rotation']
        chipno = conf['QCInfo/TrackDistanceTemplate'][...]
        index = conf['Stitch'].attrs['TemplateSource']

    src = r'D:\AllData\Big\D01666C1D3\D01666C1D3'
    output = r'D:\AllData\Big\D01666C1D3'
    src_fovs = imagespath2dict(src)
    stitch_qc = StitchQC(False, src_fovs,
                         pts, scalex, scaley, rotate, chipno, index)
    stitch_qc.set_size(24, 29)
    eval = stitch_qc.run_qc()
    print(eval)



