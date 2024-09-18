###################################################
"""reference template v2 for image, must need QC data.
create by lizepeng, 2023/4/13 14:09
"""
####################################################

import numpy as np

from sklearn.linear_model import LinearRegression
from cellbin.contrib import TemplateReference
from cellbin.utils import clog


class TemplateReferenceV2(TemplateReference):
    """
    模板推导算法V2
    分别通过点来拟合大致角度和尺度
    再通过初步校准得到局部区域的精确角度尺度值
    最终全局推导模板
    """

    MINIMIZE_METHOD = ['nelder-mead', 'slsqp', 'bfgs']

    def __init__(self, ):
        super(TemplateReferenceV2, self).__init__()

        self.scale_range = 0.7
        self.rotate_range = 35
        self.search_thresh = 500
        self.rotate_fov_min = 7

        self.set_scale_flag = False
        self.set_rotate_flag = False

        # 2023/08/28新增 用于开发对接track线模板FOV信息
        self.fov_index = None
        self.fov_best_point = None

    def get_fov_info(self, ):
        return self.fov_index, self.fov_best_point

    def set_threshold_v2(self,
                         scale_range=None,
                         rotate_range=None,
                         search_thresh=None,
                         rotate_fov_min=None):
        """
        模板推导V2阈值
        """
        if scale_range is not None:
            self.scale_range = scale_range
        if rotate_range is not None:
            self.rotate_range = rotate_range
        if search_thresh is not None:
            self.search_thresh = search_thresh
        if rotate_fov_min is not None:
            self.rotate_fov_min = rotate_fov_min

    def set_scale(self, scale_x: float, scale_y: float):
        self.scale_x = self._to_digit(scale_x)
        self.scale_y = self._to_digit(scale_y)
        assert self.scale_x is not None and self.scale_y is not None, "Input is not a number."
        self.set_scale_flag = True

    def set_rotate(self, r: float):
        self.rotation = self._to_digit(r)
        assert self.rotation is not None, "Input is not a number."
        self.set_rotate_flag = True

    def set_qc_points(self, pts):
        """
        pts: {index: [x, y, ind_x, ind_y], ...}
        """
        if self.fov_loc_array is None:
            print("Please init global location.")
            return

        assert isinstance(pts, dict), "QC Points is error."
        for ind in pts.keys():
            points = np.array(pts[ind])
            points[:, :2] = np.round(points[:, :2], 2)
            self.qc_pts[ind] = points

    def _template_correct(self, qc_pts, n=5):
        """
        Args:
            qc_pts: 已按照点数排序的点集 max->min
            n: track角度遍历搜索所需FOV数量
        """
        rotate_list = list()
        point_list = list()
        best_point = None
        index = 0

        #####角度搜索
        if not self.set_rotate_flag:
            for pts in qc_pts[:n]:
                if len(pts[1]) < 3:
                    continue
                target_points = pts[1]
                rotate = self._rotate_search(target_points, self.rotate_range)
                rotate_list.append(rotate)

            if len(rotate_list) == 0:
                return 0, np.array([0, 0, 0, 0])

            # self.rotation = max(rotate_list, key=rotate_list.count)
            # index = rotate_list.index(self.rotation)

            tmp_rot = max(rotate_list, key=rotate_list.count)
            tmp_ind = rotate_list.index(tmp_rot)
            if rotate_list.count(tmp_rot) == 1:
                for rot in rotate_list:
                    rp_count = rotate_list.count(rot + 1)
                    rm_count = rotate_list.count(rot - 1)
                    if rp_count * rm_count > 0:
                        continue
                    elif rp_count > 0:
                        tmp_rot = min([rot, rot + 1], key=rotate_list.index)
                        tmp_ind = rotate_list.index(tmp_rot)
                        break
                    elif rm_count > 0:
                        tmp_rot = min([rot, rot - 1], key=rotate_list.index)
                        tmp_ind = rotate_list.index(tmp_rot)
                        break
            self.rotation = tmp_rot
            index = tmp_ind

        #####尺度搜索
        if not self.set_scale_flag:
            scale, best_point = self._scale_search(qc_pts[index][1], self.scale_range)
            self.scale_x = self.scale_y = scale
        else:
            _, best_point = self._index_search(qc_pts[index][1])

        return index, best_point

    def _rotate_search(self, target_points, rotate_range=15):
        """
        角度搜索
        遍历角度 通过求得横向匹配点对 拟合直线得到的最小误差距离
        再和拟合直线的角度与当前角度相减 最小值即为大致角度

        Args:
            target_points:
            rotate_range: -15 ~ 15
        Return:
            rotate:
        """

        rotate = None
        rotate_min = np.Inf
        center_point = self._center_point_search(target_points)
        dis_list = list()
        rotate_dif_list = list()
        for _rotate in range(-rotate_range, rotate_range):
            line_points_x = self._cross_line_points(center_point, self.scale_x, _rotate, self.chip_no[0])
            src_points_x, _ = self.pair_to_template(line_points_x, target_points, self.search_thresh)
            src_points_x = np.unique(src_points_x, axis=0)
            line_model = LinearRegression()
            _x = src_points_x[:, 0].reshape(-1, 1)
            _y = src_points_x[:, 1].reshape(-1, 1)

            line_model.fit(_x, _y)
            y_predict = line_model.predict(_x)
            y_dis = np.sum((y_predict - _y) ** 2)
            _k = line_model.coef_[0][0]
            rotate_dif = np.abs(np.degrees(np.arctan(_k)) - _rotate)
            rotate_dif_list.append(rotate_dif)
            dis_list.append(y_dis)

        indexs = [i for i, x in enumerate(dis_list) if x == np.min(dis_list)]
        min_rotate_dif = np.array(rotate_dif_list)[indexs].min()
        index = rotate_dif_list.index(min_rotate_dif)
        return range(-rotate_range, rotate_range)[index]

    def _scale_search(self, target_points, scale_range=0.3):
        """
        尺度搜索
        Args:
            target_points:
            scale_range: 1 - scale_range ~ 1 + scale_range
        Return:
            scale:
            best_point:
        """
        scale = None
        distance = np.Inf
        scale_list = [i / 10 for i in range(int(10 - scale_range * 10), int(10 + scale_range * 10 + 1))]
        best_point = None
        for _scale in scale_list:
            self.scale_x = self.scale_y = _scale
            _distance, _best_center_point = self._index_search(target_points)
            if _distance < distance:
                distance = _distance
                scale = self.scale_x
                best_point = _best_center_point

        return scale, best_point

    def _index_search(self, target_points, center_point=None):
        """
        Args:
            target_points: 单个FOV的点集
        Return:
            best_center_point: FOV中心点的坐标及索引
        """
        best_center_point = list()
        if center_point is None:
            center_point = self._center_point_search(target_points)
        chip_len = len(self.chip_no[0])
        _distance = np.Inf
        for index_x in range(chip_len):
            for index_y in range(chip_len):
                _center_point = np.concatenate((center_point[:2], [index_x, index_y]))
                self._point_inference(_center_point, (self._range_thresh, self._range_thresh))
                distance = self.pair_to_template(target_points, self.template, self.search_thresh, dis=True)
                _judge = self._valid_scale_judge(target_points)
                if np.sum(distance) < _distance and _judge:
                    _distance = np.sum(distance)
                    best_center_point = _center_point

        return _distance, best_center_point

    def _valid_scale_judge(self, target_points, rate=1.2):
        """
        搜索尺度时，解决对于小scale尺度匹配距离过小问题
        """
        point_re, point_qc = self.pair_to_template(target_points, self.template, self.search_thresh)

        x_min = min(np.min(point_re[:, 0]), np.min(point_qc[:, 0]))
        x_max = max(np.max(point_re[:, 0]), np.max(point_qc[:, 0]))
        y_min = min(np.min(point_re[:, 1]), np.min(point_qc[:, 1]))
        y_max = max(np.max(point_re[:, 1]), np.max(point_qc[:, 1]))

        points_count = 0

        valid_temp = list()
        for point in self.template:
            if x_min <= point[0] <= x_max and \
                y_min <= point[1] <= y_max:
                points_count += 1
                valid_temp.append(point)

        if points_count / len(target_points) <= rate or \
                points_count - len(target_points) <= 10: #匹配点圈内点数量不多于1.2倍或不多于10个检点数
            return True
        elif points_count / len(target_points) <= 2 * rate or \
                points_count - len(target_points) <= 20:

            min_dif = np.inf
            for idx in range(len(self.chip_no[0])):
                re_count = len([pt for pt in point_re if pt[2] == idx])
                tp_count = len([pt for pt in valid_temp if pt[2] == idx])
                if tp_count > 0:
                    dif_rate = abs(re_count - tp_count) / tp_count
                    if dif_rate < min_dif:
                        min_dif = dif_rate

                re_count = len([pt for pt in point_re if pt[3] == idx])
                tp_count = len([pt for pt in valid_temp if pt[3] == idx])
                if tp_count > 0:
                    dif_rate = abs(re_count - tp_count) / tp_count
                    if dif_rate < min_dif:
                        min_dif = dif_rate

            if min_dif < 0.2:
                return True

        return False

    @staticmethod
    def _gradient_std_search(src_points, dst_points, reci=False):
        """
        对应点对的梯度标准差
        """
        k_list = list()
        for src_point, dst_point in zip(src_points, dst_points):
            k = (src_point[1] - dst_point[1]) / (src_point[0] - dst_point[0])
            if reci: k_list.append(1 / k)
            else: k_list.append(k)
        return np.std(k_list)

    @staticmethod
    def _center_point_search(target_points):
        """
        中心点匹配
        """
        target_points = np.array(target_points)
        x_mean = np.mean(target_points[:, 0])
        y_mean = np.mean(target_points[:, 1])
        return sorted(target_points, key=lambda x: ((x[0] - x_mean) ** 2) + (x[1] - y_mean) ** 2)[0]

    def _cross_line_points(self, center_point, scale, rotate, chip):
        """
        中心点出发的十字线点集
        """
        points = list()
        temp = center_point

        ind = 0
        while 0 < temp[0] < self._range_thresh and 0 < temp[1] < self._range_thresh:
            ind = int(ind % len(chip))
            _x = (chip[ind] * scale) * np.cos(np.radians(rotate))
            _y = (chip[ind] * scale) * np.sin(np.radians(rotate))
            temp = [temp[0] + _x, temp[1] + _y]
            points.append(temp)
            ind += 1

        ind = -1
        temp = center_point
        while 0 < temp[0] < self._range_thresh and 0 < temp[1] < self._range_thresh:
            ind = int(ind % len(chip))
            _x = (chip[ind] * scale) * np.cos(np.radians(rotate))
            _y = (chip[ind] * scale) * np.sin(np.radians(rotate))
            temp = [temp[0] - _x, temp[1] - _y]
            points.append(temp)
            ind -= 1

        return points

    def reference_template_v2(self, method_threshold=0.1):
        """模板推导算法V2"""
        self._check_parm()
        self._qc_points_to_gloabal(all_points=True)
        if len(self.qc_pts) == 0:
            self.flag_skip_reference = True
            clog.info("QC track points is None, quit template reference.")
            return
        qc_pts = sorted(self.qc_pts.items(), key=lambda x: x[1].shape[0], reverse=True)
        index, best_point = self._template_correct(qc_pts, self.rotate_fov_min)
        self.fov_index = qc_pts[index][0]

        if len(best_point) == 0:
            clog.info("Template reference failed.")
            return

        correct_scale_x = self.scale_x
        correct_scale_y = self.scale_y
        correct_rotate = self.rotation

        for method in self.MINIMIZE_METHOD:
            self.flag_skip_reference = False
            self.set_minimize_method(method=method)
            self.first_template_correct(target_points=qc_pts[index][1],
                                        index=qc_pts[index][0],
                                        center_points=best_point)

            self.fov_best_point = self.template

            clog.info(f"Reference template use method {method}.")
            self.reference_template('multi')

            valid_area, _, _, _, re_conf = self.get_template_eval()
            if valid_area > method_threshold or re_conf > method_threshold:
                break
            else:
                self.scale_x = correct_scale_x
                self.scale_y = correct_scale_y
                self.rotation = correct_rotate
                clog.info("Change reference template method then try again.")

    @staticmethod
    def pair_to_template(temp_qc, temp_re, threshold=10, dis=False):
        """
        one point of temp0 map to only one point of temp1
        Args:
            dis: 距离测量
        """
        import scipy.spatial as spt

        temp_src = np.array(temp_re)[:, :2]
        temp_dst = np.array(temp_qc)[:, :2]
        tree = spt.cKDTree(data=temp_src)
        distance, index = tree.query(temp_dst, k=1)

        if isinstance(threshold, int):
            thr_index = index[distance < threshold]
            points_qc = temp_dst[distance < threshold]
        elif isinstance(threshold, list):
            threshold1, threshold2 = threshold
            thr_index = index[(threshold1 < distance) & (distance < threshold2)]
            points_qc = temp_dst[(threshold1 < distance) & (distance < threshold2)]

        points_re = np.array(temp_re)[thr_index]

        if dis:
            return distance
        else:
            return [points_re, points_qc]


if __name__ == '__main__':
    import h5py

    ipr_path = r"E:\hanqingju\BigChip_QC\SS200001018TR_C1D3\SS200001018TR_C1D3_20230524_175345_0.1.ipr"
    pts = {}
    with h5py.File(ipr_path) as conf:
        qc_pts = conf['QCInfo/CrossPoints/']
        for i in qc_pts.keys():
            pts[i] = conf['QCInfo/CrossPoints/' + i][:]
        loc = conf['Research/Stitch/StitchFovLocation'][...]

    chipno = [[240, 300, 330, 390, 390, 330, 300, 240, 420],
              [240, 300, 330, 390, 390, 330, 300, 240, 420]]

    # chipno = [[112, 144, 208, 224, 224, 208, 144, 112, 160],
    #           [112, 144, 208, 224, 224, 208, 144, 112, 160]]

    tr = TemplateReferenceV2()

    tr.set_chipno(chipno)
    tr.set_fov_location(loc)
    tr.set_qc_points(pts)

    tr.reference_template_v2()

    dct = tr.get_template_eval()
    mat = tr.get_global_eval()
    print(1)
