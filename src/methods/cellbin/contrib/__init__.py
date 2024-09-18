import math

from .template_reference import TemplateReference
from .template_reference_v2 import TemplateReferenceV2
from .fov_aligner import FOVAligner
from .global_location import GlobalLocation


infinity = 0.00000001


def rotate(pt, angle, ori_w, ori_h, new_w, new_h):
    px, py = pt
    cx = int(new_w / 2)
    cy = int(new_h / 2)
    theta = angle
    rad = math.radians(theta)
    new_px = cx + float(px - cx) * math.cos(rad) + float(py - cy) * math.sin(rad)
    new_py = cy + -(float(px - cx) * math.sin(rad)) + float(py - cy) * math.cos(rad)
    x_offset, y_offset = (ori_w - new_w) / 2, (ori_h - new_h) / 2
    new_px += x_offset
    new_py += y_offset
    return int(new_px), int(new_py)


class Line(object):
    def __init__(self, ):
        self.coefficient = None
        self.bias = None
        self.index = 0

    def two_points(self, shape):
        h, w = shape
        if self.coefficient >= 0:
            pt0 = self.get_point_by_x(0)
            pt1 = self.get_point_by_x(w)
        else:
            pt0 = self.get_point_by_y(0)
            pt1 = self.get_point_by_y(h)
        return [pt0, pt1]

    def set_coefficient_by_rotation(self, rotation):
        self.coefficient = math.tan(math.radians(rotation))

    def init_by_point_pair(self, pt0, pt1):
        x0, y0 = pt0
        x1, y1 = pt1
        if x1 > x0:
            self.coefficient = (y1 - y0) / (x1 - x0)
        elif x1 == x0:
            self.coefficient = (y0 - y1) / infinity
        else:
            self.coefficient = (y0 - y1) / (x0 - x1)
        self.bias = y0 - self.coefficient * x0

    def init_by_point_k(self, pt0, k):
        x0, y0 = pt0
        self.coefficient = k
        self.bias = y0 - k * x0

    def rotation(self, ):
        return math.degrees(math.atan(self.coefficient))

    def get_point_by_x(self, x):
        return [x, self.coefficient * x + self.bias]

    def get_point_by_y(self, y):
        return [(y - self.bias) / self.coefficient, y]

    def line_rotate(self, angle, ori_w, ori_h, new_w, new_h):
        shape = (new_h, new_w)
        p0, p1 = self.two_points(
            shape=shape
        )
        p0_new = rotate(
            p0,
            angle,
            ori_w, ori_h, new_w, new_h
        )
        p1_new = rotate(
            p1,
            angle,
            ori_w, ori_h, new_w, new_h
        )
        self.init_by_point_pair(p0_new, p1_new)
        return self
