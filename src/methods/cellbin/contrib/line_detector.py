import math
import random
import copy
import numpy as np
from scipy.signal import argrelextrema
from sklearn.linear_model import LinearRegression

from cellbin.contrib import Line

model = LinearRegression()


def random_color():
    b = random.randint(0, 256)
    g = random.randint(0, 256)
    r = random.randint(0, 256)
    return b, g, r


class TrackLineDetector(object):
    def __init__(self):
        self.grid = 100

    def generate(self, arr):
        """
        This algorithm will not work the angle of image is more than 8 degree

        Args:
            arr (): 2D array in uint 8 or uint 16

        Returns:
            h_lines: horizontal line
            v_lines: vertical line

        """
        h_lines, v_lines = [], []

        # horizontal direction
        horizontal_candidate_pts = self.create_candidate_pts(arr, 'x')
        h_angle = self.integer_angle(horizontal_candidate_pts, 'x')
        if h_angle != -1000:
            horizontal_pts = self.select_pts_by_integer_angle(horizontal_candidate_pts, h_angle, tolerance=1)
            if len(horizontal_pts) != 0:
                horizontal_color_pts = self.classify_points(horizontal_pts, h_angle, tolerance=1)
                h_lines = self.points_to_line(horizontal_color_pts, tolerance=3)

        # vertical direction
        vertical_candidate_pts = self.create_candidate_pts(arr, 'y')
        v_angle = self.integer_angle(vertical_candidate_pts, 'y')
        if v_angle != -1000:
            vertical_pts = self.select_pts_by_integer_angle(vertical_candidate_pts, v_angle, tolerance=1)
            if len(vertical_pts) != 0:
                vertical_color_pts = self.classify_points(vertical_pts, v_angle, tolerance=1)
                v_lines = self.points_to_line(vertical_color_pts, tolerance=3)

        return h_lines, v_lines

    @staticmethod
    def points_to_line(dct, tolerance=2):
        lines = list()
        for k, v in dct.items():
            # 少于两个的拟合不成直线
            if len(v) > tolerance:
                tmp = np.array(v)
                model.fit(tmp[:, 0].reshape(-1, 1), tmp[:, 1])  # TODO: 90度直线拟合会有问题, need to be fixed
                line = Line()
                # 一个点，加上coef拟合直线
                line.init_by_point_k(v[0], model.coef_[0])
                lines.append(line)
        return lines

    def classify_points(self, candidate_pts, base_angle, tolerance=2):
        pts = copy.copy(candidate_pts)
        ind = 0
        dct = dict()
        while (len(pts) > 1):
            pts_, index = self.angle_line(base_angle, pts, tolerance)
            # 将找到的同一直线的点放入dct中
            dct[ind] = pts_
            # 根据angle_line中返回的index删除已找到的点
            pts = np.delete(np.array(pts), index, axis=0).tolist()
            ind += 1
        # 返回存有点的分类的dct
        return dct

    @staticmethod
    def angle_line(angle, points, tolerance=2):
        # 找跟points[0]在一条直线的点
        count = len(points)
        orignal_point = points[0]
        points_ = [points[0]]
        index = [0]
        for i in range(1, count):
            p = points[i]
            line = Line()
            line.init_by_point_pair(orignal_point, p)
            diff = abs(line.rotation() - angle)
            diff = (diff > 90) and (180 - diff) or diff
            if diff < tolerance:
                points_.append(p)
                index.append(i)
        # 返回这条线的所有点，以及对应的index号，index用于之后删除array中已找到的点
        return points_, index

    @staticmethod
    def select_pts_by_integer_angle(candidate_pts, base_angle, tolerance=2):
        # 过滤
        x_count = len(candidate_pts)
        # pts用来储存该方向的所有点
        pts = list()
        for i in range(0, x_count - 1):
            if len(candidate_pts[i]) > 100:
                continue
            # 遍历所有的采样区域
            pts_start = candidate_pts[i]
            pts_end = candidate_pts[i + 1]
            # 遍历pts_start中的所有的点
            for p0 in pts_start:
                # 遍历pts_end中所有点，计算p0与各点的距离
                d = [math.sqrt(pow(p0[0] - p1[0], 2) + pow(p0[1] - p1[1], 2)) for p1 in pts_end]
                # 取绝对值
                d_ = np.abs(d)
                # 找到最短的距离的index
                ind = np.where(d_ == np.min(d_))[0]
                line = Line()
                # 计算角度
                line.init_by_point_pair(p0, pts_end[ind[0]])
                # 如果角度小于tol，
                if abs(line.rotation() - base_angle) <= tolerance: pts.append(p0)
        return pts

    @staticmethod
    def integer_angle(pts, derection='x'):
        angle = -1000
        x_count = len(pts)
        # angles将保存该方向上所有采样区域的角度
        angles = list()
        for i in range(0, x_count - 1):
            if len(pts[i]) > 100:
                continue
            # 相邻的两个采样区域
            pts_start = pts[i]
            pts_end = pts[i + 1]
            for p0 in pts_start:
                # pts_start中的一个点p0对应pts_end的所有点的euclidean distance
                d = [math.sqrt(pow(p0[0] - p1[0], 2) + pow(p0[1] - p1[1], 2)) for p1 in pts_end]
                # 取绝对值
                d_ = np.abs(d)
                # 找到pts_start的该点的最小距离的index
                ind = np.where(d_ == np.min(d_))[0]
                line = Line()
                # 将p0和对应的最小距离的点输入到line.init_by_point_pair
                # 可以得到线的coef和bias
                line.init_by_point_pair(p0, pts_end[ind[0]])
                # rotation方法运用上面得到的coef计算该直线的角度
                # 并将角度记录
                angles.append(round(line.rotation()))
        if len(angles) != 0:
            x = np.array(angles) - np.min(angles)
            # 得到该方向上的角度（出现最多的角度）
            angle = np.argmax(np.bincount(x)) + np.min(angles)
        return angle

    def create_candidate_pts(self, mat, derection='x'):
        pts = list()
        h, w = mat.shape
        # direction x -> h
        # direction y -> w
        counter = (derection == 'x' and h or w)
        # self.grid -> defined by user, 采样间隔
        for i in range(0, counter, self.grid):
            # i -> 从0到h或者w
            # 设置t为当前采样间隔的x或者y
            t = i + self.grid / 2
            if derection == 'x':
                # 区域 -> i到i+采样间隔的区域
                region_mat = mat[i: i + self.grid, :w]
                # 如果区域不是取样规定长度，继续
                if region_mat.shape[0] != self.grid:
                    continue
                # 对y方向进行像素求和，并除以规定的采样间隔，可以看做成normalization
                line = np.sum(region_mat, axis=0) / self.grid
            else:
                # 这里的处理与上述基本一样，只是这里是方向y的情况
                region_mat = mat[:h, i: i + self.grid]
                if region_mat.shape[1] != self.grid:
                    continue
                line = np.sum(region_mat, axis=1) / self.grid
            # 找到该条线上的极值（最小值）
            p = argrelextrema(line, np.less_equal, order=100)
            # print(p[0].shape)
            if derection == 'x':
                pt = [[p, t] for p in p[0]]
            else:
                pt = [[t, p] for p in p[0]]
            # pts中保存的为该方向的所有点
            pts.append(pt)
        return pts


def main():
    import cv2
    image_path = r"D:\Data\tmp\Y00035MD\Y00035MD\Y00035MD_0000_0004_2023-01-30_15-50-41-868.tif"
    arr = cv2.imread(image_path, -1)
    ftl = TrackLineDetector()
    result = ftl.generate(arr)


if __name__ == '__main__':
    main()
