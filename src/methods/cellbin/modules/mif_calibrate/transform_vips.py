import os
import tifffile
import numpy as np
import math

try:
    # vipshome = r'D:\software\vips-dev-w64-all-8.12.2\vips-dev-8.12\bin'
    # os.environ['PATH'] = vipshome + ';' + os.environ['PATH']
    import pyvips
except ImportError:
    print('Import pyvips ERROR.')

# map vips formats to np dtypes
format_to_dtype = {
    'uchar': np.uint8,
    'char': np.int8,
    'ushort': np.uint16,
    'short': np.int16,
    'uint': np.uint32,
    'int': np.int32,
    'float': np.float32,
    'double': np.float64,
    'complex': np.complex64,
    'dpcomplex': np.complex128,
}

# map np dtypes to vips
dtype_to_format = {
    'uint8': 'uchar',
    'int8': 'char',
    'uint16': 'ushort',
    'int16': 'short',
    'uint32': 'uint',
    'int32': 'int',
    'float32': 'float',
    'float64': 'double',
    'complex64': 'complex',
    'complex128': 'dpcomplex',
}


# numpy array to vips image
def numpy2vips(a):
    height, width, bands = a.shape
    linear = a.reshape(width * height * bands)
    vi = pyvips.Image.new_from_memory(linear.data, width, height, bands,
                                      dtype_to_format[str(a.dtype)])
    return vi


# vips image to numpy array
def vips2numpy(vi):
    return np.ndarray(buffer=vi.write_to_memory(),
                      dtype=format_to_dtype[vi.format],
                      shape=[vi.height, vi.width, vi.bands])


def affineMatrix3D(center, shift=[0, 0], scale=1, rotation=0):
    '''
    Compute the affine transformation matrix by the angle and scale
    :param center: (cx,cy)
    :param angle: rotation
    :param scale: scale
    :return: matrix 3x3
    '''
    w, h = center[0], center[1]
    # 1. 旋转
    angle = math.pi * (rotation / 180)
    # 2. 绕中心点旋转
    # w1 = w * (1 - np.cos(angle)) - h * np.sin(angle)
    # h1 = w * np.sin(angle) + h * (1 - np.cos(angle))
    alpha = scale * math.cos(angle)
    belta = scale * math.sin(angle)
    w1 = (1 - alpha) * center[0] - belta * center[1]
    h1 = (belta * center[0]) + (1 - alpha) * center[1]
    M_rotation = np.array([[alpha, belta, w1],
                           [-belta, alpha, h1],
                           [0, 0, 1]], dtype=np.float32)
    # 尺度缩放
    # M_scale = np.array([[scale[0], 0, 0],
    #                     [0, scale[1], 0],
    #                     [0, 0, 1]], dtype=np.float32)
    # 平移
    M_shift = np.array([[1, 0, shift[1]],
                        [0, 1, shift[0]],
                        [0, 0, 1]], dtype=np.float32)

    M = M_shift @ M_rotation
    return M


class PyvipsImage(pyvips.Image):
    def __init__(self) -> None:
        self.mat = None

    def regist(self, flip=0, rot_type=None, offset=None, dst_shape=None):
        h, w = dst_shape
        x, y = offset
        arr = self.new_from_array(self.mat)
        if flip: arr = arr.fliphor()
        if rot_type is not None:
            theta = math.radians(-rot_type * 90)
            m = [np.cos(theta), -np.sin(theta), np.sin(theta), np.cos(theta)]
            arr = arr.affine(m, interpolate=pyvips.Interpolate.new("nearest"),
                             background=[0])

        arr = arr.affine([1, 0, 0, 1], interpolate=pyvips.Interpolate.new("nearest"),
                         idx=x, idy=y, oarea=[0, 0, w, h])

        mask = vips2numpy(arr)

        if mask.ndim == 3:
            if mask.shape[2] != 3:
                mask = mask[:, :, 0]
        return mask

    def calibrate(self, scale_x, scale_y, rotation, offset=None, dst_shape=None):
        h, w = dst_shape
        # x, y = offset
        # rotation = -rotation  # 与QC角度值相反
        arr = self.new_from_array(self.mat)
        # theta = math.radians(rotation)
        # m = [scale_x * np.cos(theta), -scale_x * np.sin(theta), scale_y * np.sin(theta), scale_y * np.cos(theta)]
        mat_np = vips2numpy(arr)
        mat_h, mat_w = mat_np.shape[:2]
        center = [int(mat_w / 2), int(mat_h / 2)]
        offset = [offset[0], int(offset[1])]
        H = affineMatrix3D(center, offset, scale_x, rotation)
        arr = arr.affine((H[0][0], H[0][1], H[1][0], H[1][1]),
                         odx=H[0][2],
                         ody=H[1][2],
                         oarea=[0, 0, w, h],
                         interpolate=pyvips.Interpolate.new("nearest"),
                         background=[0])

        # arr = arr.affine([1, 0, 0, 1], interpolate=pyvips.Interpolate.new("nearest"),
        #                  odx=x, ody=y, oarea=[0, 0, w, h])

        mask = vips2numpy(arr)

        if mask.ndim == 3:
            if mask.shape[2] != 3:
                mask = mask[:, :, 0]
        return mask

    def transform(self, scale_x, scale_y, rotation, stitch_shape, padding_shape):
        _h, _w = stitch_shape
        rotation = -rotation  # 与QC角度值相反
        if self.mat.ndim == 2:
            arr = self.new_from_array(self.mat)
        else:
            if self.mat.shape[0] == 2 or self.mat.shape[0] == 3:
                arr = self.new_from_array(self.mat[0, :, :])
            else:
                arr = self.new_from_array(self.mat[:, :, 0])
        theta = math.radians(rotation)
        m = [scale_x * np.cos(theta), -scale_x * np.sin(theta), scale_y * np.sin(theta), scale_y * np.cos(theta)]
        arr = arr.affine(m, interpolate=pyvips.Interpolate.new("nearest"), background=[0])
        mask = vips2numpy(arr)

        mask = mask[padding_shape[0]:padding_shape[0] + _h,
               padding_shape[1]:padding_shape[1] + _w]
        if mask.ndim == 3:
            mask = mask.reshape([mask.shape[0], mask.shape[1]])
        return mask

    def _transform(self, scale_x=1, scale_y=1, rotation=0, vipsImage=False):

        if isinstance(self.mat, np.ndarray):
            arr = self.new_from_array(self.mat)
        else:
            arr = self.mat
        theta = math.radians(rotation)
        m = [scale_x * np.cos(theta), -scale_x * np.sin(theta), scale_y * np.sin(theta), scale_y * np.cos(theta)]
        arr = arr.affine(m, interpolate=pyvips.Interpolate.new("nearest"), background=[0])
        if vipsImage:
            return arr
        else:
            mask = vips2numpy(arr)
            if mask.ndim == 3:
                if mask.shape[2] != 3:
                    mask = mask[:, :, 0]

            return mask

    # def _transform(self, scale_x, scale_y, rotation):
    #     if self.mat.ndim == 2:
    #         arr = self.new_from_array(self.mat)
    #     else:
    #         if self.mat.shape[0] == 2 or self.mat.shape[0] == 3:
    #             arr = self.new_from_array(self.mat[0, :, :])
    #         else:
    #             arr = self.new_from_array(self.mat[:, :, 0])
    #     theta = math.radians(rotation)
    #     m = [scale_x * np.cos(theta), -scale_x * np.sin(theta), scale_y * np.sin(theta), scale_y * np.cos(theta)]
    #     arr = arr.affine(m, interpolate=pyvips.Interpolate.new("nearest"), background=[0])
    #     mask = vips2numpy(arr)

    #     if mask.ndim == 3:
    #         mask = mask.reshape([mask.shape[0], mask.shape[1]])

    #     return mask

    def to_array(self, ):
        arr = vips2numpy(self.mat)
        if arr.ndim == 3:
            arr = arr[:, :, 0]
        return arr

    def save(self, save_path):
        tifffile.imwrite(save_path, self.mat)

    def load(self, file_path):
        # self.mat = tifffile.imread(file_path)
        self.mat = self.new_from_file(file_path)


# def count_time(func):
#     def int_time():
#         start_time = time.time()
#         func()
#         over_time = time.time()
#         total_time = over_time - start_time
#         print("程序运行了%s秒" % total_time)
#     return int_time

# def count_info(func):
#     def float_info():
#         pid = os.getpid()
#         p = psutil.Process(pid)
#         info_start = p.memory_full_info().uss/1024
#         func()
#         info_end=p.memory_full_info().uss/1024
#         print("程序占用了内存"+str(info_end-info_start)+"KB")
#     return float_info

# @count_time
# @profile
# def Transform(stitch_img):
#     i = Image()
#     i.mat = stitch_img
#     i.load('/hwfssz1/ST_BIOINTEL/P20Z10200N0039/06.user/lhl/02.data/temp/marmoset_stitch/T33/2_stitch/fov_stitched.tif')
#     i.transform(scale_x=1.313316299761574, scale_y=1.313316299761574, rotation=0.15923469957198236)

# if __name__ == '__main__':
#     # main()
#     t1 = threading.Thread(target=main)
#     t1.start()

#     thread_num = len(threading.enumerate())
#     print("主线程：线程数量是%d" % thread_num)
if __name__ == '__main__':
    import cv2

    img = cv2.imread(r"D:\Data\imagestudio\fov_stitched_transformed.tif")
    pi = PyvipsImage()
    flip = 1
    rot_type = 3
    offset = None
    dst_shape = None
    pi.regist()
