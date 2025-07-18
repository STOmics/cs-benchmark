import os.path

import cv2
import h5py
import cv2 as cv
import numpy as np
from typing import Union

import tifffile
from scipy.spatial.distance import cdist

from cellbin2.image import CBImage, cbimread
from cellbin2.contrib.alignment.basic import transform_points
from cellbin2.image.augmentation import f_ij_16_to_8, dapi_enhance, he_enhance
from cellbin2.contrib.alignment.basic import TemplateInfo, ChipBoxInfo
from cellbin2.contrib.template.inferencev1 import TemplateReferenceV1

pt_enhance_method = {
    'ssDNA': dapi_enhance,
    'DAPI': dapi_enhance,
    'HE': he_enhance
}


def get_tissue_corner_points(
        tissue_data: np.ndarray,
        k: int = 9
):
    _tissue = tissue_data.copy()

    _tissue = cv.dilate(
        _tissue,
        kernel=np.ones([k, k], dtype=np.uint8)
    )

    contours, _ = cv.findContours(
        _tissue,
        cv.RETR_EXTERNAL,
        cv.CHAIN_APPROX_NONE
    )

    max_contours = sorted(contours, key=cv.contourArea, reverse=True)[0]

    x, y, w, h = cv.boundingRect(max_contours)

    corner_points = np.array(
        [[x, y], [x, y + h], [x + w, y], [x + w, y + h]]
    )

    dis = cdist(
        max_contours.squeeze(),
        corner_points
    )

    result_points = max_contours.squeeze()[np.argmin(dis, axis=0)]

    return result_points


def crop_image(corner_temp_points, points, image,
               image_size, image_type,
               draw_radius, template_color, draw_thickness
               ):
    cp_image_list = list()
    coord_list = list()
    for cp in corner_temp_points:
        x, y = map(int, cp[:2])
        if x <= image_size // 2:
            x_left = 0
            x_right = image_size
        elif x + image_size // 2 > image.width:
            x_left = image.width - image_size
            x_right = image.width
        else:
            x_left = x - image_size // 2
            x_right = x + image_size // 2

        if y <= image_size // 2:
            y_up = 0
            y_down = image_size
        elif y + image_size // 2 > image.height:
            y_up = image.height - image_size
            y_down = image.height
        else:
            y_up = y - image_size // 2
            y_down = y + image_size // 2

        _ci = image.crop_image([y_up, y_down, x_left, x_right])
        coord_list.append([y_up, y_down, x_left, x_right])
        enhance_func = pt_enhance_method.get(image_type, "DAPI")
        _ctp = [i for i in points if (i[0] > x_left) and
                (i[1] > y_up) and
                (i[0] < x_right) and
                (i[1] < y_down)]
        _ctp = np.array(_ctp)[:, :2] - [x_left, y_up]
        _ci = enhance_func(_ci)

        for i in _ctp:
            cv.circle(_ci, list(map(int, i))[:2],
                      draw_radius * 2, template_color, draw_thickness)
        cp_image_list.append(_ci)

    return cp_image_list, coord_list


def template_painting(
        image_data: Union[str, np.ndarray, CBImage],
        tissue_seg_data: Union[str, np.ndarray, CBImage],
        image_type: str,
        qc_points: np.ndarray = None,
        template_points: np.ndarray = None,
        image_size: int = 2048,
        track_color: tuple = (0, 0, 255),
        template_color: tuple = (0, 255, 0),
        chip_rect_color: tuple = (0, 255, 255),
        tissue_rect_color: tuple = (255, 255, 0),
        draw_radius: int = 5,
        draw_thickness: int = 2
) -> Union[np.ndarray, list, list]:
    """

    Args:
        image_data:
        tissue_seg_data:
        image_type: str -- ssDNA, DAPI, HE
        qc_points:
        template_points:
        image_size: image height size
        track_color:
        template_color:
        chip_rect_color:
        tissue_rect_color:
        draw_radius:
        draw_thickness:

    Returns:

    """
    image = cbimread(image_data)
    if image_type in ['DAPI', 'ssDNA']:
        image = image.to_gray()

    tissue_image = cbimread(tissue_seg_data)

    _temp, _track = TemplateReferenceV1.pair_to_template(
        qc_points, template_points
    )

    corner_points = np.array([[0, 0], [0, image.height],
                              [image.width, image.height], [image.width, 0]])

    points_dis = cdist(_temp[:, :2], corner_points)
    corner_temp_points = _temp[np.argmin(points_dis, axis=0)]

    tissue_corner_points = get_tissue_corner_points(tissue_image.image)
    tissue_points_dis = cdist(_temp[:, :2], tissue_corner_points)
    corner_tissue_temp_points = _temp[np.argmin(tissue_points_dis, axis=0)]

    ########################
    # 芯片角最近track点
    cp_image_list, cp_coord_list = crop_image(
        corner_temp_points, _temp, image,
        image_size, image_type,
        draw_radius, template_color, draw_thickness
    )
    # 组织边缘track点
    tissue_image_list, tissue_coord_list = crop_image(
        corner_tissue_temp_points, _temp, image,
        image_size, image_type,
        draw_radius, template_color, draw_thickness
    )
    ########################

    track_list = _track.tolist()
    _unpair = [i for i in qc_points[:, :2].tolist() if i not in track_list]

    rate = image_size / image.image.shape[0]
    _image = image.resize_image(rate)

    if len(_image.image.shape) == 2:
        _image = cv.cvtColor(f_ij_16_to_8(_image.image), cv.COLOR_GRAY2BGR)
    else:
        _image = f_ij_16_to_8(_image.image)
    for i in np.array(_unpair):
        cv.circle(_image, list(map(int, i * rate)),
                  draw_radius, track_color, draw_thickness)

    for i in np.array(_temp):
        cv.circle(_image, list(map(int, i[:2] * rate)),
                  draw_radius, template_color, draw_thickness)

    for cc in cp_coord_list:
        y0, y1, x0, x1 = cc
        _p = np.array([[x0, y0], [x0, y1], [x1, y1], [x1, y0]], dtype=np.int32)

        cv.polylines(_image, [(_p * rate).astype(np.int32)],
                     True, chip_rect_color, draw_thickness)

    for tc in tissue_coord_list:
        y0, y1, x0, x1 = tc
        _p = np.array([[x0, y0], [x0, y1], [x1, y1], [x1, y0]], dtype=np.int32)

        cv.polylines(_image, [(_p * rate).astype(np.int32)],
                     True, tissue_rect_color, draw_thickness)

    return _image, cp_image_list, tissue_image_list


def chip_box_painting(
        image_data: Union[str, np.ndarray, CBImage],
        chip_info: Union[np.ndarray, ChipBoxInfo] = None,
        ipr_path: str = None,
        layer: str = None,
        image_size: int = 2048,
        chip_color: tuple = (0, 255, 0),
        draw_thickness: int = 5
) -> np.ndarray:
    """

    Args:
        image_data:
        chip_info:
        ipr_path:
        layer:
        image_size:
        chip_color:
        draw_thickness:

    Returns:

    """
    if ipr_path is not None:
        with h5py.File(ipr_path) as conf:
            p_lt = conf[f'{layer}/QCInfo/ChipBBox'].attrs['LeftTop']
            p_lb = conf[f'{layer}/QCInfo/ChipBBox'].attrs['LeftBottom']
            p_rb = conf[f'{layer}/QCInfo/ChipBBox'].attrs['RightBottom']
            p_rt = conf[f'{layer}/QCInfo/ChipBBox'].attrs['RightTop']
        points = np.array([p_lt, p_lb, p_rb, p_rt])

    elif chip_info is not None:
        points = chip_info.chip_box

    else:
        raise ValueError("Chip info not found.")

    image = cbimread(image_data)
    rate = image_size / image.image.shape[0]
    # _image = image.resize_image(rate)
    # _image = cv.cvtColor(f_ij_16_to_8(_image.image), cv.COLOR_GRAY2BGR)

    image = image.image
    image = f_ij_16_to_8(image)
    if len(image.shape) == 2:
        image = cv2.equalizeHist(image)
        image = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_NEAREST)
        image = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
    else:
        image = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_NEAREST)

    points = points * rate
    points = points.reshape(-1, 1, 2)
    cv.polylines(image, [points.astype(np.int32)],
                 True, chip_color, draw_thickness)

    return image


def get_view_image(
        image: Union[np.ndarray, str],
        points: np.ndarray,
        is_matrix: bool = False,
        output_path: str = "",
        size: int = 500,
        color: tuple = (0, 0, 255),
        radius: int = 10,
        thickness: int = 1
):
    """

    Args:
        image:
        points:
        is_matrix:
        size:
        color:
        radius:
        thickness:
        output_path:

    Returns:

    """
    if isinstance(image, str):
        image = tifffile.imread(image)

    image = f_ij_16_to_8(image)

    if is_matrix:
        size *= 2
        radius *= 2

    image_list = list()
    for fp in points:
        x, y = map(lambda k: int(k), fp)
        _x = _y = size

        if x < size:
            _x = x
            x = size

        if y < size:
            _y = y
            y = size

        if x > image.shape[1] - size:
            _x = 2 * size + x - image.shape[1]
            x = image.shape[1] - size

        if y > image.shape[0] - size:
            _y = 2 * size + y - image.shape[0]
            y = image.shape[0] - size

        _image = image[y - size: y + size, x - size: x + size]
        if not is_matrix:
            if _image.ndim == 3:
                _image = cv.cvtColor(cv.bitwise_not(cv.cvtColor(_image, cv.COLOR_BGR2GRAY)), cv.COLOR_GRAY2BGR)
            else:
                _image = cv.cvtColor(cv.equalizeHist(_image), cv.COLOR_GRAY2BGR)
        else:
            _image = cv.filter2D(_image, -1, np.ones((21, 21), np.float32))
            _image = (_image > 0).astype(np.uint8) * 255
            _image = cv.cvtColor(_image, cv.COLOR_GRAY2BGR)

        line1 = np.array([[_x, _y - radius], [_x, _y + radius]], np.int32).reshape((-1, 1, 2))
        line2 = np.array([[_x - radius, _y], [_x + radius, _y]], np.int32).reshape((-1, 1, 2))

        cv.circle(_image, [_x, _y], radius // 2, color[::-1], thickness)

        cv.polylines(_image, pts=[line1, line2], isClosed=False,
                     color=color, thickness=thickness, lineType=cv.LINE_8)

        image_list.append(_image)

    if os.path.isdir(output_path):
        name_list = ["left_up", "left_down", "right_down", "right_up"]
        for name, im in zip(name_list, image_list):
            cv.imwrite(os.path.join(output_path, f"{name}.tif"), im)

    return image_list


if __name__ == '__main__':
    import h5py
    from cellbin2.image import cbimwrite

    get_view_image(image = r"D:\02.data\temp\temp_cellbin2_test\trans_data_1\A00792D3\A00792D3_DAPI_stitch.tif",
                   points = np.loadtxt(r"D:\02.data\temp\temp_cellbin2_test\trans_data_1\A00792D3\A00792D3_DAPI_stitch.txt"),
                   output_path = r"D:\02.data\temp\temp_cellbin2_test\trans_data_1\A00792D3")

    # register_img = "/media/Data/dzh/data/cellbin2/test/SS200000135TL_D1_demo/SS200000135TL_D1_ssDNA_regist.tif"
    # tissue_cut = "/media/Data/dzh/data/cellbin2/test/SS200000135TL_D1_demo/SS200000135TL_D1_ssDNA_tissue_cut.tif"
    # with h5py.File("/media/Data/dzh/data/cellbin2/test/SS200000135TL_D1_demo/SS200000135TL_D1.ipr", "r") as f:
    #     template_points = f["ssDNA"]["Register"]["RegisterTemplate"][...]
    #     track_points = f["ssDNA"]["Register"]["RegisterTrackTemplate"][...]
    # _image, cp_image_list, tissue_image_list = template_painting(
    #     image_data=register_img,
    #     tissue_seg_data=tissue_cut,
    #     image_type="ssDNA",
    #     qc_points=track_points,
    #     template_points=template_points,
    # )
    # cbimwrite("/media/Data/dzh/data/cellbin2/test/SS200000135TL_D1_demo/assets/image/ssDNA_trackpoint.png", _image)
