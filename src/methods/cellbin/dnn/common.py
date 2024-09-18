from cellbin.image.augmentation import f_resize
from cellbin.image.augmentation import f_padding


# TODO: tceg utils/general.py 17

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114)):
    """
    Resize and pad image while meeting stride-multiple constraints
    该模块将非正方形图片先以长边与new_shape的比例进行resize, 再补齐短边形成正方形
    如果是正方形图片, 就是直接resize成new_shape
    Returns:
        im (array): (height, width, 3)
        ratio (array): [w_ratio, h_ratio]
        (dw, dh) (array): [w_padding h_padding]
    """
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):  # [h_rect, w_rect]
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    # Compute padding
    new_unpad = int(round(shape[0] * r)), int(round(shape[1] * r)),  # w h 这一步得到新的图片大小(未做padding)
    dw, dh = new_shape[1] - new_unpad[1], new_shape[0] - new_unpad[0]  # wh padding 通过新的大小与new_unpad的差值可得出还差多少

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = f_resize(im, new_unpad, mode='BILINEAR')

    # test_img = f_padding(im, new_shape, 'constant', constant_values=color)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im_pad = f_padding(im, top, bottom, left, right, value=color[0])
    # im_ori = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im_pad


# TODO: tceg utils/general.py 42

def scale_polys(img1_shape, polys, img0_shape, box=False, ratio_pad=None):
    # ratio_pad: [(h_raw, w_raw), (hw_ratios, wh_paddings)]
    # Rescale coords (xyxyxyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = resized / raw
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]  # h_ratios
        pad = ratio_pad[1]  # wh_paddings
    if box:
        polys[..., [0, 2]] -= pad[0]  # x padding
        polys[..., [1, 3]] -= pad[1]  # y padding
        polys[..., :4] /= gain
        clip_boxes(boxes, img0_shape)  # TODO: cseg implement this
    else:
        polys[:, [0, 2, 4, 6]] -= pad[0]  # x padding
        polys[:, [1, 3, 5, 7]] -= pad[1]  # y padding
        polys[:, :8] /= gain  # Rescale poly shape to img0_shape
    # clip_polys(polys, img0_shape)
    return polys
