from cellbin.image.augmentation import f_histogram_normalization,f_ij_16_to_8

import numpy as np
import cv2


def f_preformat(img, half=False, is_nhwc=False):
    if half and img.dtype != np.float16:
        img = np.asarray(img, dtype=np.float16)
    if is_nhwc:
        img = img.permute(0, 2, 3, 1)  # torch BCHW to numpy BHWC shape(1,320,192,3)
    return img


def f_img_process(img, img_shape=(640, 640)):
    img = f_ij_16_to_8(img)
    img = cv2.resize(img, img_shape, interpolation=cv2.INTER_LINEAR)
    if len(img.shape) == 2:
        img = np.expand_dims(img, axis=2)
        img = np.concatenate((img, img, img), axis=-1)
    img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    # img = f_histogram_normalization(img)
    img = np.divide(img, 255.0)
    img = np.array(img).astype(np.float32)
    img = np.ascontiguousarray(img)  # contiguous
    if len(img.shape) == 3:
        img = img[None]  # expand for batch dim

    return img


def _f_crop_mask(masks, boxes):
    """
    "Crop" predicted masks by zeroing out everything not in the predicted bbox.
    Vectorized by Chong (thanks Chong).

    Args:
        - masks should be a size [h, w, n] tensor of masks
        - boxes should be a size [n, 4] tensor of bbox coords in relative point form
    """

    n, h, w = masks.shape
    x1, y1, x2, y2 = np.split(boxes[:, :, None], 4, 1)  # x1 shape(1,1,n)
    r = np.arange(w, dtype=x1.dtype)[None, None, :]  # rows shape(1,w,1)
    c = np.arange(h, dtype=x1.dtype)[None, :, None]  # cols shape(h,1,1)
    return masks * ((r >= x1) * (r < x2) * (c >= y1) * (c < y2))


def _f_sigmoid(Z):
    return 1 / (1 + np.exp(-Z))  # TODO: Overflow


def f_process_mask(protos, masks_in, bboxes, shape, upsample=False):
    """
    Crop before upsample.
    proto_out: [mask_dim, mask_h, mask_w]
    out_masks: [n, mask_dim], n is number of masks after nms
    bboxes: [n, 4], n is number of masks after nms
    shape:input_image_size, (h, w)

    return: h, w, n
    """
    c, mh, mw = protos.shape  # CHW
    ih, iw = shape
    masks = _f_sigmoid((masks_in @ np.asarray(protos, dtype=float).reshape(c, -1))).reshape(-1, mh, mw)
    downsampled_bboxes = bboxes.copy()
    downsampled_bboxes[:, 0] *= mw / iw
    downsampled_bboxes[:, 2] *= mw / iw
    downsampled_bboxes[:, 3] *= mh / ih
    downsampled_bboxes[:, 1] *= mh / ih
    masks = _f_crop_mask(masks, downsampled_bboxes)  # CHW
    black = np.zeros((masks.shape[0], ih, iw))
    if upsample:
        for i in range(masks.shape[0]):
            masks2 = cv2.resize(masks[i, :, :], (iw, ih))
            masks2 = np.expand_dims(masks2, axis=0)
            masks2[masks2 > 0.5] = 1
            black = black + masks2
    return black


def f_scale_image(im1_shape, masks, im0_shape, ratio_pad=None):
    """
    img1_shape: model input shape, [h, w]
    img0_shape: origin pic shape, [h, w, 3]
    masks: [h, w, num]
    """
    if ratio_pad is None:  # calculate from im0_shape
        gain = min(im1_shape[0] / im0_shape[0], im1_shape[1] / im0_shape[1])  # gain  = old / new
        pad = (im1_shape[1] - im0_shape[1] * gain) / 2, (im1_shape[0] - im0_shape[0] * gain) / 2  # wh padding
    else:
        pad = ratio_pad[1]
    top, left = int(pad[1]), int(pad[0])  # y, x
    bottom, right = int(im1_shape[0] - pad[1]), int(im1_shape[1] - pad[0])
    if len(masks.shape) < 2:
        raise ValueError(f'"len of masks shape" should be 2 or 3, but got {len(masks.shape)}')
    masks = masks[top:bottom, left:right]
    masks = cv2.resize(masks, (im0_shape[1], im0_shape[0]))
    if len(masks.shape) == 2:
        masks = masks[:, :, None]
    return masks
