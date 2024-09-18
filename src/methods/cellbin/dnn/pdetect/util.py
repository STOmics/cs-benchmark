import numpy as np
from scipy.spatial.distance import cdist
import time

pi = 3.141592


def rbox2poly(obboxes):
    """
    Trans rbox format to poly format.
    Args:
        rboxes (array/tensor): (num_gts, [cx cy l s θ]) θ∈[-pi/2, pi/2)

    Returns:
        polys (array/tensor): (num_gts, [x1 y1 x2 y2 x3 y3 x4 y4])
    """

    center, w, h, theta = np.split(obboxes, (2, 3, 4), axis=-1)
    Cos, Sin = np.cos(theta), np.sin(theta)

    vector1 = np.concatenate(
        [w / 2 * Cos, -w / 2 * Sin], axis=-1)
    vector2 = np.concatenate(
        [-h / 2 * Sin, -h / 2 * Cos], axis=-1)

    point1 = center + vector1 + vector2
    point2 = center + vector1 - vector2
    point3 = center - vector1 - vector2
    point4 = center - vector1 + vector2
    order = obboxes.shape[:-1]
    return np.concatenate(
        [point1, point2, point3, point4], axis=-1).reshape(*order, 8)


def nms_rotate_cpu(boxes, scores, iou_threshold, max_output_size):
    """
    boxes: format (x_c, y_c, w, h, theta)
    scores: scores of all boxes
    iou_threshold:
    max_output_size: max number of output
    return: the remaining index of box
    """
    keep = []
    order = scores.argsort()[::-1]  # 对score进行从高到低排序排序, 返回index
    num = boxes.shape[0]  # 一共有多少个bbx
    suppressed = np.zeros((num), dtype=int)  # 记录不要的box的index
    for _i in range(num):
        if len(keep) >= max_output_size:  # 当输出数量达到最大值结束
            break
        i = order[_i]  # 当前index
        if suppressed[i] == 1:  # 如果在不要的里面, 那么就继续
            continue
        keep.append(i)  # 放入到keep中
        tmp_dist = cdist(boxes[i, :2].reshape(-1, 2), boxes[order[_i + 1:], :2])
        tmp = order[_i + 1:][np.where(tmp_dist <= np.min(boxes[i, 2: 4]))[1]]
        suppressed[tmp] = 1
    return np.array(keep, np.int64)


def obb_nms_np(dets, scores, iou_thr):
    """
    RIoU NMS - iou_thr.
    Args:
        dets (tensor/array): (num, [cx cy w h θ]) θ∈[-pi/2, pi/2)
        scores (tensor/array): (num)
        iou_thr (float): (1)
    Returns:
        dets (tensor): (n_nms, [cx cy w h θ])
        inds (tensor): (n_nms), nms index of dets
    """

    dets_np = dets
    scores_np = scores
    if dets_np.size == 0:  # len(dets)
        inds = np.zeros(0, dtype=np.int64)
    else:
        # same bug will happen when bboxes is too small
        too_small_np = dets_np[:, [2, 3]].min(1) < 0.001
        if too_small_np.all():  # all the bboxes is too small
            inds = np.zeros(0, dtype=np.int64)
        else:
            ori_inds_np = np.arange(dets_np.shape[0])
            ori_inds_np = ori_inds_np[~too_small_np]
            dets_np = dets_np[~too_small_np]
            scores_np = scores_np[~too_small_np]
            inds = nms_rotate_cpu(dets_np, scores_np, iou_thr, max_output_size=1000)
            inds = ori_inds_np[inds]
    return dets[inds, :], inds


def non_max_suppression_obb_np(
        prediction,
        conf_thres=0.25,
        iou_thres=0.45,
        agnostic=False,
        multi_label=False,
        max_det=1500
):
    prediction_np = prediction
    nc_np = prediction_np.shape[2] - 5 - 180
    xc_np = prediction_np[..., 4] > conf_thres
    class_index_np = nc_np + 5

    # Checks
    assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
    assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'

    # Settings
    max_wh = 4096  # min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 30.0  # seconds to quit after
    # redundant = True  # require redundant detections
    multi_label &= nc_np > 1  # multiple labels per box (adds 0.5ms/img)

    t = time.time()
    output_np = [np.zeros((0, 7))] * prediction_np.shape[0]

    for xi, x_ in enumerate(prediction_np):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x_np = x_[xc_np[xi]]
        # Cat apriori labels if autolabelling

        # If none remain process next image
        if not x_np.shape[0]:
            continue

        # Compute conf
        x_np[:, 5:class_index_np] *= x_np[:, 4:5]

        theta_pred_np = np.argmax(x_np[:, class_index_np:], 1, keepdims=True)
        theta_pred_np = np.float32(theta_pred_np - 90) / 180 * pi

        # Detections matrix nx7 (xyls, θ, conf, cls) θ ∈ [-pi/2, pi/2)
        i_np, j_np = np.where((x_np[:, 5:class_index_np] > conf_thres) == True)
        x_np = np.concatenate(
            (x_np[i_np, :4], theta_pred_np[i_np], x_np[i_np, j_np + 5, None], np.float16(j_np[:, None])), 1)

        # Check shape
        n_np = x_np.shape[0]
        if not n_np:  # no boxes
            continue
        elif n_np > max_nms:  # excess boxes
            x_np = x_np[np.argsort(x_np[:, 5])[::-1][:max_nms]]

        # Batched NMS
        c_np = x_np[:, 6:7] * (0 if agnostic else max_wh)  # classes
        rboxes_np = np.copy(x_np[:, :5])
        rboxes_np[:, :2] = rboxes_np[:, :2] + c_np
        scores_np = x_np[:, 5]
        _, i = obb_nms_np(rboxes_np, scores_np, iou_thres)
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]

        output_np[xi] = x_np[i]
        if (time.time() - t) > time_limit:
            print(f'WARNING: NMS time limit {time_limit}s exceeded')
            break  # time limit exceeded
    return output_np
