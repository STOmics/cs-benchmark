import numpy as np

def compute_overlap(boxes, query_boxes):
    N = boxes.shape[0]
    K = query_boxes.shape[0]
    overlaps = np.zeros((N, K), dtype=np.float64)
    for k in range(K):
        box_area = ((query_boxes[k, 2] - query_boxes[k, 0] + 1) * (query_boxes[k, 3] - query_boxes[k, 1] + 1))
        for n in range(N):
            iw = min(boxes[n, 2], query_boxes[k, 2]) - max(boxes[n, 0], query_boxes[k, 0]) + 1
            if iw > 0:
                ih = min(boxes[n, 3], query_boxes[k, 3]) - max(boxes[n, 1], query_boxes[k, 1]) + 1
                if ih > 0:
                    ua = (boxes[n, 2] - boxes[n, 0] + 1) * (boxes[n, 3] - boxes[n, 1] + 1) + box_area - iw * ih
                    overlaps[n, k] = iw * ih / ua
    return overlaps

def compute_overlap_3D(boxes, query_boxes):
    N = boxes.shape[0]
    K = query_boxes.shape[0]
    overlaps = np.zeros((N, K), dtype=np.float64)
    for k in range(K):
        box_volume = ((query_boxes[k, 3] - query_boxes[k, 0] + 1) * (query_boxes[k, 4] - query_boxes[k, 1] + 1) * (query_boxes[k, 5] - query_boxes[k, 2] + 1))
        for n in range(N):
            id = min(boxes[n, 3], query_boxes[k, 3]) - max(boxes[n, 0], query_boxes[k, 0]) + 1
            if id > 0:
                iw = min(boxes[n, 4], query_boxes[k, 4]) - max(boxes[n, 1], query_boxes[k, 1]) + 1
                if iw > 0:
                    ih = min(boxes[n, 5], query_boxes[k, 5]) - max(boxes[n, 2], query_boxes[k, 2]) + 1
                    if ih > 0:
                        ua = (boxes[n, 3] - boxes[n, 0] + 1) * (boxes[n, 4] - boxes[n, 1] + 1) * (boxes[n, 5] - boxes[n, 2] + 1) + box_volume - iw * ih * id
                        overlaps[n, k] = iw * ih * id / ua
    return overlaps
