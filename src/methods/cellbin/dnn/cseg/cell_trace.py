import cv2
from cellbin.image.wsi_split import SplitWSI
import numpy as np


def get_trace(mask):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    h, w = mask.shape[: 2]
    output = []
    for i in range(num_labels):
        box_w, box_h, area = stats[i][2:]
        if box_h == h and box_w == w:
            continue
        output.append([box_h, box_w, area])
    return output

def get_trace_v2(mask):
    """
    2023/09/20 @fxzhao get_trace的升级版本,分块处理,降低大数据量下的内存占用
    2023/09/21 @fxzhao 增加数据量检测,当数据较小时使用不分块方法
    """
    h, w = mask.shape[: 2]
    steps = 10000
    overlap = 1000
    if h < steps+overlap:
        return get_trace(mask)
    
    starts = np.array(range(0, h, steps))[:-1]
    starts -= overlap
    ends = starts + steps + overlap*2
    starts[0] = 0
    ends[-1] = h
    output = []
    for start,end in zip(starts, ends):
        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask[start:end,], connectivity=8)
        up_thre = overlap
        if start == 0:
            up_thre = start 
        down_thre = up_thre + steps
        if end == h:
            down_thre = end - start
        for i in range(num_labels):
            _, box_y, box_w, box_h, area = stats[i]
            if box_y < up_thre or down_thre <= box_y:
                continue
            if box_h == (end-start) and box_w == w:
                continue
            output.append([box_h, box_w, area])
    return output


def show_hist(data, xlabel, ylabel, k=1000, bins=None):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    _, _, bar_container = ax.hist(data, bins=bins, density=False,
                                  facecolor="tab:blue", edgecolor="tab:orange",
                                  range=(k, np.max(data)))

    plt.ylim(0, int(len(data) * 0.05))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()
    return


def test_hist(cell_mask):
    data = get_trace(cell_mask)

    data = np.array(data)
    d = data[:, 2]
    cellarea_1600 = np.sum(d > 1600) / d.size
    d = data[:, 0]
    cell_height_40 = np.sum(d > 40) / d.size
    d = data[:, 1]
    cell_width_40 = np.sum(d > 40) / d.size

    return cellarea_1600, cell_height_40, cell_width_40


def check_cells_with_tissue(tissue_mask, cell_mask, k):
    def check_area(img):
        h, w = img.shape[:2]
        return int(h * w) == np.sum(img > 0)

    tissue_mask[tissue_mask > 0] = 1
    cell_mask[cell_mask > 0] = 1

    flag = True
    tissue_area = np.sum(tissue_mask > 0)

    cell_mask = cv2.bitwise_and(cell_mask, cell_mask, mask=tissue_mask)
    tmp = np.subtract(tissue_mask, cell_mask)
    sp_run = SplitWSI(tmp, win_shape=(k, k), overlap=0, batch_size=1,
                      need_fun_ret=True, need_combine_ret=False, editable=False, tar_dtype=np.uint8)
    sp_run.f_set_run_fun(check_area)
    _, ret, _ = sp_run.f_split2run()
    ret = np.array(ret).flatten()
    count = np.sum(ret > 0)
    flag = count == 0
    cell_miss_area = count * np.square(k)
    return flag, cell_miss_area / tissue_area


if __name__ == '__main__':
    import sys
    import tifffile

    tissue = tifffile.imread(r"D:\stock\dataset\test\out\FP200000340BR_A1_t.tif")
    cells = tifffile.imread(r"D:\stock\dataset\test\out\FP200000340BR_A1.tif")
    ret = check_cells_with_tissue(tissue, cells, 256)
    print(ret)

    print(test_hist(cells))
    sys.exit()
