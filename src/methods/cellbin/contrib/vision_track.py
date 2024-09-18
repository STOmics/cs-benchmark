import numpy as np
import math


def get_mass(image):
    """
    2023/09/20 @fxzhao 移除类型转换,不影响结果
    """
    image = image.astype(float)  # 别动这里！！！
    image_x = np.sum(image, 0)
    xx = np.array(range(len(image_x)))
    xx_cal = xx * image_x
    x_mass = np.sum(xx_cal) / np.sum(image_x)

    image_y = np.sum(image, 1)
    yy = np.array(range(len(image_y)))
    yy_cal = yy * image_y
    y_mass = np.sum(yy_cal) / np.sum(image_y)
    return np.array([x_mass, y_mass])


def find_cross_ind(position, summation):
    result = []
    for _ in range(len(summation) - np.max(position) - 1):
        position += 1
        result.append(sum(summation[position]))
    first_ind = result.index(min(result)) + 2
    return first_ind


def find_first_tp(image, mass_center, chip_template, find_range=3000):
    min_x = int(round(mass_center[0]) - find_range)
    max_x = int(round(mass_center[0]) + find_range)
    min_y = int(round(mass_center[1]) - find_range)
    max_y = int(round(mass_center[1]) + find_range)
    find_image = image[min_y: max_y, min_x: max_x].astype(float)
    x_sum = np.sum(find_image, 0)  # vertical
    y_sum = np.sum(find_image, 1)  # horizontal
    position_x, position_y = np.cumsum(np.insert(np.array(chip_template), 0, 0, axis=1)[:, :-1], axis=1)
    x_first = find_cross_ind(position_x, x_sum)
    y_first = find_cross_ind(position_y, y_sum)
    x_first += min_x
    y_first += min_y
    return x_first, y_first


def one_to_all(template, mid_pos, length):
    step_size = sum(template)
    upper = math.ceil((length - mid_pos) / step_size)
    lower = math.ceil(mid_pos / step_size)
    interval = np.concatenate(
        (
            np.cumsum(np.tile(template, upper)),
            np.array([0]),
            np.cumsum(np.tile(-np.array(template[::-1]), lower))
        )
    )
    index = np.concatenate(
        (
            np.tile(np.arange(len(template)), lower),
            np.tile(np.arange(len(template)), upper),
            np.array([0])
        )
    )
    interval.sort()
    interval = mid_pos + interval
    interval = interval.reshape(-1, 1)
    index = index.reshape(-1, 1)
    combined = np.concatenate(
        (interval, index),
        axis=1
    )
    return combined


def find_cross(gene_exp, chip_template):
    mass_center = get_mass(gene_exp)
    x_mid, y_mid = find_first_tp(gene_exp, mass_center, chip_template)
    h, w = gene_exp.shape
    x_all = one_to_all(chip_template[0], x_mid, w)
    y_all = one_to_all(chip_template[1], y_mid, h)

    all_comb = np.concatenate(
        (
            np.tile(
                np.expand_dims(x_all, 1),
                (1, y_all.shape[0], 1)
            ),
            np.tile(
                np.expand_dims(y_all, 0),
                (x_all.shape[0], 1, 1)
            )
        ),
        axis=2
    ).reshape(-1, 4)

    all_comb[:, [1, 2]] = all_comb[:, [2, 1]]
    final_result = all_comb[
        np.logical_and.reduce(
            (
                all_comb[:, 0] <= w,
                all_comb[:, 1] <= h,
                all_comb[:, 0] >= 0,
                all_comb[:, 1] >= 0
            )
        )
    ]
    return final_result


if __name__ == '__main__':
    import cv2

    vision_image = cv2.imread(r"D:\Data\regist\FP200000340BR_D1\3_vision\FP200000340BR_D1_gene_exp.tif", -1)
    chip_template = [[240, 300, 330, 390, 390, 330, 300, 240, 420], [240, 300, 330, 390, 390, 330, 300, 240, 420]]
    new_cp = find_cross(vision_image, chip_template)
    print("asd")
