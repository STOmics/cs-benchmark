import numpy as np
from math import ceil
from sklearn.cluster import DBSCAN

from cellbin.dnn.clarity.evaler import ClarityClassifier
from cellbin.image.augmentation import f_ij_16_to_8
from cellbin.image.augmentation import f_gray2bgr
from cellbin.utils import clog

REPRESENT_4 = {
    0: 'black',
    1: 'blur',
    2: 'good',
    3: 'over_expo',
    -1: 'uncertain',
}

REPRESENT_6 = {
    0: 'black',
    1: 'good',
    2: 'first_level_blur',
    3: 'second_level_blur',
    4: 'first_level_over_expo',
    5: 'second_level_over_expo',
    -1: 'uncertain',
}

COLOR_SET = {
    'red': (255, 0, 0),
    'lightsalmon': (255, 160, 122),
    'blue': (0, 0, 255),
    'lightblue': (173, 216, 230),
    'yellow': (255, 255, 0),
    'green': (0, 128, 0),
    'pink': (255, 203, 192)
}

COLOR_4 = {
    'black': 'yellow',
    'blur': 'blue',
    'good': 'green',
    'over_expo': 'red',
    'uncertain': 'pink'
}

COLOR_6 = {
    'black': 'yellow',
    'first_level_blur': 'lightblue',
    'second_level_blur': 'blue',
    'good': 'green',
    'first_level_over_expo': 'lightsalmon',
    'second_level_over_expo': 'red',
    'uncertain': 'pink'
}

WEIGHT_MAP_6 = {
    0: 0,
    1: 1,
    2: 0.2,
    3: 0,
    4: 0.9,
    5: 0
}

WEIGHT_MAP_4 = {
    0: 0,
    1: 0,
    2: 1,
    3: 0,
}


class ClarityQC(object):
    def __init__(self, ):
        self.cl_classify = None

        self.counts = {}
        self.score = 0
        self.preds = np.array([])
        self.box_lst = []

        self.draw_img = np.array([])
        self.black_img = np.array([])
        self.fig = None
        self.topk_result = []

        self.represent = None
        self.color = None
        self.num_class = None
        self.pre_func = None
        self.weight_map = None

    def load_model(self, model_path, batch_size=2000, conf_thresh=0, gpu='-1', num_threads=0, ):
        """
        Load clarity model

        Args:
            model_path (str): weight path
            batch_size (int): batch size when inferencing
            conf_thresh (): confidence threshold
            gpu (): if use gpu
            num_threads (): number of threads used in onnx session

        Returns:

        """
        self.cl_classify = ClarityClassifier(
            weight_path=model_path,
            batch_size=batch_size,
            conf_thresh=conf_thresh,
            gpu=gpu,
            num_threads=num_threads,
        )
        self.pre_func = None
        self.num_class = self.cl_classify.output_shape[1]
        if self.num_class == 4:
            self.represent = REPRESENT_4
            self.color = COLOR_4
            self.weight_map = WEIGHT_MAP_4
        elif self.num_class == 6:
            self.represent = REPRESENT_6
            self.color = COLOR_6
            self.weight_map = WEIGHT_MAP_6

    def set_enhance_func(self, f):
        self.pre_func = f

    def run(self, img: np.ndarray, detect_channel=-1):
        """
        This function will spilit the input image into (64, 64) pieces, then classify each piece into category.
        Category is ['black', 'over_exposure', 'blur', 'good']

        Args:
            img (): stitched image after tissue cut (numpy ndarray)

        Returns:
            self.counts: counts of each category ['black', 'blur', 'good', 'over_expo']
            self.score: clarity score
            self.preds: prediction in
                - shape is ceil(image_height / (64 - _overlap)),  ceil(image_width / (64 - self._overlap), 2)
                - 2 -> 1st: class, 2nd probability
            self.boxes: the pieces coordinate
                - [[y_begin, y_end, x_begin, x_end], ...]

        """
        clog.info(f"Clarity eval input has {img.ndim} dims, using enhance func {self.pre_func}")
        if not isinstance(img, np.ndarray):
            raise Exception(f"Only accept numpy array as input")
        if img.ndim == 3:
            if detect_channel != -1:
                img = img[:, :, detect_channel]
            elif self.pre_func is not None:
                img = self.pre_func(img, need_not=True)
        if img.dtype != np.uint8:
            img = f_ij_16_to_8(img)
        if img.ndim == 2:
            img = f_gray2bgr(img)
        self.original_img = img.copy()
        # if img.ndim != 3:
        #     img = np.expand_dims(img, axis=2)
        #     img = np.concatenate((img, img, img), axis=-1)

        counts, preds, box_lst = self.cl_classify.inference(img)
        score = self.cl_classify.score_calculator(preds, self.weight_map)
        self.counts = counts
        self.score = score
        self.preds = preds
        self.box_lst = box_lst

        # clog.info(f"Clarity eval counts: {self.counts}")
        clog.info(f"Clarity eval score: {self.score}")

    def post_process(self, win_h=64, win_w=64, overlap=0):
        """
        This function will draw clarity classification result on image.

        Args:
            preds (): predictions from clarity result
                - shape is ceil(image_height / (64 - _overlap)),  ceil(image_width / (64 - self._overlap), 2)
                - 2 -> 1st: class, 2nd probability
            draw_img (): the image to draw results on
            win_h (): height of image piece
            win_w (): width of image piece
            overlap (): overlap when spliting image

        Returns:
            self.draw_img: the image to draw results on

        """
        import cv2
        if len(self.preds) == 0:
            return self.original_img
        h, w = self.original_img.shape[:2]
        y_nums = ceil(h / (win_h - overlap))
        x_nums = ceil(w / (win_w - overlap))
        self.black_img = np.zeros((self.original_img.shape[:2]), dtype=np.uint8)  # @jqc update
        for y_temp in range(y_nums):
            for x_temp in range(x_nums):
                x_begin = int(max(0, x_temp * (win_w - overlap)))
                y_begin = int(max(0, y_temp * (win_h - overlap)))
                x_end = int(min(x_begin + win_w, w))
                y_end = int(min(y_begin + win_h, h))
                if y_begin >= y_end or x_begin >= x_end:
                    continue

                cur_class, cur_score = self.preds[y_temp, x_temp]
                cur_color = COLOR_SET[self.color[self.represent[cur_class]]]
                mid_x, mid_y = int((x_begin + x_end) / 2), int((y_begin + y_end) / 2)
                cv2.putText(
                    self.original_img,
                    str(int(cur_class)) + "_" + str(round(cur_score * 100)),
                    (mid_x, mid_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.3,
                    (255, 0, 0),
                    1,
                    cv2.LINE_AA
                )

                cv2.rectangle(self.original_img, (x_begin, y_begin), (x_end, y_end), cur_color, 1)
                if cur_color == (255, 255, 0):
                    cv2.rectangle(self.black_img, (x_begin, y_begin), (x_end, y_end), cur_color, -1)
        # self.black_img[np.where(self.black_img == 0)] = 1
        # self.black_img[np.where(self.black_img == 255)] = 0
        self.black_img = (self.black_img == 0).astype(np.int8)
        draw_img = cv2.cvtColor(self.original_img, cv2.COLOR_RGB2BGR)  # if use cv2 write must do this
        self.draw_img = draw_img

    def cluster(self, top_k=10):
        """
        This function is to get clustering result based on the clarity classification result.

        Args:
            preds (): predictions from clarity classification
                - shape is ceil(image_height / (64 - _overlap)),  ceil(image_width / (64 - self._overlap), 2)
                - 2 -> 1st: class, 2nd probability
            boxes (): the pieces coordinate
                - [[y_begin, y_end, x_begin, x_end], ...]
            top_k (): top k cluster results will be returned

        Returns:
            self.fig: matplotlib figure object
                - fig.show() to show the figure
                - fig.savefig(f"{path}/xx.png") to save the figure
            self.topk_result: n x 3
                - 1st element: area percent -> cluster counts / tissue_counts
                - 2nd element: cluster length in x direction -> x length / image width
                - 3rd element: cluster length in y direction -> y length / image height

        """
        import matplotlib.pyplot as plt

        preds = self.preds.reshape(-1, 2)
        preds_class = preds[:, 0: 1]
        boxes = np.array(self.box_lst)
        max_y = np.max(boxes[:, :2])
        max_x = np.max(boxes[:, 2:4])

        fig, ax = plt.subplots()
        topk_result = []

        boxes_ = []
        tissue_counts = 0
        for class_i in range(self.num_class):
            cur_class_name = self.represent[class_i]
            if self.num_class == 4 and cur_class_name not in ['black', 'good']:
                boxes_.append(boxes[preds_class[:, 0] == class_i])
            if self.num_class == 6 and cur_class_name not in ['black', 'good', 'first_level_over_expo']:
                boxes_.append(boxes[preds_class[:, 0] == class_i])
            if cur_class_name not in ['black']:
                tissue_counts += (preds_class[:, 0] == class_i).sum()
        boxes_ = np.concatenate(boxes_)
        # boxes_1 = boxes[((preds_class[:, 0] == 1) | (preds_class[:, 0] == 3))]
        # tissue_counts_1 = ((preds_class[:, 0] == 1) | (preds_class[:, 0] == 3) | (preds_class[:, 0] == 2)).sum()
        y_mid = ((boxes_[:, 0] + boxes_[:, 1]) / 2).reshape(-1, 1)
        x_mid = ((boxes_[:, 2] + boxes_[:, 3]) / 2).reshape(-1, 1)
        X = np.hstack((x_mid, y_mid))  # [[x, y]]
        if len(X) != 0:
            # DBSCAN
            db = DBSCAN(eps=91, min_samples=4).fit(X)
            labels = db.labels_
            # Number of clusters in labels, ignoring noise if present.
            n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

            if n_clusters_ != 0:
                n_noise_ = list(labels).count(-1)

                # print("Estimated number of clusters: %d" % n_clusters_)
                # print("Estimated number of noise points: %d" % n_noise_)

                core_samples_mask = np.zeros_like(labels, dtype=bool)
                core_samples_mask[db.core_sample_indices_] = True

                unique_ls, counts = np.unique(labels[core_samples_mask], return_counts=True)
                unique_ls_sort = unique_ls[counts.argsort()[::-1]]

                topk = top_k
                topk = min(topk, n_clusters_)
                unique_ls_sort = unique_ls_sort[: topk]
                colors = [plt.cm.tab20(each) for each in np.linspace(0, 1, topk)]

                index = 1

                # draw
                for k, col in zip(unique_ls_sort, colors):
                    if k == -1:
                        # Black used for noise.
                        # col = [0, 0, 0, 1]
                        continue

                    class_member_mask = labels == k
                    xy = X[class_member_mask & core_samples_mask]

                    cluster_x = xy[:, 0]
                    cluster_minx, cluster_maxx = np.min(cluster_x), np.max(cluster_x)
                    cluster_y = xy[:, 1]
                    cluster_miny, cluster_maxy = np.min(cluster_y), np.max(cluster_y)
                    x_len = cluster_maxx - cluster_minx
                    y_len = cluster_maxy - cluster_miny
                    x_len_pt = x_len / max_x
                    y_len_pt = y_len / max_y
                    topk_result.append([len(xy) / tissue_counts, x_len_pt, y_len_pt])

                    # topk_result[index] = len(xy)
                    ax.plot(
                        xy[:, 0],
                        xy[:, 1],
                        "o",
                        markerfacecolor=tuple(col),
                        markeredgecolor="k",
                        markersize=5,
                    )
                    index += 1
                    # xy = X[class_member_mask & ~core_samples_mask]
                    # plt.plot(
                    #     xy[:, 0],
                    #     xy[:, 1],
                    #     "o",
                    #     markerfacecolor=tuple(col),
                    #     markeredgecolor="k",
                    #     markersize=5,
                    # )
                    # break
                scale = (max_x // 100) * 3
                plt.xlim([-scale, max_x + scale])
                plt.ylim([-scale, max_y + scale])
                plt.gca().invert_yaxis()
                plt.gca().set_aspect('equal', adjustable='box')

                topk_result = np.round(topk_result, 3)
                # plt.title(f"Estimated number of clusters: {n_clusters_}")
                largest = topk_result[0]
                plt.title(
                    f"Top {topk} clusters \n Largest cluster area: {largest[0]}, width: {largest[1]}, height: {largest[2]}")
            plt.tight_layout()
        self.fig = fig
        self.topk_result = topk_result


if __name__ == '__main__':
    import cv2

    weight_path = r"D:\PycharmProjects\imageqc-beta\ImageQC\models\ST_TP_Mobile_small_050_V2.onnx"
    batch_size = 2000
    clarity_qc = ClarityQC()
    clarity_qc.load_model(weight_path)
    print("asd")

    img_path = r"D:\Data\qc\new_qc_test_data\clarity\bad\test_imgs\test_output\fovs_test\fov_stitched_transform.tif"
    img = cv2.imread(img_path, -1)
    clarity_qc.run(img)
    clarity_qc.post_process()
    # cv2.imwrite(r"D:\Data\qc\SS200000302TL_B5_out\new_qc_out\test1.tif", clarity_qc.draw_img)
    # clarity_qc.cluster()
    # clarity_qc.fig.savefig(r"D:\Data\qc\SS200000302TL_B5_out\new_qc_out\test1.png",)
    # tissue_sum = counts[1] + counts[2] + counts[3]
    # topk_pt = (topk_result / tissue_sum) * 100  # topk cluster area percentage
    # print("asd")
    from cellbin.dnn.tseg.yolo.detector import TissueSegmentationYolo
    import tifffile

    seg = TissueSegmentationYolo()
    seg.f_init_model(r"D:\PycharmProjects\cellbin\cellbin\dnn\weights\tissueseg_yolo_SH_20230131_th.onnx", )
    img = tifffile.imread(img_path)
    mask = seg.f_predict(img)
    iou_result = iou(clarity_qc.black_img, mask)
