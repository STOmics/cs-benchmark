import numpy as np
from cellbin.dnn.tseg.yolo.detector import TissueSegmentationYolo
from cellbin.image.augmentation import f_ij_16_to_8
from cellbin.utils import clog


class ClassifyFOV(object):
    def __init__(self):
        self.x = None
        self._tissue_detector = None
        self._tissue_fov_map = np.array([])
        self._tissue_fov_roi = (-1, -1, -1, -1)
        self._fov_loc = None
        self._fov_size = None
        self.mask_area = 0
        self._tissue_mask = None
        self.success = False

    def set_detector(self, detector):
        """
        set tissuecut detector
        Args:
            detector(object): tissuecut detector

        Returns:

        """
        self._tissue_detector = detector

    def classify(self, mosaic, fov_loc, fov_size, expand=1, ch=0):
        """
        clsssify FOV into those with tissue and those without
        Args:
            mosaic(ndarray):img array
            fov_loc(ndarray): r x c x 2,Coordinates for the stitched tiled
            imageeither program stitched or microscopestitched
            fov_size(tuple): fov size
            expand(int):Extend X FOV centered on tissue
            ch(int):img channel index,default 0

        Returns:

        """
        self._fov_loc = fov_loc
        w, h = fov_size
        self._fov_size = fov_size
        clog.info(f"Tissue cut input has {mosaic.ndim} dims, Tissue cut is using channel: {ch}")
        if mosaic.ndim == 3:
            mosaic = mosaic[:, :, ch]
        if mosaic.dtype != 'uint8':
            mosaic = f_ij_16_to_8(mosaic)  # IMPORTANT! Tissue seg require uint 8
        self._tissue_mask = self._tissue_detector.f_predict(mosaic)
        self.mask_area = self._tissue_detector.mask_num
        row, col = fov_loc.shape[:2]
        self._tissue_fov_map = np.zeros((row, col), dtype=np.uint8)
        for i in range(row):
            for j in range(col):
                x, y = fov_loc[i, j]
                fov_mask = self._tissue_mask[y: y + h, x: x + w]
                if np.sum(fov_mask) > 10:
                    self._tissue_fov_map[i, j] = 1
        index_y, index_x = np.nonzero(self._tissue_fov_map)
        if len(index_x) != 0 and len(index_y) != 0:
            # ROI: (x_min, y_min, x_max, y_max)
            x_min, y_min, x_max, y_max = (np.min(index_x), np.min(index_y),
                                          np.max(index_x), np.max(index_y))
            col0, row0 = [col - 1, row - 1]
            x_min = (x_min > expand) and (x_min - expand) or 0
            y_min = (y_min > expand) and (y_min - expand) or 0
            x_max = (x_max + expand > col0) and col0 or (x_max + expand)
            y_max = (y_max + expand > row0) and row0 or (y_max + expand)
            self._tissue_fov_roi = (x_min, y_min, x_max, y_max)
            self.success = True
        else:
            self.success = False

    def tissue_bbox_in_mosaic(self, ):
        """

        Returns: coordinate of the tissue in mosaic

        """
        if -1 not in self._tissue_fov_roi:
            w, h = self._fov_size
            x_min, y_min, x_max, y_max = self._tissue_fov_roi
            roi_loc = self._fov_loc[y_min: y_max + 1, x_min: x_max + 1]

            x0, y0, x1, y1 = [np.min(roi_loc[:, :, 0]), np.min(roi_loc[:, :, 1]),
                              np.max(roi_loc[:, :, 0] + w), np.max(roi_loc[:, :, 1]) + h]
            return [x0, y0, x1, y1]
        else:
            return [-1, -1, -1, -1]

    @property
    def tissue_fov_roi(self):
        return self._tissue_fov_roi

    @property
    def tissue_fov_map(self):
        return self._tissue_fov_map

    @property
    def tissue_mask(self):
        return self._tissue_mask

    @property
    def tissue_detector(self):
        return self._tissue_detector


if __name__ == '__main__':
    weights_path = r"D:\fov\tissue_seg\tissueseg_yolo_SH_20230131.onnx"
    cls_fov = ClassifyFOV()
    cls_fov.clsssify(mosaic=r'', fov_loc=np.zeros((5, 5)), fov_size=(3064, 2039))
    box_fov = cls_fov.tissue_bbox_in_fovs()
    box_mosaic = cls_fov.tissue_bbox_in_mosaic()
    print('Bbox of FOV {}, Bbox of mosaic {}'.format(box_fov, box_mosaic))
