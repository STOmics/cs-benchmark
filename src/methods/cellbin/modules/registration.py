import numpy as np
from copy import deepcopy
from numba import njit,prange

from cellbin.modules import CellBinElement
from cellbin.contrib.alignment import AlignByTrack


class Registration(CellBinElement):
    def __init__(self):
        super(Registration, self).__init__()
        self.align_track = AlignByTrack()

        self.offset = [0, 0]
        self.rot90 = 0
        self.flip = False
        self.score = 0

        self.regist_img = np.array([])
        self.fov_transformed = np.array([])
        self.dist_shape = ()
        self.vision_cp = np.array([])
        self.adjusted_stitch_template = np.array([])
        self.adjusted_stitch_template_unflip = np.array([])
        # self._moving_image = self._moving_marker = None
        # self._fixed_image = self._fixed_marker = None
        # self.x_scale = self.y_scale = None
        # self.rotation = None

    # def fixed_image_shape(self, ):
    #     return self._fixed_image.shape

    def mass_registration_trans(
            self,
            fov_transformed,
            vision_image,
            chip_template,
            track_template,
            scale_x,
            scale_y,
            fov_stitched_shape,
            rotation,
            flip

    ):
        """
        This func is used to regist the transform image

        Args:
            fov_transformed (): transform image
            vision_image (): gene matrix image
            chip_template (): chip template, SS2, FP1
            track_template (): stitch template
            scale_x (): x direction scale
            scale_y (): y direction scale
            fov_stitched_shape (): stitched image shape
            rotation (): rotation
            flip (): if flip

        Returns:
            self.flip: if flip
            self.rot90: 90 degree rot times
            self.offset: offset in x, y direction
            self.score: regist score

        """
        self.align_track.set_chip_template(chip_template=chip_template)
        self.fov_transformed = fov_transformed
        self.dist_shape = vision_image.shape
        track_template_copy = deepcopy(track_template)
        self.adjusted_stitch_template = self.align_track.adjust_cross(
            stitch_template=track_template,
            scale_x=scale_x,
            scale_y=scale_y,
            fov_stitched_shape=fov_stitched_shape,
            new_shape=self.fov_transformed.shape,
            chip_template=chip_template,
            rotation=rotation
        )
        self.adjusted_stitch_template_unflip = self.align_track.adjust_cross(
            stitch_template=track_template_copy,
            scale_x=scale_x,
            scale_y=scale_y,
            fov_stitched_shape=fov_stitched_shape,
            new_shape=self.fov_transformed.shape,
            chip_template=chip_template,
            rotation=rotation,
            flip=False
        )
        self.vision_cp = self.align_track.find_track_on_vision_image(vision_image, chip_template)

        offset, rot_type, score = self.align_track.run(
            transformed_image=self.fov_transformed,
            vision_img=vision_image,
            vision_cp=self.vision_cp,
            stitch_tc=self.adjusted_stitch_template,
            flip=flip,
        )

        # result update
        self.flip = flip
        self.rot90 = rot_type
        self.offset = offset
        self.score = score

        return 0

    def mass_registration_stitch(
            self,
            fov_stitched,
            vision_image,
            chip_template,
            track_template,
            scale_x,
            scale_y,
            rotation,
            flip

    ):
        """
        This func is used to regist the stitch image

        Args:
            fov_stitched (): stitched image
            vision_image (): gene matrix image
            chip_template (): chip template, SS2, FP1
            track_template (): stitch template
            scale_x (): x direction scale
            scale_y (): y direction scale
            rotation (): rotation
            flip (): if flip


        Returns:
            self.flip: if flip
            self.rot90: 90 degree rot times
            self.offset: offset in x, y direction
            self.score: regist score

        """
        # transform
        self.fov_transformed = self.stitch_to_transform(
            fov_stitch=fov_stitched,
            scale_x=scale_x,
            scale_y=scale_y,
            rotation=rotation
        )
        fov_stitched_shape = fov_stitched.shape
        self.mass_registration_trans(
            self.fov_transformed,
            vision_image,
            chip_template,
            track_template,
            scale_x,
            scale_y,
            fov_stitched_shape,
            rotation,
            flip
        )

        return 0

    @staticmethod
    def stitch_to_transform(fov_stitch, scale_x, scale_y, rotation):
        """
        From stitched image to transform image based on provided scale and rotation

        Args:
            fov_stitch (): stitched image
            scale_x (): x direction scale
            scale_y (): y direction scale
            rotation (): rotation

        Returns:
            fov_transformed: tranformed image

        """
        from cellbin.image.transform import ImageTransform
        i_trans = ImageTransform()
        i_trans.set_image(fov_stitch)
        fov_transformed = i_trans.rot_scale(
            x_scale=scale_x,
            y_scale=scale_y,
            angle=rotation
        )
        return fov_transformed

    def transform_to_regist(self, ):
        """
        From transform image to regist image based on regist result

        Returns:
            self.regist_img: regist image

        """
        from cellbin.image.transform import ImageTransform
        i_trans = ImageTransform()
        i_trans.set_image(self.fov_transformed)
        if self.flip:
            i_trans.flip(
                flip_type='hor',
                ret_dst=False
            )
        i_trans.rot90(self.rot90, ret_dst=False)
        self.regist_img = i_trans.offset(self.offset[0], self.offset[1], self.dist_shape)

    @staticmethod
    def register_score(regist_img, vis_img):
        """
        Calculate regist score baed on gene matrix image

        Args:
            regist_img (): regist image
            vis_img (): gene matrix image

        Returns:
            regist score

        2023/09/20 @fxzhao 使用numba加速计算,并降低内存,不会再修改输入的regist_img
        """
        # regist_img[np.where(regist_img > 1)] = 1
        # total = np.sum(vis_img)
        # roi_mat = vis_img * regist_img
        # roi = np.sum(roi_mat)
        total = np.sum(vis_img)
        roi = multiply_sum(regist_img, vis_img)
        return int(roi * 100 / total)

@njit(parallel=True)
def multiply_sum(regist_img, vis_img):
    """
    2023/09/20 @fxzhao 使用numba加速计算
    """
    h, w = regist_img.shape
    line_sums = np.zeros(h)
    for i in prange(h):
        line_sum = 0
        for j in range(w):
            if regist_img[i][j] > 0:
                line_sum += vis_img[i][j]
        line_sums[i] = line_sum
    return np.sum(line_sums)

if __name__ == '__main__':
    import json
    from glob import glob
    import os
    import tifffile
    import cv2
    import numpy as np

    vipshome = r'C:\vips-dev-8.12\bin'
    os.environ['PATH'] = vipshome + ';' + os.environ['PATH']

    regist_path = r"D:\Data\qc\new_qc_test_data\regist_issue\A02177C4"
    ipr_path = glob(os.path.join(regist_path, "**.ipr"))[0]
    with h5py.File(ipr_path, "r") as f:
        # json_obj = json.load(f)
        scale_x = f["Register"].attrs["ScaleX"]
        scale_y = f["Register"].attrs["ScaleY"]
        rotation = f["Register"].attrs["Rotation"]
        # chip_template = f["ChipInfo"]["FOVTrackTemplate"]
        # offset_ori = f["AnalysisInfo"]["input_dct"]["offset"]
        # rot_ori = f["AnalysisInfo"]["input_dct"]["rot_type"]
    # fov_transformed_path = os.path.join(regist_path, '4_register', 'fov_stitched_transformed.tif')
    # fov_transformed = tifffile.imread(fov_transformed_path)
    chip_template = [[240, 300, 330, 390, 390, 330, 300, 240, 420], [240, 300, 330, 390, 390, 330, 300, 240, 420]]
    fov_stitched_path = glob(os.path.join(regist_path, '**fov_stitched.tif'))[0]
    fov_stitched = tifffile.imread(fov_stitched_path)

    # czi mouse brain -> stitch shape (2, x, x)
    if len(fov_stitched.shape) == 3:
        fov_stitched = fov_stitched[0, :, :]

    # try:
    #     gene_exp_path = glob(os.path.join(regist_path, "**raw.tif"))[0]
    # except IndexError:
    #     try:
    #         gene_exp_path = glob(os.path.join(regist_path, "3_vision", "**_gene_exp.tif"))[0]
    #     except IndexError:
    #         gene_exp_path = glob(os.path.join(regist_path, "3_vision", "**.gem.tif"))[0]

    gene_exp_path = glob(os.path.join(regist_path, "**gene.tif"))[0]
    gene_exp = cv2.imread(gene_exp_path, -1)

    track_template = np.loadtxt(glob(os.path.join(regist_path, '**template.txt'))[0])  # stitch template
    flip = True
    # im_shape = np.loadtxt(os.path.join(regist_path, "4_register", "im_shape.txt"))
    rg = Registration()
    rg.mass_registration_stitch(
        fov_stitched,
        gene_exp,
        chip_template,
        track_template,
        scale_x,
        scale_y,
        rotation,
        flip
    )
    print(rg.offset, rg.rot90, rg.score)
    rg.transform_to_regist()
    regist_img = rg.regist_img
    print("asd")
    tifffile.imwrite(os.path.join(regist_path, "new_regist_1.tif"), regist_img)
