import os
from typing import List, Any, Tuple, Union
import numpy as np
import shutil

from PIL import ImageFile, Image
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

from cellbin2.utils import clog
from cellbin2.contrib.alignment.basic import transform_points
from cellbin2.image import cbimwrite
from cellbin2.utils.common import TechType
from cellbin2.contrib.param import ChipFeature
from cellbin2.modules.metadata import ProcFile
from cellbin2.modules import naming, run_state
from cellbin2.contrib.alignment import template_00pt
from cellbin2.contrib.alignment.basic import RegistrationInfo
from cellbin2.utils import ipr
from cellbin2.modules.extract.base import FeatureExtract
from cellbin2.contrib.mask_manager import BestTissueCellMask, MaskManagerInfo
from cellbin2.contrib.param import ChipBoxInfo
from cellbin2.contrib.tissue_segmentor import TissueSegInputInfo
from cellbin2.utils.pro_monitor import process_decorator
from cellbin2.image import CBImage, cbimread
from cellbin2.utils.common import iPlaceHolder


class ImageFeatureExtract(FeatureExtract):
    def __init__(self, image_file: ProcFile, output_path: str, sn: str, im_naming: naming.DumpImageFileNaming):
        super(ImageFeatureExtract, self).__init__(image_file, output_path)
        self._fov_wh = (2000, 2000)  # 初始裁图尺寸
        self._channel_image = ipr.IFChannel()  # Union[ipr.ImageChannel, ipr.IFChannel]
        # self._naming = naming.DumpImageFileNaming(image_file.file_path)
        self._naming: naming.DumpImageFileNaming = im_naming
        # self._state = run_state.ImageRunState(image_file.file_path, dst_dir=output_path)
        # self._state = state
        self.sn = sn

    def set_channel_image(self, ci: Union[ipr.ImageChannel, ipr.IFChannel]):
        self._channel_image = ci

    def transform2regist(self, info: RegistrationInfo):
        """
        将transform图（染色图/mask）变成配准图:模板变成regist模板
        """
        self._channel_image.update_registration(info)
        dct = {
            self._naming.transformed_image: self._naming.registration_image,
            self._naming.transform_cell_mask: self._naming.cell_mask,
            self._naming.transform_cell_mask_raw: self._naming.cell_mask_raw,
            self._naming.transform_tissue_mask: self._naming.tissue_mask,
            self._naming.transform_tissue_mask_raw: self._naming.tissue_mask_raw,
            # self._naming.transform_cell_correct_mask: self._naming.cell_correct_mask,
            self._naming.transformed_template: self._naming.register_template,
            self._naming.transformed_track_template: self._naming.register_track_template
        }
        for src, dst in dct.items():
            src_path = src
            dst_path = dst
            # if os.path.exists(dst_path):
            #     continue
            if os.path.exists(src_path):
                if os.path.splitext(src_path)[1] == ".txt":  # 或其他判断
                    points, _ = transform_points(
                        src_shape=self._channel_image.Stitch.TransformShape,
                        points=np.loadtxt(src_path),
                        rotation=(4 - info.counter_rot90) * 90,
                        flip=0 if info.flip else -1,
                        offset=info.offset
                    )
                    np.savetxt(dst_path, points)
                    if dst == self._naming.register_template:
                        self._channel_image.Register.RegisterTemplate = points
                    if dst == self._naming.register_track_template:
                        self._channel_image.Register.RegisterTrackTemplate = points
                else:
                    dst_image = cbimread(src_path).trans_image(
                        flip_lr=info.flip, rot90=info.counter_rot90, offset=info.offset, dst_size=info.dst_shape)
                    cbimwrite(dst_path, dst_image)
        # for img in [self._naming.transform_cell_mask, self._naming.transform_tissue_mask,
        #             self._naming.transform_cell_correct_mask]:
        #     img_path = os.path.join(self.output_path, img)
        #     if os.path.exists(img_path):
        #         os.remove(img_path)

    def stitched2regist(self, trans_matrix: np.ndarray, dst_siz: Tuple[int, int]):
        """
        将stitched图（染色图/mask）变成配准图:芯片框变成regist模板
        """
        pass

    def extract4transform(
            self,
            scale: Tuple[float, float],
            rotation: float,
            offset: Tuple[float, float] = (0., 0.)
    ):
        # stitch
        if not os.path.exists(self._naming.stitch_image):
            shutil.copy2(self._image_file.file_path,  self._naming.stitch_image)

        if not os.path.exists(self._naming.transformed_image):
            transform_image = cbimread(self._image_file.file_path).trans_image(scale=scale, rotate=rotation, offset=offset)
            cbimwrite(self._naming.transformed_image, transform_image)
            self._channel_image.Stitch.TransformShape = transform_image.shape
        else:
            transform_image = Image.open(self._naming.transformed_image)
            self._channel_image.Stitch.TransformShape = transform_image.size[::-1]

        if self._image_file.registration.trackline:
            # transform
            info = self._channel_image.Stitch.ScopeStitch

            if self._channel_image.QCInfo.TrackCrossQCPassFlag:
                stitch_template = self._channel_image.Stitch.TemplatePoint
                qc_template = self._channel_image.QCInfo.CrossPoints.stack_points
                self._channel_image.Stitch.TrackPoint = qc_template

                stitch_trans_template, _ = transform_points(
                    src_shape=(info.GlobalHeight, info.GlobalWidth),
                    scale=scale, rotation=-rotation,
                    points=stitch_template,
                    offset=offset
                )

                qc_trans_template, _ = transform_points(
                    src_shape=(info.GlobalHeight, info.GlobalWidth),
                    scale=scale, rotation=-rotation,
                    points=qc_template,
                    offset=offset
                )

                self._channel_image.Stitch.TrackPoint = qc_template
                self._channel_image.Stitch.TransformTemplate = stitch_trans_template
                self._channel_image.Stitch.TransformTrackPoint = qc_trans_template
                # 输出：参数写入ipr、txt、tif
                np.savetxt(self._naming.transformed_template, stitch_trans_template)
                np.savetxt(self._naming.transformed_track_template, qc_trans_template)

            if self._channel_image.QCInfo.ChipDetectQCPassFlag:
                # stitch chip box -> transform chip box
                stitch_chip_box = self._channel_image.QCInfo.ChipBBox.get().chip_box
                trans_chip_box, _ = transform_points(
                    src_shape=(info.GlobalHeight, info.GlobalWidth),
                    scale=(1 / self._channel_image.QCInfo.ChipBBox.ScaleX,
                           1 / self._channel_image.QCInfo.ChipBBox.ScaleY),
                    rotation=-self._channel_image.QCInfo.ChipBBox.Rotation,
                    points=stitch_chip_box,
                    offset=offset
                )
                trans_chip_box_info = ChipBoxInfo()
                trans_chip_box_info.set_chip_box(trans_chip_box)
                trans_chip_box_info.ScaleX, trans_chip_box_info.ScaleY = 1.0, 1.0
                trans_chip_box_info.Rotation = 0.0
                self._channel_image.Stitch.TransformChipBBox.update(trans_chip_box_info)
        final_tissue_mask = None
        final_cell_mask = None
        tissue_mask = None
        cell_mask = None
        if self._image_file.tissue_segmentation:
            if not self._naming.transform_tissue_mask_raw.exists():
                tissue_mask = self._tissue_segmentation(
                    image_path=self._naming.transformed_image,
                    res_path=self._naming.transform_tissue_mask_raw
                )
            else:
                tissue_mask_p = self._naming.transform_tissue_mask_raw
                tissue_mask = cbimread(tissue_mask_p, only_np=True)
            final_tissue_mask = tissue_mask
        if self._image_file.cell_segmentation:
            if not self._naming.transform_cell_mask_raw.exists():
                cell_mask = self._cell_segmentation(
                    image_path=self._naming.transformed_image,
                    res_path=self._naming.transform_cell_mask_raw
                )
            else:
                cell_mask_p = self._naming.transform_cell_mask_raw
                cell_mask = cbimread(cell_mask_p, only_np=True)
            final_cell_mask = cell_mask
        if tissue_mask is not None and cell_mask is not None:
            input_data = MaskManagerInfo(
                tissue_mask=tissue_mask,
                cell_mask=cell_mask,
                chip_box=self._channel_image.Stitch.TransformChipBBox.get(),
                method=1,
                stain_type=self._image_file.tech
            )
            btcm = BestTissueCellMask.get_best_tissue_cell_mask(input_data=input_data)
            final_tissue_mask = btcm.best_tissue_mask
            final_cell_mask = btcm.best_cell_mask
        if final_cell_mask is not None:
            cbimwrite(
                output_path=self._naming.transform_cell_mask,
                files=final_cell_mask
            )
        if final_tissue_mask is not None:
            cbimwrite(
                output_path=self._naming.transform_tissue_mask,
                files=final_tissue_mask
            )

    def extract4stitched(self, ):
        image = cbimread(self._image_file.file_path)
        if self._image_file.tech is TechType.IF:
            self._channel_image = ipr.IFChannel()
        else:
            self._channel_image = ipr.ImageChannel()
        self._channel_image.update_basic_info(
            chip_name=self._param_chip.chip_name,
            channel=image.channel,
            width=image.width,
            height=image.height,
            stain_type=self._image_file.tech_type,
            depth=image.depth
        )

        # 估计 & 第一次更新裁图尺寸
        cut_siz, est_scale = self._estimate_fov_size()

        if self._image_file.chip_detect:
            chip_info: ChipBoxInfo = self._detect_chip()
            self._channel_image.QCInfo.ChipDetectQCPassFlag = 1 if chip_info.IsAvailable else 0
            if chip_info.IsAvailable:
                # 第二次更新裁图尺寸
                scale = (chip_info.ScaleY + chip_info.ScaleX) / 2
                clog.info('Using the image chip box, calculate scale == {}'.format(scale))
                cut_siz = (int(self._fov_wh[0] * scale), int(self._fov_wh[1] * scale))
                clog.info('Estimate2 FOV-WH from {} to {}'.format(self._fov_wh, cut_siz))
        self._channel_image.ImageInfo.FOVHeight = cut_siz[1]
        self._channel_image.ImageInfo.FOVWidth = cut_siz[0]
        if self._image_file.quality_control:
            self._clarity()

        if self._image_file.registration.trackline:
            img_tpl = self._inference_template(cut_siz=cut_siz, est_scale=est_scale)
            if img_tpl.trackcross_qc_pass_flag:
                self._channel_image.QCInfo.TrackCrossQCPassFlag = 1
                clog.info('Template Scale is {}, rotation is {}'.format(
                    (img_tpl.scale_x, img_tpl.scale_y), img_tpl.rotation))

        if self._image_file.chip_detect and self._param_chip.is_after_230508():  # 满足配准前置的条件
            if chip_info.IsAvailable and img_tpl.trackcross_qc_pass_flag:
                clog.info('The chip-data meets the pre-registration conditions')
                self._pre_registration()
        cpf = True if self._channel_image.QCInfo.ChipDetectQCPassFlag == 1 else False
        tcf = True if self._channel_image.QCInfo.TrackCrossQCPassFlag == 1 else False
        self._channel_image.QCInfo.QCPassFlag = (cpf or tcf)

        clog.info('ImageQC result is {}'.format(self._channel_image.QCInfo.QCPassFlag))

    @process_decorator('GiB')
    def _pre_registration(self, ):
        moving_image = ChipFeature()
        moving_image.tech_type = self._image_file.tech_type
        moving_image.set_mat(cbimread(self._image_file.file_path))
        moving_image.set_template(self._channel_image.stitched_template_info)
        moving_image.set_chip_box(self._channel_image.box_info)
        moving_image.set_point00(self._param_chip.zero_zero_point)
        moving_image.set_anchor_point(tuple(self._param_chip.zero_zero_chip_point))

        res = template_00pt.template_00pt_align(moving_image=moving_image,
                                                ref=self._param_chip.fov_template,
                                                dst_shape=(self._param_chip.height, self._param_chip.width))

        # TODO 临时测试用
        with open(os.path.join(self.output_path, 'register_00pt.txt'), 'w') as f:
            f.writelines(f"offset: {res.offset} \n")

        # res.register_mat.write(os.path.join(self.output_path, "register_00pt.tif"))

        # self._channel_image.update_registration(res)

    @process_decorator('GiB')
    def _estimate_fov_size(self, ):
        scale = self._scale_estimate()  # 尺度估计
        clog.info('Using the image and chip prior size, calculate scale == {}'.format(scale))
        wh = (int(self._fov_wh[0] * scale), int(self._fov_wh[1] * scale))
        clog.info('Estimate1 FOV-WH from {} to {}'.format(self._fov_wh, wh))
        return wh, scale

    def _scale_estimate(self, ):
        image = cbimread(self._image_file.file_path)
        mx = max(image.width, image.height) / max(self._param_chip.width, self._param_chip.height)
        my = min(image.width, image.height) / min(self._param_chip.width, self._param_chip.height)
        return (mx + my) / 2

    @process_decorator('GiB')
    def _detect_chip(self, ) -> ChipBoxInfo:
        from cellbin2.contrib import chip_detector

        actual_size = self._param_chip.norm_chip_size
        info = chip_detector.detect_chip(file_path=self._image_file.file_path,
                                         cfg=self._config.chip_detector,
                                         stain_type=self._image_file.tech,
                                         actual_size=actual_size)
        self._channel_image.QCInfo.ChipBBox.update(box=info)

        return info

    @process_decorator('GiB')
    def _inference_template(self, cut_siz: Tuple[int, int], est_scale: float, overlap=0.0):
        from cellbin2.contrib import inference

        points_info, template_info = inference.template_inference(
            ref=self._param_chip.fov_template,
            track_points_config=self._config.track_points,
            track_lines_config=self._config.track_lines,
            template_v1_config=self._config.template_ref_v1,
            template_v2_config=self._config.template_ref_v2,
            file_path=self._image_file.file_path,
            stain_type=self._image_file.tech,
            fov_wh=cut_siz,
            est_scale=est_scale,
            overlap=overlap)
        self._channel_image.update_template_points(points_info=points_info, template_info=template_info)
        # np.savetxt(os.path.join(self.output_path, 'stitched.txt'), img_tpl.template_points)

        return template_info

    @process_decorator('GiB')
    def _clarity(self, ):
        from cellbin2.contrib import clarity

        c = clarity.run_detect(
            img_file=self._image_file.file_path,
            cfg=self._config.clarity,
            stain_type=self._image_file.tech
        )
        self._channel_image.QCInfo.update_clarity(cut_siz=64, overlap=0, score=int(c.score * 100), pred=c.preds)

    @process_decorator('GiB')
    def _tissue_segmentation(self, image_path: str, res_path: str):
        from cellbin2.contrib.tissue_segmentor import segment4tissue
        tissue_input = TissueSegInputInfo(
            weight_path_cfg=self._config.tissue_segmentation,
            input_path=image_path,
            stain_type=self._image_file.tech,
        )
        tissue_mask_output = segment4tissue(tissue_input)
        tissue_mask = tissue_mask_output.tissue_mask
        cbimwrite(res_path, tissue_mask)
        return tissue_mask

    @process_decorator('GiB')
    def _cell_segmentation(self, image_path: str, res_path: str):

        if self._image_file.tech == TechType.IF:
            from cellbin2.contrib import cellpose_segmentor

            cell_mask = cellpose_segmentor.segment4cell(
                input_path=image_path,
                cfg=self._config.cell_segmentation,
            )
        else:
            from cellbin2.contrib import cell_segmentor

            cell_mask = cell_segmentor.segment4cell(
                input_path=image_path,
                cfg=self._config.cell_segmentation,
                s_type=self._image_file.tech,
                fast=False, gpu=0)
        cbimwrite(res_path, cell_mask)
        return cell_mask

    def _adjust_mask(self, ):
        # TODO：更优的细胞分割结果输出
        clog.info('Adjust mask need update, please wait a moment..')
        pass

    @staticmethod
    def _expand_mask(mask: np.typing.NDArray, distance: int, process: int) -> np.typing.NDArray[np.uint8]:
        from cellbin2.contrib.fast_correct import run_fast_correct

        if distance < 1:
            clog.info(f"distance is: {distance} which is less than 0, return mask as it is")
            return mask

        fast_mask = run_fast_correct(
            mask=mask,
            dis=distance,
            process=process
        )
        return fast_mask

    def export_record(self, tag='record.ipr'):
        self._channel_image.write(os.path.join(self.output_path, tag), extra={})

    @property
    def channel_image(self, ):
        return self._channel_image


def main():
    from cellbin.modules.metadata import default_image
    from cellbin.utils.config import Config

    config_file = r'D:\01.code\cellbin2\dev\cellbin2\cellbin\config\cellbin.yaml'
    weights_root = r'D:\01.code\cellbin2\weights'

    image_file = default_image(file_path=r"D:\02.data\temp\A03599D1\fov_stitched_DAPI.tif",
                               tech_type='DAPI', clarity=False)
    ife = ImageFeatureExtract(image_file=image_file, output_path=r'D:\02.data\temp\A03599D1\cellbin2')
    ife.set_config(Config(config_file, weights_root))
    ife.extract4stitched()
    param = ife.channel_image.Register
    ife.extract4transform(scale=(1 / param.ScaleX, 1 / param.ScaleY),
                          rotation=param.Rotation, offset=(param.OffsetX, param.OffsetY))
    ife.export_record()


if __name__ == '__main__':
    main()
