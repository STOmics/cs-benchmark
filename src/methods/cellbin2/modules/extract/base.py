from typing import Union
from cellbin2.image import CBImage
import numpy as np
from cellbin2.contrib.param import ChipBoxInfo, TemplateInfo
from cellbin2.utils.config import Config
from cellbin2.modules.metadata import ProcFile
from cellbin2.utils.stereo_chip import StereoChip


class FeatureExtract(object):
    def __init__(self, image_file: ProcFile, output_path: str):
        self._image_file = image_file
        self._param_chip = StereoChip()
        self._config: Config
        self._chip_box: ChipBoxInfo = ChipBoxInfo()
        self._template: TemplateInfo = TemplateInfo()
        self._mat: Union[str, np.ndarray, CBImage] = ''
        self.output_path: str = output_path

    def set_chip_param(self, chip: StereoChip):
        self._param_chip = chip

    def set_config(self, cfg: Config):
        self._config = cfg

    @property
    def chip_box(self, ):
        return self._chip_box

    def set_chip_box(self, chip_box):
        self._chip_box = chip_box

    @property
    def mat(self, ):
        return self._mat

    @property
    def template(self, ):
        return self._template
