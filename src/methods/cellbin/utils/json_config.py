import os
from json import dumps, loads, load

from cellbin.utils import clog


class JSONObject:
    def __init__(self, d):
        self.__dict__ = d

    def __getattr__(self, item):
        return None


def json_2_dict(file_path: str) -> dict:
    with open(file_path, 'r', encoding='utf-8') as fd:
        return load(fd)


def dict_deserialize(dct: dict) -> JSONObject:
    str_dct = dumps(dct, indent=2)
    clog.info(str_dct)
    return loads(str_dct, object_hook=JSONObject)


def json_deserialize(file_path: str) -> JSONObject:
    dct = json_2_dict(file_path)
    return dict_deserialize(dct)


class TrackPtConfig(object):
    def __init__(self):
        self.track_point_score_threshold = 0
        self.track_point_first_level_threshold = 5
        self.track_point_second_level_threshold = 20
        self.track_point_good_threshold = 5
        self.most_freq_angle_threshold = 0.05


class WeightsConfig(object):
    def __init__(self, track_pt_detect, tissue_seg, clarity_eval):
        self.track_pt_detect = track_pt_detect
        self.tissue_seg = tissue_seg
        self.clarity_eval = clarity_eval


class ModelWeightsConfig(object):
    def __init__(self):
        self.SSDNA = WeightsConfig(
            track_pt_detect="points_detect_yolov5obb_SSDNA_20220513_pytorch.onnx",
            tissue_seg="tissueseg_yolo_SH_20230131_th.onnx",
            clarity_eval="clarity_eval_mobilev3small05064_DAPI_20230202_pytorch.onnx"
        )
        self.DAPI = WeightsConfig(
            track_pt_detect="points_detect_yolov5obb_SSDNA_20220513_pytorch.onnx",
            tissue_seg="tissueseg_yolo_SH_20230131_th.onnx",
            clarity_eval="clarity_eval_mobilev3small05064_DAPI_20230202_pytorch.onnx"
        )
        self.HE = WeightsConfig(
            track_pt_detect="points_detect_yolov5obb_SSDNA_20220513_pytorch.onnx",
            tissue_seg="tissueseg_yolo_SH_20230131_th.onnx",
            clarity_eval="clarity_eval_mobilev3small05064_DAPI_20230202_pytorch.onnx"
        )


class TrackLineConfig(object):
    def __init__(self):
        self.track_line_score_threshold = 0
        self.track_line_topk = 10


class ClarityConfig(object):
    def __init__(self):
        self.clarity_score_threshold = 0.85
        self.cluster_area_threshold = 0.05
        self.cluster_width_threshold = 0.2
        self.cluster_height_threshold = 0.2


class GlobalTemplateConfig(object):
    def __init__(self):
        self.template_pair_points_threshold = 10
        self.template_pair_points_qc_threshold = 5
        self.template_range_image_size_threshold = 5000
        self.template_pair_correct_threshold = 20
        self.template_cluster_num_threshold = 10
        self.template_v2_scale_range_threshold = 0.7
        self.template_v2_rotate_range_threshold = 35
        self.template_v2_search_range_threshold = 500
        self.template_v2_rotate_fov_min_threshold = 7


class RunningConfig(object):
    def __init__(self):
        self.pt_running_process = 5
        self.stitch_running_process = 5


class Operation(object):
    def __init__(self):
        self.tissue_segment = True
        self.track_pt_detect = True
        self.track_line_detect = True
        self.stitch_qc = True
        self.clarity_qc = True


class Channels(object):
    def __init__(self, pt_detect=-1, tissue_seg=-1, line_detect=-1, fft=0, clarity=-1):
        self.pt_detect = pt_detect
        self.tissue_seg = tissue_seg
        self.line_detect = line_detect
        self.fft = fft
        self.clarity = clarity


class ChannelConfig(object):
    def __init__(self):
        self.SSDNA = Channels()
        self.DAPI = Channels()
        self.HE = Channels(fft=1, tissue_seg=0)
        self.CZI = Channels(pt_detect=1, line_detect=1, fft=0, clarity=0)


class ConfigReader(object):
    def __init__(self):
        self.track_pt_config = TrackPtConfig()
        self.track_line_config = TrackLineConfig()
        self.clarity_config = ClarityConfig()
        self.global_template_config = GlobalTemplateConfig()
        self.channel_config = ChannelConfig()
        self.running_config = RunningConfig()
        self.operation = Operation()
        self.model_weights_config = ModelWeightsConfig()

    def load_from_json(self, json_path: str):
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"{json_path} not exist!")
        obj = json_deserialize(json_path)
        dct = self.__dict__.copy()
        for attr in dct:
            if hasattr(obj, attr):
                setattr(self, attr, getattr(obj, attr))


if __name__ == '__main__':
    json_path = r"D:\PycharmProjects\pipeline-research\src\qc_config.json"
    config = ConfigReader()
    config.load_from_json(json_path)
    print(config)
