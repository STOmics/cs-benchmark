import os

from cellbin.utils import clog

WEIGHTS = {
    'clarity_eval_mobilev3small05064_DAPI_20230202_pytorch.onnx': "https://github.com/STOmics/StereoCell_v2.0/releases/download/V1/clarity_eval_mobilev3small05064_DAPI_20230202_pytorch.onnx",
    'clarity_eval_mobilev3small05064_DAPI_20230608_pytorch.onnx': "https://github.com/STOmics/StereoCell_v2.0/releases/download/V1/clarity_eval_mobilev3small05064_DAPI_20230608_pytorch.onnx",
    'points_detect_yolov5obb_SSDNA_20220513_pytorch.onnx': "https://github.com/STOmics/StereoCell_v2.0/releases/download/V1/points_detect_yolov5obb_SSDNA_20220513_pytorch.onnx",
    'tissueseg_yolo_SH_20230131_th.onnx': "https://github.com/STOmics/StereoCell_v2.0/releases/download/V1/tissueseg_yolo_SH_20230131_th.onnx",
    'tissueseg_bcdu_SDI_220822_tf.onnx': "https://github.com/STOmics/StereoCell_v2.0/releases/download/V1/tissueseg_bcdu_SDI_220822_tf.onnx",
    'tissueseg_bcdu_SDI_230523_tf.onnx': "https://github.com/STOmics/StereoCell_v2.0/releases/download/V1/tissueseg_bcdu_SDI_230523_tf.onnx",
    'tissueseg_bcdu_H_221101_tf.onnx': "https://github.com/STOmics/StereoCell_v2.0/releases/download/V1/tissueseg_bcdu_H_230602_tf.onnx",
    'tissueseg_bcdu_rna_220909_tf.onnx': "https://github.com/STOmics/StereoCell_v2.0/releases/download/V1/tissueseg_bcdu_rna_220909_tf.onnx",
    'cellseg_bcdu_SHDI_221008_tf.onnx': "https://github.com/STOmics/StereoCell_v2.0/releases/download/V1/cellseg_bcdu_SHDI_221008_tf.onnx",
}


# pwd = os.path.dirname(__file__)

# Just for test
# checkpoints = {
#     'ClarityEvaler': {
#         'Local': 'ST_TP_Mobile_small_050.onnx',
#         'Remote': ''
#     },
#     'PointsDetector': {
#         'Local': '',
#         'Remote': ''
#     },
#     'PointsFilter': {
#         'Local': '',
#         'Remote': ''
#     },
#     'TissueYolo': {
#         'Local': '',
#         'Remote': ''
#     },
#     'TissueBCDU': {
#         'Local': '',
#         'Remote': ''
#     },
#     'CellBCDU': {
#         'Local': '',
#         'Remote': ''
#     }
# }


def auto_download_weights(save_dir, names):
    for k, url in WEIGHTS.items():
        if k not in names:
            continue
        weight = os.path.join(save_dir, k)
        if not os.path.exists(weight):
            try:
                import requests
                clog.info('Download {} from remote {}'.format(k, url))
                r = requests.get(url)
                with open(os.path.join(save_dir, k), 'wb') as fd:
                    fd.write(r.content)
            except Exception as e:
                clog.error('FAILED! (Download {} from remote {})'.format(k, url))
                print(e)
                return 1
        else:
            clog.info('{} already exists'.format(k))
    return 0


if __name__ == '__main__':
    save_dir = r"D:\Data\weights_2"
    names = WEIGHTS.keys()
    auto_download_weights(
        save_dir=save_dir,
        names=names,
    )
