import cellprofiler_core.pipeline
import cellprofiler_core.preferences
import cellprofiler_core.utilities.java
import pathlib
import os
import argparse


work_path = os.path.abspath('.')
def cellprofiler_method(para, args):
    cellprofiler_core.preferences.set_headless()
    cellprofiler_core.utilities.java.start_java()

    pipeline = cellprofiler_core.pipeline.Pipeline()
    if para.img_type == 'he':
        pipeline.load(os.path.join(work_path, 'src/methods/cellprofiler/cellsegmentation_HE.cppipe'))
    elif para.img_type == 'mif':
        pipeline.load(os.path.join(work_path, 'src/methods/cellprofiler/cellsegmentation_mif.cppipe'))
    else:
        pipeline.load(os.path.join(work_path, 'src/methods/cellprofiler/cellsegmentation_erode.cppipe'))
    if not os.path.exists(para.output):
            os.makedirs(para.output)
    cellprofiler_core.preferences.set_default_output_directory(para.output)

    # Read images from the specified image_path
    file_list = list(pathlib.Path(para.image_path).glob('*.tif'))
    files = [file.as_uri() for file in file_list]
    pipeline.read_file_list(files)
    output_measurements = pipeline.run()

    cellprofiler_core.utilities.java.stop_java()

USAGE = 'cellprofiler'
PROG_VERSION = 'v0.0.1'

def main():
    arg_parser = argparse.ArgumentParser(usage=USAGE)
    arg_parser.add_argument("--version", action="version", version=PROG_VERSION)
    arg_parser.add_argument("-o", "--output", action="store", dest="output",
                            type=str, default=None, help="Save path of stitch result files.")
    arg_parser.add_argument("-i", "--image_path", action="store", dest="image_path",
                            type=str, default=None, help="FOV images storage location.")
    arg_parser.add_argument("-g", "--is_gpu", action="store", dest="is_gpu",
                            type=bool, default=False, help="Use GPU or not.")
    arg_parser.add_argument("-t", "--img_type", help="ss/he")
    arg_parser.set_defaults(func=cellprofiler_method)
    (para, args) = arg_parser.parse_known_args()
    # weights = os.path.join(os.path.abspath('.'), "weights/cellpose")
    # models_logger.info('Load Model - Cellpose from {}'.format(weights))
    
    # os.environ["CELLPOSE_LOCAL_MODELS_PATH"] = weights
    para.func(para, args)
 


if __name__ == '__main__':
    import sys
    
    return_code = main()
    sys.exit(return_code)
