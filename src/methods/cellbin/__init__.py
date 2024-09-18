import os

version = '1.2.8'
__version__ = version

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHT_DIR = os.path.join(ROOT_DIR, "weights")
PIPELINE_CONFIG_PATH = os.path.join(ROOT_DIR, "test", "scripts", "pipeline_config.json")
QC_CONFIG_PATH = os.path.join(ROOT_DIR, "test", "scripts", "qc_config.json")