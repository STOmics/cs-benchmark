from cellbin.modules import CellBinElement, StainType
from cellbin.dnn.tseg.bcdu.detector import TissueSegmentationBcdu
from cellbin.utils import clog


# class StainType(enum.Enum):
#     ssDNA = 'ssdna'
#     DAPI = 'dapi'
#     HE = 'HE'
#     mIF = 'mIF'
#     rna = 'rna'


class TissueSegmentation(CellBinElement):
    def __init__(self, model_path, stype='ssdna', gpu="-1", num_threads=0):
        """
        Args:
            model_path(str): network model file path
            stype(str):ssdna or rna
            gpu(str): gpu index
            num_threads(int): default is 0,When you use the CPU,
            you can use it to control the maximum number of threads
        """
        super(TissueSegmentation, self).__init__()
        self._stype = stype
        self._INPUT_SIZE = (512, 512, 1)
        self._model_path = model_path
        self._gpu = gpu
        self._num_threads = num_threads

        self._tiss_seg = TissueSegmentationBcdu(img_type=self._stype, input_size=self._INPUT_SIZE,
                                                gpu=self._gpu, num_threads=self._num_threads)
        clog.info("start loading model weight")
        self._tiss_seg.f_init_model(model_path=self._model_path)
        clog.info("end loading model weight")

    def run(self, img):
        """
        run tissue predict
        Args:
            img(ndarray): img array

        Returns(ndarray):tissue mask

        """
        clog.info("start tissue seg")
        mask = self._tiss_seg.f_predict(img)
        clog.info("end tissue seg")
        return mask
