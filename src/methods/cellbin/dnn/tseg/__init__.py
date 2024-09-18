from abc import ABC, abstractmethod


class TissueSegmentation(ABC):
    @abstractmethod
    def f_predict(self, img):
        """
        input img output cell mask
        :param img:CHANGE
        :return: 掩模大图
        """
        return
