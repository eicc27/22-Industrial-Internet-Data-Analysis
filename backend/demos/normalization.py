import numpy as np

from ..preproc.normalization import Norm, NormMethods
class NormPlot(Norm):
    def __init__(self, method: NormMethods) -> None:
        data = np.linspace()
        super().__init__(data, column_index, method)