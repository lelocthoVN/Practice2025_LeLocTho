from enum import Enum
from DWT_DCT_SVD.dwt_dct_svd import EmbedDwtDctSvd
from DWT_DCT.dwt_dct import DWTDCTSystem
from LSB.lsb import LSBSystem


class WatermarkType(Enum):
    DWT_DCT_SVD = 1
    DWT_DCT = 2
    LSB = 3


class WatermarkSystem:
    def __init__(self, watermark_type, args):
        self.watermark_type = watermark_type
        if self.watermark_type == WatermarkType.DWT_DCT_SVD:
            self.wm_sys = EmbedDwtDctSvd(*args)
        elif self.watermark_type == WatermarkType.DWT_DCT:
            self.wm_sys = DWTDCTSystem(*args)
        elif self.watermark_type == WatermarkType.LSB:
            self.wm_sys = LSBSystem(*args)
        else:
            raise Exception("Invalid Watermark Type")

    def encode(self, args):
        return self.wm_sys.encode(*args)

    def decode(self, args):
        return self.wm_sys.decode(*args)
