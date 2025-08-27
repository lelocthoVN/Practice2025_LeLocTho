from enum import Enum
import os.path
import cv2
import matplotlib.pyplot as plt
import numpy as np
import utils
from Distortion.distortion import jpeg_compress, contract, cut, rotation
from watermark_system import WatermarkSystem, WatermarkType
from Distortion.distortion import DistortionType

DATASET_PATH = "../dataset/Water Bodies Dataset"


def stable_distortion(image_name, watermark_type, distortion_type=DistortionType.JPEG_COMPRESSION, is_using_mask=True):
    image_path = os.path.join(DATASET_PATH, "Images", image_name)
    mask_path = os.path.join(DATASET_PATH, "Masks", image_name)
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    second_params_name = ['JPEG-Quality', 'Scale', 'Cut', 'Rotation']
    functions_distortion = [jpeg_compress, contract, cut, rotation]

    C = img[:, :, 0]
    if is_using_mask:
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)/255
    else:
        mask = np.ones_like(C)
    K = 6512

    value_to_hide = 20
    wm_bits = utils.decimal_to_binary_array(value_to_hide)
    second_params = []
    scales = np.arange(20, 42, 2)
    C = cv2.resize(C, (384, 384))
    mask = cv2.resize(mask, (384, 384))

    if distortion_type == DistortionType.JPEG_COMPRESSION:
        second_params = np.arange(10, 100, 10)
    elif distortion_type == DistortionType.CONTRAST:
        second_params = np.arange(1.1, 1.5, 0.05)
    elif distortion_type == DistortionType.CUT:
        second_params = np.arange(0.1, 1.0, 0.1)
    elif distortion_type == DistortionType.ROTATION:
        second_params = np.arange(1, 50, 5)

    is_success = np.zeros((len(scales), len(second_params)))
    psnr_values = np.zeros((len(scales), len(second_params)))

    for i, scale in enumerate(scales):
        for j, second_param in enumerate(second_params):
            wm_sys = WatermarkSystem(watermark_type, (6512, wm_bits, scale, 4))
            CW = wm_sys.encode((C, mask))
            CW = np.clip(CW, 0, 255)
            if distortion_type == DistortionType.CUT:
                input_distortion = (C, CW)
            else:
                input_distortion = (CW, )
            CW_mod = functions_distortion[distortion_type.value](
                *input_distortion, second_param)
            bits_extracted = wm_sys.decode((CW_mod, mask))
            value_extracted = utils.binary_array_to_decimal(bits_extracted)

            psnr = utils.calculate_psnr(CW, CW_mod)
            is_success[i][j] = (value_extracted == value_to_hide)
            psnr_values[i][j] = psnr
    plt.figure()
    for i, scale in enumerate(scales):
        for j, second_param in enumerate(second_params):
            color = 'green' if is_success[i, j] else 'red'
            plt.scatter(scale, second_param, color=color)
            psnr_value = psnr_values[i, j]
            plt.text(scale, second_param,
                     f'{psnr_value:.2f}', fontsize=9, ha='right', color='blue')
    plt.title(
        f'Success Indicator for Alpha and {second_params_name[distortion_type.value]}')
    plt.xlabel('Alpha')
    plt.ylabel(second_params_name[distortion_type.value])
    plt.grid(True)


if __name__ == "__main__":
    # stable_distortion("water_body_14.jpg", WatermarkType.DWT_DCT_SVD, DistortionType.JPEG_COMPRESSION, True)
    # stable_distortion("water_body_14.jpg", WatermarkType.DWT_DCT_SVD, DistortionType.JPEG_COMPRESSION, False)
    # stable_distortion("water_body_14.jpg", WatermarkType.DWT_DCT_SVD, DistortionType.CONTRAST, True)
    # stable_distortion("water_body_14.jpg", WatermarkType.DWT_DCT_SVD, DistortionType.CONTRAST, False)
    # stable_distortion("water_body_14.jpg", WatermarkType.DWT_DCT_SVD, DistortionType.CUT, True)
    # stable_distortion("water_body_14.jpg", WatermarkType.DWT_DCT_SVD, DistortionType.CUT, False)
    # stable_distortion("water_body_14.jpg", WatermarkType.DWT_DCT_SVD, DistortionType.ROTATION, True)
    # stable_distortion("water_body_14.jpg", WatermarkType.DWT_DCT_SVD, DistortionType.ROTATION, False)

    # stable_distortion("water_body_14.jpg", WatermarkType.MAX_DCT, DistortionType.JPEG_COMPRESSION, True)
    # stable_distortion("water_body_14.jpg", WatermarkType.MAX_DCT, DistortionType.JPEG_COMPRESSION, False)
    # stable_distortion("water_body_14.jpg", WatermarkType.MAX_DCT, DistortionType.CONTRAST, True)
    # stable_distortion("water_body_14.jpg", WatermarkType.MAX_DCT, DistortionType.CONTRAST, False)
    # stable_distortion("water_body_14.jpg", WatermarkType.MAX_DCT, DistortionType.CUT, True)
    # stable_distortion("water_body_14.jpg", WatermarkType.MAX_DCT, DistortionType.CUT, False)
    # stable_distortion("water_body_14.jpg", WatermarkType.MAX_DCT, DistortionType.ROTATION, True)
    # stable_distortion("water_body_14.jpg", WatermarkType.MAX_DCT, DistortionType.ROTATION, False)

    # stable_distortion("water_body_14.jpg", WatermarkType.DWT_DCT, DistortionType.JPEG_COMPRESSION, True)
    # stable_distortion("water_body_14.jpg", WatermarkType.DWT_DCT, DistortionType.JPEG_COMPRESSION, False)
    # stable_distortion("water_body_14.jpg", WatermarkType.DWT_DCT, DistortionType.CONTRAST, True)
    # stable_distortion("water_body_14.jpg", WatermarkType.DWT_DCT, DistortionType.CONTRAST, False)
    # stable_distortion("water_body_14.jpg", WatermarkType.DWT_DCT, DistortionType.CUT, True)
    # stable_distortion("water_body_14.jpg", WatermarkType.DWT_DCT, DistortionType.CUT, False)
    # stable_distortion("water_body_14.jpg", WatermarkType.DWT_DCT, DistortionType.ROTATION, True)
    # stable_distortion("water_body_14.jpg", WatermarkType.DWT_DCT, DistortionType.ROTATION, False)
    stable_distortion("water_body_14.jpg", WatermarkType.JAWS,
                      DistortionType.JPEG_COMPRESSION, False)

    plt.show()
