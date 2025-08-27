import os.path
from enum import Enum
import cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from base_work.data import DataTestModel
from watermark.Distortion.distortion import jpeg_compress, contract, cut, rotation
from watermark.JAWS.jaws import SystemJASW
from base_work import utils

DATASET_PATH = "../../dataset/Water Bodies Dataset"

class Distortion(Enum):
    JPEG_COMPRESSION = 0
    CONTRACT = 1
    CUT = 2
    ROTATION = 3

def compare_extract_variances(m = 128):
    img_path = os.path.join(DATASET_PATH, "Images")
    dataset = DataTestModel(384, img_path, is_transformed=False, is_tensor_input=False)
    mask_path = os.path.join(DATASET_PATH, "Masks")
    dataset_mask = DataTestModel(384, mask_path, is_transformed=False, is_tensor_input=False)

    K = 6512
    success_count_normalize = 0
    success_count_non_normalize = 0
    for i in tqdm(range(500), desc="Processing", unit="iter"):
        wmBit = np.random.randint(0, 2, 7)
        JAWS = SystemJASW(K, wmBit, m, 0.01)
        C = dataset[i][:,:,0]
        mask = dataset_mask[i][:,:,0]//255
        CW = JAWS.encode(C, mask)
        value_extracted, is_success = JAWS.decode(CW, True)

        value_extracted = utils.binary_array_to_decimal(value_extracted)
        value_to_hide = utils.binary_array_to_decimal(wmBit)
        success_count_normalize += int(value_extracted == value_to_hide)

        value_extracted, is_success = JAWS.decode(CW, False)
        value_extracted = utils.binary_array_to_decimal(value_extracted)
        success_count_non_normalize += int(value_extracted==value_to_hide)

    return success_count_normalize, success_count_non_normalize

def stable_distortion(image_name, distortion_type=Distortion.JPEG_COMPRESSION, is_using_mask=True):
    image_path = os.path.join(DATASET_PATH, "Images", image_name)
    mask_path = os.path.join(DATASET_PATH, "Masks", image_name)
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    second_params_name = ['JPEG-Quality', 'Scale', 'Cut', 'Rotation']
    functions_distortion = [jpeg_compress, contract, cut, rotation]
    C = img[:,:,0]
    if is_using_mask:
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)/255
    else:
        mask = np.ones_like(C)

    C = cv2.resize(C, (384, 384))
    mask = cv2.resize(mask, (384, 384))

    K = 6512
    M = 128
    value_to_hide = 20
    second_params = []
    alphas = np.arange(0.002, 0.12, 0.01)

    if distortion_type == Distortion.JPEG_COMPRESSION:
        second_params = np.arange(10, 100, 10)
    elif distortion_type == Distortion.CONTRACT:
        second_params = np.arange(1.1, 1.5, 0.1)
    elif distortion_type == Distortion.CUT:
        second_params = np.arange(0.1, 0.99, 0.1)
    elif distortion_type == Distortion.ROTATION:
        second_params = np.arange(1, 99.8, 8.9)

    is_success = np.zeros((len(alphas), len(second_params)))
    psnr_values = np.zeros((len(alphas), len(second_params)))

    for i, alpha in enumerate(alphas):
        for j, second_param in enumerate(second_params):
            JAWS = SystemJASW(K, value_to_hide, M, alpha)
            CW = JAWS.encode(C, mask)
            CW = np.clip(CW, 0, 255)
            if distortion_type == Distortion.CUT:
                input_distortion = (C, CW)
            else:
                input_distortion = (CW, )
            CW_mod = functions_distortion[distortion_type.value](*input_distortion, second_param)
            value_extracted, success = JAWS.decode(CW_mod, True)
            psnr = utils.calculate_psnr(C/255, CW_mod/255)
            is_success[i][j] = success
            psnr_values[i][j] = psnr

    plt.figure()
    for i, alpha in enumerate(alphas):
        for j, second_param in enumerate(second_params):
            color = 'green' if is_success[i, j] else 'red'
            plt.scatter(alpha, second_param, color=color)
            psnr_value = psnr_values[i, j]
            plt.text(alpha, second_param, f'{psnr_value:.2f}', fontsize=9, ha='right', color='blue')

    plt.title(f'Success Indicator for Alpha and {second_params_name[distortion_type.value]}')
    plt.xlabel('Alpha')
    plt.ylabel(second_params_name[distortion_type.value])
    plt.grid(True)


if __name__ == "__main__":
    print(compare_extract_variances(128))

    # image_name = "water_body_4.jpg"
    # image_path = os.path.join(DATASET_PATH, "Images", image_name)
    # mask_path = os.path.join(DATASET_PATH, "Masks", image_name)
    # img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    # img = cv2.resize(img, (384, 384))
    # red_channel = img[:,:,2]
    # mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)/255
    # mask = cv2.resize(mask, (384, 384))
    #
    # rotate_red_channel = rotation(red_channel, 1)
    # print(utils.calculate_psnr(red_channel/255, rotate_red_channel/255))
    #
    # plt.figure()
    # plt.imshow(red_channel, cmap='gray')
    # plt.figure()
    # plt.imshow(rotate_red_channel, cmap='gray')
    # plt.show()
    #


    # stable_distortion("water_body_14.jpg", Distortion.JPEG_COMPRESSION, True)
    # stable_distortion("water_body_14.jpg", Distortion.JPEG_COMPRESSION, False)

    # stable_distortion("water_body_14.jpg", Distortion.CONTRACT, True)
    # stable_distortion("water_body_14.jpg", Distortion.CONTRACT, False)

    # stable_distortion("water_body_14.jpg", Distortion.CUT, True)
    # stable_distortion("water_body_14.jpg", Distortion.CUT, False)
    #
    # stable_distortion("water_body_14.jpg", Distortion.ROTATION, True)
    # stable_distortion("water_body_14.jpg", Distortion.ROTATION, False)
    # plt.show()