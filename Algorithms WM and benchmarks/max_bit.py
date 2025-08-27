from watermark_system import WatermarkSystem, WatermarkType
import utils
from base_work.data import DataTestModel
import os
import matplotlib.pyplot as plt
import numpy as np

DATASET_PATH = "../dataset/Water Bodies Dataset"


def get_success_extract(dataset, dataset_mask, watermark_type, wm_len=4, is_using_mask=True):
    wm_bits = np.random.randint(0, 2, wm_len)
    if watermark_type == WatermarkType.DWT_DCT_SVD:
        args_wm = (wm_bits, 16, 4)
    elif watermark_type == WatermarkType.MAX_DCT:
        args_wm = (wm_bits, 16, 4)
    elif watermark_type == WatermarkType.DWT_DCT:
        args_wm = (6512, wm_bits, 25, 4)
    elif watermark_type == WatermarkType.JAWS:
        args_wm = (6512, wm_bits, 128, 0.04)

    wm_sys = WatermarkSystem(watermark_type, args_wm)
    len_dataset = 10
    num_success = 0
    for i in range(len_dataset):
        c = dataset[i][:, :, 0]
        if not is_using_mask:
            mask = np.ones_like(c)
        else:
            mask = dataset_mask[i]//255
        cw = wm_sys.encode((c, mask))
        if watermark_type == WatermarkType.JAWS:
            bits_extracted, _ = wm_sys.decode((cw, True))
            bits_extracted = np.array(bits_extracted)
            hidden_values = utils.binary_array_to_decimal(wm_bits)
            extracted_values = utils.binary_array_to_decimal(bits_extracted)
            if hidden_values == extracted_values:
                num_success += 1
        else:
            bits_extracted = np.array(wm_sys.decode((cw, mask)))
            if np.max(np.abs(bits_extracted - wm_bits)) == 0:
                num_success += 1
        print(utils.score_extract(wm_bits, bits_extracted))
    return num_success


if __name__ == "__main__":
    img_path = os.path.join(DATASET_PATH, "Images")
    mask_path = os.path.join(DATASET_PATH, "Masks")
    dataset = DataTestModel(
        384, img_path, is_transformed=False, is_tensor_input=False)
    dataset_mask = DataTestModel(
        384, mask_path, is_transformed=False, is_tensor_input=False)
    wm_len = 4
    x = 0
    ls = np.zeros(10, dtype=int)
    for i in range(1000):
        curr_ratio = np.count_nonzero(dataset_mask[i][:, :, 0]//255)/384**2
        ls[int(curr_ratio//0.1)] += 1

    print(ls)
    print(len(ls))
    print(np.sum(ls))
    # while True:
    # num_success = get_success_extract(dataset, dataset_mask, WatermarkType.JAWS, wm_len, is_using_mask=True)
    # print(f"Success: {num_success}")
    # print(f"WM length: {wm_len}")
    # wm_len=wm_len*2
    # if wm_len > 64:
    #     break
    #
