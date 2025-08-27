from pathlib import Path
from PIL import Image
import math
import os.path
import numpy as np
import cv2

# from base_work.data import DataTestModel
from watermark_system import WatermarkSystem, WatermarkType
from Distortion.distortion import (DistortionType, jpeg_compress, contract, cut, rotation, resized_crop,
                                   random_erasing, adjust_contrast, gaussian_blur, gaussian_noise)
import utils
from sklearn.metrics import roc_curve
from tqdm import tqdm
import pandas as pd
from scipy.interpolate import interp1d

DATASET_PATH = "E:\\waves-data\\main\\diffusiondb\\real"
results_path = "E:\\waves-data\\main\\results"

distortion_levels = {
    DistortionType.JPEG_COMPRESSION: np.arange(10, 100, 10),
    DistortionType.CONTRAST: np.arange(1.1, 1.5, 0.05),
    DistortionType.CUT: np.arange(0.1, 1., 0.1),
    DistortionType.ROTATION: np.arange(1, 50, 5),
    DistortionType.RESIZED_CROP: np.arange(0.5, 1., 0.05),
    DistortionType.RANDOM_ERASING: np.arange(0.05, 0.3, 0.03),
    DistortionType.ADJUST_CONTRAST: np.arange(0.2, 1., 0.1),
    DistortionType.GAUSSIAN_BLUR: np.arange(5, 22, 2),
    DistortionType.GAUSSIAN_NOISE: np.arange(0.02, 0.1, 0.01),
}

distortion_function = {
    DistortionType.JPEG_COMPRESSION: jpeg_compress,
    DistortionType.CONTRAST: contract,
    DistortionType.CUT: cut,
    DistortionType.ROTATION: rotation,
    DistortionType.RESIZED_CROP: resized_crop,
    DistortionType.RANDOM_ERASING: random_erasing,
    DistortionType.ADJUST_CONTRAST: adjust_contrast,
    DistortionType.GAUSSIAN_BLUR: gaussian_blur,
    DistortionType.GAUSSIAN_NOISE: gaussian_noise,
}


def robustness_by_psnr(dataset, watermark_type, distortion_type, is_using_mask=False):
    psnr_values = []
    tpr_at_0_1_fpr = []

    len_dataset = 50
    distortion_levels_ = distortion_levels[distortion_type]
    distortion_function_ = distortion_function[distortion_type]

    # if is_using_mask:
    #     mask_path = os.path.join(DATASET_PATH, "Masks")
    #     dataset_mask = DataTestModel(384, mask_path, is_transformed=False, is_tensor_input=False)

    for distortion_level_ in distortion_levels_:
        psnr_values_current_level = []
        scores = []
        y_true = []
        for i in tqdm(range(len_dataset), desc="Processing"):
            np.random.seed(None)
            if watermark_type == WatermarkType.JAWS or is_using_mask:
                wm_bits = np.random.randint(0, 2, 7)
            else:
                wm_bits = np.random.randint(0, 2, 32)
            args_wm = (wm_bits, )
            if watermark_type == WatermarkType.DWT_DCT_SVD:
                args_wm = (wm_bits, 36, 4)
            elif watermark_type == WatermarkType.MAX_DCT:
                args_wm = (wm_bits, 36, 4)
            elif watermark_type == WatermarkType.DWT_DCT:
                args_wm = (6512, wm_bits, 25, 4)
            elif watermark_type == WatermarkType.JAWS:
                args_wm = (6512, wm_bits, 128, 0.04)
            elif watermark_type == WatermarkType.LSB:
                args_wm = (6512, wm_bits, 0, 16)
            wm_sys = WatermarkSystem(watermark_type, args_wm)
            c = dataset[i][:, :, 0]

            # if not is_using_mask:
            mask = np.ones_like(c)
            # else:
            #     mask = dataset_mask[i][:, :, 0]//255
            cw = wm_sys.encode((c, mask))

            cw = np.clip(cw, 0, 255)
            if distortion_type == DistortionType.CUT:
                args_distortion_cw = (c, cw, distortion_level_)
                args_distortion_c = (c, c, distortion_level_)
            else:
                args_distortion_cw = (cw, distortion_level_)
                args_distortion_c = (c, distortion_level_)
            cw_distorted = distortion_function_(*args_distortion_cw)
            c_distorted = distortion_function_(*args_distortion_c)
            psnr_cw = utils.calculate_psnr(cw, cw_distorted)
            if psnr_cw == float('inf'):
                continue

            if watermark_type == WatermarkType.JAWS:
                extracted_bit, _ = wm_sys.decode((cw_distorted, True))
            else:
                extracted_bit = wm_sys.decode((cw_distorted, mask))
            score = utils.score_extract(wm_bits, extracted_bit)
            scores.append(score)
            y_true.append(1)

            if watermark_type == WatermarkType.JAWS:
                extracted_bit_empty_container, _ = wm_sys.decode(
                    (c_distorted, True))
            else:
                extracted_bit_empty_container = wm_sys.decode(
                    (c_distorted, mask))
            score_empty_container = utils.score_extract(
                wm_bits, extracted_bit_empty_container)
            scores.append(score_empty_container)
            y_true.append(0)
            psnr_values_current_level.append(psnr_cw)

        psnr_avg = np.mean(psnr_values_current_level)
        fpr, tpr, _ = roc_curve(y_true, scores)
        f_interp = interp1d(fpr, tpr, kind='linear', fill_value="extrapolate")
        psnr_values.append(psnr_avg)
        tpr_at_0_1_fpr.append(f_interp(0.01))

        print({
            'psnr_values': psnr_values,
            'tpr_at_0_1_fpr': tpr_at_0_1_fpr
        })

    data = {
        'psnr_values': psnr_values,
        'tpr_at_0_1_fpr': tpr_at_0_1_fpr
    }
    df = pd.DataFrame(data)
    output_file = f"output_data_{watermark_type.name}_{distortion_type.name}{'_HAS_MASK' if is_using_mask else ''}.csv"
    output_file = os.path.join(results_path, output_file)
    df.to_csv(output_file, index=False)
    print(f"Data has been saved to {output_file}")


def generate_square_mask_grid(image, num_clusters, area_fraction):
    H, W = image.shape[:2]
    total_pixels = H * W
    target_area = area_fraction * total_pixels

    area_per_square = target_area / num_clusters
    side_float = math.sqrt(area_per_square)
    side = int(math.floor(side_float))

    if side < 1:
        raise ValueError(
            "Kích thước hình vuông quá nhỏ (side < 1). Hãy tăng area_fraction hoặc giảm num_clusters")
    if side > min(H, W):
        raise ValueError(
            "Kích thước hình vuông lớn hơn ảnh. Giảm area_fraction hoặc num_clusters")

    per_row = W // side
    max_rows = H // side
    capacity = per_row * max_rows
    if capacity < num_clusters:
        raise ValueError(
            f"Không đủ chỗ đặt {num_clusters} hình vuông (chỉ đặt được {capacity}). Giảm side, area_fraction hoặc num_clusters")

    mask = np.zeros((H, W), dtype=np.uint8)

    for i in range(num_clusters):
        row = i // per_row
        col = i % per_row
        x1 = col * side
        y1 = row * side
        x2 = x1 + side
        y2 = y1 + side
        cv2.rectangle(mask, (x1, y1), (x2, y2), color=1, thickness=-1)

    return mask


def load_dataset_images(image_dir, size=(384, 384), limit=50):
    # """
    # Đọc và resize ảnh RGB từ thư mục, trả về list NumPy arrays.

    # Args:
    #     image_dir (str or Path): Đường dẫn đến thư mục ảnh.
    #     size (tuple): Kích thước cần resize về (width, height).
    #     limit (int): Số ảnh cần đọc (nếu có).

    # Returns:
    #     List[np.ndarray]: Danh sách ảnh RGB dạng (H, W, 3)
    # """
    image_dir = Path(image_dir)
    dataset = []
    supported_exts = (".png", ".jpg", ".jpeg")

    image_paths = sorted([p for p in image_dir.iterdir()
                         if p.suffix.lower() in supported_exts])

    for i, image_path in enumerate(image_paths):
        if i >= limit:
            break
        try:
            with Image.open(image_path) as img:
                img = img.convert("RGB")
                img = img.resize(size)
                img_array = np.array(img)
                dataset.append(img_array)
        except Exception as e:
            print(f"Error {image_path.name}: {e}")

    print(f"Da them {len(dataset)} anh tu {image_dir}")
    return dataset


if __name__ == '__main__':
    # img_path = "../dataset/Water Bodies Dataset/Images/water_body_3.jpg"
    # masks_path = "../dataset/Water Bodies Dataset/Masks/water_body_3.jpg"

    # for percent in np.arange(0.0001, 0.001, 0.0001):
    #     rho = 0
    #     for i in range(5):
    #         np.random.seed(None)
    #         wm_bits = np.random.randint(0, 2, 14)
    #         C = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)[2]
    #         C = np.resize(C, (384, 384)) / 255.
    #         mask = generate_square_mask_grid(C, 1, percent)
    #         wm = WatermarkSystem(WatermarkType.JAWS,
    #                              (6512, wm_bits, 128, 0.04))
    #         Cw = wm.encode((C, mask))
    #         extract, _ = wm.decode((Cw, True))

    #         # print(wm_bits)
    #         # print(extract)

    #         rho += utils.score_extract(wm_bits, extract)

    #     print(rho/5, percent)

    dataset = load_dataset_images(DATASET_PATH, size=(384, 384), limit=50)

    # robustness_by_psnr(dataset, watermark_type=WatermarkType.DWT_DCT_SVD,
    #                    distortion_type=DistortionType.JPEG_COMPRESSION)
    # robustness_by_psnr(dataset, watermark_type=WatermarkType.DWT_DCT_SVD,
    #                    distortion_type=DistortionType.CONTRAST)
    # robustness_by_psnr(dataset, watermark_type=WatermarkType.DWT_DCT_SVD,
    #                    distortion_type=DistortionType.CUT)
    # robustness_by_psnr(dataset, watermark_type=WatermarkType.DWT_DCT_SVD,
    #                    distortion_type=DistortionType.ROTATION)

    robustness_by_psnr(dataset, watermark_type=WatermarkType.LSB,
                       distortion_type=DistortionType.JPEG_COMPRESSION)
    robustness_by_psnr(dataset, watermark_type=WatermarkType.LSB,
                       distortion_type=DistortionType.CONTRAST)
    robustness_by_psnr(dataset, watermark_type=WatermarkType.LSB,
                       distortion_type=DistortionType.CUT)
    robustness_by_psnr(dataset, watermark_type=WatermarkType.LSB,
                       distortion_type=DistortionType.ROTATION)

    # robustness_by_psnr(dataset, watermark_type=WatermarkType.MAX_DCT, distortion_type=DistortionType.JPEG_COMPRESSION)
    # robustness_by_psnr(dataset, watermark_type=WatermarkType.MAX_DCT, distortion_type=DistortionType.CONTRAST)
    # robustness_by_psnr(dataset, watermark_type=WatermarkType.MAX_DCT, distortion_type=DistortionType.CUT)
    # robustness_by_psnr(dataset, watermark_type=WatermarkType.MAX_DCT, distortion_type=DistortionType.ROTATION)

    # robustness_by_psnr(dataset, watermark_type=WatermarkType.DWT_DCT,
    #                    distortion_type=DistortionType.JPEG_COMPRESSION)
    # robustness_by_psnr(dataset, watermark_type=WatermarkType.DWT_DCT,
    #                    distortion_type=DistortionType.CONTRAST)
    # robustness_by_psnr(dataset, watermark_type=WatermarkType.DWT_DCT,
    #                    distortion_type=DistortionType.CUT)
    # robustness_by_psnr(dataset, watermark_type=WatermarkType.DWT_DCT,
    #                    distortion_type=DistortionType.ROTATION)

    # robustness_by_psnr(dataset, watermark_type=WatermarkType.JAWS, distortion_type=DistortionType.JPEG_COMPRESSION)
    # robustness_by_psnr(dataset, watermark_type=WatermarkType.JAWS, distortion_type=DistortionType.CONTRAST)
    # robustness_by_psnr(dataset, watermark_type=WatermarkType.JAWS, distortion_type=DistortionType.CUT)
    # robustness_by_psnr(dataset, watermark_type=WatermarkType.JAWS, distortion_type=DistortionType.ROTATION)

    # robustness_by_psnr(dataset, watermark_type=WatermarkType.JAWS, distortion_type=DistortionType.RESIZED_CROP, is_using_mask=False)
    # robustness_by_psnr(dataset, watermark_type=WatermarkType.DWT_DCT, distortion_type=DistortionType.RESIZED_CROP, is_using_mask=False)
    # robustness_by_psnr(dataset, watermark_type=WatermarkType.DWT_DCT_SVD, distortion_type=DistortionType.RESIZED_CROP, is_using_mask=False)
    # robustness_by_psnr(dataset, watermark_type=WatermarkType.MAX_DCT, distortion_type=DistortionType.RESIZED_CROP, is_using_mask=False)

    # robustness_by_psnr(dataset, watermark_type=WatermarkType.JAWS, distortion_type=DistortionType.RANDOM_ERASING, is_using_mask=False)
    # robustness_by_psnr(dataset, watermark_type=WatermarkType.DWT_DCT, distortion_type=DistortionType.RANDOM_ERASING, is_using_mask=False)
    # robustness_by_psnr(dataset, watermark_type=WatermarkType.DWT_DCT_SVD, distortion_type=DistortionType.RANDOM_ERASING, is_using_mask=False)
    # robustness_by_psnr(dataset, watermark_type=WatermarkType.MAX_DCT, distortion_type=DistortionType.RANDOM_ERASING, is_using_mask=False)

    # robustness_by_psnr(dataset, watermark_type=WatermarkType.JAWS, distortion_type=DistortionType.ADJUST_CONTRAST, is_using_mask=False)
    # robustness_by_psnr(dataset, watermark_type=WatermarkType.DWT_DCT, distortion_type=DistortionType.ADJUST_CONTRAST, is_using_mask=False)
    # robustness_by_psnr(dataset, watermark_type=WatermarkType.DWT_DCT_SVD, distortion_type=DistortionType.ADJUST_CONTRAST, is_using_mask=False)
    # robustness_by_psnr(dataset, watermark_type=WatermarkType.MAX_DCT, distortion_type=DistortionType.ADJUST_CONTRAST, is_using_mask=False)

    # robustness_by_psnr(dataset, watermark_type=WatermarkType.JAWS, distortion_type=DistortionType.GAUSSIAN_BLUR, is_using_mask=False)
    # robustness_by_psnr(dataset, watermark_type=WatermarkType.DWT_DCT, distortion_type=DistortionType.GAUSSIAN_BLUR, is_using_mask=False)
    # robustness_by_psnr(dataset, watermark_type=WatermarkType.DWT_DCT_SVD, distortion_type=DistortionType.GAUSSIAN_BLUR, is_using_mask=False)
    # robustness_by_psnr(dataset, watermark_type=WatermarkType.MAX_DCT, distortion_type=DistortionType.GAUSSIAN_BLUR, is_using_mask=False)
    #
    # robustness_by_psnr(dataset, watermark_type=WatermarkType.JAWS, distortion_type=DistortionType.GAUSSIAN_NOISE, is_using_mask=False)
    # robustness_by_psnr(dataset, watermark_type=WatermarkType.DWT_DCT, distortion_type=DistortionType.GAUSSIAN_NOISE, is_using_mask=False)
    # robustness_by_psnr(dataset, watermark_type=WatermarkType.DWT_DCT_SVD, distortion_type=DistortionType.GAUSSIAN_NOISE, is_using_mask=False)
    # robustness_by_psnr(dataset, watermark_type=WatermarkType.MAX_DCT, distortion_type=DistortionType.GAUSSIAN_NOISE, is_using_mask=False)

    # ------------------------------------------ Now with mask ------------------------------------------

    # robustness_by_psnr(dataset, watermark_type=WatermarkType.DWT_DCT_SVD, distortion_type=DistortionType.JPEG_COMPRESSION, is_using_mask=True)
    # robustness_by_psnr(dataset, watermark_type=WatermarkType.DWT_DCT_SVD, distortion_type=DistortionType.CONTRAST, is_using_mask=True)
    # robustness_by_psnr(dataset, watermark_type=WatermarkType.DWT_DCT_SVD, distortion_type=DistortionType.CUT, is_using_mask=True)
    # robustness_by_psnr(dataset, watermark_type=WatermarkType.DWT_DCT_SVD, distortion_type=DistortionType.ROTATION, is_using_mask=True)

    # robustness_by_psnr(dataset, watermark_type=WatermarkType.DWT_DCT, distortion_type=DistortionType.JPEG_COMPRESSION, is_using_mask=True)
    # robustness_by_psnr(dataset, watermark_type=WatermarkType.DWT_DCT, distortion_type=DistortionType.CONTRAST, is_using_mask=True)
    # robustness_by_psnr(dataset, watermark_type=WatermarkType.DWT_DCT, distortion_type=DistortionType.CUT, is_using_mask=True)
    # robustness_by_psnr(dataset, watermark_type=WatermarkType.DWT_DCT, distortion_type=DistortionType.ROTATION, is_using_mask=True)

    # robustness_by_psnr(dataset, watermark_type=WatermarkType.JAWS, distortion_type=DistortionType.JPEG_COMPRESSION, is_using_mask=True)
    # robustness_by_psnr(dataset, watermark_type=WatermarkType.JAWS, distortion_type=DistortionType.CONTRAST, is_using_mask=True)
    # robustness_by_psnr(dataset, watermark_type=WatermarkType.JAWS, distortion_type=DistortionType.CUT, is_using_mask=True)
    # robustness_by_psnr(dataset, watermark_type=WatermarkType.JAWS, distortion_type=DistortionType.ROTATION, is_using_mask=True)

    # robustness_by_psnr(dataset,watermark_type=WatermarkType.MAX_DCT, distortion_type=DistortionType.JPEG_COMPRESSION, is_using_mask=True)
    # robustness_by_psnr(dataset, watermark_type=WatermarkType.MAX_DCT, distortion_type=DistortionType.CONTRAST, is_using_mask=True)
    # robustness_by_psnr(dataset, watermark_type=WatermarkType.MAX_DCT, distortion_type=DistortionType.CUT, is_using_mask=True)
    # robustness_by_psnr(dataset, watermark_type=WatermarkType.MAX_DCT, distortion_type=DistortionType.ROTATION, is_using_mask=True)

    # robustness_by_psnr(dataset, watermark_type=WatermarkType.JAWS, distortion_type=DistortionType.RESIZED_CROP, is_using_mask=True)
    # robustness_by_psnr(dataset, watermark_type=WatermarkType.DWT_DCT, distortion_type=DistortionType.RESIZED_CROP, is_using_mask=True)
    # robustness_by_psnr(dataset, watermark_type=WatermarkType.DWT_DCT_SVD, distortion_type=DistortionType.RESIZED_CROP, is_using_mask=True)
    # robustness_by_psnr(dataset, watermark_type=WatermarkType.MAX_DCT, distortion_type=DistortionType.RESIZED_CROP, is_using_mask=True)

    # robustness_by_psnr(dataset, watermark_type=WatermarkType.JAWS, distortion_type=DistortionType.RANDOM_ERASING, is_using_mask=True)
    # robustness_by_psnr(dataset, watermark_type=WatermarkType.DWT_DCT, distortion_type=DistortionType.RANDOM_ERASING, is_using_mask=True)
    # robustness_by_psnr(dataset, watermark_type=WatermarkType.DWT_DCT_SVD, distortion_type=DistortionType.RANDOM_ERASING, is_using_mask=True)
    # robustness_by_psnr(dataset, watermark_type=WatermarkType.MAX_DCT, distortion_type=DistortionType.RANDOM_ERASING, is_using_mask=True)

    # robustness_by_psnr(dataset, watermark_type=WatermarkType.JAWS, distortion_type=DistortionType.ADJUST_CONTRAST, is_using_mask=True)
    # robustness_by_psnr(dataset, watermark_type=WatermarkType.DWT_DCT, distortion_type=DistortionType.ADJUST_CONTRAST, is_using_mask=True)
    # robustness_by_psnr(dataset, watermark_type=WatermarkType.DWT_DCT_SVD, distortion_type=DistortionType.ADJUST_CONTRAST, is_using_mask=True)
    # robustness_by_psnr(dataset, watermark_type=WatermarkType.MAX_DCT, distortion_type=DistortionType.ADJUST_CONTRAST, is_using_mask=True)

    # robustness_by_psnr(dataset, watermark_type=WatermarkType.JAWS, distortion_type=DistortionType.GAUSSIAN_BLUR, is_using_mask=True)
    # robustness_by_psnr(dataset, watermark_type=WatermarkType.DWT_DCT, distortion_type=DistortionType.GAUSSIAN_BLUR, is_using_mask=True)
    # robustness_by_psnr(dataset, watermark_type=WatermarkType.DWT_DCT_SVD, distortion_type=DistortionType.GAUSSIAN_BLUR, is_using_mask=True)
    # robustness_by_psnr(dataset, watermark_type=WatermarkType.MAX_DCT, distortion_type=DistortionType.GAUSSIAN_BLUR, is_using_mask=True)
    #
    # robustness_by_psnr(dataset, watermark_type=WatermarkType.JAWS, distortion_type=DistortionType.GAUSSIAN_NOISE, is_using_mask=True)
    # robustness_by_psnr(dataset, watermark_type=WatermarkType.DWT_DCT, distortion_type=DistortionType.GAUSSIAN_NOISE, is_using_mask=True)
    # robustness_by_psnr(dataset, watermark_type=WatermarkType.DWT_DCT_SVD, distortion_type=DistortionType.GAUSSIAN_NOISE, is_using_mask=True)
    # robustness_by_psnr(dataset, watermark_type=WatermarkType.MAX_DCT, distortion_type=DistortionType.GAUSSIAN_NOISE, is_using_mask=True)

    # ------------------------------------------ Now with Regeneration attack ------------------------------------------
    # regen_attack_by_psnr(dataset, watermark_type=WatermarkType.JAWS, regen_attack_type=RegenAttackType.RegenVae, is_using_mask=False)
    # regen_attack_by_psnr(dataset, watermark_type=WatermarkType.DWT_DCT, regen_attack_type=RegenAttackType.RegenVae, is_using_mask=False)
    # regen_attack_by_psnr(dataset, watermark_type=WatermarkType.DWT_DCT_SVD, regen_attack_type=RegenAttackType.RegenVae, is_using_mask=False)
    # regen_attack_by_psnr(dataset, watermark_type=WatermarkType.MAX_DCT, regen_attack_type=RegenAttackType.RegenVae, is_using_mask=False)

    # regen_attack_by_psnr(dataset, watermark_type=WatermarkType.JAWS, regen_attack_type=RegenAttackType.RegenVae, is_using_mask=True)
    # regen_attack_by_psnr(dataset, watermark_type=WatermarkType.DWT_DCT, regen_attack_type=RegenAttackType.RegenVae, is_using_mask=True)
    # regen_attack_by_psnr(dataset, watermark_type=WatermarkType.DWT_DCT_SVD, regen_attack_type=RegenAttackType.RegenVae, is_using_mask=True)
    # regen_attack_by_psnr(dataset, watermark_type=WatermarkType.MAX_DCT, regen_attack_type=RegenAttackType.RegenVae, is_using_mask=True)
