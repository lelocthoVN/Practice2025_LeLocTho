from pathlib import Path
from PIL import Image
import math
import os.path
import numpy as np
import cv2
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
    """
    Evaluate the robustness of a watermarking system against a specific distortion type
    Arg:
        dataset: The dataset to evaluate.
        watermark_type: The type of watermarking to use.
        distortion_type: The type of distortion to apply.
        is_using_mask: Whether to use a mask for the watermark.
    Return:
        A tuple containing the average PSNR values and the True Positive Rate (TPR) at 0.1% False Positive Rate (FPR).
    """
    psnr_values = []
    tpr_at_0_1_fpr = []

    len_dataset = 50
    distortion_levels_ = distortion_levels[distortion_type]
    distortion_function_ = distortion_function[distortion_type]

    for distortion_level_ in distortion_levels_:
        psnr_values_current_level = []
        scores = []
        y_true = []
        for i in tqdm(range(len_dataset), desc="Processing"):
            np.random.seed(None)
            if is_using_mask:
                wm_bits = np.random.randint(0, 2, 7)
            else:
                wm_bits = np.random.randint(0, 2, 32)
            args_wm = (wm_bits, )
            if watermark_type == WatermarkType.DWT_DCT_SVD:
                args_wm = (wm_bits, 36, 4)
            elif watermark_type == WatermarkType.DWT_DCT:
                args_wm = (6512, wm_bits, 25, 4)
            elif watermark_type == WatermarkType.LSB:
                args_wm = (6512, wm_bits, 0, 16)
            wm_sys = WatermarkSystem(watermark_type, args_wm)
            c = dataset[i][:, :, 0]
            mask = np.ones_like(c)
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
            extracted_bit = wm_sys.decode((cw_distorted, mask))
            score = utils.score_extract(wm_bits, extracted_bit)
            scores.append(score)
            y_true.append(1)
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


def load_dataset_images(image_dir, size=(384, 384), limit=50):
    """
    Read and resize RGB images from a directory, return a list of NumPy arrays.

    Args:
        image_dir (str or Path): Path to the image directory.
        size (tuple): Target resize size (width, height).
        limit (int): Number of images to read (if any).

    Returns:
        List[np.ndarray]: List of RGB images as (H, W, 3) arrays.
    """
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
    dataset = load_dataset_images(DATASET_PATH, size=(384, 384), limit=50)

    # --------------- LSB --------------
    # robustness_by_psnr(dataset, watermark_type=WatermarkType.LSB,
    #                    distortion_type=DistortionType.JPEG_COMPRESSION)
    # robustness_by_psnr(dataset, watermark_type=WatermarkType.LSB,
    #                    distortion_type=DistortionType.CONTRAST)
    # robustness_by_psnr(dataset, watermark_type=WatermarkType.LSB,
    #                    distortion_type=DistortionType.CUT)
    # robustness_by_psnr(dataset, watermark_type=WatermarkType.LSB,
    #                    distortion_type=DistortionType.ROTATION)
    # robustness_by_psnr(dataset, watermark_type=WatermarkType.LSB,
    #                    distortion_type=DistortionType.RESIZED_CROP)
    # robustness_by_psnr(dataset, watermark_type=WatermarkType.LSB,
    #                    distortion_type=DistortionType.RANDOM_ERASING)
    # robustness_by_psnr(dataset, watermark_type=WatermarkType.LSB,
    #                    distortion_type=DistortionType.ADJUST_CONTRAST)
    # robustness_by_psnr(dataset, watermark_type=WatermarkType.LSB,
    #                    distortion_type=DistortionType.GAUSSIAN_BLUR)
    # robustness_by_psnr(dataset, watermark_type=WatermarkType.LSB,
    #                    distortion_type=DistortionType.GAUSSIAN_NOISE)

    # --------------- DWT_DCT --------------
    # robustness_by_psnr(dataset, watermark_type=WatermarkType.DWT_DCT,
    #                    distortion_type=DistortionType.JPEG_COMPRESSION)
    # robustness_by_psnr(dataset, watermark_type=WatermarkType.DWT_DCT,
    #                    distortion_type=DistortionType.CONTRAST)
    # robustness_by_psnr(dataset, watermark_type=WatermarkType.DWT_DCT,
    #                    distortion_type=DistortionType.CUT)
    # robustness_by_psnr(dataset, watermark_type=WatermarkType.DWT_DCT,
    #                    distortion_type=DistortionType.ROTATION)
    # robustness_by_psnr(dataset, watermark_type=WatermarkType.DWT_DCT,
    #                    distortion_type=DistortionType.RESIZED_CROP)
    # robustness_by_psnr(dataset, watermark_type=WatermarkType.DWT_DCT,
    #                    distortion_type=DistortionType.RANDOM_ERASING)
    # robustness_by_psnr(dataset, watermark_type=WatermarkType.DWT_DCT,
    #                    distortion_type=DistortionType.ADJUST_CONTRAST)
    # robustness_by_psnr(dataset, watermark_type=WatermarkType.DWT_DCT,
    #                    distortion_type=DistortionType.GAUSSIAN_BLUR)
    # robustness_by_psnr(dataset, watermark_type=WatermarkType.DWT_DCT,
    #                    distortion_type=DistortionType.GAUSSIAN_NOISE)

    # --------------- DWT_DCT_SVD --------------
    robustness_by_psnr(dataset, watermark_type=WatermarkType.DWT_DCT_SVD,
                       distortion_type=DistortionType.JPEG_COMPRESSION)
    robustness_by_psnr(dataset, watermark_type=WatermarkType.DWT_DCT_SVD,
                       distortion_type=DistortionType.CONTRAST)
    robustness_by_psnr(dataset, watermark_type=WatermarkType.DWT_DCT_SVD,
                       distortion_type=DistortionType.CUT)
    robustness_by_psnr(dataset, watermark_type=WatermarkType.DWT_DCT_SVD,
                       distortion_type=DistortionType.ROTATION)
    robustness_by_psnr(dataset, watermark_type=WatermarkType.DWT_DCT_SVD,
                       distortion_type=DistortionType.RESIZED_CROP)
    robustness_by_psnr(dataset, watermark_type=WatermarkType.DWT_DCT_SVD,
                       distortion_type=DistortionType.RANDOM_ERASING)
    robustness_by_psnr(dataset, watermark_type=WatermarkType.DWT_DCT_SVD,
                       distortion_type=DistortionType.ADJUST_CONTRAST)
    robustness_by_psnr(dataset, watermark_type=WatermarkType.DWT_DCT_SVD,
                       distortion_type=DistortionType.GAUSSIAN_BLUR)
    robustness_by_psnr(dataset, watermark_type=WatermarkType.DWT_DCT_SVD,
                       distortion_type=DistortionType.GAUSSIAN_NOISE)
