import cv2
import numpy as np
import io
from scipy.ndimage import rotate
from enum import Enum
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms.functional as F
import utils
import random


class DistortionType(Enum):
    JPEG_COMPRESSION = 0
    CONTRAST = 1
    CUT = 2
    ROTATION = 3
    RESIZED_CROP = 4
    RANDOM_ERASING = 5
    ADJUST_CONTRAST = 6
    GAUSSIAN_BLUR = 7
    GAUSSIAN_NOISE = 8


def contract(cw, scale):
    mod_container = np.clip(cw * scale, 0, 255)
    return mod_container


def jpeg_compress(cw, quality=75):
    pre_process_img = np.clip(cw, 0, 255)
    image_array = pre_process_img.astype(np.uint8)
    image = Image.fromarray(image_array)
    output_io = io.BytesIO()
    image.save(output_io, format='JPEG', quality=int(quality))
    compressed_data = output_io.getvalue()
    output_io.close()
    input_io = io.BytesIO(compressed_data)
    image = Image.open(input_io)
    decompressed_array = np.array(image, dtype=np.uint8)
    return decompressed_array


def cut(c, cw, retain_part=0.8):
    N1, N2 = c.shape
    max_n1 = int(N1 * np.sqrt(retain_part))
    max_n2 = int(N2 * np.sqrt(retain_part))
    cw_mod = np.copy(c)
    cw_mod[:max_n1, :max_n2] = np.copy(cw[:max_n1, :max_n2])
    return cw_mod


def rotation(cw, angle):
    return rotate(cw, angle, reshape=False)


def resized_crop(img_array_grayscale, ratio):
    image = Image.fromarray(img_array_grayscale.astype('uint8')).convert('L')
    width, height = image.size
    target_height = int(height * ratio)
    target_width = int(width * ratio)
    i = random.randint(0, height - target_height)
    j = random.randint(0, width - target_width)
    distorted_image = F.resized_crop(
        image, i, j, target_height, target_width, (width, height))
    distorted_image_array = np.array(distorted_image)
    return distorted_image_array


def random_erasing(cw, ratio=0.05):
    res = np.copy(cw)
    N1, N2 = cw.shape
    max_n1 = int(N1 * ratio)
    max_n2 = int(N2 * ratio)
    x1 = random.randint(0, N1 - max_n1)
    y1 = random.randint(0, N2 - max_n2)
    x2 = x1 + max_n1
    y2 = y1 + max_n2
    res[x1:x2, y1:y2] = 0
    return res


def adjust_contrast(img_array, factor):
    mean = np.mean(img_array)
    img_contrast = (img_array - mean) * factor + mean
    return np.clip(img_contrast, 0, 255).astype(np.uint8)


def gaussian_blur(img_array, kernel_size=4.):
    return cv2.GaussianBlur(img_array, (int(kernel_size), int(kernel_size)), 0)


def gaussian_noise(img_array, sigma=0.03):
    img_norm = img_array/255
    noise = np.random.normal(0, sigma, img_array.shape)
    noisy_image = img_norm + noise
    return np.clip(noisy_image*255, 0, 255)


if __name__ == "__main__":
    img_test_path = "E:/waves-data/main/diffusiondb/real/img (1).png"
    img = cv2.imread(img_test_path, cv2.IMREAD_UNCHANGED)
    img = cv2.resize(img, (384, 384))
    img = img[:, :, 0]
    img_distorted = gaussian_noise(img, 0.04)

    print("---- Mang anh goc ---- \n", img)
    print("---- Mang anh bi bien dang ---- \n", img_distorted)
    print("PSNR = ", utils.calculate_psnr(img, img_distorted))

    plt.figure()
    plt.imshow(img_distorted, cmap='gray')
    plt.figure()
    plt.imshow(img, cmap='gray')
    plt.show()
