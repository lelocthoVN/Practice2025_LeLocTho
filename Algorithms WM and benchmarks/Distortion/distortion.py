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


def contract(img_array, scale):
    """
    Adjusts the contrast of an image by scaling its pixel values.

    Args:
        img_array (np.ndarray): The input image array.
        scale (float): The scaling factor for contrast adjustment.

    Returns:
        np.ndarray: The image array with adjusted contrast.
    """
    mod_container = np.clip(img_array * scale, 0, 255)
    return mod_container


def jpeg_compress(img_array, quality=75):
    """
    Compresses an image using JPEG compression and returns the decompressed result.

    Args:
        img_array (np.ndarray): The input image array.
        quality (int): The compression quality (0-100). Higher is better.

    Returns:
        np.ndarray: The decompressed image array.
    """
    image_array = np.clip(img_array, 0, 255).astype(np.uint8)
    image = Image.fromarray(image_array)
    output_io = io.BytesIO()
    image.save(output_io, format='JPEG', quality=int(quality))
    compressed_data = output_io.getvalue()
    output_io.close()
    input_io = io.BytesIO(compressed_data)
    image = Image.open(input_io)
    decompressed_array = np.array(image, dtype=np.uint8)
    return decompressed_array


def cut(original_image, watermarked_image, retain_part=0.8):
    """
    Replaces a part of the watermarked image with a portion of the original image.

    Args:
        original_image (np.ndarray): The original image array.
        watermarked_image (np.ndarray): The watermarked image array.
        retain_part (float): The ratio of the image to retain from the watermarked version.

    Returns:
        np.ndarray: The resulting image with a portion replaced.
    """
    h, w = original_image.shape
    max_h = int(h * np.sqrt(retain_part))
    max_w = int(w * np.sqrt(retain_part))
    watermarked_image_mod = np.copy(watermarked_image)
    watermarked_image_mod[:max_h, :max_w] = np.copy(
        original_image[:max_h, :max_w])
    return watermarked_image_mod


def rotation(img_array, angle):
    """
    Rotates an image by a given angle.

    Args:
        img_array (np.ndarray): The input image array.
        angle (float): The rotation angle in degrees.

    Returns:
        np.ndarray: The rotated image array.
    """
    return rotate(img_array, angle, reshape=False)


def resized_crop(img_array_grayscale, ratio):
    """
    Performs a random resized crop on a grayscale image.

    Args:
        img_array_grayscale (np.ndarray): The grayscale image array.
        ratio (float): The scaling ratio for the crop.

    Returns:
        np.ndarray: The cropped and resized image array.
    """
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


def random_erasing(img_array, ratio=0.05):
    """
    Erases a random rectangular area of an image by setting pixel values to zero.

    Args:
        cw (np.ndarray): The input image array.
        ratio (float): The ratio of the area to erase.

    Returns:
        np.ndarray: The image array with a random area erased.
    """
    res = np.copy(img_array)
    N1, N2 = img_array.shape
    max_n1 = int(N1 * ratio)
    max_n2 = int(N2 * ratio)
    x1 = random.randint(0, N1 - max_n1)
    y1 = random.randint(0, N2 - max_n2)
    x2 = x1 + max_n1
    y2 = y1 + max_n2
    res[x1:x2, y1:y2] = 0
    return res


def adjust_contrast(img_array, factor):
    """
    Adjusts the contrast of an image based on a given factor.

    Args:
        img_array (np.ndarray): The input image array.
        factor (float): The contrast adjustment factor.

    Returns:
        np.ndarray: The image array with adjusted contrast.
    """
    mean = np.mean(img_array)
    img_contrast = (img_array - mean) * factor + mean
    return np.clip(img_contrast, 0, 255).astype(np.uint8)


def gaussian_blur(img_array, kernel_size=4.):
    """
    Applies a Gaussian blur filter to an image.

    Args:
        img_array (np.ndarray): The input image array.
        kernel_size (float): The size of the Gaussian kernel.

    Returns:
        np.ndarray: The blurred image array.
    """
    return cv2.GaussianBlur(img_array, (int(kernel_size), int(kernel_size)), 0)


def gaussian_noise(img_array, sigma=0.03):
    """
    Adds Gaussian noise to an image.

    Args:
        img_array (np.ndarray): The input image array.
        sigma (float): The standard deviation of the noise.

    Returns:
        np.ndarray: The noisy image array.
    """
    img_norm = img_array/255
    noise = np.random.normal(0, sigma, img_array.shape)
    noisy_image = img_norm + noise
    return np.clip(noisy_image*255, 0, 255)


if __name__ == "__main__":
    img_test_path = "E:/waves-data/main/diffusiondb/real/img (1).png"
    img = cv2.imread(img_test_path, cv2.IMREAD_UNCHANGED)
    img = cv2.resize(img, (384, 384))
    img = img[:, :, 0]
    img_distorted = rotation(img, 20)

    print("---- Original Image Array ---- \n", img)
    print("---- Distorted Image Array ---- \n", img_distorted)
    print("PSNR = ", utils.calculate_psnr(img, img_distorted))

    plt.figure()
    plt.imshow(img_distorted, cmap='gray')
    plt.title("Distorted Image")
    plt.figure()
    plt.imshow(img, cmap='gray')
    plt.title("Original Image")
    plt.show()
