import cv2
import numpy as np
import matplotlib.pyplot as plt


def read_img(img_path):
    img_arr = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    if len(img_arr.shape) == 3:
        img_arr = cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB)
    return img_arr


def show_img(img_array):
    plt.figure(figsize=(4, 4))
    if img_array.shape[-1] != 3:
        plt.imshow(img_array, cmap='gray')
    else:
        plt.imshow(img_array)


def show_image_by_ax(ax, arr, title):
    ax.set_title(title)
    if len(arr.shape) == 3 and arr.shape[2] == 3:
        ax.imshow(arr)
    else:
        ax.imshow(arr, cmap='gray')


def show_images(imgs, cols, window_title):
    n = len(imgs)
    rows = (n + cols - 1) // cols
    fig = plt.figure(figsize=(3*cols, 3*rows))
    plt.suptitle(window_title, fontsize=16)

    for i, (img, title) in enumerate(imgs):
        ax_img = plt.subplot(rows, cols, i+1)
        show_image_by_ax(ax_img, img, title)
        ax_img.axis('off')

    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    fig.tight_layout()
    plt.show()


def concat_channels(r, g, b):
    if len(r.shape) == 2:
        r = np.expand_dims(r, axis=-1)
    if len(g.shape) == 2:
        g = np.expand_dims(g, axis=-1)
    if len(b.shape) == 2:
        b = np.expand_dims(b, axis=-1)
    img = np.concatenate((r, g, b), axis=-1)
    return img


def save_img(path, img_arr):
    img_to_save = cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB)
    cv2.imwrite(path, img_to_save)


def binary_array_to_decimal(binary_array):
    return int("".join(map(str, binary_array)), 2)


def decimal_to_binary_array(decimal_number, arr_len=None):
    if arr_len is not None:
        binary_string = bin(decimal_number)[2:].zfill(arr_len)
        return [int(bit) for bit in binary_string]
    return [int(bit) for bit in bin(decimal_number)[2:]]


def calculate_psnr(original, distorted):
    original = original.astype(np.float64)
    distorted = distorted.astype(np.float64)
    mse = np.mean((original - distorted) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.
    psnr = 10 * np.log10((max_pixel ** 2) / mse)
    return psnr


def score_extract(true_bits, extracted_bits):
    true_bits = np.array(true_bits)
    extracted_bits = np.array(extracted_bits)
    M = max(len(true_bits), len(extracted_bits))
    true_bits = np.pad(true_bits, (M-len(true_bits), 0),
                       'constant', constant_values=0)
    extracted_bits = np.pad(
        extracted_bits, (M-len(extracted_bits), 0), 'constant', constant_values=0)
    if true_bits.sum() == 0 and extracted_bits.sum() == 0:
        return 1.0
    return sum(true_bits*extracted_bits) / (np.sqrt(sum(true_bits**2))*np.sqrt(sum(extracted_bits**2)) + 1e-10)


if __name__ == "__main__":
    print(decimal_to_binary_array(12))
    print(binary_array_to_decimal([1, 1, 0, 0]))
