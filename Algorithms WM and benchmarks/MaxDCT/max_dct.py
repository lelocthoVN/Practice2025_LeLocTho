import numpy as np
import cv2
import pywt
import matplotlib.pyplot as plt
import utils

DATA_PATH = "../../dataset/Water Bodies Dataset"
cloud_body_img = r"..\..\dataset\Water Bodies Dataset\Images\water_body_3.jpg"
cloud_body_mask = r"..\..\dataset\Water Bodies Dataset\Masks\water_body_3.jpg"


class EmbedMaxDct(object):
    def __init__(self, watermarks, scales=36, block=4):
        self._watermarks = watermarks
        self._wmLen = len(self._watermarks)
        self._scales = scales
        self._block = block

    def encode(self, img_grayscale, binary_mask):
        img_to_encode = np.copy(img_grayscale)
        (row, col) = img_grayscale.shape
        valid_row = row // 4 * 4
        valid_col = col // 4 * 4
        img_to_encode = img_to_encode[:valid_row, :valid_col]
        mask_cropped = binary_mask[:valid_row, :valid_col]
        if mask_cropped.max() > 1:
            mask_cropped = mask_cropped / 255.0
        ca1, (h1, v1, d1) = pywt.dwt2(img_to_encode, 'haar')
        mask_downsampled = cv2.resize(
            mask_cropped, (img_grayscale.shape[1]//2, img_grayscale.shape[0]//2), interpolation=cv2.INTER_NEAREST)
        if self._scales > 0:
            self.encode_frame(ca1, self._scales, mask_downsampled)
        img_to_encode = pywt.idwt2((ca1, (h1, v1, d1)), 'haar')
        return img_to_encode

    def decode(self, img_grayscale, binary_mask):
        img_to_decode = np.copy(img_grayscale)
        (row, col) = img_grayscale.shape
        valid_row = row // 4 * 4
        valid_col = col // 4 * 4
        img_to_decode = img_to_decode[:valid_row, :valid_col]
        mask_cropped = binary_mask[:valid_row, :valid_col]
        if mask_cropped.max() > 1:
            mask_cropped = mask_cropped / 255.0
        ca1, (h1, v1, d1) = pywt.dwt2(img_to_decode, 'haar')
        mask_downsampled = cv2.resize(
            mask_cropped, (ca1.shape[1], ca1.shape[0]), interpolation=cv2.INTER_NEAREST)
        scores = [[] for _ in range(self._wmLen)]
        scores = self.decode_frame(ca1, self._scales, scores, mask_downsampled)
        avgScores = [np.array(s).mean() if len(s) > 0 else 0 for s in scores]
        bits = (np.array(avgScores) * 255 > 127)
        bits = bits.astype(np.uint8)
        return bits

    def decode_frame(self, frame, scale, scores, mask_downsampled):
        (row, col) = frame.shape
        num = 0
        for i in range(row//self._block):
            for j in range(col//self._block):
                block = frame[i*self._block: i*self._block + self._block,
                              j*self._block: j*self._block + self._block]
                block_mask = mask_downsampled[i*self._block: i*self._block + self._block,
                                              j*self._block: j*self._block + self._block]
                if np.mean(block_mask) > 0.5:
                    score = self.infer_dct_matrix(block, scale)
                    wmBit = num % self._wmLen
                    scores[wmBit].append(score)
                num = num + 1
        return scores

    def diffuse_dct_matrix(self, block, wm_bit, scale):
        pos = np.argmax(abs(block.flatten()[1:])) + 1
        i, j = pos // self._block, pos % self._block

        val = block[i][j]
        if val >= 0.0:
            block[i][j] = (val // scale + 0.25 + 0.5 * wm_bit) * scale
        else:
            val = abs(val)
            block[i][j] = -1.0 * (val // scale + 0.25 + 0.5 * wm_bit) * scale
        return block

    def infer_dct_matrix(self, block, scale):
        pos = np.argmax(abs(block.flatten()[1:])) + 1
        i, j = pos // self._block, pos % self._block
        val = block[i][j]
        if val < 0:
            val = abs(val)
        if (val % scale) > 0.5 * scale:
            return 1
        else:
            return 0

    def encode_frame(self, frame, scale, mask_downsampled):
        (row, col) = frame.shape
        num = 0
        for i in range(row//self._block):
            for j in range(col//self._block):
                block = frame[i*self._block: i*self._block + self._block,
                              j*self._block: j*self._block + self._block]
                block_mask = mask_downsampled[i*self._block: i*self._block + self._block,
                                              j*self._block: j*self._block + self._block]
                if np.mean(block_mask) > 0.5:
                    wmBit = self._watermarks[(num % self._wmLen)]
                    diffusedBlock = self.diffuse_dct_matrix(
                        block, wmBit, scale)
                    frame[i*self._block: i*self._block + self._block,
                          j*self._block: j*self._block + self._block] = diffusedBlock
                num = num+1


if __name__ == "__main__":
    image = cv2.imread(cloud_body_img, cv2.IMREAD_UNCHANGED)
    image = cv2.resize(image, (384, 384))
    c = image[:, :, 2]
    wm_bits = np.random.randint(0, 2, size=32)

    sys_watermark = EmbedMaxDct(wm_bits, 36, 4)
    mask = cv2.imread(cloud_body_mask, cv2.IMREAD_GRAYSCALE) / 255
    mask = cv2.resize(mask, (384, 384))
    cw = sys_watermark.encode(c, mask)
    print(utils.calculate_psnr(c, cw))
    # plt.figure()
    # plt.imshow(c, cmap='gray')
    #
    # plt.figure()
    # plt.imshow(cw, cmap='gray')
    #
    # plt.figure()
    # plt.imshow(np.abs(c - cw), cmap='gray')
    # plt.show()
