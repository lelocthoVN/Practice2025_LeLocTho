import cv2
import matplotlib.pyplot as plt
import pywt
import numpy as np
import utils
import os

DATA_PATH = "E:/waves-data/main/diffusiondb/real"


class DWTDCTSystem:
    """
    A system for embedding and extracting a watermark using a
    combination of Discrete Wavelet Transform (DWT) and Discrete Cosine Transform (DCT).
    """

    def __init__(self, k, watermark, alpha=15, block_size=4, mid_band_lo=0.30, mid_band_hi=0.70):
        """
        Args:
            k (int): Seed for pseudo-random number generation.
            watermark (np.ndarray): The watermark to be embedded.
            alpha (float): The embedding strength factor.
            block_size (int): The size of the DCT blocks.
            mid_band_lo (float): The lower bound for selecting mid-band frequencies.
            mid_band_hi (float): The upper bound for selecting mid-band frequencies.
        """
        self.k = k
        self.N = 384
        self.block_size = block_size
        self.mid_band_lo = mid_band_lo
        self.mid_band_hi = mid_band_hi

        # TRUE mid-band per zigzag (skip DC & highest freqs)
        self.mid_band_indices = self.get_mid_band_indices(
            self.block_size, mid_band_lo, mid_band_hi)
        self.num_mid_band = len(self.mid_band_indices)

        # PN size fixed as in your original code (frozen by N and block size)
        self.L = self.N ** 2 // 16 // (self.block_size ** 2)
        self.pn = self.generate_pn_sequences(self.L * self.num_mid_band)
        self.watermark = watermark
        self.watermark_len = len(self.watermark)
        self.alpha = alpha

    @staticmethod
    def zigzag_indices(n):
        """
        Generates the zigzag order indices for an n x n matrix.
        """
        indices = np.array([(i, j) for i in range(n) for j in range(n)])
        sorted_indices = sorted(indices, key=lambda x: (
            x[0] + x[1], x[1] if (x[0] + x[1]) % 2 else -x[0]))
        return sorted_indices

    @staticmethod
    def get_mid_band_indices(n, lo=0.30, hi=0.70):
        """
        Select true mid-band from zigzag by keeping (lo..hi) fraction of indices.
        """
        zigzag = DWTDCTSystem.zigzag_indices(n)
        L = len(zigzag)
        start = int(max(0, min(L - 1, np.floor(L * lo))))
        end = int(max(start + 1, min(L, np.ceil(L * hi))))
        return zigzag[start:end]

    def generate_pn_sequences(self, size):
        """
        Generates two pseudo-random number (PN) sequences.
        """
        np.random.seed(self.k)
        pn0 = np.random.choice([-1, 1], size)
        pn1 = np.random.choice([-1, 1], size)
        return [pn0, pn1]

    def block_dct(self, hl2_input):
        """
        Applies a block-based DCT on the input image.
        """
        h, w = hl2_input.shape
        dct_blocks = np.zeros((h, w), dtype=np.float32)
        for i in range(0, h, self.block_size):
            for j in range(0, w, self.block_size):
                block = hl2_input[i:i + self.block_size, j:j + self.block_size]
                dct_blocks[i:i + self.block_size, j:j +
                           self.block_size] = cv2.dct(np.float32(block))
        return dct_blocks

    def block_idct(self, dct_blocks):
        """
        Applies a block-based IDCT on the DCT coefficients.
        """
        h, w = dct_blocks.shape
        hl2_input_revert = np.zeros((h, w), dtype=np.float32)
        for i in range(0, h, self.block_size):
            for j in range(0, w, self.block_size):
                block = dct_blocks[i:i + self.block_size,
                                   j:j + self.block_size]
                hl2_input_revert[i:i + self.block_size, j:j +
                                 self.block_size] = cv2.idct(np.float32(block))
        return hl2_input_revert

    @staticmethod
    def round_image(img):
        """
        Clips, rounds, and converts an image array to uint8 format.
        """
        img = np.clip(img, 0, 255)
        img = np.round(img).astype(np.uint8)
        return img

    # ---- EMBED/EXTRACT ----
    def embedd_to_mid_bands(self, dct_blocks):
        """
        Embeds the watermark into the mid-band frequencies of DCT blocks.
        """
        h, w = dct_blocks.shape
        idx_wm = 0
        dct_blocks_embedded = np.copy(dct_blocks)
        for i in range(0, h, self.block_size):
            for j in range(0, w, self.block_size):
                block = dct_blocks_embedded[i:i +
                                            self.block_size, j:j + self.block_size]
                bit = int(self.watermark[idx_wm % self.watermark_len])
                base = idx_wm * self.num_mid_band
                for k, (x, y) in enumerate(self.mid_band_indices):
                    block[x, y] += self.alpha * self.pn[bit][base + k]
                idx_wm += 1
        return dct_blocks_embedded

    def extract_from_mid_bands(self, dct_blocks_embedded):
        """
        Extracts the watermark from the mid-band frequencies of DCT blocks.
        """
        h, w = dct_blocks_embedded.shape
        idx_wm = 0
        scores = [[] for _ in range(self.watermark_len)]
        for i in range(0, h, self.block_size):
            for j in range(0, w, self.block_size):
                block = dct_blocks_embedded[i:i +
                                            self.block_size, j:j + self.block_size]
                mid_band = np.array([block[x, y]
                                    for x, y in self.mid_band_indices])
                pn0_part = self.pn[0][idx_wm * self.num_mid_band:idx_wm *
                                      self.num_mid_band + self.num_mid_band]
                pn1_part = self.pn[1][idx_wm * self.num_mid_band:idx_wm *
                                      self.num_mid_band + self.num_mid_band]
                corr_pn0 = np.sum(mid_band * pn0_part)
                corr_pn1 = np.sum(mid_band * pn1_part)
                scores[idx_wm % self.watermark_len].append(
                    int(corr_pn1 > corr_pn0))
                idx_wm += 1
        avgScores = [np.array(s).mean() if len(s) > 0 else 0 for s in scores]
        bits = np.uint8(np.array(avgScores) * 255 > 127)
        return bits

    # ---- Public API ----
    def encode(self, img, mask=None):
        """
        Embeds the watermark into the input image.
        """
        if mask is None and isinstance(img, tuple) and len(img) == 2:
            img, mask = img

        LL, (LH, HL, HH) = pywt.dwt2(img, 'haar')
        LL2, (LH2, HL2, HH2) = pywt.dwt2(HL, 'haar')
        block_dct = self.block_dct(HL2)
        block_dct_embedded = self.embedd_to_mid_bands(block_dct)
        HL2_revert = self.block_idct(block_dct_embedded)
        HL_revert = pywt.idwt2((LL2, (LH2, HL2_revert, HH2)), 'haar')
        img_revert = pywt.idwt2((LL, (LH, HL_revert, HH)), 'haar')
        img_revert = self.round_image(img_revert)
        return img_revert

    def decode(self, img_watermarked, mask=None):
        """
        Extracts the watermark from a watermarked image.
        """
        if mask is None and isinstance(img_watermarked, tuple) and len(img_watermarked) == 2:
            img_watermarked, mask = img_watermarked

        LL, (LH, HL, HH) = pywt.dwt2(img_watermarked, 'haar')
        LL2, (LH2, HL2, HH2) = pywt.dwt2(HL, 'haar')
        block_dct = self.block_dct(HL2)
        extracted_information = self.extract_from_mid_bands(block_dct)
        return extracted_information


if __name__ == "__main__":
    img_test_path = os.path.join(DATA_PATH, "img (1).png")
    image = cv2.imread(img_test_path, cv2.IMREAD_UNCHANGED)
    assert image is not None, f"Not found: {img_test_path}"
    image = cv2.resize(image, (384, 384), interpolation=cv2.INTER_AREA)
    c = image[:, :, 2]
    wm_bits = np.random.randint(0, 2, size=32).astype(np.uint8)

    sys_watermark = DWTDCTSystem(
        k=6512,
        watermark=wm_bits,
        alpha=5,  # 15..25
        block_size=4,
        mid_band_lo=0.30,
        mid_band_hi=0.70,
    )
    cw = sys_watermark.encode(c)
    extracted_watermark = sys_watermark.decode(cw)

    # --- Metrics ---
    psnr = utils.calculate_psnr(c, cw)
    acc = (wm_bits == extracted_watermark).mean()
    print(f"PSNR: {psnr:.2f} dB")
    print("wm_bits :", wm_bits.tolist())
    print("extracted :", extracted_watermark.tolist())
    print(f"Bit accuracy : {acc*100:.2f}%")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(c, cmap="gray")
    axes[0].set_title("Original Image")
    axes[0].axis("off")
    axes[1].imshow(cw, cmap="gray")
    axes[1].set_title("Watermarked Image")
    axes[1].axis("off")
    axes[2].imshow(cv2.absdiff(c, cw), cmap="gray")
    axes[2].set_title("Absolute Difference")
    axes[2].axis("off")
    plt.suptitle(
        f"PSNR: {psnr:.2f} dB | Acc: {acc*100:.2f}%")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
