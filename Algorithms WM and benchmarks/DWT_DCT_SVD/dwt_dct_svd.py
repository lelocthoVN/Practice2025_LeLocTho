import utils
import numpy as np
import cv2
import pywt
import os
import matplotlib.pyplot as plt


DATA_PATH = "E:/waves-data/main/diffusiondb/real"


class EmbedDwtDctSvd:
    """
    A watermarking system using a combination of DWT, DCT, and SVD
    to embed and extract information from a grayscale image.
    """

    def __init__(self, watermarks, scale=25, block=4):
        """
        Args:
            watermarks (np.ndarray): A binary array representing the watermark.
            scale (float): The quantization step size for QIM.
            block (int): The size of the DCT blocks.
        """
        self._watermarks = watermarks
        self._wmLen = len(self._watermarks)
        self._scale = scale
        self._block = block

        # Use zigzag to select the mid-band frequency range for the 4x4 block
        self._mb_lo = 0.30
        self._mb_hi = 0.70
        self._mb_idx = self._get_mid_band_indices(
            self._block, self._mb_lo, self._mb_hi)

    @staticmethod
    def _zigzag_indices(n):
        """
        Generates indices for a zigzag scan of an n x n matrix.
        """
        indices = [(i, j) for i in range(n) for j in range(n)]
        return sorted(indices, key=lambda x: (x[0] + x[1], x[1] if ((x[0] + x[1]) % 2) else -x[0]))

    @classmethod
    def _get_mid_band_indices(cls, n, lo=0.30, hi=0.70):
        """
        Selects a subset of zigzag indices corresponding to the mid-band
        frequencies based on lo and hi percentages.
        """
        zigzag = cls._zigzag_indices(n)
        L = len(zigzag)
        start = int(max(0, min(L - 1, np.floor(L * lo))))
        end = int(max(start + 1, min(L, np.ceil(L * hi))))
        return zigzag[start:end]

    def encode(self, img_grayscale, binary_mask=None):
        """
        Embeds the watermark into the grayscale image.

        Args:
            img_grayscale (np.ndarray): The input grayscale image.
            binary_mask (np.ndarray, optional): A mask to determine which
                                                blocks to embed. Defaults to None.

        Returns:
            np.ndarray: The watermarked image.
        """
        img_to_encode = np.copy(img_grayscale)
        (row, col) = img_grayscale.shape
        valid_row = row // 4 * 4
        valid_col = col // 4 * 4
        img_to_encode = img_to_encode[:valid_row, :valid_col]

        if binary_mask is None:
            mask_cropped = np.ones_like(img_to_encode, dtype=np.float32)
        else:
            mask_cropped = binary_mask[:valid_row, :valid_col]

        ca1, (h1, v1, d1) = pywt.dwt2(img_to_encode, 'haar')
        mask_downsampled = cv2.resize(
            mask_cropped, (ca1.shape[1], ca1.shape[0]
                           ), interpolation=cv2.INTER_NEAREST
        )
        if self._scale > 0:
            self.encode_frame(ca1, self._scale, mask_downsampled)
        img_to_encode = pywt.idwt2((ca1, (h1, v1, d1)), 'haar')
        img_to_encode = np.clip(np.rint(img_to_encode),
                                0, 255).astype(np.uint8)
        return img_to_encode

    def decode(self, img_grayscale, binary_mask=None):
        """
        Extracts the watermark from the watermarked grayscale image.

        Args:
            img_grayscale (np.ndarray): The input watermarked grayscale image.
            binary_mask (np.ndarray, optional): A mask to determine which
                                                blocks to extract. Defaults to None.

        Returns:
            np.ndarray: The extracted watermark.
        """
        img_to_decode = np.copy(img_grayscale)
        (row, col) = img_grayscale.shape
        valid_row = row // 4 * 4
        valid_col = col // 4 * 4
        img_to_decode = img_to_decode[:valid_row, :valid_col]

        if binary_mask is None:
            mask_cropped = np.ones_like(img_to_decode, dtype=np.float32)
        else:
            mask_cropped = binary_mask[:valid_row, :valid_col]

        ca1, (h1, v1, d1) = pywt.dwt2(img_to_decode, 'haar')
        mask_downsampled = cv2.resize(
            mask_cropped, (ca1.shape[1], ca1.shape[0]
                           ), interpolation=cv2.INTER_NEAREST
        )

        scores = [[] for _ in range(self._wmLen)]
        scores = self.decode_frame(ca1, self._scale, scores, mask_downsampled)
        avgScores = [np.array(s).mean() if len(s) > 0 else 0 for s in scores]
        bits = (np.array(avgScores) * 255 > 127).astype(np.uint8)
        return np.array(bits)

    def encode_frame(self, frame, scale, binary_mask):
        """
        Encodes the watermark into a single DWT subband.

        Args:
            frame (np.ndarray): The DWT subband (e.g., LL1).
            scale (float): Quantization step size.
            binary_mask (np.ndarray): The downsampled binary mask.
        """
        (row, col) = frame.shape
        num = 0
        for i in range(row // self._block):
            for j in range(col // self._block):
                block = frame[i * self._block:(i + 1) * self._block,
                              j * self._block:(j + 1) * self._block]
                mask_block = binary_mask[i * self._block:(i + 1) * self._block,
                                         j * self._block:(j + 1) * self._block]
                if np.mean(mask_block) > 0.5:
                    wmBit = self._watermarks[num % self._wmLen]
                    diffusedBlock = self.diffuse_dct_svd(block, wmBit, scale)
                    frame[i * self._block:(i + 1) * self._block,
                          j * self._block:(j + 1) * self._block] = diffusedBlock
                num += 1

    def decode_frame(self, frame, scale, scores, binary_mask):
        """
        Decodes the watermark from a single DWT subband.

        Args:
            frame (np.ndarray): The watermarked DWT subband.
            scale (float): Quantization step size.
            scores (list): A list to store correlation scores for each bit.
            binary_mask (np.ndarray): The downsampled binary mask.

        Returns:
            list: The updated scores list.
        """
        (row, col) = frame.shape
        num = 0
        for i in range(row // self._block):
            for j in range(col // self._block):
                block = frame[i * self._block:(i + 1) * self._block,
                              j * self._block:(j + 1) * self._block]
                mask_block = binary_mask[i * self._block:(i + 1) * self._block,
                                         j * self._block:(j + 1) * self._block]
                if np.mean(mask_block) > 0.5:
                    score = self.infer_dct_svd(block, scale)
                    wmBit = num % self._wmLen
                    scores[wmBit].append(score)
                num += 1
        return scores

    # ================= DCT+SVD =================
    @staticmethod
    def _idct_from_dct_matrix(dct_mat):
        """
        Performs a block-based IDCT.
        """
        return cv2.idct(dct_mat.astype(np.float32))

    def diffuse_dct_svd(self, block, wmBit, scale):
        """
        Embeds a single watermark bit into a block's mid-band frequencies
        by modifying the largest singular value.
        """
        # DCT
        dct_block = cv2.dct(np.float32(block))
        masked = np.zeros_like(dct_block, dtype=np.float32)
        for (x, y) in self._mb_idx:
            masked[x, y] = dct_block[x, y]
        # SVD on the mid-band part
        u, s, v = np.linalg.svd(masked, full_matrices=False)
        s[0] = (s[0] // scale + 0.25 + 0.5 * int(wmBit)) * scale
        masked_mod = (u @ np.diag(s) @ v).astype(np.float32)
        dct_mod = dct_block.copy()
        for (x, y) in self._mb_idx:
            dct_mod[x, y] = masked_mod[x, y]
        return self._idct_from_dct_matrix(dct_mod)

    def infer_dct_svd(self, block, scale):
        """
        Decodes a single watermark bit from a block by analyzing the
        quantization of the largest singular value.
        """
        dct_block = cv2.dct(np.float32(block))

        masked = np.zeros_like(dct_block, dtype=np.float32)
        for (x, y) in self._mb_idx:
            masked[x, y] = dct_block[x, y]

        u, s, v = np.linalg.svd(masked, full_matrices=False)
        # Check which quantization interval s[0] falls into
        return int((s[0] % scale) > scale * 0.5)


if __name__ == "__main__":
    img_path = os.path.join(DATA_PATH, "img (2).png")
    image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    assert image is not None, f"Not found: {img_path}"
    image = cv2.resize(image, (384, 384), interpolation=cv2.INTER_AREA)

    c = image[:, :, 2]
    wm_bits = np.random.randint(0, 2, size=32).astype(np.uint8)
    sys_watermark = EmbedDwtDctSvd(wm_bits, scale=36, block=4)
    cw = sys_watermark.encode(c, binary_mask=None)
    bits = sys_watermark.decode(cw, binary_mask=None)

    # --- Metrics ---
    psnr = utils.calculate_psnr(c, cw)
    acc = (wm_bits == bits).mean()

    print(f"PSNR (R): {psnr:.2f} dB")
    print("wm_bits     :", wm_bits.tolist())
    print("extracted   :", bits.tolist())
    print(f"Bit accuracy: {acc*100:.2f}%")

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
