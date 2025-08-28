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

    def __init__(self, watermarks, scale=25, block=4, subband="HL2", svd_k=2):
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
        self._subband = subband  # "LL1" or "HL2"
        self._svd_k = max(1, int(svd_k))

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
        img_to_encode = np.copy(img_grayscale).astype(np.float32)
        H, W = img_to_encode.shape[:2]
        H = (H // self._block) * self._block
        W = (W // self._block) * self._block
        img_to_encode = img_to_encode[:H, :W]

        if binary_mask is None:
            mask_cropped = np.ones_like(img_to_encode, dtype=np.float32)
        else:
            mask_cropped = binary_mask[:H, :W].astype(np.float32)

        if self._subband.upper() == "LL1":
            # DWT cấp 1 → nhúng ở LL1 (giữ chế độ cũ)
            LL1, (LH1, HL1, HH1) = pywt.dwt2(img_to_encode, 'haar')
            mLL = cv2.resize(
                mask_cropped, (LL1.shape[1], LL1.shape[0]), interpolation=cv2.INTER_NEAREST)
            self._encode_frame_inplace(LL1, mLL)
            img_rec = pywt.idwt2((LL1, (LH1, HL1, HH1)), 'haar')
        else:
            # DWT cấp 2 → nhúng ở HL2 (công bằng với DWT_DCT)
            LL1, (LH1, HL1, HH1) = pywt.dwt2(img_to_encode, 'haar')
            LL2, (LH2, HL2, HH2) = pywt.dwt2(HL1, 'haar')
            mHL2 = cv2.resize(
                mask_cropped, (HL2.shape[1], HL2.shape[0]), interpolation=cv2.INTER_NEAREST)
            self._encode_frame_inplace(HL2, mHL2)
            HL1_rec = pywt.idwt2((LL2, (LH2, HL2, HH2)), 'haar')
            img_rec = pywt.idwt2((LL1, (LH1, HL1_rec, HH1)), 'haar')

        img_rec = np.clip(np.rint(img_rec), 0, 255).astype(np.uint8)
        return img_rec

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
        img_to_decode = np.copy(img_grayscale).astype(np.float32)
        H, W = img_to_decode.shape[:2]
        H = (H // self._block) * self._block
        W = (W // self._block) * self._block
        img_to_decode = img_to_decode[:H, :W]

        if binary_mask is None:
            mask_cropped = np.ones_like(img_to_decode, dtype=np.float32)
        else:
            mask_cropped = binary_mask[:H, :W].astype(np.float32)

        if self._subband.upper() == "LL1":
            LL1, (LH1, HL1, HH1) = pywt.dwt2(img_to_decode, 'haar')
            mLL = cv2.resize(
                mask_cropped, (LL1.shape[1], LL1.shape[0]), interpolation=cv2.INTER_NEAREST)
            scores = self._decode_frame(LL1, mLL)
        else:
            LL1, (LH1, HL1, HH1) = pywt.dwt2(img_to_decode, 'haar')
            LL2, (LH2, HL2, HH2) = pywt.dwt2(HL1, 'haar')
            mHL2 = cv2.resize(
                mask_cropped, (HL2.shape[1], HL2.shape[0]), interpolation=cv2.INTER_NEAREST)
            scores = self._decode_frame(HL2, mHL2)

        # gom theo vị trí (nhóm r), rồi threshold như cũ
        avgScores = [np.mean(s) if len(s) else 0 for s in scores]
        bits = (np.array(avgScores) * 255 > 127).astype(np.uint8)
        return bits

    def _encode_frame_inplace(self, frame, mask_down):
        """
        Encodes the watermark into a single DWT subband.

        Args:
            frame (np.ndarray): The DWT subband (e.g., LL1).
            mask_down (np.ndarray): The downsampled binary mask.
        """
        H, W = frame.shape
        num = 0
        for i in range(H // self._block):
            for j in range(W // self._block):
                block = frame[i*self._block:(i+1)*self._block,
                              j*self._block:(j+1)*self._block]
                mblk = mask_down[i*self._block:(i+1)*self._block,
                                 j*self._block:(j+1)*self._block]
                if np.mean(mblk) > 0.5:
                    wmBit = int(self._watermarks[num % self._wmLen])
                    frame[i*self._block:(i+1)*self._block,
                          j*self._block:(j+1)*self._block] = self._embed_block_dct_svd(block, wmBit)
                num += 1

    def _decode_frame(self, frame, mask_down):
        """
        Decodes the watermark from a single DWT subband.

        Args:
            frame (np.ndarray): The watermarked DWT subband.
            mask_down (np.ndarray): The downsampled binary mask.

        Returns:
            list: The updated scores list.
        """
        H, W = frame.shape
        scores = [[] for _ in range(self._wmLen)]
        num = 0
        for i in range(H // self._block):
            for j in range(W // self._block):
                block = frame[i*self._block:(i+1)*self._block,
                              j*self._block:(j+1)*self._block]
                mblk = mask_down[i*self._block:(i+1)*self._block,
                                 j*self._block:(j+1)*self._block]
                if np.mean(mblk) > 0.5:
                    score = self._infer_block_dct_svd(block)
                    idx = num % self._wmLen
                    scores[idx].append(score)
                num += 1
        return scores

    # ================= DCT+SVD =================
    def _embed_block_dct_svd(self, block, wmBit):
        dct_blk = cv2.dct(block.astype(np.float32))
        masked = np.zeros_like(dct_blk, dtype=np.float32)
        for (x, y) in self._mb_idx:
            masked[x, y] = dct_blk[x, y]

        U, S, Vt = np.linalg.svd(masked, full_matrices=False)

        K = min(self._svd_k, len(S))
        for k in range(K):
            S[k] = (np.floor(S[k] / self._scale) +
                    0.25 + 0.5 * wmBit) * self._scale

        masked_mod = (U @ np.diag(S) @ Vt).astype(np.float32)
        dct_mod = dct_blk.copy()
        for (x, y) in self._mb_idx:
            dct_mod[x, y] = masked_mod[x, y]

        return cv2.idct(dct_mod)

    def _infer_block_dct_svd(self, block):
        dct_blk = cv2.dct(block.astype(np.float32))
        masked = np.zeros_like(dct_blk, dtype=np.float32)
        for (x, y) in self._mb_idx:
            masked[x, y] = dct_blk[x, y]

        U, S, Vt = np.linalg.svd(masked, full_matrices=False)

        K = min(self._svd_k, len(S))
        votes = [(S[k] % self._scale) > (0.5 * self._scale) for k in range(K)]

        return int(np.mean(votes) > 0.5)


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
