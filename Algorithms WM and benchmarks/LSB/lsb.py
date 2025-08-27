import numpy as np
import os
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import utils

DATA_PATH = "E:/waves-data/main/diffusiondb/real"


class LSBSystem:
    def __init__(self, k, watermark_bits, bit_plane=0, reps=16):
        """
        A watermarking system that embeds and extracts a binary watermark
        in the least significant bit (LSB) planes of a grayscale image.
        """
        self.k = int(k)
        self.wm = np.asarray(watermark_bits, dtype=np.uint8).ravel()
        self.L = int(self.wm.size)
        self.bit_plane = int(bit_plane)
        self.reps = int(reps)

    @staticmethod
    def _ensure_gray2d(x):
        """
        Checks if the input is a 2D grayscale image array.
        """
        x = np.asarray(x)
        if x.ndim != 2:
            raise ValueError(
                "LSBSystem: Only 2D grayscale images are supported.")
        return x

    @staticmethod
    def _norm_mask(mask, h, w):
        """
        Normalizes the mask to a binary array of the correct dimensions.
        """
        if mask is None:
            return np.ones((h, w), dtype=np.uint8)
        m = np.asarray(mask)
        if m.shape != (h, w):
            m = np.ones((h, w), dtype=np.uint8)
        return (m > 0).astype(np.uint8)

    def _select_positions(self, h, w, mask):
        """
        Selects and shuffles positions (1D indices) based on the seed k.
        """
        roi = np.flatnonzero(mask.ravel() > 0)
        if roi.size == 0:
            raise ValueError("Mask is empty — no positions to embed.")
        rng = np.random.RandomState(self.k)
        perm = rng.permutation(roi.size)
        return roi[perm]  # Shuffled positions

    def _effective_reps(self, capacity):
        """
        Calculates the effective repetition count, ensuring there's enough
        space in the image for all watermark bits.
        """
        max_reps = capacity // self.L
        if max_reps < 1:
            raise ValueError(
                f"Insufficient capacity: requires at least {self.L} pixels, found only {capacity}."
            )
        return min(self.reps, int(max_reps))

    def encode(self, img, mask=None):
        """
        Embeds the watermark into the image.
        """
        if mask is None and isinstance(img, tuple) and len(img) == 2:
            img, mask = img

        x = self._ensure_gray2d(img)
        h, w = x.shape
        m = self._norm_mask(mask, h, w)

        out = np.clip(np.rint(x), 0, 255).astype(np.uint8, copy=True)

        pos = self._select_positions(h, w, m)
        reps_eff = self._effective_reps(pos.size)
        need = self.L * reps_eff
        pos = pos[:need]

        # Repeat watermark bits to match the number of positions
        bits_rep = np.repeat(self.wm, reps_eff).astype(np.uint8)

        # Embed into the specified bit-plane
        vals = out.ravel()[pos]
        clear_mask = np.uint8(~(1 << self.bit_plane) & 0xFF)
        vals = (vals & clear_mask) | (bits_rep << self.bit_plane)
        out.ravel()[pos] = vals

        return out

    def decode(self, img_wm, mask=None):
        """
        Extracts the watermark from the watermarked image.
        """
        if mask is None and isinstance(img_wm, tuple) and len(img_wm) == 2:
            img_wm, mask = img_wm

        x = self._ensure_gray2d(img_wm)
        h, w = x.shape
        m = self._norm_mask(mask, h, w)

        pos = self._select_positions(h, w, m)
        reps_eff = self._effective_reps(pos.size)
        need = self.L * reps_eff
        pos = pos[:need]

        vals = np.asarray(x, dtype=np.uint8).ravel()[pos]
        bits = (vals >> self.bit_plane) & 1  # 0/1
        bits = bits.reshape(self.L, reps_eff)

        decided = (bits.sum(axis=1) > (reps_eff / 2)).astype(np.uint8)
        return decided


if __name__ == "__main__":
    img_test_path = os.path.join(DATA_PATH, "img (40).png")
    image = cv2.imread(img_test_path, cv2.IMREAD_UNCHANGED)
    assert image is not None, f"Not found: {img_test_path}"

    image = cv2.resize(image, (384, 384), interpolation=cv2.INTER_AREA)

    c = image[:, :, 2]
    wm_bits = np.random.randint(0, 2, size=32).astype(np.uint8)

    # --- Create LSB System ---
    # k: seed for position randomization
    # bit_plane=0: embed in LSB
    # reps=16: repeat each bit 16 times for majority voting (more robust)
    wm_sys = LSBSystem(k=6512, watermark_bits=wm_bits, bit_plane=0, reps=16)
    mask = np.ones_like(c, dtype=np.uint8)
    cw = wm_sys.encode((c, mask))
    bits = wm_sys.decode((cw, mask))
    # --- Metrics ---
    psnr = utils.calculate_psnr(c, cw)
    acc = (wm_bits == bits).mean()

    print(f"PSNR: {psnr:.2f} dB")
    print("wm_bits   :", wm_bits.tolist())
    print("extracted :", bits.tolist())
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
