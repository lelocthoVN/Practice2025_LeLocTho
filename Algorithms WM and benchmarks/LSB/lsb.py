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
        k              : seed để xáo trộn vị trí (RandomState)
        watermark_bits : mảng/list 0/1, độ dài L
        bit_plane      : 0 = LSB, 1 = bit thứ 2, ...
        reps           : số pixel dùng cho mỗi bit (đa số phiếu)
        """
        self.k = int(k)
        self.wm = np.asarray(watermark_bits, dtype=np.uint8).ravel()
        self.L = int(self.wm.size)
        self.bit_plane = int(bit_plane)
        self.reps = int(reps)

    # ---------------- utils ----------------
    @staticmethod
    def _ensure_gray2d(x):
        x = np.asarray(x)
        if x.ndim != 2:
            raise ValueError("LSBSystem: chỉ hỗ trợ ảnh xám 2D.")
        return x

    @staticmethod
    def _norm_mask(mask, h, w):
        if mask is None:
            return np.ones((h, w), dtype=np.uint8)
        m = np.asarray(mask)
        if m.shape != (h, w):
            # WAVES thường truyền mask đồng kích thước; nếu khác, fallback toàn 1
            m = np.ones((h, w), dtype=np.uint8)
        return (m > 0).astype(np.uint8)

    def _select_positions(self, h, w, mask):
        """Chọn & xáo trộn vị trí (1D index) theo seed k."""
        roi = np.flatnonzero(mask.ravel() > 0)
        if roi.size == 0:
            raise ValueError("Mask rỗng — không có vị trí để nhúng.")
        rng = np.random.RandomState(self.k)
        perm = rng.permutation(roi.size)
        return roi[perm]  # vị trí sau khi shuffle

    def _effective_reps(self, capacity):
        """reps thực tế nếu ảnh nhỏ (đảm bảo L * reps_eff <= capacity)."""
        max_reps = capacity // self.L
        if max_reps < 1:
            raise ValueError(
                f"Không đủ dung lượng: cần ≥ {self.L} pixel, chỉ có {capacity}."
            )
        return min(self.reps, int(max_reps))

    # ---------------- API ----------------
    def encode(self, img, mask=None):
        # WAVES có thể gọi encode((img, mask))
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

        # Lặp bit → vector bit cho từng vị trí
        bits_rep = np.repeat(self.wm, reps_eff).astype(np.uint8)

        # Ghi vào bit-plane
        vals = out.ravel()[pos]
        clear_mask = np.uint8(~(1 << self.bit_plane) & 0xFF)
        vals = (vals & clear_mask) | (bits_rep << self.bit_plane)
        out.ravel()[pos] = vals

        return out  # uint8

    def decode(self, img_wm, mask=None):
        # WAVES có thể gọi decode((img, mask))
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
    img_test_path = os.path.join(DATA_PATH, "img (1).png")
    image = cv2.imread(img_test_path, cv2.IMREAD_UNCHANGED)
    assert image is not None, f"Not found: {img_test_path}"

    # Resize về 384x384 để đồng bộ với pipeline hiện tại
    image = cv2.resize(image, (384, 384), interpolation=cv2.INTER_AREA)

    # Lấy kênh R làm ảnh xám (OpenCV là BGR)
    c = image[:, :, 2]

    # --- Watermark bits ---
    wm_bits = np.random.randint(0, 2, size=32).astype(np.uint8)

    # --- Tạo hệ LSB ---
    #   k: seed xáo trộn vị trí
    #   bit_plane=0: nhúng vào LSB
    #   reps=16: mỗi bit lặp 16 pixel để bỏ phiếu đa số (bền hơn)
    wm_sys = LSBSystem(k=6512, watermark_bits=wm_bits, bit_plane=0, reps=16)

    # Mask toàn 1 để tương thích giao diện WAVES (encode((img, mask)))
    mask = np.ones_like(c, dtype=np.uint8)

    # --- Encode & Decode ---
    cw = wm_sys.encode((c, mask))         # trả về uint8 cùng kích thước
    bits = wm_sys.decode((cw, mask))         # mảng bit 0/1 độ dài 32

    # --- Metrics ---
    psnr = utils.calculate_psnr(c, cw)
    acc = (wm_bits == bits).mean()
    ber = 1.0 - acc

    print(f"PSNR: {psnr:.2f} dB")
    print("wm_bits   :", wm_bits.tolist())
    print("extracted :", bits.tolist())
    print(f"Bit accuracy: {acc*100:.2f}% | BER: {ber*100:.2f}%")

    # --- Visualize ---
    plt.figure(figsize=(10, 3))
    plt.subplot(1, 3, 1)
    plt.imshow(c,  cmap="gray")
    plt.title("Ảnh gốc (R)")
    plt.axis("off")
    plt.subplot(1, 3, 2)
    plt.imshow(cw, cmap="gray")
    plt.title("Ảnh sau nhúng (LSB)")
    plt.axis("off")
    plt.subplot(1, 3, 3)
    plt.imshow(cv2.absdiff(c, cw), cmap="gray")
    plt.title("|img - wm|")
    plt.axis("off")
    plt.suptitle(
        f"PSNR: {psnr:.2f} dB | Acc: {acc*100:.2f}% | BER: {ber*100:.2f}%")
    plt.tight_layout()
    plt.show()
