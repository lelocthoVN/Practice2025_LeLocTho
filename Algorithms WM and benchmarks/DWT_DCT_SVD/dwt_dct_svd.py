import utils
import numpy as np
import cv2
import pywt
import os
import matplotlib.pyplot as plt


DATA_PATH = "E:/waves-data/main/diffusiondb/real"
# cloud_body_img = r"..\..\dataset\Water Bodies Dataset\Images\water_body_3.jpg"
# cloud_body_mask = r"..\..\dataset\Water Bodies Dataset\Masks\water_body_3.jpg"


class EmbedDwtDctSvd:
    def __init__(self, watermarks, scale=25, block=4):
        self._watermarks = watermarks
        self._wmLen = len(self._watermarks)
        self._scale = scale
        self._block = block

        # --- mid-band config (không thay đổi chữ ký __init__) ---
        # Dùng zigzag để chọn dải trung tần cho block 4x4 (mặc định 30%..70% giữa)
        self._mb_lo = 0.30
        self._mb_hi = 0.70
        self._mb_idx = self._get_mid_band_indices(
            self._block, self._mb_lo, self._mb_hi)

    # ================= Helpers: zigzag & mid-band =================
    @staticmethod
    def _zigzag_indices(n):
        indices = [(i, j) for i in range(n) for j in range(n)]
        return sorted(indices, key=lambda x: (x[0] + x[1], x[1] if ((x[0] + x[1]) % 2) else -x[0]))

    @classmethod
    def _get_mid_band_indices(cls, n, lo=0.30, hi=0.70):
        zz = cls._zigzag_indices(n)
        L = len(zz)
        start = int(max(0, min(L - 1, np.floor(L * lo))))
        end = int(max(start + 1, min(L, np.ceil(L * hi))))
        return zz[start:end]

    # ================= API không đổi =================
    def encode(self, img_grayscale, binary_mask=None):
        img_to_encode = np.copy(img_grayscale)
        (row, col) = img_grayscale.shape
        valid_row = row // 4 * 4
        valid_col = col // 4 * 4
        img_to_encode = img_to_encode[:valid_row, :valid_col]

        # nếu không có mask, dùng toàn 1 đúng kích thước ảnh (sau crop)
        if binary_mask is None:
            mask_cropped = np.ones_like(img_to_encode, dtype=np.float32)
        else:
            mask_cropped = binary_mask[:valid_row, :valid_col]

        ca1, (h1, v1, d1) = pywt.dwt2(img_to_encode, 'haar')

        # đưa mask xuống cùng kích thước subband ca1
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

    # ================= Nâng cấp “bên trong” DCT+SVD =================
    @staticmethod
    def _idct_from_dct_matrix(dct_mat):
        return cv2.idct(dct_mat.astype(np.float32))

    def diffuse_dct_svd(self, block, wmBit, scale):
        """
        Nâng cấp:
        - Chỉ thao tác trên MID-BAND của DCT block (DC & high giữ nguyên).
        - SVD trên ma trận DCT đã 'mask mid-band'.
        - QIM trên singular value lớn nhất s[0] (giữ cấu trúc & nguyên lý code cũ).
        """
        # DCT
        dct_block = cv2.dct(np.float32(block))

        # Tạo bản 'masked' chỉ chứa mid-band
        masked = np.zeros_like(dct_block, dtype=np.float32)
        for (x, y) in self._mb_idx:
            masked[x, y] = dct_block[x, y]

        # SVD trên phần mid-band
        u, s, v = np.linalg.svd(masked, full_matrices=False)

        # QIM (giữ công thức cũ): tâm 0-bit tại 0.25*scale, 1-bit tại 0.75*scale
        s[0] = (s[0] // scale + 0.25 + 0.5 * int(wmBit)) * scale

        # Khôi phục phần mid-band đã chỉnh
        masked_mod = (u @ np.diag(s) @ v).astype(np.float32)

        # Gộp: mid-band lấy từ masked_mod; DC & high giữ nguyên từ dct_block
        dct_mod = dct_block.copy()
        for (x, y) in self._mb_idx:
            dct_mod[x, y] = masked_mod[x, y]

        # IDCT -> block không gian
        return self._idct_from_dct_matrix(dct_mod)

    def infer_dct_svd(self, block, scale):
        """
        Giải mã đối xứng: SVD trên phần mid-band rồi quyết định bit từ s[0] % scale.
        """
        dct_block = cv2.dct(np.float32(block))

        masked = np.zeros_like(dct_block, dtype=np.float32)
        for (x, y) in self._mb_idx:
            masked[x, y] = dct_block[x, y]

        u, s, v = np.linalg.svd(masked, full_matrices=False)
        return int((s[0] % scale) > scale * 0.5)

    # @staticmethod
    # def diffuse_dct_svd(block, wmBit, scale):
    #     dct_block = cv2.dct(np.float32(block))
    #     u, s, v = np.linalg.svd(dct_block)
    #     s[0] = (s[0] // scale + 0.25 + 0.5 * wmBit) * scale
    #     modified_block = cv2.idct(np.dot(u, np.dot(np.diag(s), v)))
    #     return modified_block

    # @staticmethod
    # def infer_dct_svd(block, scale):
    #     dct_block = cv2.dct(np.float32(block))
    #     u, s, v = np.linalg.svd(dct_block)
    #     score = int((s[0] % scale) > scale * 0.5)
    #     return score


if __name__ == "__main__":
    img_path = os.path.join(DATA_PATH, "img (1).png")
    image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    assert image is not None, f"Not found: {img_path}"
    image = cv2.resize(image, (384, 384), interpolation=cv2.INTER_AREA)

    c = image[:, :, 2]
    wm_bits = np.random.randint(0, 2, size=32).astype(np.uint8)

    sys_watermark = EmbedDwtDctSvd(wm_bits, scale=36, block=4)

    # mask = cv2.imread(cloud_body_mask, cv2.IMREAD_GRAYSCALE) / 255
    # mask = cv2.resize(mask, (384, 384))

    cw = sys_watermark.encode(c, binary_mask=None)
    bits = sys_watermark.decode(cw, binary_mask=None)

    # --- Chỉ số ---
    psnr = utils.calculate_psnr(c, cw)
    acc = (wm_bits == bits).mean()
    ber = 1.0 - acc

    print(f"PSNR (R): {psnr:.2f} dB")
    print("wm_bits     :", wm_bits.tolist())
    print("extracted   :", bits.tolist())
    print(f"Bit accuracy: {acc*100:.2f}%   |   BER: {ber*100:.2f}%")

    # --- Sai khác tuyệt đối ---
    # diff = cv2.absdiff(c, cw)

    # --- Visualize ---
    plt.figure(figsize=(10, 3))
    plt.subplot(1, 3, 1)
    plt.imshow(c, cmap="gray")
    plt.title("Ảnh gốc")
    plt.axis("off")
    plt.subplot(1, 3, 2)
    plt.imshow(cw,  cmap="gray")
    plt.title("Ảnh sau nhúng")
    plt.axis("off")
    plt.subplot(1, 3, 3)
    plt.imshow(cv2.absdiff(c, cw), cmap="gray")
    plt.title("|img - wm|")
    plt.axis("off")
    plt.suptitle(
        f"PSNR: {psnr:.2f} dB | Acc: {acc*100:.2f}% | BER: {ber*100:.2f}%")
    plt.tight_layout()
    plt.show()

    # plt.figure()
    # plt.imshow(c, cmap='gray')
    #
    # plt.figure()
    # plt.imshow(cw, cmap='gray')
    #
    # plt.figure()
    # plt.imshow(np.abs(c - cw), cmap='gray')
    # plt.show()

# plt.figure()
    # plt.imshow(np.abs(img[:,:,0]-cw), cmap='gray')
    # plt.show()
    #
    # plt.figure()
    # plt.imshow(mask, cmap='gray')
    # plt.show()
    #
