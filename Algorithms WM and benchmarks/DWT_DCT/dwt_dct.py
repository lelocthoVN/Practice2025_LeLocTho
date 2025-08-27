# import cv2
# import matplotlib.pyplot as plt
# import pywt
# import numpy as np
# import utils
# DATA_PATH = "../../dataset/Water Bodies Dataset"
# cloud_body_img = r"..\..\dataset\Water Bodies Dataset\Images\water_body_3.jpg"
# # cloud_body_mask = r"..\..\dataset\Water Bodies Dataset\Masks\water_body_3.jpg"


# class DWTDCTSystem:
#     def __init__(self, k, watermark, alpha=15, block_size=4):
#         self.k = k
#         self.N = 384
#         self.block_size = block_size
#         self.mid_band_indices = self.get_mid_band_indices(self.block_size)
#         self.num_mid_band = len(self.mid_band_indices)
#         self.L = self.N ** 2 // 16 // (self.block_size ** 2)
#         self.pn = self.generate_pn_sequences(self.L * self.num_mid_band)
#         self.watermark = watermark
#         self.watermark_len = len(self.watermark)
#         self.alpha = alpha

#     @staticmethod
#     def zigzag_indices(n):
#         indices = np.array([(i, j) for i in range(n) for j in range(n)])
#         sorted_indices = sorted(indices, key=lambda x: (
#             x[0] + x[1], x[1] if (x[0] + x[1]) % 2 else -x[0]))
#         return sorted_indices

#     @staticmethod
#     def get_mid_band_indices(n):
#         zigzag_indices = DWTDCTSystem.zigzag_indices(n)
#         L = len(zigzag_indices)
#         mid_band = zigzag_indices
#         return mid_band

#     def generate_pn_sequences(self, size):
#         np.random.seed(self.k)
#         pn0 = np.random.choice([-1, 1], size)
#         pn1 = np.random.choice([-1, 1], size)
#         return [pn0, pn1]

#     def block_dct(self, hl2_input):
#         h, w = hl2_input.shape
#         dct_blocks = np.zeros((h, w), dtype=np.float32)
#         for i in range(0, h, self.block_size):
#             for j in range(0, w, self.block_size):
#                 block = hl2_input[i:i + self.block_size, j:j + self.block_size]
#                 dct_blocks[i:i + self.block_size, j:j +
#                            self.block_size] = cv2.dct(np.float32(block))
#         return dct_blocks

#     def block_idct(self, dct_blocks):
#         h, w = dct_blocks.shape
#         hl2_input_revert = np.zeros((h, w), dtype=np.float32)
#         for i in range(0, h, self.block_size):
#             for j in range(0, w, self.block_size):
#                 block = dct_blocks[i:i + self.block_size,
#                                    j:j + self.block_size]
#                 hl2_input_revert[i:i + self.block_size, j:j +
#                                  self.block_size] = cv2.idct(np.float32(block))
#         return hl2_input_revert

#     @staticmethod
#     def round_image(img):
#         img = np.clip(img, 0, 255)
#         img = np.round(img).astype(np.uint8)
#         return img

#     def embedd_to_mid_bands(self, dct_blocks, mask):
#         h, w = dct_blocks.shape
#         idx_wm = 0
#         dct_blocks_embedded = np.copy(dct_blocks)
#         for i in range(0, h, self.block_size):
#             for j in range(0, w, self.block_size):
#                 block = dct_blocks_embedded[i:i +
#                                             self.block_size, j:j + self.block_size]
#                 mask_block = mask[i:i + self.block_size, j:j + self.block_size]
#                 if np.mean(mask_block) > 0.5:
#                     for k, (x, y) in enumerate(self.mid_band_indices):
#                         block[x, y] += self.alpha * self.pn[int(
#                             self.watermark[idx_wm % self.watermark_len])][idx_wm * self.num_mid_band + k]
#                 #     for k, (x, y) in enumerate(self.mid_band_indices):
#                 #         block[x, y] += self.alpha * self.pn[int(self.watermark[idx_wm % self.watermark_len])] \
#                 #                              [idx_wm*self.num_mid_band+k]
#                 idx_wm += 1
#         return dct_blocks_embedded

#     def extract_from_mid_bands(self, dct_blocks_embedded, mask):
#         h, w = dct_blocks_embedded.shape
#         idx_wm = 0
#         scores = [[] for _ in range(self.watermark_len)]
#         for i in range(0, h, self.block_size):
#             for j in range(0, w, self.block_size):
#                 block = dct_blocks_embedded[i:i +
#                                             self.block_size, j:j + self.block_size]
#                 mask_block = mask[i:i + self.block_size, j:j + self.block_size]
#                 if np.mean(mask_block) > 0.5:
#                     mid_band = np.array([block[i, j]
#                                         for i, j in self.mid_band_indices])
#                     pn0_part = self.pn[0][idx_wm * self.num_mid_band:idx_wm *
#                                           self.num_mid_band + self.num_mid_band]
#                     pn1_part = self.pn[1][idx_wm * self.num_mid_band:idx_wm *
#                                           self.num_mid_band + self.num_mid_band]
#                     corr_pn0 = np.sum(mid_band * pn0_part)
#                     corr_pn1 = np.sum(mid_band * pn1_part)
#                     scores[idx_wm % self.watermark_len].append(
#                         int(corr_pn1 > corr_pn0))
#                 idx_wm += 1
#         avgScores = [np.array(s).mean() if len(s) > 0 else 0 for s in scores]
#         bits = np.uint8(np.array(avgScores) * 255 > 127)
#         return bits

#     def encode(self, img, mask):
#         LL, (LH, HL, HH) = pywt.dwt2(img, 'haar')
#         LL2, (LH2, HL2, HH2) = pywt.dwt2(HL, 'haar')
#         block_dct = self.block_dct(HL2)
#         mask_downsampled = cv2.resize(
#             mask, (self.N // 4, self.N // 4), interpolation=cv2.INTER_NEAREST)
#         block_dct_embedded = self.embedd_to_mid_bands(
#             block_dct, mask_downsampled)
#         HL2_revert = self.block_idct(block_dct_embedded)
#         HL_revert = pywt.idwt2((LL2, (LH2, HL2_revert, HH2)), 'haar')
#         img_revert = pywt.idwt2((LL, (LH, HL_revert, HH)), 'haar')
#         img_revert = self.round_image(img_revert)
#         return img_revert

#     def decode(self, img_watermarked, mask):
#         LL, (LH, HL, HH) = pywt.dwt2(img_watermarked, 'haar')
#         LL2, (LH2, HL2, HH2) = pywt.dwt2(HL, 'haar')
#         block_dct = self.block_dct(HL2)
#         mask_downsampled = cv2.resize(
#             mask, (self.N // 4, self.N // 4), interpolation=cv2.INTER_NEAREST)
#         extracted_information = self.extract_from_mid_bands(
#             block_dct, mask_downsampled)
#         return extracted_information


# #
# # def get_mask(img):
# #     model_path = "C:/Users/phamn/PycharmProjects/WatermarkIntoWater/models/model_UnetPlusPlus_epoch_10_lr_0.0001.pth"
# #     model = SegmentationWaterModel("UnetPlusPlus", "resnet34", in_channels=3, out_classes=1).to(device)
# #     model.load_state_dict(torch.load(model_path, weights_only=True))
# #     model.eval()
# #     with torch.no_grad():
# #         model_input = img.transpose((2, 0, 1))
# #         model_input = torch.tensor(np.array([model_input/255]), dtype=torch.float32).to(device)
# #         pred = model.forward(model_input)
# #         pred = torch.nn.functional.sigmoid(pred)
# #         pred = (pred > 0.5).float()
# #     pred = pred.cpu().numpy()
# #     return pred[0, 0, :, :]

# if __name__ == "__main__":
#     image = cv2.imread(cloud_body_img, cv2.IMREAD_UNCHANGED)
#     image = cv2.resize(image, (384, 384))
#     c = image[:, :, 2]
#     wm_bits = np.random.randint(0, 2, size=32)
#     sys_watermark = DWTDCTSystem(6512, wm_bits, 25, 4)
#     mask = cv2.imread(cloud_body_mask, cv2.IMREAD_GRAYSCALE) / 255
#     mask = cv2.resize(mask, (384, 384))
#     cw = sys_watermark.encode(c, mask)
#     extracted_watermark = sys_watermark.decode(cw, mask)

#     print(utils.calculate_psnr(c, cw))
#     print(wm_bits)
#     print(extracted_watermark)
#     #
#     # plt.figure()
#     # plt.imshow(c, cmap='gray')
#     # plt.figure()
#     # plt.imshow(cw, cmap='gray')
#     # plt.figure()
#     # plt.imshow(np.abs(c - cw), cmap='gray')
#     # plt.show()

#     # cw = sys_watermark.encode(image[:, :, 0], mask)
#     # cw_full_color = np.copy(image)
#     # cw_full_color[:,:,0] = cw
#     # extracted_watermark = sys_watermark.decode(cw, mask)
#     #
#     # print(utils.calculate_psnr(image[:,:,0], cw))
#     # print(utils.binary_array_to_decimal(wm_bits)==utils.binary_array_to_decimal(extracted_watermark))

#     # plt.figure()
#     # plt.imshow(np.abs(image[:,:,0]-cw), cmap='gray')
#     # plt.figure()
#     # plt.imshow(mask, cmap='gray')
#     # plt.figure()
#     # plt.imshow(cw, cmap='gray')
#     # plt.figure()
#     # plt.imshow(image, cmap='gray')
#     #
#     # plt.show()


import cv2
import matplotlib.pyplot as plt
import pywt
import numpy as np
import utils
import os

DATA_PATH = "E:/waves-data/main/diffusiondb/real"


class DWTDCTSystem:
    def __init__(self, k, watermark, alpha=15, block_size=4, mid_band_lo=0.30, mid_band_hi=0.70):
        self.k = k
        self.N = 384
        self.block_size = block_size
        # --- TRUE mid-band per zigzag (skip DC & highest freqs) ---
        self.mid_band_indices = self.get_mid_band_indices(
            self.block_size, mid_band_lo, mid_band_hi)
        self.num_mid_band = len(self.mid_band_indices)
        # PN size fixed as in your original code (frozen by N and block size)
        self.L = self.N ** 2 // 16 // (self.block_size ** 2)
        self.pn = self.generate_pn_sequences(self.L * self.num_mid_band)
        self.watermark = watermark
        self.watermark_len = len(self.watermark)
        self.alpha = alpha

    # ---- zigzag + mid-band selection ----
    @staticmethod
    def zigzag_indices(n):
        indices = np.array([(i, j) for i in range(n) for j in range(n)])
        sorted_indices = sorted(indices, key=lambda x: (
            x[0] + x[1], x[1] if (x[0] + x[1]) % 2 else -x[0]))
        return sorted_indices

    @staticmethod
    def get_mid_band_indices(n, lo=0.30, hi=0.70):
        """Select true mid-band from zigzag by keeping (lo..hi) fraction of indices."""
        zz = DWTDCTSystem.zigzag_indices(n)
        L = len(zz)
        start = int(max(0, min(L - 1, np.floor(L * lo))))
        end = int(max(start + 1, min(L, np.ceil(L * hi))))
        return zz[start:end]

    # ---- PN ----
    def generate_pn_sequences(self, size):
        np.random.seed(self.k)
        pn0 = np.random.choice([-1, 1], size)
        pn1 = np.random.choice([-1, 1], size)
        return [pn0, pn1]

    # ---- block DCT/IDCT ----
    def block_dct(self, hl2_input):
        h, w = hl2_input.shape
        dct_blocks = np.zeros((h, w), dtype=np.float32)
        for i in range(0, h, self.block_size):
            for j in range(0, w, self.block_size):
                block = hl2_input[i:i + self.block_size, j:j + self.block_size]
                dct_blocks[i:i + self.block_size, j:j +
                           self.block_size] = cv2.dct(np.float32(block))
        return dct_blocks

    def block_idct(self, dct_blocks):
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
        img = np.clip(img, 0, 255)
        img = np.round(img).astype(np.uint8)
        return img

    # ---- EMBED/EXTRACT ----
    def embedd_to_mid_bands(self, dct_blocks):
        h, w = dct_blocks.shape
        idx_wm = 0
        dct_blocks_embedded = np.copy(dct_blocks)
        for i in range(0, h, self.block_size):
            for j in range(0, w, self.block_size):
                block = dct_blocks_embedded[i:i +
                                            self.block_size, j:j + self.block_size]
                # always embed (no mask filtering)
                bit = int(self.watermark[idx_wm % self.watermark_len])
                base = idx_wm * self.num_mid_band
                for k, (x, y) in enumerate(self.mid_band_indices):
                    block[x, y] += self.alpha * self.pn[bit][base + k]
                idx_wm += 1
        return dct_blocks_embedded

    def extract_from_mid_bands(self, dct_blocks_embedded):
        h, w = dct_blocks_embedded.shape
        idx_wm = 0
        scores = [[] for _ in range(self.watermark_len)]
        for i in range(0, h, self.block_size):
            for j in range(0, w, self.block_size):
                block = dct_blocks_embedded[i:i +
                                            self.block_size, j:j + self.block_size]
                # collect mid-band coefficients
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
        # keep your original thresholding style
        bits = np.uint8(np.array(avgScores) * 255 > 127)
        return bits

    # ---- Public API ----
    def encode(self, img, mask=None):
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
        if mask is None and isinstance(img_watermarked, tuple) and len(img_watermarked) == 2:
            img_watermarked, mask = img_watermarked

        LL, (LH, HL, HH) = pywt.dwt2(img_watermarked, 'haar')
        LL2, (LH2, HL2, HH2) = pywt.dwt2(HL, 'haar')
        block_dct = self.block_dct(HL2)
        extracted_information = self.extract_from_mid_bands(block_dct)
        return extracted_information


if __name__ == "__main__":

    # --- Load image ---
    img_test_path = os.path.join(DATA_PATH, "img (1).png")
    image = cv2.imread(img_test_path, cv2.IMREAD_UNCHANGED)
    assert image is not None, f"Not found: {img_test_path}"

    # Resize to match N=384 expected by this class (keeps PN length fixed)
    image = cv2.resize(image, (384, 384), interpolation=cv2.INTER_AREA)

    # Use the R channel (OpenCV is BGR)
    c = image[:, :, 2]

    # --- Prepare watermark bits ---
    wm_bits = np.random.randint(0, 2, size=32).astype(np.uint8)

    # --- Create system (no mask, true mid-band) ---
    sys_watermark = DWTDCTSystem(
        k=6512,
        watermark=wm_bits,
        alpha=5,  # try 15..25 for visual quality vs robustness
        block_size=4,
        mid_band_lo=0.30,  # keep mid 40% of zigzag
        mid_band_hi=0.70,
    )

    # --- Encode & Decode ---
    cw = sys_watermark.encode(c)  # no mask
    extracted_watermark = sys_watermark.decode(cw)

    # --- Metrics ---
    psnr = utils.calculate_psnr(c, cw)
    acc = (wm_bits == extracted_watermark).mean()

    print(f"PSNR: {psnr:.2f} dB")
    print("wm_bits :", wm_bits.tolist())
    print("extracted :", extracted_watermark.tolist())
    print(f"Bit accuracy : {acc*100:.2f}%")

    # --- Optional: visualize & save ---
    plt.figure()
    plt.imshow(c, cmap='gray')
    plt.title('Original channel (R)')
    plt.figure()
    plt.imshow(cw, cmap='gray')
    plt.title('Watermarked channel (R)')
    plt.figure()
    plt.imshow(cv2.absdiff(c, cw), cmap='gray')
    plt.title('Abs diff')
    plt.show()

    # Save full-color image with watermarked R channel
    # watermarked_color = image.copy()
    # watermarked_color[:, :, 2] = cw
    # cv2.imwrite('watermarked_color.jpg', watermarked_color)


# ------------------------------------

    # image = cv2.imread(cloud_body_img, cv2.IMREAD_UNCHANGED)
    # image = cv2.resize(image, (384, 384))
    # c = image[:, :, 2]

    # wm_bits = np.random.randint(0, 2, size=32)
    # sys_watermark = DWTDCTSystem(
    #     6512, wm_bits, alpha=25, block_size=4, mid_band_lo=0.30, mid_band_hi=0.70)

    # cw = sys_watermark.encode(c)
    # extracted_watermark = sys_watermark.decode(cw)

    # print(utils.calculate_psnr(c, cw))
    # print(wm_bits)
    # print(extracted_watermark)

    # # Visualization (optional)
    # plt.figure(); plt.imshow(c, cmap='gray'); plt.title('Original channel')
    # plt.figure(); plt.imshow(cw, cmap='gray'); plt.title('Watermarked channel')
    # plt.figure(); plt.imshow(np.abs(c - cw), cmap='gray'); plt.title('Abs diff')
    # plt.show()
