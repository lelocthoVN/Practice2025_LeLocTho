import cv2
import math
import numpy as np
import utils
import matplotlib.pyplot as plt
from scipy.signal import convolve2d

cloud_body_img = r"..\..\dataset\Water Bodies Dataset\Images\water_body_3.jpg"
cloud_body_mask = r"..\..\dataset\Water Bodies Dataset\Masks\water_body_3.jpg"


class SystemJASW:
    def __init__(self, K, wm_bit, M, alpha):
        self.K = K
        self.M = M
        self.alpha = alpha
        self.wm_bit = wm_bit
        self.wm = utils.binary_array_to_decimal(wm_bit)

    @staticmethod
    def corr2d_unnormalized(base: np.ndarray, pattern: np.ndarray) -> np.ndarray:
        assert len(base.shape) == 2 and len(pattern.shape) == 2
        N1, N2 = pattern.shape
        M1, M2 = base.shape
        h_flipped = np.flip(pattern)
        base_padded = np.pad(base, ((0, N1 - 1), (0, N2 - 1)))
        h_padded = np.pad(h_flipped, ((0, M1 - 1), (0, M2 - 1)))
        F = np.fft.fft2(base_padded)
        H = np.fft.fft2(h_padded)
        G = np.fft.ifft2(F * H)
        return np.real(G)

    @staticmethod
    def phase_only_filter(x):
        magnitude = np.abs(x)
        magnitude[magnitude == 0] = 1
        return x / magnitude

    @staticmethod
    def corr2d_normalized(base: np.ndarray, pattern: np.ndarray) -> np.ndarray:
        assert len(base.shape) == 2 and len(pattern.shape) == 2
        N1, N2 = pattern.shape
        M1, M2 = base.shape
        h_flipped = np.flip(pattern)
        base_padded = np.pad(base, ((0, N1 - 1), (0, N2 - 1)))
        h_padded = np.pad(h_flipped, ((0, M1 - 1), (0, M2 - 1)))
        F = np.fft.fft2(base_padded)
        H = np.fft.fft2(h_padded)
        G = np.fft.ifft2(SystemJASW.phase_only_filter(F)
                         * SystemJASW.phase_only_filter(H))
        return np.real(G)

    @staticmethod
    def conv(img_input, mask_input):
        return convolve2d(img_input, mask_input, boundary="symm", mode="same")

    @staticmethod
    def generate_sample_P(K, M):
        np.random.seed(K)
        P = np.random.normal(0, 1, (M, M))
        return P

    @staticmethod
    def shift(P, delta_x, delta_y):
        return np.roll(np.roll(P, delta_x, axis=0), delta_y, axis=1)

    @staticmethod
    def compute_W(P, b):
        arr = utils.decimal_to_binary_array(b, 14)
        shifted_P = SystemJASW.shift(P, utils.binary_array_to_decimal(
            arr[:7]), utils.binary_array_to_decimal(arr[7:14]))
        W_t = P - shifted_P
        return W_t

    def encode(self, img, mask):
        img_norm = img/255
        P = SystemJASW.generate_sample_P(self.K, self.M)
        W = SystemJASW.compute_W(P, self.wm)

        container_wm_ed = np.zeros_like(img_norm)
        N1, N2 = container_wm_ed.shape
        for row in range(N1):
            for col in range(N2):
                container_wm_ed[row][col] = img_norm[row][col] + \
                    self.alpha * mask[row][col] * W[row % self.M][col % self.M]

        plt.figure()
        plt.imshow(container_wm_ed-img_norm, cmap='gray')

        return container_wm_ed*255

    def decode(self, cw, normalize_corr=False):
        cw_norm = cw/255
        P = SystemJASW.generate_sample_P(self.K, self.M)
        N1, N2 = cw_norm.shape
        W_hat = np.zeros((self.M, self.M))
        S = math.floor(N1 / self.M) * math.floor(N2 / self.M)
        for m1 in range(self.M):
            for m2 in range(self.M):
                for row in range(math.floor(N1 / self.M)):
                    for col in range(math.floor(N2 / self.M)):
                        W_hat[m1][m2] += cw_norm[row *
                                                 self.M + m1][col * self.M + m2]
        W_hat = W_hat / S
        if normalize_corr:
            B = SystemJASW.corr2d_normalized(W_hat, P)
        else:
            B = SystemJASW.corr2d_unnormalized(W_hat, P)

        x = np.argmin(B)
        row_x = x // B.shape[0]
        col_x = x % B.shape[1]
        y = np.argmax(B)
        row_y = y // B.shape[0]
        col_y = y % B.shape[1]

        shift_x = row_x - row_y
        shift_y = col_x - col_y
        shift_x = (shift_x + self.M * 2) % self.M
        shift_y = (shift_y + self.M * 2) % self.M
        print(utils.decimal_to_binary_array(self.wm, 14))
        extracted_wm = utils.decimal_to_binary_array(
            shift_x, 7) + utils.decimal_to_binary_array(shift_y, 7)
        print(extracted_wm)
        return extracted_wm, extracted_wm == self.wm


if __name__ == "__main__":
    img = cv2.imread(cloud_body_img, cv2.IMREAD_UNCHANGED)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mask = cv2.imread(cloud_body_mask, cv2.IMREAD_GRAYSCALE)/255
    K = 6512
    M = 128
    hidden_information = 121
    wm_bit = utils.decimal_to_binary_array(hidden_information)
    JAWS = SystemJASW(K=K, M=M, alpha=0.02, wm_bit=wm_bit)
    CW = JAWS.encode(img[:, :, 0], mask)
    extracted, _ = JAWS.decode(CW, True)

    CW = np.clip(CW, 0, 255)

    print(utils.binary_array_to_decimal(extracted) == hidden_information)
    print(img)
    print(CW)
    print(utils.calculate_psnr(img[:, :, 0], CW))

    plt.figure()
    plt.imshow(img[:, :, 0], cmap='gray')
    plt.figure()
    plt.imshow(CW, cmap='gray')
    plt.figure()
    plt.imshow(np.abs(CW-img[:, :, 0]), cmap='gray')
    plt.show()
    #
