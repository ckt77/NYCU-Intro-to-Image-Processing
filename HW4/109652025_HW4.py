import cv2
import numpy as np
from matplotlib import pyplot as plt


# test1

# Fourier Transform (Spectrum)
img_1 = cv2.imread("test1.tif", cv2.IMREAD_GRAYSCALE)
f_1 = np.fft.fft2(img_1)
fshift_1 = np.fft.fftshift(f_1)
spectrum_1 = 20 * np.log(np.abs(fshift_1))
plt.subplot(121)
plt.imshow(spectrum_1, cmap='gray')
plt.title('test1 spectrum')
plt.axis('off')

# Parameters
row_1, col_1 = img_1.shape
center_row_1, center_col_1 = row_1 // 2, col_1 // 2
radius_row, radius_col = (20, 5)

# Filter
IBRF_1 = np.ones(img_1.shape)
IBRF_1[:center_row_1 - radius_row,
       center_col_1 - radius_col:center_col_1 + radius_col] = 0
IBRF_1[center_row_1 + radius_row:,
       center_col_1 - radius_col:center_col_1 + radius_col] = 0
filtered_spectrum_1 = IBRF_1 * spectrum_1
plt.subplot(122)
plt.imshow(filtered_spectrum_1, cmap='gray')
plt.title('test1 filtered spectrum')
plt.axis('off')
plt.show()

# Final Image
f_ishift_1 = fshift_1 * IBRF_1
f_ishift_1 = np.fft.ifftshift(f_ishift_1)
img_1_back = np.fft.ifft2(f_ishift_1)
img_1_back = np.abs(img_1_back)

plt.subplot(121)
plt.imshow(img_1, cmap='gray')
plt.title('test1 origin')
plt.axis('off')
plt.subplot(122)
plt.imshow(img_1_back, cmap='gray')
plt.axis('off')
plt.title('test1 result')
plt.show()


# test2

# Fourier Transform (Spectrum)
img_2 = cv2.imread("test2.tif", cv2.IMREAD_GRAYSCALE)
f_2 = np.fft.fft2(img_2)
fshift_2 = np.fft.fftshift(f_2)
spectrum_2 = 20 * np.log(np.abs(fshift_2))
plt.subplot(121)
plt.imshow(spectrum_2, cmap='gray')
plt.title('test2 spectrum')
plt.axis('off')

# Parameters
row_2, col_2 = img_2.shape
center_2 = [(83, 54), (41, 54), (163, 54), (205, 54),
            (40, 110), (80, 110), (160, 110), (202, 110)]
C_2, W_2, n_2 = (5, 12, 2)

# Filter
IBRF_2 = np.ones(img_2.shape)
for i in range(row_2):
    for j in range(col_2):
        for k in center_2:
            if (abs(((i - k[0]) ** 2 + (j - k[1]) ** 2) ** (1 / 2) - C_2) < W_2 / 2):
                IBRF_2[i, j] = 0
filtered_spectrum_2 = IBRF_2 * spectrum_2
plt.subplot(122)
plt.imshow(filtered_spectrum_2, cmap='gray')
plt.title('test2 filtered spectrum')
plt.axis('off')
plt.show()

# Final Image
fshift_2 = fshift_2 * IBRF_2
f_ishift_2 = np.fft.ifftshift(fshift_2)
img_2_back = np.fft.ifft2(f_ishift_2)
img_2_back = np.abs(img_2_back)

plt.subplot(121)
plt.imshow(img_2, cmap='gray')
plt.title('test2 origin')
plt.axis('off')
plt.subplot(122)
plt.imshow(img_2_back, cmap='gray')
plt.axis('off')
plt.title('test2 result')
plt.show()
