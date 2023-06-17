import cv2
import numpy as np
import math


# 1. Histogram Equalization - Q1.jpg
img_1 = cv2.imread("Q1.jpg")
height_1, width_1 = img_1.shape[:2]
cdf_1 = np.zeros(256)
MN_1 = height_1 * width_1

for i in range(height_1):
    for j in range(width_1):
        cdf_1[img_1[i, j, 0]] += 1

cdf_1[0] /= MN_1

for i in range(1, 256):
    cdf_1[i] /= MN_1
    cdf_1[i] += cdf_1[i - 1]

for i in range(height_1):
    for j in range(width_1):
        color_value = img_1[i, j, 0]
        img_1[i, j, 0] = round(255 * cdf_1[color_value])
        img_1[i, j, 1] = round(255 * cdf_1[color_value])
        img_1[i, j, 2] = round(255 * cdf_1[color_value])

cv2.imwrite('HW2_Q1.jpg', img_1)

# 2. Histogram Specification - Transform the histogram of Q1.jpg to the histogram of Q2.jpg.
img_1 = cv2.imread("Q1.jpg")
img_2 = cv2.imread("Q2.jpg")
height_2, width_2 = img_2.shape[:2]
cdf_2 = np.zeros(256)
MN_2 = height_2 * width_2

for i in range(height_2):
    for j in range(width_2):
        cdf_2[img_2[i, j, 0]] += 1

cdf_2[0] /= MN_2

for i in range(1, 256):
    cdf_2[i] /= MN_2
    cdf_2[i] += cdf_2[i - 1]

inverse_map = np.full((256), -1)

for i in range(255, -1, -1):
    inverse_map[round(255 * cdf_2[i])] = i

for i in range(height_1):
    for j in range(width_1):
        color_value = round(255 * cdf_1[img_1[i, j, 0]])
        if inverse_map[color_value] == -1:
            k = 1
            while True:
                if color_value - k >= 0 and inverse_map[color_value - k] != -1:
                    img_1[i, j, 0], img_1[i, j, 1], img_1[i, j,
                                                          2] = inverse_map[color_value - k], inverse_map[color_value - k], inverse_map[color_value - k]
                    break

                elif color_value + k <= 255 and inverse_map[color_value + k] != -1:
                    img_1[i, j, 0], img_1[i, j, 1], img_1[i, j,
                                                          2] = inverse_map[color_value + k], inverse_map[color_value + k], inverse_map[color_value + k]
                    break

                k += 1
        else:
            img_1[i, j, 0], img_1[i, j, 1], img_1[i, j,
                                                  2] = inverse_map[color_value], inverse_map[color_value], inverse_map[color_value]

cv2.imwrite('HW2_Q2.jpg', img_1)

# 3.Gaussian Filter (K=1, size=5x5, σ=25) - Q3.jpg
img_3 = cv2.imread("Q3.jpg")
img_3 = cv2.cvtColor(img_3, cv2.COLOR_BGR2GRAY)
height_3, width_3 = img_3.shape
result_img = np.zeros([height_3, width_3, 3], dtype=np.uint8)
img_3_pad = np.pad(img_3, ((2, 2), (2, 2)))
kernel = np.zeros([5, 5])

for i in range(5):
    for j in range(5):
        kernel[i, j] = math.exp(-((i - 2) ** 2 + (j - 2) ** 2) / 1250)
denominator = np.sum(kernel)

for i in range(height_3):
    for j in range(width_3):
        color_value = 0
        for a in range(5):
            for b in range(5):
                color_value += kernel[a, b] * \
                    img_3_pad[i + a, j + b] / denominator
        color_value = int(color_value)
        result_img[i, j, 0], result_img[i, j, 1], result_img[i,
                                                             j, 2] = color_value, color_value, color_value

cv2.imwrite('HW2_Q3.jpg', result_img)
