import cv2
import numpy as np


def read_yaxis_point(x):
    return img[x, y1].astype(np.int32), img[x, y2].astype(
        np.int32), img[x, y3].astype(np.int32), img[x, y4].astype(np.int32)


def bicubic(p1, p2, p3, p4, distance):
    return (-1 / 2 * p1 + 3 / 2 * p2 - 3 / 2 * p3 + 1 / 2 * p4) * (distance ** 3) + \
        (p1 - 5 / 2 * p2 + 2 * p3 - 1 / 2 * p4) * \
        (distance ** 2) + (-1 / 2 * p1 + 1 / 2 * p3) * distance + p2


def overflow_address(color_value):
    if color_value > 255:
        return 255
    elif color_value < 0:
        return 0
    else:
        return color_value


# 讀取圖片
img = cv2.imread("test.jpg")
height, width, channels = img.shape
piece_height, piece_width = height // 3, width // 3
pieces = []

# 將圖片分成九宮格
for i in range(3):
    for j in range(3):
        x, y = j * piece_width, i * piece_height
        piece = img[y: y + piece_height, x: x + piece_width]
        pieces.append(piece)

# 將此格用 Bicubic interpolation 放大2倍，只需保留放大後左上角和原格子大小相同的部分 (10%)
enlarge_img = np.zeros([piece_height, piece_width, 3], dtype=np.uint8)

for i in range(piece_height, piece_height * 2):
    for j in range(piece_width, piece_width * 2):
        x = (i - 120) / 2 + 120
        y = (j - 200) / 2 + 200
        x1, y1 = int(x) - 1, int(y) - 1
        x2, y2 = x1 + 1, y1 + 1
        x3, y3 = x2 + 1, y2 + 1
        x4, y4 = x3 + 1, y3 + 1
        dx = x - x2
        dy = y - y2

        p1, p2, p3, p4 = read_yaxis_point(x1)
        q1 = bicubic(p1, p2, p3, p4, dy)

        p1, p2, p3, p4 = read_yaxis_point(x2)
        q2 = bicubic(p1, p2, p3, p4, dy)

        p1, p2, p3, p4 = read_yaxis_point(x3)
        q3 = bicubic(p1, p2, p3, p4, dy)

        p1, p2, p3, p4 = read_yaxis_point(x4)
        q4 = bicubic(p1, p2, p3, p4, dy)

        b, g, r = bicubic(q1, q2, q3, q4, dx)
        b, g, r = overflow_address(b), overflow_address(g), overflow_address(r)
        enlarge_img[i - 120, j - 200] = b, g, r

pieces[4] = enlarge_img

# 交換指定兩格的內容 (20%)
pieces[0], pieces[2] = pieces[2], pieces[0]

# 將此格轉成灰階影像 (10%)
piece = pieces[6].astype(np.int32)

for i in range(piece_height):
    for j in range(piece_width):
        b, g, r = piece[i, j]
        piece[i, j] = (b + g + r) // 3

pieces[6] = piece.astype(np.uint8)

# 將此格轉為灰階影像，再把灰階的Intensity resolution降為4 (256→4) (10%)
piece = pieces[8].astype(np.int32)

for i in range(piece_height):
    for j in range(piece_width):
        b, g, r = piece[i, j]
        gray_value = (b + g + r) // 3

        if gray_value <= 63:
            piece[i, j] = 0
        elif gray_value <= 127:
            piece[i, j] = 64
        elif gray_value <= 191:
            piece[i, j] = 128
        else:
            piece[i, j] = 192

pieces[8] = piece.astype(np.uint8)

# 紅色濾鏡: 只保留此格圖片的紅色區塊，其餘轉為灰階 (10%)
piece = pieces[3].astype(np.int32)

for i in range(piece_height):
    for j in range(piece_width):
        b, g, r = piece[i, j]

        if r <= 150 or r * 0.6 <= b or r * 0.6 <= g:
            piece[i, j] = (b + g + r) // 3

pieces[3] = piece.astype(np.uint8)

# 黃色濾鏡: 只保留此格圖片的黃色區塊，其餘轉為灰階 (10%)
piece = pieces[5].astype(np.int32)

for i in range(piece_height):
    for j in range(piece_width):
        b, g, r = piece[i, j]

        if (g + r) * 0.3 <= b or abs(g - r) >= 50:
            piece[i, j] = (b + g + r) // 3

pieces[5] = piece.astype(np.uint8)

# 將此格的綠色值放大2倍 (10%)
piece = pieces[7].astype(np.int32)

for i in range(piece_height):
    for j in range(piece_width):
        piece[i, j, 1] = overflow_address(piece[i, j, 1] * 2)

pieces[7] = piece.astype(np.uint8)

# 將此格用 Bilinear interpolation 放大2倍，只需保留放大後左上角和原格子大小相同的部分 (10%)
piece = pieces[1]
enlarge_img = np.zeros([piece_height, piece_width, 3], dtype=np.uint8)

for i in range(piece_height):
    for j in range(piece_width):
        r, c = i / 2, j / 2
        r1, c1 = int(r), int(c)
        r2, c2 = r1 + 1, c1 + 1
        frac_r, frac_c = r - r1, c - c1
        top = (1 - frac_c) * piece[r1, c1] + frac_c * piece[r1, c2]
        bottom = (1 - frac_c) * piece[r2, c1] + frac_c * piece[r2, c2]
        enlarge_img[i, j] = (1 - frac_r) * top + frac_r * bottom

pieces[1] = enlarge_img

# 合併九宮格並輸出結果
new_img = cv2.vconcat([cv2.hconcat(pieces[:3]), cv2.hconcat(
    pieces[3:6]), cv2.hconcat(pieces[6:])])
cv2.imwrite("output.png", new_img)
