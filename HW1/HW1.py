import cv2
import numpy as np

img = cv2.imread('img/test.jpg')

x = len(img[0])  # 600
y = len(img)  # 360

# exchange position
temp = img[0:y//3, 0:x//3].copy()
img[0:y//3, 0:x//3] = img[0:y//3, x//3*2:x]
img[0:y//3, x//3*2:x] = temp

# gray scale


def gray_scale(img, y, x):
    avg = (int(img[y, x, 0])+int(img[y, x, 1])+int(img[y, x, 2]))/3
    img[y, x] = np.uint8(avg)
    return img


for i in range(y//3*2, y):
    for j in range(0, x//3):
        img = gray_scale(img, i, j)

# intensity resolution
for i in range(y//3*2, y):
    for j in range(x//3*2, x):
        avg = (int(img[i, j, 0])+int(img[i, j, 1])+int(img[i, j, 2]))/3
        avg = int(avg/64)*64
        img[i, j] = np.uint8(avg)

# color filter - red
for i in range(y//3, y//3*2):
    for j in range(x//3):
        b, g, r = img[i, j]
        if r <= 150 or r*0.6 <= b or r*0.6 <= g:
            img = gray_scale(img, i, j)

# color filter - yellow
for i in range(y//3, y//3*2):
    for j in range(x//3*2, x):
        b, g, r = img[i, j]
        if (int(g)+int(r))*0.3 <= int(b) or abs(int(g)-int(r)) >= 50:
            img = gray_scale(img, i, j)

# channel operation
for i in range(y//3*2, y):
    for j in range(x//3, x//3*2):
        if img[i, j, 1] > 127:
            img[i, j, 1] = 255
        else:
            img[i, j, 1] *= 2

# bilinear interpolation


def bilinear_interpolation(img, y, x):
    h = int(y//3)
    w = int(x//3)

    temp = img[0:y//3, x//3:x//3*2].copy()

    for i in range(h):
        for j in range(w):
            xf = (j+0.5)/2-0.5
            yf = (i+0.5)/2-0.5
            if xf < 0:
                xf = 0
            if yf < 0:
                yf = 0
            xi = min(int(xf), w-2)
            yi = min(int(yf), h-2)
            dx = xf-xi
            dy = yf-yi

            new_b = np.uint8(temp[yi, xi, 0]*(1-dx)*(1-dy)+temp[yi, xi+1, 0]*(dx)
                             * (1-dy)+temp[yi+1, xi, 0]*(1-dx)*(dy)+temp[yi+1, xi+1, 0]*(dx)*(dy))
            new_g = np.uint8(temp[yi, xi, 1]*(1-dx)*(1-dy)+temp[yi, xi+1, 1]*(dx)
                             * (1-dy)+temp[yi+1, xi, 1]*(1-dx)*(dy)+temp[yi+1, xi+1, 1]*(dx)*(dy))
            new_r = np.uint8(temp[yi, xi, 2]*(1-dx)*(1-dy)+temp[yi, xi+1, 2]*(dx)
                             * (1-dy)+temp[yi+1, xi, 2]*(1-dx)*(dy)+temp[yi+1, xi+1, 2]*(dx)*(dy))

            img[i, j+w] = [new_b, new_g, new_r]
    return img


img = bilinear_interpolation(img, y, x)

# bicubic interpolation


def bicubic_interpolation(img, y, x):
    h = y//3
    w = x//3
    temp = img[y//3:y//3*2, x//3:x//3*2].copy()

    for i in range(h):
        for j in range(w):
            xf = (j+0.5)/2-0.5
            yf = (i+0.5)/2-0.5
            if xf < 0:
                xf = 0
            if yf < 0:
                yf = 0
            xi = min(int(xf), w-2)
            yi = min(int(yf), h-2)
            dx = xf-xi
            dy = yf-yi

            new_b = []
            new_g = []
            new_r = []

            for k in range(-1, 3):
                r = []
                g = []
                b = []
                for m in range(-1, 3):
                    if 0 <= xi+m < w and 0 <= yi+k < h:
                        b.append(temp[yi+k, xi+m, 0])
                        g.append(temp[yi+k, xi+m, 1])
                        r.append(temp[yi+k, xi+m, 2])
                    else:
                        b.append(0)
                        g.append(0)
                        r.append(0)
                new_b.append((-0.5*b[0]+1.5*b[1]-1.5*b[2]+0.5*b[3])*dx**3 +
                             (b[0]-2.5*b[1]+2*b[2]-0.5*b[3])*dx**2+(-0.5*b[0]+0.5*b[2])*dx+b[1])
                new_g.append((-0.5*g[0]+1.5*g[1]-1.5*g[2]+0.5*g[3])*dx**3 +
                             (g[0]-2.5*g[1]+2*g[2]-0.5*g[3])*dx**2+(-0.5*g[0]+0.5*g[2])*dx+g[1])
                new_r.append((-0.5*r[0]+1.5*r[1]-1.5*r[2]+0.5*r[3])*dx**3 +
                             (r[0]-2.5*r[1]+2*r[2]-0.5*r[3])*dx**2+(-0.5*r[0]+0.5*r[2])*dx+r[1])
            img[y//3+i, x//3+j, 0] = np.clip((-0.5*new_b[0]+1.5*new_b[1]-1.5*new_b[2]+0.5*new_b[3])*dy**3+(
                new_b[0]-2.5*new_b[1]+2*new_b[2]-0.5*new_b[3])*dy**2+(-0.5*new_b[0]+0.5*new_b[2])*dy+new_b[1], 0, 255)
            img[y//3+i, x//3+j, 1] = np.clip((-0.5*new_g[0]+1.5*new_g[1]-1.5*new_g[2]+0.5*new_g[3])*dy**3+(
                new_g[0]-2.5*new_g[1]+2*new_g[2]-0.5*new_g[3])*dy**2+(-0.5*new_g[0]+0.5*new_g[2])*dy+new_g[1], 0, 255)
            img[y//3+i, x//3+j, 2] = np.clip((-0.5*new_r[0]+1.5*new_r[1]-1.5*new_r[2]+0.5*new_r[3])*dy**3+(
                new_r[0]-2.5*new_r[1]+2*new_r[2]-0.5*new_r[3])*dy**2+(-0.5*new_r[0]+0.5*new_r[2])*dy+new_r[1], 0, 255)
    return img


img = bicubic_interpolation(img, y, x)

cv2.imshow('Result', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('img/result.jpg', img)
