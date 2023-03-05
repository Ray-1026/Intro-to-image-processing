import cv2
import numpy as np

img=cv2.imread('test.jpg')

x=len(img[0])   #600
y=len(img)      #360

#exchange position
temp=img[0:y//3, 0:x//3].copy()
img[0:y//3, 0:x//3]=img[0:y//3, x//3*2:x]
img[0:y//3, x//3*2:x]=temp

#gray scale
def gray_scale(img, y, x):
    avg=(int(img[y, x, 0])+int(img[y, x, 1])+int(img[y, x, 2]))/3
    img[y, x]=np.uint8(avg)
    return img

for i in range(y//3*2, y):
    for j in range(0, x//3):
        img=gray_scale(img, i, j)

#intensity resolution
for i in range(y//3*2, y):
    for j in range(x//3*2, x):
        avg=(int(img[i, j, 0])+int(img[i, j, 1])+int(img[i, j, 2]))/3
        avg=int(avg/64)*64
        img[i, j]=np.uint8(avg)

#color filter - red
for i in range(y//3, y//3*2):
    for j in range(x//3):
        b, g, r=img[i, j]
        if r<=150 or r*0.6<=b or r*0.6<=g:
            img=gray_scale(img, i, j)

#color filter - yellow
for i in range(y//3, y//3*2):
    for j in range(x//3*2, x):
        b, g, r=img[i, j]
        if (int(g)+int(r))*0.3<=int(b) or abs(int(g)-int(r))>=50:
            img=gray_scale(img, i, j)

#channel operation
for i in range(y//3*2, y):
    for j in range(x//3, x//3*2):
        if img[i, j, 1]>127:
            img[i, j, 1]=255
        else:
            img[i, j, 1]*=2

#bilinear interpolation
def bilinear_interpolation(img, y, x):
    h=int(y//3)
    w=int(x//3)

    temp=img[0:y//3, x//3:x//3*2].copy()

    for i in range(1, h):
        for j in range(1, w):
            xf=(j+0.5)/2-0.5
            yf=(i+0.5)/2-0.5
            xi=min(int(xf), w-2)
            yi=min(int(yf), h-2)
            dx=xf-xi
            dy=yf-yi

            new_b=np.uint8(temp[yi, xi, 0]*(1-dx)*(1-dy)+temp[yi, xi+1, 0]*(dx)*(1-dy)+temp[yi+1, xi, 0]*(1-dx)*(dy)+temp[yi+1, xi+1, 0]*(dx)*(dy))
            new_g=np.uint8(temp[yi, xi, 1]*(1-dx)*(1-dy)+temp[yi, xi+1, 1]*(dx)*(1-dy)+temp[yi+1, xi, 1]*(1-dx)*(dy)+temp[yi+1, xi+1, 1]*(dx)*(dy))
            new_r=np.uint8(temp[yi, xi, 2]*(1-dx)*(1-dy)+temp[yi, xi+1, 2]*(dx)*(1-dy)+temp[yi+1, xi, 2]*(1-dx)*(dy)+temp[yi+1, xi+1, 2]*(dx)*(dy))
            
            img[i, j+w]=[new_b, new_g, new_r]
    return img

img=bilinear_interpolation(img, y, x)

#bicubic interpolation
def Weight(x):
    a=-0.5
    x=abs(x)
    if x<=1:
        return (a+2)*(x**3)-(a+3)*(x**2)+1
    elif 1<x<2:
        return a*(x**3)-5*a*(x**2)+8*a*x-4*a
    else:
        return 0

def bicubic_interpolation(img, y, x):
    h=y//3
    w=x//3
    temp=img[y//3:y//3*2, x//3:x//3*2].copy()

    for i in range(h):
        for j in range(w):
            xf=(j+0.5)/2-0.5
            yf=(i+0.5)/2-0.5
            xi=int(xf)
            yi=int(yf)
            dx=xf-xi
            dy=yf-yi

            res_b, res_g, res_r=0, 0, 0
            for k in range(-1, 3):
                for m in range(-1, 3):
                    if 0<=yi+k<h and 0<=xi+m<w:
                        res_b+=temp[yi+k, xi+m, 0]*Weight(k-dy)*Weight(m-dx)
                        res_g+=temp[yi+k, xi+m, 1]*Weight(k-dy)*Weight(m-dx)
                        res_r+=temp[yi+k, xi+m, 2]*Weight(k-dy)*Weight(m-dx)
            img[y//3+i, x//3+j]=[np.clip(res_b, 0, 255), np.clip(res_g, 0, 255), np.clip(res_r, 0, 255)]
    return img

img=bicubic_interpolation(img, y, x)

cv2.imshow('Result', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('result.jpg', img)