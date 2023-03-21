import cv2
import numpy as np

q1=cv2.imread('Q1.jpg', cv2.IMREAD_GRAYSCALE)
q2=cv2.imread('Q2.jpg', cv2.IMREAD_GRAYSCALE)
q3=cv2.imread('Q3.jpg', cv2.IMREAD_GRAYSCALE)

def generateCDF(img):
    y, x=img.shape
    pdf=np.zeros(256)
    cdf=np.zeros(256)

    for i in range(y):
        for j in range(x):
            pdf[img[i, j]]+=1
    pdf/=(x*y)
    for i in range(256):
        cdf[i]=np.round(np.sum(pdf[0:i+1])*255)
    return cdf

# histogram equalization
q1_heq=q1.copy()
q1_cdf=generateCDF(q1)

for i in range(q1.shape[0]):
    for j in range(q1.shape[1]):
        q1_heq[i, j]=q1_cdf[q1[i, j]]

# histogram specification
q1_hspec=q1.copy()
q2_cdf=generateCDF(q2)

for i in range(256):
    diff=np.abs(q2_cdf-q1_cdf[i])
    idx=diff.argmin()
    q1_cdf[i]=idx

for i in range(q1.shape[0]):
    for j in range(q1.shape[1]):
        q1_hspec[i, j]=q1_cdf[q1[i, j]]

# gaussian filter
def GaussianKernel(K=1, size=5, sigma=25):
    gauss_filter=np.zeros((size, size))
    total=0
    for i in range(-2, 3):
        for j in range(-2, 3):
            gauss_filter[i+2, j+2]=K*np.exp(-(i**2+j**2)/(2*sigma**2))
            total+=gauss_filter[i+2, j+2]
    gauss_filter/=total
    return gauss_filter

def Convolution(img):
    kernel=GaussianKernel()
    kernel_size=kernel.shape[0]
    y, x=img.shape
    vec=np.zeros((x*y, kernel_size*kernel_size))
    img=np.pad(img, 2, 'constant')
    flat_kernel=kernel.ravel()

    for i in range(y):
        for j in range(x):
            vec[i*x+j, :]=img[i:i+kernel_size, j:j+kernel_size].ravel()
    res=np.round(flat_kernel@vec.T).astype('uint8')
    return res.reshape((y, x))

q3_gaussian_filter=Convolution(q3)

cv2.imshow('Histogram Equalization', q1_heq)
cv2.imshow('Histogram Specification', q1_hspec)
cv2.imshow('Gaussian Filter', q3_gaussian_filter)
cv2.imwrite('Q1_heq.jpg', q1_heq)
cv2.imwrite('Q1_hspec.jpg', q1_hspec)
cv2.imwrite('Q3_GF.jpg', q3_gaussian_filter)
cv2.waitKey(0)
cv2.destroyAllWindows()