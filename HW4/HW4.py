import cv2
import numpy as np

test1 = cv2.imread("img/test1.tif", cv2.IMREAD_GRAYSCALE)
test2 = cv2.imread("img/test2.tif", cv2.IMREAD_GRAYSCALE)

m1, n1 = test1.shape
m2, n2 = test2.shape

# FFT
test1_ft = np.fft.fftshift(np.fft.fft2(test1))
test2_ft = np.fft.fftshift(np.fft.fft2(test2))

# Magnitude
test1_ft_mag = np.clip(20*np.log(np.abs(test1_ft)), 0, 255).astype('uint8')
test2_ft_mag = np.clip(20*np.log(np.abs(test2_ft)), 0, 255).astype('uint8')


# function


def Vertical_Notch_Reject(shape, width=[0, 0], height=[0, 0]):
    m, n = shape
    res = np.ones((m, n))
    res[height[0]:height[1], width[0]:width[1]] = 0
    return res


def Ideal_Notch_Reject(shape, points, d0):
    m, n = shape
    res = np.ones((m, n))
    for i in range(m):
        for j in range(n):
            for p in points:
                u0, v0 = p
                d1 = ((i-u0)**2+(j-v0)**2)**0.5
                if d1 <= d0:
                    res[i, j] = 0
    return res


# notch reject filter for test1
test1_filter = Vertical_Notch_Reject(
    test1.shape, [334, 341], [0, 315])*Vertical_Notch_Reject(test1.shape, [334, 341], [345, 662])
test1_filtered = test1_ft_mag*test1_filter.astype('uint8')

# notch reject filter for test2
bignoice_center = [[83, 54], [163, 54], [83, 114], [163, 114]]
smallnoice_center = [[43, 54], [203, 54], [43, 114], [203, 114]]
test2_filter = Ideal_Notch_Reject(test2.shape, bignoice_center, 8) * \
    Ideal_Notch_Reject(test2.shape, smallnoice_center, 8)
test2_filtered = test2_ft_mag*test2_filter.astype('uint8')

# Inverse FFT
test1_ifft = np.clip(
    np.round(np.abs(np.fft.ifft2(np.fft.ifftshift(test1_ft*test1_filter)))), 0, 255).astype('uint8')
test2_ifft = np.clip(
    np.round(np.abs(np.fft.ifft2(np.fft.ifftshift(test2_ft*test2_filter)))), 0, 255).astype('uint8')


cv2.imshow("test1 spectrum", test1_ft_mag)
cv2.imshow("test2 spectrum", test2_ft_mag)
cv2.imshow("test1 filtered spectrum", test1_filtered)
cv2.imshow("test2 filtered spectrum", test2_filtered)
cv2.imshow("test1 result", test1_ifft)
cv2.imshow("test2 result", test2_ifft)

cv2.imwrite("img/test1_spectrum.tif", test1_ft_mag)
cv2.imwrite("img/test2_spectrum.tif", test2_ft_mag)
cv2.imwrite("img/test1_filtered_spectrum.tif", test1_filtered)
cv2.imwrite("img/test2_filtered_spectrum.tif", test2_filtered)
cv2.imwrite("img/test1_result.tif", test1_ifft)
cv2.imwrite("img/test2_result.tif", test2_ifft)
cv2.waitKey(0)
cv2.destroyAllWindows()
