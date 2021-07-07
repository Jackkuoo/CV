import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

def GaussianFilter(frequency, highPass=False):

    size = 6 * frequency + 1
    if not size % 2:
        size = size + 1

    gaussian = lambda i,j: np.exp(-1.0 * ((i - size//2)**2 + (j - size//2)**2) / (2 * frequency**2))
    k = np.array([[1-gaussian(i,j) if highPass else gaussian(i,j) for j in range(size)] for i in range(size)])
    return k/np.sum(k)

def filterDFT(img, filterH):
    k_h, k_w = filterH.shape[0],filterH.shape[1]
    start_h,start_w = (img.shape[0] - k_h) // 2, (img.shape[1] - k_w) // 2
    pad_filter = np.zeros(img.shape[:2])
    pad_filter[start_h : start_h + k_h, start_w : start_w + k_w] = filterH # pad the filter

    filt_fft = np.fft.fft2(pad_filter)

    if len(img.shape) == 3:
        result = np.zeros(img.shape)
        for color in range(3):
            img_fft = np.fft.fft2(img[:, :, color])
            result[:, :, color] = np.fft.fftshift(np.fft.ifft2(img_fft * filt_fft)).real # apply the filter
        return result
    else:
        img_fft = np.fft.fft2(img)
        result_img = np.fft.ifft2(img_fft * filt_fft).real # apply the filter
        return np.fft.fftshift(result_img)

def subsampling(img):
    newImg = np.zeros((img.shape[0]//2,img.shape[1]//2,3)) if len(img.shape) == 3 else np.zeros((img.shape[0]//2,img.shape[1]//2))

    for i in range(newImg.shape[0]):
        for j in range(newImg.shape[1]):
            newImg[i][j] = img[2*i][2*j]
    return newImg
    
def upsampling(img,old_result):
    padc, padr = np.array(old_result.shape[:2]) - np.array(img.shape[:2])*2
    col_idx = (np.ceil(np.arange(1, 1 + img.shape[0]*2)//2) - 1).astype(int)
    row_idx = (np.ceil(np.arange(1, 1 + img.shape[1]*2)//2) - 1).astype(int)
    result = img[:, row_idx][col_idx, :]

    if len(img.shape) == 3:
        return np.pad(result,((padc,0),(padr,0),(0,0)),"constant")
    else:
        return np.pad(result,((padc,0),(padr,0)),"constant")


def img_to_spectrum(img):
    if len(img.shape) == 3:
        result = np.zeros(img.shape)
        for color in range(3):
            result[:, :, color] = np.fft.fft2(img[:, :, color]).real
        return normalize(np.log(1+np.abs(np.fft.fftshift(result)))) # normalizing result
    else:
        return normalize(np.log(1+np.abs(np.fft.fftshift(np.fft.fft2(img))))) # normalizing result

def normalize(img):
    img_min = img.min()
    img_max = img.max()
    return (img - img_min) / (img_max - img_min)

if __name__ == "__main__":
    root = os.path.join('hw2_data','task1,2_hybrid_pyramid')
    name='4_einstein.bmp'
    step=5
    img1 = cv2.imread(os.path.join(root,name))
    result = img1
    for i in range(step):
        old_result = result
        result = filterDFT(result,GaussianFilter(2))
        #cv2.imwrite(os.path.join('result','task2',f'gaussian_{i}.jpg'),result)
        cv2.imshow(result)
        gaussian_spectrum = img_to_spectrum(result)
        plt.imshow(gaussian_spectrum)
        #plt.savefig(os.path.join('result','task2',f'gaussian_sp_{i}.jpg'))
        plt.show()

        if i == 5:
            Laplacian = result
        else:
            result = subsampling(result)
            result2 = upsampling(result,old_result)
            Laplacian = old_result - result2
        cv2.imwrite(os.path.join('result','task2',f'Laplacian_{i}.jpg'),Laplacian)
        cv2.imshow(Laplacian)

        Laplacian_spectrum = img_to_spectrum(Laplacian)
        plt.imshow(Laplacian_spectrum)
        #plt.savefig(os.path.join('result','task2',f'Laplacian_sp_{i}.jpg'))
        plt.show()
