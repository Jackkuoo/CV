import os
import math
import numpy as np
import cv2
import matplotlib.pyplot as plt
manager = plt.get_current_fig_manager()
manager.window.showMaximized()


def hybrid(img1,img2,cutoff_frequency,Filter):
    assert img1.shape==img2.shape,'shape not match'
    h,w,c=img1.shape
    lowPassed = convolution(img1, Filter(h,w,cutoff_frequency,lowPass=True))
    highPassed = convolution(img2, Filter(h,w,cutoff_frequency, lowPass=False))
    return highPassed+lowPassed,lowPassed,highPassed

def idealFilter(h,w,cutoff_frequency,lowPass):
    x0,y0=w//2,h//2
    if lowPass:
        H=np.zeros((h,w))
        for x in range(x0-cutoff_frequency, x0+cutoff_frequency):
            for y in range(int(y0-math.sqrt(cutoff_frequency**2-(x-x0)**2)),int(y0+math.sqrt(cutoff_frequency**2-(x-x0)**2))):
                    H[y,x]=1
    else:
        H=np.ones((h,w))
        for x in range(x0-cutoff_frequency,x0+cutoff_frequency):
            for y in range(int(y0-math.sqrt(cutoff_frequency**2-(x-x0)**2)),int(y0+math.sqrt(cutoff_frequency**2-(x-x0)**2))):
                    H[y,x]=0
    return H

def GaussianFilter(h,w,cutoff_frequency, lowPass):
    x0,y0=w//2,h//2
    if lowPass:
        H=np.zeros((h,w))
        for x in range(w):
            for y in range(h):
                H[y,x]=math.exp(-1*((x-x0)**2+(y-y0)**2)/(2*cutoff_frequency**2))
    else:
        H=np.ones((h,w))
        for x in range(w):
            for y in range(h):
                H[y,x]-=math.exp(-1*((x-x0)**2+(y-y0)**2)/(2*cutoff_frequency**2))
    return H

def convolution(image, H):
    h,w,c=image.shape
    image_sp = np.zeros((h,w,c))
    result = np.zeros((h,w,c))
    for channel in range(c):
        image_ = image[:,:,channel] / 255
        for i in range(h):
            for j in range(w):
                image_[i,j]=((-1)**(i+j))*image_[i,j]

        F=(np.fft.fft2(image_))
        image_sp[:,:,channel]=20*np.log(np.abs(F))
        result[:,:,channel]=np.absolute(np.fft.ifft2(F*H))
    return result

def normalize(img):
    img_min = img.min()
    img_max = img.max()
    return (img - img_min) / (img_max - img_min)

if __name__ == "__main__":
  root = os.path.join('hw2_data','task1,2_hybrid_pyramid')
  name1 = np.array(['0_Afghan_girl_after.jpg','1_bicycle.bmp','2_bird.bmp','3_cat.bmp','4_einstein.bmp','5_fish.bmp','6_makeup_after.jpg','7_mouse.jpg'])
  name2 = np.array(['0_Afghan_girl_before.jpg','1_motorcycle.bmp','2_plane.bmp','3_dog.bmp','4_marilyn.bmp','5_submarine.bmp','6_makeup_before.jpg','7_rabbit.jpg'])
  cutoff_frequencies=[5,6,7,8,9,10] # bigger value -> clearer lowPass & less highPass

  for i in range(8):
    img1 = cv2.imread(os.path.join(root,name1[i]))
    img2 = cv2.imread(os.path.join(root,name2[i]))

    if name1[i]=='6_makeup_after.jpg' and name2[i]=='6_makeup_before.jpg':
      img1=img1[:-1,:-1,:]

    for cutoff_frequency in cutoff_frequencies:
      for name,Filter in zip(['ideal','gaussian'],[idealFilter,GaussianFilter]):
        result,lowPassed,highPassed = hybrid(img1,img2,cutoff_frequency,Filter)
        plt.subplot(231), plt.imshow(img1[:,:,::-1]), plt.xticks([]), plt.yticks([]), plt.title('image1')
        plt.subplot(234), plt.imshow(img2[:, :, ::-1]), plt.xticks([]), plt.yticks([]), plt.title('image2')
        plt.subplot(232), plt.imshow(normalize(lowPassed)[:,:,::-1]), plt.xticks([]), plt.yticks([]), plt.title('lowPass')
        plt.subplot(235), plt.imshow(normalize(highPassed)[:,:,::-1]), plt.xticks([]), plt.yticks([]), plt.title('highPass')
        plt.subplot(133), plt.imshow(normalize(result)[:,:,::-1]), plt.xticks([]), plt.yticks([]), plt.title(f'result (freq={cutoff_frequency})')
        #plt.savefig(os.path.join('result','task1',f'{name1.split(".")[0]}_{name2.split(".")[0]}_cutoff{cutoff_frequency}.jpg'))
        plt.show()