import os
import matplotlib.pyplot as plt
import numpy as np
import cv2

def ncc(g,f):
    g=g-g.mean(axis=0)
    f=f-f.mean(axis=0)
    return np.sum((g * f)/(np.linalg.norm(g)) * (np.linalg.norm(f)))

def Align(target,x,t):
    mini = float("-inf")
    col=np.linspace(-t,t,2*t,dtype=int)
    row=np.linspace(-t,t,2*t,dtype=int)

    for i in col:
        for j in row:
            diff = ncc(target,np.roll(x,[i,j],axis=(0,1)))
            if diff > mini:
                mini = diff
                offset = [i,j]
    return offset

if __name__ == "__main__":

    root = os.path.join('hw2_data','task3_colorizing')
    allimg = ['cathedral.jpg','emir.tif','icon.tif','lady.tif','melons.tif','monastery.jpg','nativity.jpg','onion_church.tif','three_generations.tif','tobolsk.jpg','train.tif','village.tif','workshop.tif']
    for name in allimg:
        img=cv2.imread(os.path.join(root,name),0)
        w, h = img.shape[:2]
        img = img[int(w * 0.01):int(w - w * 0.01), int(h * 0.01):int(h - h * 0.01)]
        w, h = img.shape[:2]
        
        height = w // 3
        blue = img[0:height, :]
        green = img[height:2 * height, :]
        red = img[2 * height:3 * height, :]
        offset_g = Align(blue,green,10)
        offset_r = Align(blue,red,10)
        print(offset_g)
        print(offset_r)
        green=np.roll(green,offset_g,axis=(0,1))
        red=np.roll(red,offset_r,axis=(0,1))
        result = np.concatenate((red[:,:,None],green[:,:,None],blue[:,:,None]),axis=2)
        plt.imshow(result)
        #plt.savefig(os.path.join('result','task3',f'{name.split(".")[0]}'))
        #plt.waitforbuttonpress(0)
        plt.show()