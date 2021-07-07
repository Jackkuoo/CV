import os
import cv2
import numpy as np
import collections
from scipy.spatial.distance import cdist
import cyvlfeat as vlfeat  # https://github.com/menpo/cyvlfeat/blob/master/cyvlfeat/sift/dsift.py


def loadImgs(filepath):
    img_list=[]

    class_names=os.listdir(filepath)
    for class_name in class_names:
        class_path = os.path.join(filepath,class_name)
        image_names = os.listdir(class_path)
        for image_name in image_names:
            try:
                img_path = os.path.join(class_path,image_name)
                img = cv2.imread(img_path)
                img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                img_list.append(img)
            except:
                pass

    return img_list

def get_all_features(img_list):
    """
    :param imgs: a list of images in an category
    :return: (#keypoints from all images,128) ndarray
    """
    keypoints=None
    describes=None
    for img in img_list:
        kp,des=vlfeat.sift.dsift(img,step=8,fast=True,float_descriptors=True)
        des=des[:int(len(des)*0.01)]  # randomly choose features from lots of features
        describes=np.vstack((describes,des)) if describes is not None else des

    return describes

def img2histogram(img,vocabulary_size,centers):
    """represent image by frequencies of visual words in vocabulary
    :param img: an (200x200) ndarray img
    :param vocabulary_size:
    :param centers: (vocabulary_size,128) center points
    :return: (vocabulary_size)
    """
    kp,des=vlfeat.sift.dsift(img,step=8,fast=True,float_descriptors=True)
    distances=cdist(des,centers,'euclidean')
    counter=collections.Counter(np.argmin(distances,axis=1))
    re=np.zeros(vocabulary_size)
    for i in counter:
        re[i]=counter[i]

    return re

def get_histograms(img_list,vocabulary_size,centers):
    """
    :param img_list: list of all categories images
    :param vocabulary_size:
    :param centers: (vocabulary_size,128) ndarray
    :return: (#images,vocabulary_size) ndarray
    """
    histograms=np.zeros((len(img_list),vocabulary_size))
    for i,img in enumerate(img_list):
        histograms[i]=img2histogram(img,vocabulary_size,centers)

    return histograms

def knn(indices):
    """
    :param indics: (#test images,k) ndarray
    """
    indices = indices // 100
    k=indices.shape[1]
    acc=0
    for i in range(len(indices)):
        target_class=i//10
        predict_class=collections.Counter(indices[i]).most_common(1)[0][0]
        if target_class==predict_class:
            acc+=1

    return acc/len(indices)


train_path=os.path.join('hw5_data','train')
test_path=os.path.join('hw5_data','test')
category=list(os.listdir(train_path))
vocabulary_size=100
k=7


if __name__=='__main__':
    # load images
    train_imgs=loadImgs(train_path)
    test_imgs=loadImgs(test_path)

    # visual vocabulary
    vocabulary = None
    for i,c in enumerate(category):
        describes=get_all_features(train_imgs[i*100:i*100+100])
        vocabulary=np.vstack((vocabulary,describes)) if vocabulary is not None else describes
    print(f'vocabulary shape: {vocabulary.shape}')

    # k-means
    centers = vlfeat.kmeans.kmeans(vocabulary,vocabulary_size)

    # train & test histograms
    train_histograms=get_histograms(train_imgs,vocabulary_size,centers)
    test_histograms=get_histograms(test_imgs,vocabulary_size,centers)

    # knn
    indices=np.argsort(cdist(test_histograms,train_histograms,'euclidean'),axis=1)
    acc=knn(indices[:,:k])
    print(f'vocabulary_size:{vocabulary_size}')
    print(f'acc: {acc*100:.2f}%')
