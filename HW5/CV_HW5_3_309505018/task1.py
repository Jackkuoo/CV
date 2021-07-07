import os
import cv2
import numpy as np
from scipy.spatial.distance import cdist


train_path=os.path.join('hw5_data','train')
test_path=os.path.join('hw5_data','test')
class_names = os.listdir(train_path)

def get_feature(filepath):
    features = []
    for c in class_names:
        class_path = os.path.join(filepath, c)
        image_names = os.listdir(class_path)
        for image_name in image_names:
            if image_name.endswith('.jpg'):
                img_path = os.path.join(class_path, image_name)
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = cv2.resize(img, (16, 16))
                feature = np.reshape(img, (1, -1))
                # normalize
                feature = feature - feature.mean()
                feature /= np.linalg.norm(feature)
                features.append(feature[0])

    return np.asarray(features)


if __name__=='__main__':

    features = get_feature(train_path)
    test_features = get_feature(test_path)
    distances = cdist(test_features, features, 'euclidean')

    # knn
    acc = 0
    for i in range(len(test_features)):
        gt = i//10
        sorted_idx = np.argsort(distances[i])
        predict = sorted_idx[0]//100
        if gt == predict:
            acc += 1

    print(f'acc: {acc/len(test_features)*100:.2f}%')
