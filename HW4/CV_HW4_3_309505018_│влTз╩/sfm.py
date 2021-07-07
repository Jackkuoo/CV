import os
import numpy as np
import cv2
from mpl_toolkits.mplot3d import Axes3D
import matlab
import matplotlib.pyplot as plt

def img2keypointsandfeature(img):
    sift = cv2.xfeatures2d.SIFT_create()
    keypoints, features = sift.detectAndCompute(img, None)
    keypoints = np.float32([i.pt for i in keypoints])
    return keypoints, features

def valid_matching(keypoints1, features1, keypoints2, features2, ratio):
    ##### opencv feature matching #####
    # match_instance = cv2.DescriptorMatcher_create("BruteForce")
    # All_Matches = match_instance.knnMatch(features1, features2, 2)
    # valid_matches = []
    # for val in All_Matches:
    #     if len(val) == 2 and val[0].distance < val[1].distance * ratio:
    #         valid_matches.append((val[0].trainIdx, val[0].queryIdx))
    # print(valid_matches)
    raw_match = []
    match_dist = []
    # find the closest and the second closest features
    for i in range(features1.shape[0]):
        if np.linalg.norm(features1[i] - features2[0]) < np.linalg.norm(features1[i] - features2[1]):
            closest = np.linalg.norm(features1[i] - features2[0])
            second = np.linalg.norm(features1[i] - features2[1])
            c, s = 0, 1
        else:
            closest = np.linalg.norm(features1[i] - features2[1])
            second = np.linalg.norm(features1[i] - features2[0])
            c, s = 1, 0

        for j in range(2, features2.shape[0]):
            dist = np.linalg.norm(features1[i] - features2[j])
            if dist < second:
                if dist < closest:
                    second = closest
                    closest = dist
                    s = c
                    c = j
                else:
                    second = dist
                    s = j
        raw_match.append([c, s])
        match_dist.append([closest, second])

    valid_match = []
    valid_kp1 = []
    valid_kp2 = []
    for i, m in enumerate(raw_match):
        closest, second = match_dist[i]
        # to eliminate ambiguous matches
        if closest < ratio * second:
            valid_kp1.append(keypoints1[i])
            valid_kp2.append(keypoints2[m[0]])

    return np.asarray(valid_kp1),np.asarray(valid_kp2)

def feature_point_matching(img1,img2,ratio):
    """
    :returns:
        keypoints1: (N,2) ndarray
        keypoints2: (N,2) ndarray
    """
    keypoints1,features1=img2keypointsandfeature(img1)
    keypoints2,features2=img2keypointsandfeature(img2)
    return valid_matching(keypoints1,features1,keypoints2,features2,ratio)

def normalize_coordinate(points):
    """ Scale and translate image points so that centroid of the points
        are at the origin and avg distance to the origin is equal to sqrt(2).
    :param points: (3,8) ndarray
    """
    x = points[0]
    y = points[1]
    center = points.mean(axis=1)  # mean of each row
    cx = x - center[0] # center the points
    cy = y - center[1]
    dist = np.sqrt(np.power(cx, 2) + np.power(cy, 2))
    scale = np.sqrt(2) / dist.mean()
    T = np.array([
        [scale, 0, -scale * center[0]],
        [0, scale, -scale * center[1]],
        [0,     0,                  1]
        ])
    return T, T@points

def compute_fundamental_matrix(x,x_):
    """
    :param x: (3,8) ndarray
    :param x_: (3,8) ndarray
    """
    #Each row in the A is [x'*x, x'*y, x', y'*x, y'*y, y', x, y, 1]
    A = np.zeros((8,9))
    for i in range(8):
        A[i]=[ x_[0, i]*x[0, i], x_[0, i]*x[1, i], x_[0, i], x_[1, i]*x[0, i], x_[1, i]*x[1, i], x_[1, i], x[0, i], x[1, i], 1 ]
    
    # A@f=0
    U, S, V = np.linalg.svd(A)
    F = V[-1].reshape(3, 3)

    # det(F)=0 constrain
    U, S, V = np.linalg.svd(F)
    S[-1] = 0
    F = U @ np.diag(S) @ V
    return F

def compute_fundamental_matrix_normalized(p1,p2):
    """
    :param p1: (8,3) ndarray
    :param p2: (8,3) ndarray
    """
    # preprocess image coordinates
    T1,p1_normalized = normalize_coordinate(p1.T)
    T2,p2_normalized = normalize_coordinate(p2.T)

    F = compute_fundamental_matrix(p1_normalized,p2_normalized)

    F = T2.T @ F @ T1
    return F/F[-1,-1]

def get_fundamental_matrix(keypoints1,keypoints2,threshold):
    """
    :param keypoints1: (N,2) ndarray
    :param keypoints2: (N,2) ndarray
    :param threshold: |x'Fx| < threshold as inliers
    """
    rs = np.random.RandomState(seed = 0)
    N=len(keypoints1)
    keypoints1=np.hstack((keypoints1,np.ones((N,1))))
    keypoints2=np.hstack((keypoints2,np.ones((N,1))))

    best_cost=1e9
    best_F=None
    best_inlier_idxs=None
    # find best F with RANSAC
    for _ in range(2000):
        choose_idx=rs.choice(N, 8, replace=False)  # sample 8 correspondence feature points
        # get F
        F=compute_fundamental_matrix_normalized(keypoints1[choose_idx,:],keypoints2[choose_idx,:])

        # select indices with accepted points, Sampson distance as error.
        Fx1=(keypoints1@F).T
        Fx2=(keypoints2@F).T
        denom = Fx1[0] ** 2 + Fx1[1] ** 2 + Fx2[0] ** 2 + Fx2[1] ** 2
        errors = np.diag(keypoints2 @ F @ keypoints1.T) ** 2 / denom
        inlier_idxs=np.where(errors<threshold)[0]

        cost = np.sum(errors[errors<threshold]) + (N-len(inlier_idxs))*threshold
        if cost < best_cost:
            best_cost=cost
            best_F=F
            best_inlier_idxs=inlier_idxs

    best_F = best_F.T

    return best_F, best_inlier_idxs

def norm_line(lines):
    a = lines[0,:]
    b = lines[1,:]
    length = np.sqrt(a**2 + b**2)
    return lines / length

def drawlines(img1, img2, lines, pts1, pts2):
    '''
    :param img1: image on which we draw the epilines for the points in img2
    :param lines: corresponding epilines
    '''
    r,c = img1.shape
    img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2RGB)
    img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2RGB)
    for r, pt1, pt2 in zip(lines, pts1, pts2):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[2]/r[1] ])
        x1, y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv2.line(img1, (x0,y0), (x1,y1), color, 1)
        img1 = cv2.circle(img1, tuple(pt1), 5, color, -1)
        img2 = cv2.circle(img2, tuple(pt2), 5, color, -1)
    return img1, img2

def draw_epilines(gray1, gray2, inlier1, inlier2, F, filename):
    """
    :param inlier1: (N,2) ndarray
    :param inlier2: (N,2) ndarray
    """
    lines1_unnorm= F @ np.hstack((inlier2,np.ones((inlier2.shape[0],1)))).T
    lines1 = norm_line(lines1_unnorm)
    img1, img2 = drawlines(gray1, gray2, lines1.T, inlier1.astype(np.int), inlier2.astype(np.int))

    lines2_unnorm = F.T @ np.hstack((inlier1,np.ones((inlier1.shape[0],1)))).T
    lines2 = norm_line(lines2_unnorm)
    img3, img4 = drawlines(gray2, gray1, lines2.T, inlier2.astype(np.int), inlier1.astype(np.int))

    plt.subplot(221), plt.imshow(img1)
    plt.subplot(222), plt.imshow(img2)
    plt.subplot(223), plt.imshow(img4)
    plt.subplot(224), plt.imshow(img3)
    plt.savefig(filename)

def plot(final3d):
    fig = plt.figure()
    ax = Axes3D(fig)
    for i in range(len(final3d)):
        ax.scatter(final3d[i,0], final3d[i,1], final3d[i,2])
    ax.set_xlabel('x axis')
    ax.set_ylabel('y axis')
    ax.set_zlabel('z axis')
    ax.view_init(elev = 135, azim = 90)
    plt.show()

def compute_P_from_essential(E):
    """ Compute the second camera matrix (assuming P1 = [I 0])
        from an essential matrix. E = [t]R
    :returns: list of 4 possible camera matrices.
    """
    U, S, V = np.linalg.svd(E)

    # Ensure rotation matrix are right-handed with positive determinant
    if np.linalg.det(np.dot(U, V)) < 0:
        V = -V

    # create 4 possible camera matrices
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    P2s = [np.vstack(((U @ W @ V).T, U[:, 2])).T,
           np.vstack(((U @ W @ V).T,-U[:, 2])).T,
           np.vstack(((U @ W.T @ V).T, U[:, 2])).T,
           np.vstack(((U @ W.T @ V).T,-U[:, 2])).T]

    return P2s

def skew_sym_mat(x):
    return np.array([
        [0, -x[2], x[1]],
        [x[2], 0, -x[0]],
        [-x[1], x[0], 0]
    ])

def best_P2(P1,P2s,pt1,pt2):
    """
    :param P1: Extrinsic matrix from camera1
    :param P2s: Extrinsic matrixs from camera2
    :param pt1: (3,N) ndarray
    :param pt2: (3,N) ndarray
    :return: [(N,3) ndarray, (N,3) ndarray, (N,3) ndarray, (N,3) ndarray]
    """
    index = -1
    for i, P2 in enumerate(P2s):
        # (pt1 x P1) * X = 0
        # (pt2 x P2) * X = 0
        A = np.vstack((skew_sym_mat(pt1[:, 0]) @ P1,
                       skew_sym_mat(pt2[:, 0]) @ P2))
        U, S, V = np.linalg.svd(A)
        P = np.ravel(V[-1, :4])
        v1 = P / P[3]  # X solution

        P2_h = np.linalg.inv(np.vstack([P2, [0, 0, 0, 1]]))
        v2 = np.dot(P2_h[:3, :4], v1)

        if v1[2] > 0 and v2[2] > 0:
            index = i

    return P2s[index]

def choose_best_threeD(h1,h2,P1,P2):
    P2 = np.linalg.inv(np.vstack([P2, [0, 0, 0, 1]]))[:3, :4]

    n_point = h1.shape[1]
    res = np.ones((n_point, 3))

    for i in range(n_point):
        A = np.asarray([
            (h1[0, i] * P1[2, :] - P1[0, :]),
            (h1[1, i] * P1[2, :] - P1[1, :]),
            (h2[0, i] * P2[2, :] - P2[0, :]),
            (h2[1, i] * P2[2, :] - P2[1, :])
        ])

        U,S,V = np.linalg.svd(A)
        res[i, :] = V[-1,:-1]/ V[-1,-1]

    return res

def ndarray2matlab(x):
    return matlab.double(x.tolist())

def viz_3d_matplotlib(pt_3d):
    X = pt_3d[0,:]
    Y = pt_3d[1,:]
    Z = pt_3d[2,:]

    fig = plt.figure(1)
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(X,Y,Z,s=1,cmap='gray')
    
    plt.show()

if __name__=='__main__':
    img1_path=os.path.join('Statue1.bmp') #Mesona1.JPG  Statue1.bmp
    img2_path=os.path.join('Statue2.bmp') #Mesona2.JPG  Statue2.bmp
    ratio=0.7
    threshold=0.01
    img1=cv2.imread(img1_path)
    img2=cv2.imread(img2_path)
    img1=cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
    img2=cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)

    K1=np.array([[5426.566895, 0.678017, 330.096680],
                [0.000000, 5423.133301, 648.950012],
                [0.000000,    0.000000,   1.000000]])
    K2=np.array([[5426.566895, 0.678017, 387.430023],
                [0.000000, 5423.133301, 620.616699],
                [0.000000,    0.000000,   1.000000]])

    # 1. get the correspondence across images
    keys1,keys2 = feature_point_matching(img1,img2,ratio=ratio)

    # 2. get the Fundamental matrix by correspondence
    F,inlier_idxs = get_fundamental_matrix(keys1,keys2,threshold=threshold)
    inlier1 = keys1[inlier_idxs]
    inlier2 = keys2[inlier_idxs]

    # 3. draw epipolar lines
    print(f'# correspondence: {len(keys1)}')
    print(f'# inliers: {len(inlier_idxs)}')
    draw_epilines(img1, img2, inlier1, inlier2, F, 'epilines.png')

    # 4. four possible P2
    E = K1.T @ F @ K2
    print('F:')
    print(F)
    print('E:')
    print(E)
    P1 = np.hstack((np.eye(3),np.zeros((3,1)))) # first camera matrix
    P2s = compute_P_from_essential(E)  # second camera matrix
    
    # # 5. four possible 3D points from P1 & P2
    i1T = inlier1.T
    i2T = inlier2.T
    pt1 = np.linalg.inv(K1) @ np.asarray(np.vstack([i1T, np.ones(i1T.shape[1])]))
    pt2 = np.linalg.inv(K2) @ np.asarray(np.vstack([i2T, np.ones(i2T.shape[1])]))
    P2 = best_P2(P1,P2s,pt1,pt2)

    # # 6. find the most appropriate
    threeD=choose_best_threeD(pt1,pt2,P1,P2)
    plot(threeD)


    threeD = np.array(threeD)
    # viz_3d(threeD)
    viz_3d_matplotlib(threeD)
    print("ndarray2matlab(threeD)",ndarray2matlab(threeD))
    print("ndarray2matlab(K1@P1):",ndarray2matlab(K1@P1))
    print("img1_path:",img1_path)
    # 7. call matlab
    # eng = matlab.engine.start_matlab()
    # eng.obj_main(ndarray2matlab(threeD), ndarray2matlab(threeD), ndarray2matlab(K1@P1), img1_path, 1, nargout=0)
    # eng.quit()