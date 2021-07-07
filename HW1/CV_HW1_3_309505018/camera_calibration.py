#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import camera_calibration_show_extrinsics as show
from PIL import Image
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
# (8,6) is for the given testing images.
# If you use the another data (e.g. pictures you take by your smartphone), 
# you need to set the corresponding numbers.
corner_x = 7
corner_y = 7
objp = np.zeros((corner_x*corner_y,3), np.float32)
objp[:,:2] = np.mgrid[0:corner_x, 0:corner_y].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d points in real world space
imgpoints = [] # 2d points in image plane.

# Make a list of calibration images
images = glob.glob('data/*.jpg')

# Step through the list and search for chessboard corners
print('Start finding chessboard corners...')
for idx, fname in enumerate(images):
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    plt.imshow(gray)

    #Find the chessboard corners
    print('find the chessboard corners of',fname)
    ret, corners = cv2.findChessboardCorners(gray, (corner_x,corner_y), None)

    # If found, add object points, image points
    if ret == True:
        objpoints.append(objp)
        imgpoints.append(corners)

        # Draw and display the corners
        cv2.drawChessboardCorners(img, (corner_x,corner_y), corners, ret)
        plt.imshow(img)

print('Camera calibration...')
img_size = (img.shape[1], img.shape[0])


# In[86]:


#calculate H of each picture by Pm=0
Homography_list=[]
for i in range(len(images)):
    P = np.zeros((corner_x*corner_y*2,9),np.float32)
    for j in range(corner_x*corner_y):
        u = j*2
        P_point = objpoints[i][j]+np.array([0,0,1])
        P[u,0:3] = P_point 
        P[u+1,3:6] = P_point 
        P[u,6:]= P_point * (-imgpoints[i][j,0,0])
        P[u+1,6:]= P_point *  (-imgpoints[i][j,0,1])
        
    _,_,V=np.linalg.svd(P)
    Homography = V[-1,:] / V[-1,-1]
    Homography = Homography.reshape((3,3))
    Homography_list.append(Homography)
    
#print(Homography_list)


# In[95]:


#calculate B by Vb=0 
#b = argmin Vb
V=np.zeros((len(images)*2,6),np.float32)
for pic in range(len(images)):
    Homography=Homography_list[pic]
    V[pic*2,:]=np.array([Homography[0,0] * Homography[0,1]                                ,
                         Homography[0,0] * Homography[1,1] + Homography[0,1] * Homography[1,0],
                         Homography[0,0] * Homography[2,1] + Homography[2,0] * Homography[0,1],
                         Homography[1,0] * Homography[1,1]                                    ,
                         Homography[1,0] * Homography[2,1] + Homography[2,0] * Homography[1,1],
                         Homography[2,0] * Homography[2,1]]                                   ,
                         np.float32)
    V[pic*2+1,:]=np.array([Homography[0,0]**2 - Homography[0,1]**2                                ,
                           2*(Homography[0,0] * Homography[1,0] - Homography[0,1]*Homography[1,1]),
                           2*(Homography[0,0] * Homography[2,0] - Homography[0,1]*Homography[2,1]),
                           Homography[1,0]**2 - Homography[1,1]**2                                ,
                           2*(Homography[1,0] * Homography[2,0] - Homography[1,1]*Homography[2,1]),
                           Homography[2,0]**2 - Homography[2,1]**2]                               ,
                           np.float32)
_,_,V=np.linalg.svd(V)
B=V[-1,:]
B=np.array([[B[0],B[1],B[2]],
            [B[1],B[3],B[4]],
            [B[2],B[4],B[5]]],np.float32)

# change B to positive definite 
if B[0,0]<0:
    B=-B
#print(B)


# In[96]:


#calculate K from B by cholesky 
K_inv=np.linalg.cholesky(B).T
K=np.linalg.inv(K_inv)
K/=K[-1,-1]
print(K)


# In[107]:


#calculate [R t] from K and H
extrinsics=np.zeros((len(images),6))
for i in range(len(Homography_list)):
    Homography = Homography_list[i]
    _lambda = 1 / np.linalg.norm(K_inv @ Homography[:,0])
    ex = _lambda * K_inv @ Homography
    r1 = ex[:,0:1]
    r2 = ex[:,1:2]
    r3 = np.cross(r1.T,r2.T).T

    R = np.hstack((r1, r2, r3))
    t = ex[:,2:3]
    rot_vec,_ = cv2.Rodrigues(R)
    extrinsics[i,:] = np.concatenate((rot_vec,t)).reshape(-1)

#print(extrinsics)


# In[108]:


# show the camera extrinsics
print('Show the camera extrinsics')
# plot setting
# You can modify it for better visualization
fig = plt.figure(figsize=(10, 10))
ax = fig.gca(projection='3d')
# camera setting
camera_matrix = K
cam_width = 0.064/0.1
cam_height = 0.032/0.1
scale_focal = 1600
# chess board setting
board_width = 7
board_height = 7
square_size = 1
# display
# True -> fix board, moving cameras
# False -> fix camera, moving boards
min_values, max_values = show.draw_camera_boards(ax, camera_matrix, cam_width, cam_height,
                                                scale_focal, extrinsics, board_width,
                                                board_height, square_size, True)

X_min = min_values[0]
X_max = max_values[0]
Y_min = min_values[1]
Y_max = max_values[1]
Z_min = min_values[2]
Z_max = max_values[2]
max_range = np.array([X_max-X_min, Y_max-Y_min, Z_max-Z_min]).max() / 2.0

mid_x = (X_max+X_min) * 0.5
mid_y = (Y_max+Y_min) * 0.5
mid_z = (Z_max+Z_min) * 0.5
ax.set_xlim(mid_x - max_range, mid_x + max_range)
ax.set_ylim(mid_y - max_range, 0)
ax.set_zlim(mid_z - max_range, mid_z + max_range)

ax.set_xlabel('x')
ax.set_ylabel('z')
ax.set_zlabel('-y')
ax.set_title('Extrinsic Parameters Visualization')
#plt.savefig('opencv_ourdata_Extrinsic Parameters Visualization.png')
plt.show()

#animation for rotating plot
"""
for angle in range(0, 360):
    ax.view_init(30, angle)
    plt.draw()
    plt.pause(.001)
"""


# In[ ]:





# In[ ]:





# In[ ]:




