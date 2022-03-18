"""
https://stackoverflow.com/questions/35942095/opencv-strange-rotation-and-translation-matrices-from-decomposehomographymat

"""

import cv2
import numpy as np


# set up a virtual camera
f = 1
w = 640
h = 480

K = np.array([[f, 0, w/2],
              [0, f, h/2],
              [0, 0,   1]])
dist_coffs = np.zeros(5)
# set transformation from 1st to 2nd camera (assume K is unchanged)
#rvecDeg = np.array([[45, 12, 66]]).T
#rvecDeg = np.array([[45, 0, 0]]).T
rvecDeg = np.array([[0, 0, 0]]).T  # Vectores de Rotación
#t = np.array([[100.0, 200, 300]]).T
t = np.array([[0.0, 0.0, 0.0]]).T

print("-------------------------------------------\n")
print("Ground truth:\n")

print("K = \n" + str(K))
print("rvec = \n" + str(rvecDeg))
print("t = \n" + str(t))

# set up points on a plane
# Cuadro de L=100 con esquina inferior izquierda centrada en el origen
p3d = np.array([[0, 0, 1],
                [100, 0, 1],
                [0, 100, 1],
                [100, 100, 1]], dtype=np.float64)

# Cuadro de L=100 centrado en el origen
p3d = np.array([[-50, -50, 1],
                [50, -50, 1],
                [50, 50, 1],
                [-50, 50, 1]], dtype=np.float64)


# project on both cameras

Q, _ = cv2.projectPoints(p3d,
                         np.zeros((3, 1)),
                         np.zeros((3, 1)),
                         K,
                         dist_coffs)
# Puntos Q
print ("Puntos Q---------")
print (Q)

P, _ = cv2.projectPoints(p3d,
                         rvecDeg*np.pi/180,
                         t,
                         K,
                         dist_coffs)

print ("Puntos P---------")
print (P)


# find homography
H, _ = cv2.findHomography(Q, P)

print("-------------------------------------------\n")
print("Estimated H = \n" + str(H))


# check by reprojection
P_ = cv2.perspectiveTransform(Q, H)

sumError = 0

for i in range(P.shape[0]):
    sumError += np.linalg.norm(P[i] - P_[i])


print("-------------------------------------------\n")
print("Average reprojection error = "+str(sumError/P.shape[0]))


# decompose using identity as internal parameters matrix
num_res, Rs, ts, n = cv2.decomposeHomographyMat(H, K)

print("-------------------------------------------\n")
print("Estimated decomposition:\n\n")
for i, Rt in enumerate(zip(Rs, ts)):
    R, t = Rt
    print("option " + str(i+1))
    print("rvec = ")
    rvec, _ = cv2.Rodrigues(R)
    print(rvec*180/np.pi)
    print("t = ")
    print(t)
    
img = np.ones((h,w,3))
img2 = np.ones((h,w,3))        
img = cv2.line(img,(int(w/2),0),(int(w/2),h),(0,0,255),1)
img = cv2.line(img,(0,int(h/2)),(w,int(h/2)),(0,255,0),1)

img2 = cv2.line(img2,(int(w/2),0),(int(w/2),h),(0,0,255),1)
img2 = cv2.line(img2,(0,int(h/2)),(w,int(h/2)),(0,255,0),1)

center_coordinates = (int(Q[0][0][0]),int(Q[0][0][1]))
img = cv2.circle(img, center_coordinates, 5, (0,0,255), -1)

center_coordinates = (int(P[0][0][0]),int(P[0][0][1]))
img2 = cv2.circle(img2, center_coordinates, 5, (0,0,255), -1)

center_coordinates = (int(Q[1][0][0]),int(Q[1][0][1]))
img = cv2.circle(img, center_coordinates, 5, (0,255,0), -1)

center_coordinates = (int(P[1][0][0]),int(P[1][0][1]))
img2 = cv2.circle(img2, center_coordinates, 5, (0,255,0), -1)

center_coordinates = (int(Q[2][0][0]),int(Q[2][0][1]))
img = cv2.circle(img, center_coordinates, 5, (255,0,0), -1)

center_coordinates = (int(P[2][0][0]),int(P[2][0][1]))
img2 = cv2.circle(img2, center_coordinates, 5, (255,0,0), -1)

center_coordinates = (int(Q[3][0][0]),int(Q[3][0][1]))
img = cv2.circle(img, center_coordinates, 5, (0,255,2550), -1)

center_coordinates = (int(P[3][0][0]),int(P[3][0][1]))
img2 = cv2.circle(img2, center_coordinates, 5, (0,255,2550), -1)

#print (Q[3][0][0],Q[3][0][1])


# Dibujar los EJES
cv2.imshow("puto1",img)
cv2.imshow("puto2",img2)
cv2.waitKey()
