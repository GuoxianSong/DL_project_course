import numpy as np
import ults
import cv2

shape_,exp_,eular_,translate_,scale_=ults.Get3Dmm("Sample/afw_134212_1_aug_17.txt")

points = ults.CalculateDense(shape_,exp_) *scale_
R = ults.eulerAnglesToRotationMatrix(eular_)
points =np.matmul(points,R)
# ults.CreateObj(ults.Assemble(points)/10000,'fake')
lmk = ults.Landmark(points)[:,:2] +np.array([translate_[0]  ,translate_[1] ]  )

tmp = np.zeros((450,450),dtype=np.uint8)
lmk = np.array(lmk,dtype=np.int)

for i in range(len(lmk)):
    tmp[450-lmk[i,1],lmk[i,0]]=255
cv2.imwrite("tmp.jpg",tmp)


