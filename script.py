import numpy as np
import ults
import cv2


tru = np.loadtxt('tmp/reg_3dmm.txt')
shape = tru[0,:100]
exp = tru[0,100:179]
points = ults.CalculateDense(shape,exp)
ults.CreateObj(ults.Assemble(points)/10000,'fake')