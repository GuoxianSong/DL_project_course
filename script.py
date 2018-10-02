import numpy as np
import ults
import cv2

# shape_,exp_,eular_,translate_,scale_=ults.Get3Dmm("Sample/afw_134212_1_aug_17.txt")
# landmark = np.loadtxt("Sample/afw_134212_1_aug_17_landmark.txt")
# image= cv2.imread("Sample/afw_134212_1_aug_17.jpg")
tmp_ = np.loadtxt("tmp/gx.txt")
for i in range(len(tmp_)):
    tmp =tmp_[i,:]
    points = ults.CalculateDense(tmp[0:100],tmp[100:179])
    R = ults.eulerAnglesToRotationMatrix(tmp[179:182])
    points =np.matmul(points,R)
    ults.CreateObj(points/10000,"gx/"+str(i))


net_img_size = 224
def Normalize2(image, landmark_, translation, scale):
    xmin = np.min(landmark_[:, 0])
    xmax = np.max(landmark_[:, 0])
    ymin = np.min(landmark_[:, 1])
    ymax = np.max(landmark_[:, 1])
    old_cx = (xmin + xmax) / 2
    old_cy = (ymin + ymax) / 2;
    cx = (net_img_size - 1) / 2.0
    cy = (net_img_size - 1) * 2.0 / 5.0;
    length = ((xmax - xmin) ** 2 + (ymax - ymin) ** 2) ** 0.5
    length *= 1.2
    ori_crop_scale = net_img_size / length
    new_scale = scale * ori_crop_scale
    image = cv2.resize(image, (0, 0), fx=ori_crop_scale, fy=ori_crop_scale)
    old_cx = old_cx * ori_crop_scale
    old_cy = old_cy * ori_crop_scale

    start_x = int(old_cx - cx)
    start_y = int(old_cy - cy)
    crop_image = image[start_y:start_y + 224, start_x:start_x + 224]
    shape_ = np.shape(crop_image)
    print(shape_)
    tmp = np.zeros((224,224,3),dtype=np.uint8)
    tmp[:shape_[0],:shape_[1],:] = crop_image
    translation = translation * ori_crop_scale
    translation[0] = translation[0] - start_x
    translation[1] = translation[1] - (len(image) - 224-start_y)

    # landmark_=landmark_*ori_crop_scale
    # tmp = np.zeros((224,224),dtype=np.uint8)
    # for i in range(68):
    #     tmp[ int(landmark_[i,1] - start_y),int(landmark_[i,0] - start_x)  ] = 255;
    # cv2.imwrite("landmarl.jpg",tmp)

    return tmp, translation, new_scale

# crop_image, translation, new_scale=Normalize2(image,landmark,translate_,scale_)
# cv2.imwrite("tmp.jpg",crop_image)

# print()
#
# lmk = ults.Landmark(points*new_scale)[:,:2] +np.array([translation[0],translation[1]])
# tmp = np.zeros((224,224),dtype=np.uint8)
# lmk = np.array(lmk,dtype=np.int)
# for i in range(len(lmk)):
#     tmp[224-lmk[i,1],lmk[i,0]]=255
# cv2.imwrite("tmp_.jpg",tmp)