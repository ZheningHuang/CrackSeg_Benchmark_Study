from glob import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os 

#path of all images. In each directory, the order of images need to be the same after sorted. 

gt_dir1="HEA12D_dataset/Asphalt_pavement_FD/Masks/*"

gt_dir2="HEA12D_dataset/Asphalt_pavement_VD/Masks/*"

gt_dir3="HEA12D_dataset/Concrete_pavement_FD/Masks/*"

gt_dir4="HEA12D_dataset/Concrete_pavement_VD/Masks/*"

gt_file1=sorted(glob(gt_dir1))

gt_file2=sorted(glob(gt_dir2))

gt_file3=sorted(glob(gt_dir3))

gt_file4=sorted(glob(gt_dir4))

save_dir=".\Mask_binarized"
for i in range (len(gt_file1)):
        
    gt1=cv2.imread(gt_file1[i],cv2.IMREAD_GRAYSCALE)
    (thresh_pred, gt1) = cv2.threshold(gt1, 128, 255, cv2.THRESH_BINARY| cv2.THRESH_OTSU)
    image_name_1 = gt_file1[i].split("\\")[-1]
    PATH_1=os.path.join(save_dir,image_name_1)
    cv2.imwrite(PATH_1,gt1)
    

for i in range (len(gt_file2)):
    gt2=cv2.imread(gt_file2[i],cv2.IMREAD_GRAYSCALE)
    (thresh_pred, gt2) = cv2.threshold(gt2, 128, 255, cv2.THRESH_BINARY| cv2.THRESH_OTSU)
    image_name_2 = gt_file2[i].split("\\")[-1]
    PATH_2=os.path.join(save_dir,image_name_2)
    cv2.imwrite(PATH_2,gt2)
for i in range (len(gt_file3)):

    gt3=cv2.imread(gt_file3[i],cv2.IMREAD_GRAYSCALE)
    (thresh_pred, gt3) = cv2.threshold(gt3, 128, 255, cv2.THRESH_BINARY| cv2.THRESH_OTSU)
    image_name_3 = gt_file3[i].split("\\")[-1]
    PATH_3=os.path.join(save_dir,image_name_3)
    cv2.imwrite(PATH_3,gt3)

for i in range (len(gt_file4)):
    gt4=cv2.imread(gt_file4[i],cv2.IMREAD_GRAYSCALE)
    (thresh_pred, gt4) = cv2.threshold(gt4, 128, 255, cv2.THRESH_BINARY| cv2.THRESH_OTSU)
    image_name_4 = gt_file4[i].split("\\")[-1]
    PATH_4=os.path.join(save_dir,image_name_4)
    cv2.imwrite(PATH_4,gt4)