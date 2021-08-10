from glob import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt

#path of all images. In each directory, the order of images need to be the same after sorted. 
img_dir1="HEA12D_dataset/Asphalt_pavement_FD/Raw_images/*"
gt_dir1="HEA12D_dataset/Asphalt_pavement_FD/Masks/*"
img_dir2="HEA12D_dataset/Asphalt_pavement_VD/Raw_images/*"
gt_dir2="HEA12D_dataset/Asphalt_pavement_VD/Masks/*"
img_dir3="HEA12D_dataset/Concrete_pavement_FD/Raw_images/*"
gt_dir3="HEA12D_dataset/Concrete_pavement_FD/Masks/*"
img_dir4="HEA12D_dataset/Concrete_pavement_VD/Raw_images/*"
gt_dir4="HEA12D_dataset/Concrete_pavement_VD/Masks/*"


img_file1=sorted(glob(img_dir1))
gt_file1=sorted(glob(gt_dir1))

img_file2=sorted(glob(img_dir2))
gt_file2=sorted(glob(gt_dir2))

img_file3=sorted(glob(img_dir3))
gt_file3=sorted(glob(gt_dir3))

img_file4=sorted(glob(img_dir4))
gt_file4=sorted(glob(gt_dir4))

fig, ax = plt.subplots(2,4, figsize=(12,12))

img1=cv2.imread(img_file1[0])
gt1=cv2.imread(gt_file1[0],cv2.IMREAD_GRAYSCALE)
(thresh_pred, gt1) = cv2.threshold(gt1, 128, 255, cv2.THRESH_BINARY| cv2.THRESH_OTSU)
gt1=255-gt1
ax[0, 0].imshow(img1)
ax[0, 0].set_yticks([])
ax[0, 0].set_xticks([])
ax[0, 0].title.set_text('Asphalt,Forwarding')

ax[0, 1].imshow(gt1,cmap=plt.get_cmap('gray'),vmin=0,vmax=255)
ax[0, 1].set_yticks([])
ax[0, 1].set_xticks([])
ax[0, 1].title.set_text('Asphalt,Forwarding (mask)')
####
img2=cv2.imread(img_file2[0])
gt2=cv2.imread(gt_file2[0],cv2.IMREAD_GRAYSCALE)
(thresh_pred, gt2) = cv2.threshold(gt2, 128, 255, cv2.THRESH_BINARY| cv2.THRESH_OTSU)
gt2=255-gt2
ax[0, 2].imshow(img2)
ax[0, 2].set_yticks([])
ax[0, 2].set_xticks([])
ax[0, 2].title.set_text('Asphalt,Vertical')
ax[0, 3].imshow(gt2,cmap=plt.get_cmap('gray'),vmin=0,vmax=255)
ax[0, 3].set_yticks([])
ax[0, 3].set_xticks([])
ax[0, 3].title.set_text('Asphalt,Vertical (mask)')
####
img3=cv2.imread(img_file3[0])
gt3=cv2.imread(gt_file3[0],cv2.IMREAD_GRAYSCALE)
(thresh_pred, gt3) = cv2.threshold(gt3, 128, 255, cv2.THRESH_BINARY| cv2.THRESH_OTSU)
gt3=255-gt3
ax[1, 0].imshow(img3)
ax[1, 0].set_yticks([])
ax[1, 0].set_xticks([])
ax[1, 0].title.set_text('Concrete,Forwarding')
ax[1, 1].imshow(gt3,cmap=plt.get_cmap('gray'),vmin=0,vmax=255)
ax[1, 1].set_yticks([])
ax[1, 1].set_xticks([])
ax[1, 1].title.set_text('Concrete,Forwarding (mask)')
####
img4=cv2.imread(img_file4[0])
gt4=cv2.imread(gt_file4[0],cv2.IMREAD_GRAYSCALE)
gt4=255-gt4
(thresh_pred, gt4) = cv2.threshold(gt4, 128, 255, cv2.THRESH_BINARY| cv2.THRESH_OTSU)
ax[1, 2].imshow(img4)
ax[1, 2].set_yticks([])
ax[1, 2].set_xticks([])
ax[1, 2].title.set_text('Concrete,Vertical')
ax[1, 3].imshow(gt4,cmap=plt.get_cmap('gray'),vmin=0,vmax=255)
ax[1, 3].set_yticks([])
ax[1, 3].set_xticks([])
ax[1, 3].title.set_text('Concrete,Vertical (mask)')

fig.tight_layout()
plt.show() 



