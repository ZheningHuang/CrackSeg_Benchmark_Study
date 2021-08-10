from glob import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt



fig, ax = plt.subplots(2,6, figsize=(12,12))

#path of all images. In each directory, the order of images need to be the same after sorted. 
img_dir="HEA12D_dataset/Asphalt_pavement_FD/Raw_images/*"
gt_dir="HEA12D_dataset/Asphalt_pavement_FD/Masks/*"
result1_dir="Results/Asphalt_pavement_FD/model1/*"
result2_dir="Results/Asphalt_pavement_FD/model2/*"
result3_dir="Results/Asphalt_pavement_FD/model3/*"
result4_dir="Results/Asphalt_pavement_FD/model4/*"


img_file=sorted(glob(img_dir))
gt_file=sorted(glob(gt_dir))
result1_file=sorted(glob(result1_dir))
result2_file=sorted(glob(result2_dir))
result3_file=sorted(glob(result3_dir))
result4_file=sorted(glob(result4_dir))

i=0
img=cv2.imread(img_file[i])

gt=cv2.imread(gt_file[i],cv2.IMREAD_GRAYSCALE)
(thresh_pred, gt) = cv2.threshold(gt, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
gt=(255-gt)

result1=cv2.imread(result1_file[i],cv2.IMREAD_GRAYSCALE)
(thresh_pred, result1) = cv2.threshold(result1, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
result1=(255-result1)

result2=cv2.imread(result2_file[i],cv2.IMREAD_GRAYSCALE)
(thresh_pred, result2) = cv2.threshold(result2, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
result2=(255-result2)

result3=cv2.imread(result3_file[i],cv2.IMREAD_GRAYSCALE)
(thresh_pred, result3) = cv2.threshold(result3, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
result3=(255-result3)

result4=cv2.imread(result4_file[i],cv2.IMREAD_GRAYSCALE)
(thresh_pred, result4) = cv2.threshold(result4, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
result4=(255-result4)

img=cv2.resize(img,(result3.shape[1],result3.shape[0]))
gt=cv2.resize(gt,(result3.shape[1],result3.shape[0]))
result1=cv2.resize(result1,(result3.shape[1],result3.shape[0]))
result2=cv2.resize(result2,(result3.shape[1],result3.shape[0]))
result4=cv2.resize(result4,(result3.shape[1],result3.shape[0]))

ax[i, 0].imshow(img)
ax[i, 0].set_yticks([])
ax[i, 0].set_xticks([])
#ax[i, 0].axis('off')
ax[i, 1].imshow(gt,cmap=plt.get_cmap('gray'),vmin=0,vmax=255)
ax[i, 1].set_yticks([])
ax[i, 1].set_xticks([])
#ax[i, 1].axis('off')
ax[i, 2].imshow(result1,cmap=plt.get_cmap('gray'),vmin=0,vmax=255)
ax[i, 2].set_yticks([])
ax[i, 2].set_xticks([])
#ax[i, 2].axis('off')
ax[i, 3].imshow(result2,cmap=plt.get_cmap('gray'),vmin=0,vmax=255)
ax[i, 3].set_yticks([])
ax[i, 3].set_xticks([])
#ax[i, 3].axis('off')
ax[i, 4].imshow(result3,cmap=plt.get_cmap('gray'),vmin=0,vmax=255)
ax[i, 4].set_yticks([])
ax[i, 4].set_xticks([])
#ax[i, 4].axis('off')
ax[i, 5].imshow(result4,cmap=plt.get_cmap('gray'),vmin=0,vmax=255)
ax[i, 5].set_yticks([])
ax[i, 5].set_xticks([])
#ax[i, 5].axis('off')


fig.tight_layout()
plt.show() 

difference_1 = cv2.subtract(gt, result1)
difference_2 = cv2.subtract(gt, result2)
difference_3 = cv2.subtract(gt, result3)
difference_4 = cv2.subtract(gt, result4)

result_post_1=result1+difference_1*0.1
result_post_2=result1+difference_2*0.1
result_post_3=result1+difference_3*0.1
result_post_4=result1+difference_4*0.1

ax[2, 2].imshow(result_post_1)
ax[2, 2].set_yticks([])
ax[2, 2].set_xticks([])
#ax[i, 2].axis('off')
ax[2, 3].imshow(result_post_2)
ax[2, 3].set_yticks([])
ax[2, 3].set_xticks([])
#ax[i, 3].axis('off')
ax[2, 4].imshow(result_post_3)
ax[2, 4].set_yticks([])
ax[2, 4].set_xticks([])
#ax[i, 4].axis('off')
ax[2, 5].imshow(result_post_4)
ax[2, 5].set_yticks([])
ax[2, 5].set_xticks([])
#ax[i, 5].axis('off')

fig.tight_layout()
plt.show() 