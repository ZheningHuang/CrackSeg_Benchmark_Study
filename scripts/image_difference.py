from glob import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt

fig, ax = plt.subplots(2,4, figsize=(12,12))

#path of all images. In each directory, the order of images need to be the same after sorted. 
img_dir="HEA12D_dataset/Asphalt_pavement_VD/Raw_images/*"
gt_dir="HEA12D_dataset/Asphalt_pavement_VD/Masks/*"
result1_dir="Results/Asphalt_pavement_VD/model1/*"
result2_dir="Results/Asphalt_pavement_VD/model2/*"
result3_dir="Results/Asphalt_pavement_VD/model3/*"
result4_dir="Results/Asphalt_pavement_VD/model4/*"

img_file=sorted(glob(img_dir))
gt_file=sorted(glob(gt_dir))
result1_file=sorted(glob(result1_dir))
result2_file=sorted(glob(result2_dir))
result3_file=sorted(glob(result3_dir))
result4_file=sorted(glob(result4_dir))


def detailresult(result1,gt):
    #false postive
    difference_1_ = cv2.subtract(result1,gt)
    difference_1_3d = cv2.cvtColor(difference_1_,cv2.COLOR_GRAY2RGB)
    difference_1_3d[difference_1_ == 255] = [0, 255, 255]

    #false negative
    difference_1_reverse=cv2.subtract(gt,result1)
    difference_1_reverse_3d = cv2.cvtColor(difference_1_reverse,cv2.COLOR_GRAY2RGB)
    difference_1_reverse_3d[difference_1_reverse == 255] = [255, 0, 0]

    #Ture postive 
    tp_1=cv2.subtract(result1,difference_1_)
    tp_1_3d=cv2.cvtColor(tp_1,cv2.COLOR_GRAY2RGB)
    #tp_1_3d[tp_1 == 255] = [0, 255, 0]

    overall_1=difference_1_3d+tp_1_3d+difference_1_reverse_3d

    overall_1=[255,255,255]-overall_1
    return (overall_1)


i=6

img=cv2.imread(img_file[i])

gt=cv2.imread(gt_file[i],cv2.IMREAD_GRAYSCALE)
(thresh_pred, gt) = cv2.threshold(gt, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)\

result1=cv2.imread(result1_file[i],cv2.IMREAD_GRAYSCALE)
(thresh_pred, result1) = cv2.threshold(result1, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

result2=cv2.imread(result2_file[i],cv2.IMREAD_GRAYSCALE)
(thresh_pred, result2) = cv2.threshold(result2, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

result3=cv2.imread(result3_file[i],cv2.IMREAD_GRAYSCALE)
(thresh_pred, result3) = cv2.threshold(result3, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

result4=cv2.imread(result4_file[i],cv2.IMREAD_GRAYSCALE)
(thresh_pred, result4) = cv2.threshold(result4, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

img=cv2.resize(img,(result3.shape[1],result3.shape[0]))
gt=cv2.resize(gt,(result3.shape[1],result3.shape[0]))
result1=cv2.resize(result1,(result3.shape[1],result3.shape[0]))
result2=cv2.resize(result2,(result3.shape[1],result3.shape[0]))
result4=cv2.resize(result4,(result3.shape[1],result3.shape[0]))

result1=detailresult(result1,gt)
result2=detailresult(result2,gt)
result3=detailresult(result3,gt)
result4=detailresult(result4,gt)


i=0
ax[i, 0].imshow((255-gt),cmap=plt.get_cmap('gray'),vmin=0,vmax=255)
ax[i, 0].set_yticks([])
ax[i, 0].set_xticks([])
#ax[i, 1].axis('off')
ax[i, 1].imshow(result1,cmap=plt.get_cmap('gray'),vmin=0,vmax=255)
ax[i, 1].set_yticks([])
ax[i, 1].set_xticks([])
#ax[i, 2].axis('off')
ax[i, 2].imshow(result2,cmap=plt.get_cmap('gray'),vmin=0,vmax=255)
ax[i, 2].set_yticks([])
ax[i, 2].set_xticks([])
#ax[i, 3].axis('off')
ax[i, 3].imshow(result3,cmap=plt.get_cmap('gray'),vmin=0,vmax=255)
ax[i, 3].set_yticks([])
ax[i, 3].set_xticks([])
#ax[i, 4].axis('off')

i=11
img=cv2.imread(img_file[i])
gt=cv2.imread(gt_file[i],cv2.IMREAD_GRAYSCALE)
(thresh_pred, gt) = cv2.threshold(gt, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)\

result1=cv2.imread(result1_file[i],cv2.IMREAD_GRAYSCALE)
(thresh_pred, result1) = cv2.threshold(result1, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

result2=cv2.imread(result2_file[i],cv2.IMREAD_GRAYSCALE)
(thresh_pred, result2) = cv2.threshold(result2, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

result3=cv2.imread(result3_file[i],cv2.IMREAD_GRAYSCALE)
(thresh_pred, result3) = cv2.threshold(result3, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)


result4=cv2.imread(result4_file[i],cv2.IMREAD_GRAYSCALE)
(thresh_pred, result4) = cv2.threshold(result4, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

img=cv2.resize(img,(result3.shape[1],result3.shape[0]))
gt=cv2.resize(gt,(result3.shape[1],result3.shape[0]))
result1=cv2.resize(result1,(result3.shape[1],result3.shape[0]))
result2=cv2.resize(result2,(result3.shape[1],result3.shape[0]))
result4=cv2.resize(result4,(result3.shape[1],result3.shape[0]))

result1=detailresult(result1,gt)
result2=detailresult(result2,gt)
result3=detailresult(result3,gt)
result4=detailresult(result4,gt)


i=1
ax[i, 0].imshow((255-gt),cmap=plt.get_cmap('gray'),vmin=0,vmax=255)
ax[i, 0].set_yticks([])
ax[i, 0].set_xticks([])
#ax[i,1].axis('off')
ax[i, 1].imshow(result1,cmap=plt.get_cmap('gray'),vmin=0,vmax=255)
ax[i, 1].set_yticks([])
ax[i, 1].set_xticks([])
#ax[i, 2].axis('off')
ax[i, 2].imshow(result2,cmap=plt.get_cmap('gray'),vmin=0,vmax=255)
ax[i, 2].set_yticks([])
ax[i, 2].set_xticks([])
#ax[i, 3].axis('off')
ax[i, 3].imshow(result3,cmap=plt.get_cmap('gray'),vmin=0,vmax=255)
ax[i, 3].set_yticks([])
ax[i, 3].set_xticks([])
#ax[i, 4].axis('off')


cols = ['Groung Truth', 'DeepCrack', 'VGG16+UNet','ResNet34+UNet']
for axes, col in zip(ax[0], cols):
    axes.set_title(col)




fig.tight_layout()
plt.show()