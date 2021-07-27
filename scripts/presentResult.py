from glob import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt

#path of all images. In each directory, the order of images need to be the same after sorted. 
img_dir="Datasets/Asphalt_Pavement_Images/Ashplat_Row_Images/*"
gt_dir="Datasets/Asphalt_Pavement_Images/Segmentation_Mask/*"
result1_dir="Datasets/DetectionResults/Model_1/*"
result2_dir="Datasets/DetectionResults/Model_2/*"
result3_dir="Datasets/DetectionResults/Model_3/*"
#result4_dir=""

#load images
img_file=sorted(glob(img_dir))
gt_file=sorted(glob(gt_dir))
result1_file=sorted(glob(result1_dir))
result2_file=sorted(glob(result2_dir))
result3_file=sorted(glob(result3_dir))
#result4_file=sorted(glob(result4_dir))

totalNumOfimage=len(img_file)

column_number=4

row_number=5
fig, ax = plt.subplots(column_number,row_number, figsize=(20,12))
cols = ['Raw Image', 'Groung Truth', 'DeepCrack', 'VGG16+UNet','ResNet+UNet']
for axes, col in zip(ax[0], cols):
    axes.set_title(col,size="large")

for i in range (column_number):
    img=cv2.imread(img_file[i])
    gt=cv2.imread(gt_file[i])
    result1=cv2.imread(result1_file[i])
    result2=cv2.imread(result2_file[i])
    result3=cv2.imread(result3_file[i])
    ax[i, 0].imshow(img)
    ax[i, 0].axis('off')
    ax[i, 1].imshow(gt)
    ax[i, 1].axis('off')
    ax[i, 2].imshow(result1)
    ax[i, 2].axis('off')
    ax[i, 3].imshow(result2)
    ax[i, 3].axis('off')
    ax[i, 4].imshow(result3)
    ax[i, 4].axis('off')


fig.tight_layout()
#plt.show()

fig.savefig('detection result comparison.png')

