from glob import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt

#path of all images. In each directory, the order of images need to be the same after sorted. 
img_dir="HEA12D_dataset/Asphalt_pavement_FD/Raw_images/*"
gt_dir="HEA12D_dataset/Asphalt_pavement_FD/Masks/*"
result1_dir="Results/Asphalt_pavement_FD/model1/*"
result2_dir="Results/Asphalt_pavement_FD/model2/*"
result3_dir="Results/Asphalt_pavement_FD/model3/*"
result4_dir="Results/Asphalt_pavement_FD/model4/*"

#load images

def plotresults(img_dir,gt_dir,result1_dir,result2_dir,result3_dir,result4_dir,name):

    img_file=sorted(glob(img_dir))
    gt_file=sorted(glob(gt_dir))
    result1_file=sorted(glob(result1_dir))
    result2_file=sorted(glob(result2_dir))
    result3_file=sorted(glob(result3_dir))
    result4_file=sorted(glob(result4_dir))
    totalNumOfimage=len(img_file)
    column_number=6
    row_number=4

    fig, ax = plt.subplots(column_number,row_number, figsize=(12,12))
    cols = ['Raw Image', 'Groung Truth', 'DeepCrack', 'VGG16+UNet','ResNet34+UNet','Localized thresholding']
    for axes, col in zip(ax[:,0], cols):
        axes.set_ylabel(col,size="large")


    for i in range (row_number):

        img=cv2.imread(img_file[i+5])
        
        gt=cv2.imread(gt_file[i+5],cv2.IMREAD_GRAYSCALE)
        #gt=(255-gt)
        result1=cv2.imread(result1_file[i+5],cv2.IMREAD_GRAYSCALE)
        (thresh_pred, result1) = cv2.threshold(result1, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        #result1=(255-result1)
        
        result2=cv2.imread(result2_file[i+5],cv2.IMREAD_GRAYSCALE)
        (thresh_pred, result2) = cv2.threshold(result2, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        #result2=(255-result2)
        
        result3=cv2.imread(result3_file[i+5],cv2.IMREAD_GRAYSCALE)
        (thresh_pred, result3) = cv2.threshold(result3, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        #result3=(255-result3)
        
        result4=cv2.imread(result4_file[i+5],cv2.IMREAD_GRAYSCALE)
        (thresh_pred, result4) = cv2.threshold(result4, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        #result4=(255-result4)

        img=cv2.resize(img,(result3.shape[1],result3.shape[0]))
        gt=cv2.resize(gt,(result3.shape[1],result3.shape[0]))
        result1=cv2.resize(result1,(result3.shape[1],result3.shape[0]))
        result2=cv2.resize(result2,(result3.shape[1],result3.shape[0]))
        result4=cv2.resize(result4,(result3.shape[1],result3.shape[0]))

        ax[0, i].imshow(img)
        ax[0, i].set_yticks([])
        ax[0, i].set_xticks([])
        #ax[0, i].axis('off')
        ax[1, i].imshow(gt)
        ax[1, i].set_yticks([])
        ax[1, i].set_xticks([])
        #ax[1, i].axis('off')
        ax[2, i].imshow(result1)
        ax[2, i].set_yticks([])
        ax[2, i].set_xticks([])
        #ax[2, i].axis('off')
        ax[3, i].imshow(result2)
        ax[3, i].set_yticks([])
        ax[3, i].set_xticks([])
        #ax[3, i].axis('off')
        ax[4, i].imshow(result3)
        ax[4, i].set_yticks([])
        ax[4, i].set_xticks([])
        #ax[4, i].axis('off')
        ax[5, i].imshow(result4)
        ax[5, i].set_yticks([])
        ax[5, i].set_xticks([])
        #ax[5, i].axis('off')

    fig.tight_layout()
    #plt.show()

    fig.savefig(name)



#path of all images. In each directory, the order of images need to be the same after sorted. 
img_dir="HEA12D_dataset/Asphalt_pavement_FD/Raw_images/*"
gt_dir="HEA12D_dataset/Asphalt_pavement_FD/Masks/*"
result1_dir="Results/Asphalt_pavement_FD/model1/*"
result2_dir="Results/Asphalt_pavement_FD/model2/*"
result3_dir="Results/Asphalt_pavement_FD/model3/*"
result4_dir="Results/Asphalt_pavement_FD/model4/*"

#load images
name = "apfd.png"

plotresults(img_dir,gt_dir,result1_dir,result2_dir,result3_dir,result4_dir,name)

########################

img_dir="HEA12D_dataset/Asphalt_pavement_VD/Raw_images/*"
gt_dir="HEA12D_dataset/Asphalt_pavement_VD/Masks/*"
result1_dir="Results/Asphalt_pavement_VD/model1/*"
result2_dir="Results/Asphalt_pavement_VD/model2/*"
result3_dir="Results/Asphalt_pavement_VD/model3/*"
result4_dir="Results/Asphalt_pavement_VD/model4/*"

#load images
name = "apvd.png"

plotresults(img_dir,gt_dir,result1_dir,result2_dir,result3_dir,result4_dir,name)

img_dir="HEA12D_dataset/Concrete_pavement_FD/Raw_images/*"
gt_dir="HEA12D_dataset/Concrete_pavement_FD/Masks/*"
result1_dir="Results/Concrete_pavement_FD/model1/*"
result2_dir="Results/Concrete_pavement_FD/model2/*"
result3_dir="Results/Concrete_pavement_FD/model3/*"
result4_dir="Results/Concrete_pavement_FD/model4/*"

#load images
name = "CPfd.png"

plotresults(img_dir,gt_dir,result1_dir,result2_dir,result3_dir,result4_dir,name)


img_dir="HEA12D_dataset/Concrete_pavement_VD/Raw_images/*"
gt_dir="HEA12D_dataset/Concrete_pavement_VD/Masks/*"
result1_dir="Results/Concrete_pavement_VD/model1/*"
result2_dir="Results/Concrete_pavement_VD/model2/*"
result3_dir="Results/Concrete_pavement_VD/model3/*"
result4_dir="Results/Concrete_pavement_VD/model4/*"

#load images
name = "cpvd.png"

plotresults(img_dir,gt_dir,result1_dir,result2_dir,result3_dir,result4_dir,name)