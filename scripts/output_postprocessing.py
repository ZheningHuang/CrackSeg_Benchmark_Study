#This code can be used to convert detection output to binary images and reverse the color (from black background to white background)

from glob import glob
import cv2
import numpy as np
import os

tolerance=2

def binarize_reverse(pred_mask_dir,save_dir):    
    pred_mask_files=sorted(glob(pred_mask_dir))
    for i in range (len(pred_mask_files)):
        path_pred=pred_mask_files[i]
        image_name = pred_mask_files[i].split("\\")[-1]
        path_pred_raw = cv2.imread(path_pred, cv2.IMREAD_GRAYSCALE)
        (thresh_pred, im_bw_pred) = cv2.threshold(path_pred_raw, 254, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        #predict_inverse= (255-im_bw_pred) #reverse images
        PATH=os.path.join(save_dir,image_name)
        print (PATH)
        cv2.imwrite(PATH,im_bw_pred)
        im_bw_pred_tol=np.zeros(im_bw_pred.shape)
        for ix,iy in np.ndindex(im_bw_pred.shape):
            if im_bw_pred[ix,iy]==255:
                ix_low_boun=max(0,ix-tolerance)
                ix_up_boun=min(im_bw_pred.shape[0],ix+tolerance)
                iy_low_boun=max(0,iy-tolerance)
                iy_up_boun=min(im_bw_pred.shape[1],iy+tolerance)
                im_bw_pred_tol[ix_low_boun:ix_up_boun,iy_low_boun:iy_up_boun]=255
        cv2.imwrite(PATH,im_bw_pred_tol)

'''
output_mask_dir1="Mask_post/Asphalt_pavement_FD/"
pred_mask_dir1="HEA12D_dataset/Asphalt_pavement_FD/Masks/*" #remember to add a /* add the end of directory
output_mask_dir2="Mask_post/Asphalt_pavement_VD/"
pred_mask_dir2="HEA12D_dataset/Asphalt_pavement_VD/Masks/*" #remember to add a /* add the end of directory
output_mask_dir3="Mask_post/Concrete_pavement_FD/"
pred_mask_dir3="HEA12D_dataset/Concrete_pavement_FD/Masks/*" #remember to add a /* add the end of directory
output_mask_dir4="Mask_post/Concrete_pavement_VD/"
pred_mask_dir4="HEA12D_dataset/Concrete_pavement_VD/Masks/*" #remember to add a /* add the end of directory

binarize_reverse(pred_mask_dir1,output_mask_dir1)
binarize_reverse(pred_mask_dir2,output_mask_dir2)
binarize_reverse(pred_mask_dir3,output_mask_dir3)
binarize_reverse(pred_mask_dir4,output_mask_dir4)
'''

output_mask_dir1="Results_post/Concrete_pavement_VD/model1/"
pred_mask_dir1="Results/Concrete_pavement_VD/model1/*" #remember to add a /* add the end of directory
output_mask_dir2="Results_post/Concrete_pavement_VD/model2/"
pred_mask_dir2="Results/Concrete_pavement_VD/model2/*" #remember to add a /* add the end of directory
output_mask_dir3="Results_post/Concrete_pavement_VD/model3/"
pred_mask_dir3="Results/Concrete_pavement_VD/model3/*" #remember to add a /* add the end of directory
output_mask_dir4="Results_post/Concrete_pavement_VD/model4/"
pred_mask_dir4="Results/Concrete_pavement_VD/model4/*" #remember to add a /* add the end of directory

binarize_reverse(pred_mask_dir1,output_mask_dir1)
binarize_reverse(pred_mask_dir2,output_mask_dir2)
binarize_reverse(pred_mask_dir3,output_mask_dir3)
binarize_reverse(pred_mask_dir4,output_mask_dir4)
####
output_mask_dir1="Results_post/Concrete_pavement_FD/model1/"
pred_mask_dir1="Results/Concrete_pavement_FD/model1/*" #remember to add a /* add the end of directory
output_mask_dir2="Results_post/Concrete_pavement_FD/model2/"
pred_mask_dir2="Results/Concrete_pavement_FD/model2/*" #remember to add a /* add the end of directory
output_mask_dir3="Results_post/Concrete_pavement_FD/model3/"
pred_mask_dir3="Results/Concrete_pavement_FD/model3/*" #remember to add a /* add the end of directory
output_mask_dir4="Results_post/Concrete_pavement_FD/model4/"
pred_mask_dir4="Results/Concrete_pavement_FD/model4/*" #remember to add a /* add the end of directory

binarize_reverse(pred_mask_dir1,output_mask_dir1)
binarize_reverse(pred_mask_dir2,output_mask_dir2)
binarize_reverse(pred_mask_dir3,output_mask_dir3)
binarize_reverse(pred_mask_dir4,output_mask_dir4)


####
output_mask_dir1="Results_post/Asphalt_pavement_VD/model1/"
pred_mask_dir1="Results/Asphalt_pavement_VD/model1/*" #remember to add a /* add the end of directory
output_mask_dir2="Results_post/Asphalt_pavement_VD/model2/"
pred_mask_dir2="Results/Asphalt_pavement_VD/model2/*" #remember to add a /* add the end of directory
output_mask_dir3="Results_post/Asphalt_pavement_VD/model3/"
pred_mask_dir3="Results/Asphalt_pavement_VD/model3/*" #remember to add a /* add the end of directory
output_mask_dir4="Results_post/Asphalt_pavement_VD/model4/"
pred_mask_dir4="Results/Asphalt_pavement_VD/model4/*" #remember to add a /* add the end of directory

binarize_reverse(pred_mask_dir1,output_mask_dir1)
binarize_reverse(pred_mask_dir2,output_mask_dir2)
binarize_reverse(pred_mask_dir3,output_mask_dir3)
binarize_reverse(pred_mask_dir4,output_mask_dir4)

####
output_mask_dir1="Results_post/Asphalt_pavement_FD/model1/"
pred_mask_dir1="Results/Asphalt_pavement_FD/model1/*" #remember to add a /* add the end of directory
output_mask_dir2="Results_post/Asphalt_pavement_FD/model2/"
pred_mask_dir2="Results/Asphalt_pavement_FD/model2/*" #remember to add a /* add the end of directory
output_mask_dir3="Results_post/Asphalt_pavement_FD/model3/"
pred_mask_dir3="Results/Asphalt_pavement_FD/model3/*" #remember to add a /* add the end of directory
output_mask_dir4="Results_post/Asphalt_pavement_FD/model4/"
pred_mask_dir4="Results/Asphalt_pavement_FD/model4/*" #remember to add a /* add the end of directory

binarize_reverse(pred_mask_dir1,output_mask_dir1)
binarize_reverse(pred_mask_dir2,output_mask_dir2)
binarize_reverse(pred_mask_dir3,output_mask_dir3)
binarize_reverse(pred_mask_dir4,output_mask_dir4)