from glob import glob
import cv2
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, jaccard_score, precision_score, recall_score


tole=2


def evaluate(gt_mask_dir,pred_mask_dir):
    SCORE=[]
    f1_value_overall=[]
    #jac_value_overall=[]
    recall_value_overall=[]
    precision_value_overall=[]
    gt_mask_files=sorted(glob(gt_mask_dir))
    pred_mask_files=sorted(glob(pred_mask_dir))
    for i in range (len(gt_mask_files)):
        path_gt=gt_mask_files[i]
        path_pred=pred_mask_files[i]
        image_name = gt_mask_files[i].split("/")[-1]
        path_pred_raw = cv2.imread(path_pred, cv2.IMREAD_GRAYSCALE)
        gt_raw = cv2.imread(path_gt, cv2.IMREAD_GRAYSCALE)
        #gt_raw = cv2.resize (gt_raw,(path_pred_raw.shape[1],path_pred_raw.shape[0]))
        (thresh_pred, im_bw_pred) = cv2.threshold(path_pred_raw, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        (thresh_gt, im_bw_gt) = cv2.threshold(gt_raw, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        #tolerence
        im_bw_pred_tol=np.zeros(im_bw_pred.shape)
        for ix,iy in np.ndindex(im_bw_pred.shape):
            if im_bw_pred[ix,iy]==255:
                ix_low_boun=max(0,ix-tole)
                ix_up_boun=min(im_bw_pred.shape[0],ix+tole)
                iy_low_boun=max(0,iy-tole)
                iy_up_boun=min(im_bw_pred.shape[1],iy+tole)
                im_bw_pred_tol[ix_low_boun:ix_up_boun,iy_low_boun:iy_up_boun]=255
        #tolerence
        im_bw_gt_tol=np.zeros(im_bw_gt.shape)
        for ix,iy in np.ndindex(im_bw_pred.shape):
            if im_bw_gt[ix,iy]==255:
                ix_low_boun=max(0,ix-tole)
                ix_up_boun=min(im_bw_gt.shape[0],ix+tole)
                iy_low_boun=max(0,iy-tole)
                iy_up_boun=min(im_bw_gt.shape[1],iy+tole)
                im_bw_gt_tol[ix_low_boun:ix_up_boun,iy_low_boun:iy_up_boun]=255


        mask_raw=im_bw_gt.flatten()/255
        prediction_raw=im_bw_pred.flatten()/255

        mask_thick=im_bw_gt_tol.flatten()/255
        prediction_thick=im_bw_pred_tol.flatten()/255
        
        recall_value = recall_score(mask_raw, prediction_thick, labels=[0, 1], average="binary")
        precision_value = precision_score(mask_thick, prediction_raw, labels=[0, 1], average="binary")
        f1_value= 2* precision_value*recall_value/(recall_value+precision_value)

        '''
        f1_value = f1_score(mask, prediction, labels=[0, 1], average="binary")
        jac_value = jaccard_score(mask, prediction, labels=[0, 1], average="binary")
        recall_value = recall_score(mask, prediction, labels=[0, 1], average="binary")
        precision_value = precision_score(mask, prediction, labels=[0, 1], average="binary")
        '''
        #acc_value_overall.append(acc_value)
        f1_value_overall.append(f1_value)
        #jac_value_overall.append(jac_value)
        recall_value_overall.append(recall_value)
        precision_value_overall.append(precision_value)
        SCORE.append([image_name,precision_value,recall_value,f1_value])
    
    #acc_value_ave=np.mean(acc_value_overall, axis=0)
    f1_value_ave=np.mean(f1_value_overall, axis=0)
    #jac_value_ave=np.mean(jac_value_overall, axis=0)
    recall_value_ave=np.mean(recall_value_overall, axis=0)
    precision_value_ave=np.mean(precision_value_overall, axis=0)
    
    #print ("acc_value_ave is: ", acc_value_ave)
    print ("f1_value_ave is: ", f1_value_ave)
    #print ("jac_value_ave is: ", jac_value_ave)
    print ("recall_value_ave is: ", recall_value_ave)
    print ("precision_value_ave is: ", precision_value_ave)
    overall_score=[f1_value_ave,recall_value_ave,precision_value_ave]
    return SCORE, overall_score


gt_mask_dir="Datasets/Asphalt_Pavement_Images/Segmentation_Mask/*"


pred_mask_dir_1="Datasets/DetectionResults/Model_1/*"
pred_mask_dir_2="Datasets/DetectionResults/Model_2/*"
#pred_mask_dir_3=""
#pred_mask_dir_4=""
#pred_mask_dir_5=""

SCORE_1, overall_score_1 = evaluate(gt_mask_dir,pred_mask_dir_1)
print(SCORE_1)
print(overall_score_1)

SCORE_2, overall_score_2 = evaluate(gt_mask_dir,pred_mask_dir_2)
print(SCORE_2)
print(overall_score_2)

#SCORE_3, overall_score_3 = evaluate(gt_mask_dir,pred_mask_dir_3)
#print(SCORE_3)
#print(overall_score_3)
#SCORE_1, overall_score_4 = evaluate(gt_mask_dir,pred_mask_dir_4)
#print(SCORE_4)
#print(overall_score_4)




'''

#image_name = gt_mask_files[i].split("/")[-1]
path_pred_raw = cv2.imread(path_pred, cv2.IMREAD_GRAYSCALE)
gt_raw = cv2.imread(path_gt, cv2.IMREAD_GRAYSCALE)
#gt_raw = cv2.resize (gt_raw,(path_pred_raw.shape[1],path_pred_raw.shape[0]))
(thresh_pred, im_bw_pred) = cv2.threshold(path_pred_raw, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
(thresh_gt, im_bw_gt) = cv2.threshold(gt_raw, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

im_bw_pred_tol=np.zeros(im_bw_pred.shape)

for ix,iy in np.ndindex(im_bw_pred.shape):
    if im_bw_pred[ix,iy]==255:
        ix_low_boun=max(0,ix-tole)
        ix_up_boun=min(im_bw_pred.shape[0],ix+tole)
        iy_low_boun=max(0,iy-tole)
        iy_up_boun=min(im_bw_pred.shape[1],iy+tole)
        im_bw_pred_tol[ix_low_boun:ix_up_boun,iy_low_boun:iy_up_boun]=255

im_bw_gt_tol=np.zeros(im_bw_gt.shape)

for ix,iy in np.ndindex(im_bw_pred.shape):
    if im_bw_gt[ix,iy]==255:
        ix_low_boun=max(0,ix-tole)
        ix_up_boun=min(im_bw_gt.shape[0],ix+tole)
        iy_low_boun=max(0,iy-tole)
        iy_up_boun=min(im_bw_gt.shape[1],iy+tole)
        im_bw_gt_tol[ix_low_boun:ix_up_boun,iy_low_boun:iy_up_boun]=255


mask_raw=im_bw_gt.flatten()/255
prediction_raw=im_bw_pred.flatten()/255

mask_thick=im_bw_gt_tol.flatten()/255
prediction_thick=im_bw_pred_tol.flatten()/255



#f1_value_raw = f1_score(mask, prediction, labels=[0, 1], average="binary")
#jac_value = jaccard_score(mask, prediction, labels=[0, 1], average="binary")
recall_value = recall_score(mask_raw, prediction_raw, labels=[0, 1], average="binary")
precision_value = precision_score(mask_raw, prediction_raw, labels=[0, 1], average="binary")

recall_value_enhenced = recall_score(mask_raw, prediction_thick, labels=[0, 1], average="binary")
precision_value_enhenced = precision_score(mask_thick, prediction_raw, labels=[0, 1], average="binary")

print ("recall orginal" ,recall_value, "recall",recall_value_enhenced)
print ("precision orginal" ,precision_value, "precision", precision_value_enhenced)

'''

