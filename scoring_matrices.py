from glob import glob
import cv2
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, jaccard_score, precision_score, recall_score

def evaluate(gt_mask_dir,pred_mask_dir):
    SCORE=[]
    acc_value_overall=[]
    f1_value_overall=[]
    jac_value_overall=[]
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
        gt_raw = cv2.resize (gt_raw,(path_pred_raw.shape[1],path_pred_raw.shape[0]))
        (thresh_pred, im_bw_pred) = cv2.threshold(path_pred_raw, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        (thresh_gt, im_bw_gt) = cv2.threshold(gt_raw, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        mask=im_bw_gt.flatten()/255
        prediction=im_bw_pred.flatten()/255
        acc_value = accuracy_score(mask, prediction)
        f1_value = f1_score(mask, prediction, labels=[0, 1], average="binary")
        jac_value = jaccard_score(mask, prediction, labels=[0, 1], average="binary")
        recall_value = recall_score(mask, prediction, labels=[0, 1], average="binary")
        precision_value = precision_score(mask, prediction, labels=[0, 1], average="binary")
        acc_value_overall.append(acc_value)
        f1_value_overall.append(f1_value)
        jac_value_overall.append(jac_value)
        recall_value_overall.append(recall_value)
        precision_value_overall.append(precision_value)
        SCORE.append([image_name,acc_value, f1_value, jac_value, recall_value, precision_value])
    acc_value_ave=np.mean(acc_value_overall, axis=0)
    f1_value_ave=np.mean(f1_value_overall, axis=0)
    jac_value_ave=np.mean(jac_value_overall, axis=0)
    recall_value_ave=np.mean(recall_value_overall, axis=0)
    precision_value_ave=np.mean(precision_value_overall, axis=0)
    print ("acc_value_ave is: ", acc_value_ave)
    print ("f1_value_ave is: ", f1_value_ave)
    print ("jac_value_ave is: ", jac_value_ave)
    print ("recall_value_ave is: ", recall_value_ave)
    print ("precision_value_ave is: ", precision_value_ave)
    overall_score=[acc_value_ave,f1_value_ave,jac_value_ave,recall_value_ave,precision_value_ave]
    return SCORE, overall_score


gt_mask_dir="Datasets/Asphalt_Pavement_Images/Segmentation_Mask/*"


pred_mask_dir_1="Datasets/DetectionResults/Model_3/*"
#pred_mask_dir_2=""
#pred_mask_dir_3=""
#pred_mask_dir_4=""
#pred_mask_dir_5=""

SCORE_1, overall_score_1 = evaluate(gt_mask_dir,pred_mask_dir_1)
print(SCORE_1)
print(overall_score_1)

#SCORE_2, overall_score_2 = evaluate(gt_mask_dir,pred_mask_dir_2)
#print(SCORE_2)
#print(overall_score_2)
#SCORE_3, overall_score_3 = evaluate(gt_mask_dir,pred_mask_dir_3)
#print(SCORE_3)
#print(overall_score_3)
#SCORE_1, overall_score_4 = evaluate(gt_mask_dir,pred_mask_dir_4)
#print(SCORE_4)
#print(overall_score_4)


