from glob import glob
import cv2
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, jaccard_score, precision_score, recall_score

def evaluate(gt_mask_dir,pred_mask_dir,post_gt_mask_dir,post_pred_mask_dir):
    
    #new list for storing results
    Overall_Score=[]
    f1_value_overall=[]
    recall_value_overall=[]
    precision_value_overall=[]

    #load images including: orginal mask, original output, postprocessed mask, post output
    gt_mask_files=sorted(glob(gt_mask_dir))
    pred_mask_files=sorted(glob(pred_mask_dir))
    post_gt_mask_files=sorted(glob(post_gt_mask_dir))
    post_pred_mask_files=sorted(glob(post_pred_mask_dir))
    print (len(gt_mask_files),len(pred_mask_files),len(post_gt_mask_files),len(post_pred_mask_files))
    for i in range (len(gt_mask_files)):
        # read images
        path_gt=gt_mask_files[i]
        path_pred=pred_mask_files[i]
        post_path_gt=post_gt_mask_files[i]
        path_pred=post_pred_mask_files[i]    
        #get the name of images
        image_name = gt_mask_files[i].split("\\")[-1]
        #  convert images to array
        path_pred_raw = cv2.imread(path_pred, cv2.IMREAD_GRAYSCALE)
        gt_raw = cv2.imread(path_gt, cv2.IMREAD_GRAYSCALE)
        post_gt = cv2.imread(post_path_gt, cv2.IMREAD_GRAYSCALE)
        post_path_pred_raw = cv2.imread(path_pred, cv2.IMREAD_GRAYSCALE)

        #resize images for compatibility
        gt_raw = cv2.resize (gt_raw,(path_pred_raw.shape[1],path_pred_raw.shape[0]))
        post_gt = cv2.resize (post_gt,(path_pred_raw.shape[1],path_pred_raw.shape[0]))

        #binarize raw mask and raw predicts
        (thresh_pred, im_bw_pred) = cv2.threshold(path_pred_raw, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        (thresh_gt, im_bw_gt) = cv2.threshold(gt_raw, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU) 
        (thresh_pred, post_gt) = cv2.threshold(post_gt, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        (thresh_gt, post_path_pred_raw) = cv2.threshold(post_path_pred_raw, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU) 
        #flatten for evaluation
        mask_raw=im_bw_gt.flatten()/255
        prediction_raw=im_bw_pred.flatten()/255
        mask_thick=post_gt.flatten()/255
        prediction_thick=post_path_pred_raw.flatten()/255
        ## calculate with sklearn
        recall_value = recall_score(mask_raw, prediction_thick, labels=[0, 1], average="binary")
        precision_value = precision_score(mask_thick, prediction_raw, labels=[0, 1], average="binary")
        f1_value= 2 * precision_value*recall_value/(recall_value+precision_value)
        ## store values
        f1_value_overall.append(f1_value)
        recall_value_overall.append(recall_value)
        precision_value_overall.append(precision_value)
        Overall_Score.append([image_name,precision_value,recall_value,f1_value])
    print (Overall_Score)
    #calculate means
    f1_value_ave=np.mean(f1_value_overall, axis=0)
    recall_value_ave=np.mean(recall_value_overall, axis=0)
    precision_value_ave=np.mean(precision_value_overall, axis=0)
    #print values
    print ("f1_value_ave is: ", f1_value_ave)
    print ("recall_value_ave is: ", recall_value_ave)
    print ("precision_value_ave is: ", precision_value_ave)
    overall_score_average=[recall_value_ave,precision_value_ave,f1_value_ave]
    print ("recall, precision,f1 value for this model is,", overall_score_average)
    return Overall_Score, overall_score_average

print("########################CPVD######################")
gt_mask_dir="HEA12D_dataset/Concrete_pavement_VD/Masks/*"
post_gt_mask_dir="Mask_post/Concrete_pavement_VD/*"

print ("CPVD Model1")
pred_mask_dir1="Results/Concrete_pavement_VD/model1/*"
post_pred_mask_dir1="Results_post/Concrete_pavement_VD/model1/*"
evaluate(gt_mask_dir,pred_mask_dir1,post_gt_mask_dir,post_pred_mask_dir1)

print ("CPVD Model2")
pred_mask_dir2="Results/Concrete_pavement_VD/model2/*"
post_pred_mask_dir2="Results_post/Concrete_pavement_VD/model2/*"
evaluate(gt_mask_dir,pred_mask_dir2,post_gt_mask_dir,post_pred_mask_dir2)

print ("CPVD Model3")
pred_mask_dir3="Results/Concrete_pavement_VD/model3/*"
post_pred_mask_dir3="Results_post/Concrete_pavement_VD/model3/*"
evaluate(gt_mask_dir,pred_mask_dir3,post_gt_mask_dir,post_pred_mask_dir3)

print ("CPVD Model4")
pred_mask_dir4="Results/Concrete_pavement_VD/model4/*"
post_pred_mask_dir4="Results_post/Concrete_pavement_VD/model4/*"
evaluate(gt_mask_dir,pred_mask_dir4,post_gt_mask_dir,post_pred_mask_dir4)

print("########################CPFD######################")
gt_mask_dir="HEA12D_dataset/Concrete_pavement_FD/Masks/*"
post_gt_mask_dir="Mask_post/Concrete_pavement_FD/*"

print ("CPFD Model1")
pred_mask_dir1="Results/Concrete_pavement_FD/model1/*"
post_pred_mask_dir1="Results_post/Concrete_pavement_FD/model1/*"
evaluate(gt_mask_dir,pred_mask_dir1,post_gt_mask_dir,post_pred_mask_dir1)

print ("CPFD Model2")
pred_mask_dir2="Results/Concrete_pavement_FD/model2/*"
post_pred_mask_dir2="Results_post/Concrete_pavement_FD/model2/*"
evaluate(gt_mask_dir,pred_mask_dir2,post_gt_mask_dir,post_pred_mask_dir2)

print ("CPFD Model3")
pred_mask_dir3="Results/Concrete_pavement_FD/model3/*"
post_pred_mask_dir3="Results_post/Concrete_pavement_FD/model3/*"
evaluate(gt_mask_dir,pred_mask_dir3,post_gt_mask_dir,post_pred_mask_dir3)

print ("CPFD Model4")
pred_mask_dir4="Results/Concrete_pavement_FD/model4/*"
post_pred_mask_dir4="Results_post/Concrete_pavement_FD/model4/*"
evaluate(gt_mask_dir,pred_mask_dir4,post_gt_mask_dir,post_pred_mask_dir4)


print("########################APVD######################")
gt_mask_dir="HEA12D_dataset/Asphalt_pavement_VD/Masks/*"
post_gt_mask_dir="Mask_post/Asphalt_pavement_VD/*"

print ("APVD Model1")
pred_mask_dir1="Results/Asphalt_pavement_VD/model1/*"
post_pred_mask_dir1="Results_post/Asphalt_pavement_VD/model1/*"
evaluate(gt_mask_dir,pred_mask_dir1,post_gt_mask_dir,post_pred_mask_dir1)

print ("APVD Model2")
pred_mask_dir2="Results/Asphalt_pavement_VD/model2/*"
post_pred_mask_dir2="Results_post/Asphalt_pavement_VD/model2/*"
evaluate(gt_mask_dir,pred_mask_dir2,post_gt_mask_dir,post_pred_mask_dir2)

print ("APVD Model3")
pred_mask_dir3="Results/Asphalt_pavement_VD/model3/*"
post_pred_mask_dir3="Results_post/Asphalt_pavement_VD/model3/*"
evaluate(gt_mask_dir,pred_mask_dir3,post_gt_mask_dir,post_pred_mask_dir3)

print ("APVD Model4")
pred_mask_dir4="Results/Asphalt_pavement_VD/model4/*"
post_pred_mask_dir4="Results_post/Asphalt_pavement_VD/model4/*"
evaluate(gt_mask_dir,pred_mask_dir4,post_gt_mask_dir,post_pred_mask_dir4)

print("########################APFD######################")
gt_mask_dir="HEA12D_dataset/Asphalt_pavement_FD/Masks/*"
post_gt_mask_dir="Mask_post/Asphalt_pavement_FD/*"

print ("APFD Model1")
pred_mask_dir1="Results/Asphalt_pavement_FD/model1/*"
post_pred_mask_dir1="Results_post/Asphalt_pavement_FD/model1/*"
evaluate(gt_mask_dir,pred_mask_dir1,post_gt_mask_dir,post_pred_mask_dir1)

print ("APFD Model2")
pred_mask_dir2="Results/Asphalt_pavement_FD/model2/*"
post_pred_mask_dir2="Results_post/Asphalt_pavement_FD/model2/*"
evaluate(gt_mask_dir,pred_mask_dir2,post_gt_mask_dir,post_pred_mask_dir2)

print ("APFD Model3")
pred_mask_dir3="Results/Asphalt_pavement_FD/model3/*"
post_pred_mask_dir3="Results_post/Asphalt_pavement_FD/model3/*"
evaluate(gt_mask_dir,pred_mask_dir3,post_gt_mask_dir,post_pred_mask_dir3)

print ("APFD Model4")
pred_mask_dir4="Results/Asphalt_pavement_FD/model4/*"
post_pred_mask_dir4="Results_post/Asphalt_pavement_FD/model4/*"
evaluate(gt_mask_dir,pred_mask_dir4,post_gt_mask_dir,post_pred_mask_dir4)

