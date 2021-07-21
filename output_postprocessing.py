#This code can be used to convert detection output to binary images and reverse the color (from black background to white background)

from glob import glob
import cv2
import numpy as np
import os

def binarize_reverse(pred_mask_dir,save_dir):    
    pred_mask_files=sorted(glob(pred_mask_dir))
    for i in range (len(pred_mask_files)):
        path_pred=pred_mask_files[i]
        image_name = pred_mask_files[i].split("/")[-1]
        path_pred_raw = cv2.imread(path_pred, cv2.IMREAD_GRAYSCALE)
        (thresh_pred, im_bw_pred) = cv2.threshold(path_pred_raw, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        predict_inverse= (255-im_bw_pred) #reverse images
        PATH=os.path.join(save_dir,image_name)
        print (PATH)
        cv2.imwrite(PATH,predict_inverse)

    
output_mask_dir="output/"
pred_mask_dir="inverse_trail/*" #remember to add a /* add the end of directory

binarize_reverse(pred_mask_dir,output_mask_dir)