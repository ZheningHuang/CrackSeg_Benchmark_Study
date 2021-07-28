#code to combine cropped images together
from glob import glob
import numpy as np
import cv2 
import os



# directory for source images and output locations
croped_image_folder=".\Model4\concrete_vd\*"
path_to_save=".\Model4_merged"
files_cropped=sorted(glob(croped_image_folder))
num_images=int(len(files_cropped)/24)
for i in range(num_images):
  print (i)
  nam_img=files_cropped[i*24].split("\\")[-1]
  print (nam_img)
  crop_1_1=cv2.imread(files_cropped[i*24])
  crop_1_2=cv2.imread(files_cropped[i*24+1])
  crop_1_3=cv2.imread(files_cropped[i*24+2])
  crop_1_4=cv2.imread(files_cropped[i*24+3])
  crop_1_5=cv2.imread(files_cropped[i*24+4])
  crop_1_6=cv2.imread(files_cropped[i*24+5])
  crop_2_1=cv2.imread(files_cropped[i*24+6])
  crop_2_2=cv2.imread(files_cropped[i*24+7])
  crop_2_3=cv2.imread(files_cropped[i*24+8])
  crop_2_4=cv2.imread(files_cropped[i*24+9])
  crop_2_5=cv2.imread(files_cropped[i*24+10])
  crop_2_6=cv2.imread(files_cropped[i*24+11])
  crop_3_1=cv2.imread(files_cropped[i*24+12])
  crop_3_2=cv2.imread(files_cropped[i*24+13])
  crop_3_3=cv2.imread(files_cropped[i*24+14])
  crop_3_4=cv2.imread(files_cropped[i*24+15])
  crop_3_5=cv2.imread(files_cropped[i*24+16])
  crop_3_6=cv2.imread(files_cropped[i*24+17])
  crop_4_1=cv2.imread(files_cropped[i*24+18])
  crop_4_2=cv2.imread(files_cropped[i*24+19])
  crop_4_3=cv2.imread(files_cropped[i*24+20])
  crop_4_4=cv2.imread(files_cropped[i*24+21])
  crop_4_5=cv2.imread(files_cropped[i*24+22])
  crop_4_6=cv2.imread(files_cropped[i*24+23])

  h_img_1 = cv2.hconcat([crop_1_1,crop_1_2,crop_1_3,crop_1_4,crop_1_5,crop_1_6])
  h_img_2 = cv2.hconcat([crop_2_1,crop_2_2,crop_2_3,crop_2_4,crop_2_5,crop_2_6])
  h_img_3 = cv2.hconcat([crop_3_1,crop_3_2,crop_3_3,crop_3_4,crop_3_5,crop_3_6])
  h_img_4 = cv2.hconcat([crop_4_1,crop_4_2,crop_4_3,crop_4_4,crop_4_5,crop_4_6])
  combined_image = cv2.vconcat([h_img_1,h_img_2,h_img_3,h_img_4])
  cv2.imwrite(os.path.join(path_to_save,nam_img+"_combined.jpg"),combined_image)

print ("All done")