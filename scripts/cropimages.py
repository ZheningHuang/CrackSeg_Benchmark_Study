from glob import glob
import numpy as np
import cv2 

#directory that contains images you want to crop. 

image_dir="/content/drive/MyDrive/Ashplat/*"
image_files=sorted(glob(image_dir))

for i in range (len(image_files)):
  raw_image=cv2.imread(image_files[i])
  name_image=image_files[i].split("/")[-1].split(".")[0]
  a=0
  b=0
  h=270
  w=320
  crop_img_1_1 = raw_image[a:a+h, b:b+w]
  crop_img_1_2 = raw_image[a:a+h, b+w:b+2*w]
  crop_img_1_3 = raw_image[a:a+h, b+2*w:b+3*w]
  crop_img_1_4= raw_image[a:a+h, b+3*w:b+4*w]
  crop_img_1_5 = raw_image[a:a+h, b+4*w:b+5*w]
  crop_img_1_6 = raw_image[a:a+h, b+5*w:b+6*w]
  crop_img_2_1 = raw_image[a+h:a+2*h, b:b+w]
  crop_img_2_2 = raw_image[a+h:a+2*h, b+w:b+2*w]
  crop_img_2_3 = raw_image[a+h:a+2*h, b+2*w:b+3*w]
  crop_img_2_4 = raw_image[a+h:a+2*h, b+3*w:b+4*w]
  crop_img_2_5 = raw_image[a+h:a+2*h, b+4*w:b+5*w]
  crop_img_2_6 = raw_image[a+h:a+2*h, b+5*w:b+6*w]
  crop_img_3_1 = raw_image[a+2*h:a+3*h, b:b+w]
  crop_img_3_2 = raw_image[a+2*h:a+3*h, b+w:b+2*w]
  crop_img_3_3 = raw_image[a+2*h:a+3*h, b+2*w:b+3*w]
  crop_img_3_4 = raw_image[a+2*h:a+3*h, b+3*w:b+4*w]
  crop_img_3_5 = raw_image[a+2*h:a+3*h, b+4*w:b+5*w]
  crop_img_3_6 = raw_image[a+2*h:a+3*h, b+5*w:b+6*w]
  crop_img_4_1 = raw_image[a+3*h:a+4*h, b:b+w]
  crop_img_4_2 = raw_image[a+3*h:a+4*h, b+w:b+2*w]
  crop_img_4_3 = raw_image[a+3*h:a+4*h, b+2*w:b+3*w]
  crop_img_4_4 = raw_image[a+3*h:a+4*h, b+3*w:b+4*w]
  crop_img_4_5 = raw_image[a+3*h:a+4*h, b+4*w:b+5*w]
  crop_img_4_6 = raw_image[a+3*h:a+4*h, b+5*w:b+6*w]

  #path for targeted images
  
  name_path="/content/drive/MyDrive/Asphalt_cropped"

  
  cv2.imwrite(os.path.join(name_path,name_image+"_1_1.jpg"),crop_img_1_1)
  cv2.imwrite(os.path.join(name_path,name_image+"_1_2.jpg"),crop_img_1_2)
  cv2.imwrite(os.path.join(name_path,name_image+"_1_3.jpg"),crop_img_1_3)
  cv2.imwrite(os.path.join(name_path,name_image+"_1_4.jpg"),crop_img_1_4)
  cv2.imwrite(os.path.join(name_path,name_image+"_1_5.jpg"),crop_img_1_5)
  cv2.imwrite(os.path.join(name_path,name_image+"_1_6.jpg"),crop_img_1_6)
  cv2.imwrite(os.path.join(name_path,name_image+"_2_1.jpg"),crop_img_2_1)
  cv2.imwrite(os.path.join(name_path,name_image+"_2_2.jpg"),crop_img_2_2)
  cv2.imwrite(os.path.join(name_path,name_image+"_2_3.jpg"),crop_img_2_3)
  cv2.imwrite(os.path.join(name_path,name_image+"_2_4.jpg"),crop_img_2_4)
  cv2.imwrite(os.path.join(name_path,name_image+"_2_5.jpg"),crop_img_2_5)
  cv2.imwrite(os.path.join(name_path,name_image+"_2_6.jpg"),crop_img_2_6)
  cv2.imwrite(os.path.join(name_path,name_image+"_3_1.jpg"),crop_img_3_1)
  cv2.imwrite(os.path.join(name_path,name_image+"_3_2.jpg"),crop_img_3_2)
  cv2.imwrite(os.path.join(name_path,name_image+"_3_3.jpg"),crop_img_3_3)
  cv2.imwrite(os.path.join(name_path,name_image+"_3_4.jpg"),crop_img_3_4)
  cv2.imwrite(os.path.join(name_path,name_image+"_3_5.jpg"),crop_img_3_5)
  cv2.imwrite(os.path.join(name_path,name_image+"_3_6.jpg"),crop_img_3_6)
  cv2.imwrite(os.path.join(name_path,name_image+"_4_1.jpg"),crop_img_4_1)
  cv2.imwrite(os.path.join(name_path,name_image+"_4_2.jpg"),crop_img_4_2)
  cv2.imwrite(os.path.join(name_path,name_image+"_4_3.jpg"),crop_img_4_3)
  cv2.imwrite(os.path.join(name_path,name_image+"_4_4.jpg"),crop_img_4_4)
  cv2.imwrite(os.path.join(name_path,name_image+"_4_5.jpg"),crop_img_4_5)
  cv2.imwrite(os.path.join(name_path,name_image+"_4_6.jpg"),crop_img_4_6)

