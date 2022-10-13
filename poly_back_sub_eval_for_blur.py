from logging.handlers import DatagramHandler
from unittest import result
import cv2
import numpy as np
import os.path
import xml.etree.ElementTree as ET
import csv
import os
from skimage import filters

# import pandas as pd
from pandas import DataFrame
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.metrics
plt.style.use('seaborn')

lst = []
header = ["file_name","Object Type","Number of edge pixels","height","width","ratio","Blurred?"]

filepath = "dataset/sk500/train.txt"
with open(filepath) as f:
        lines = f.readlines()
        # print(lines)
        for line in lines:
            image, label = (line.strip().split(' '))
            if "poisson" not in image:
               if(os.path.exists(f'dataset/sk500/images/{image}')):
                   # print("Hello")
                     img = cv2.imread(f'dataset/sk500/images/{image}')
                   
                     if(os.path.exists(f'dataset/sk500/labels/{label}')): 
                          # print("Bye")
                            doc_tree = ET.parse(f'dataset/sk500/labels/{label}')
                            root = doc_tree.getroot()
                            objects = root.findall('object')
                          
                            for i, o in enumerate(list(objects)):
                                 
                                   name = o.find('name').text
                                   if name=="national_id_0":
                                      polygon = o.find('polygon') # reading bound box
                                
                                 # Top Left
                                      x1 , y1 = int(float((polygon.find('x1').text))) , int(float((polygon.find('y1').text)))
                                      top_left_point = [x1,y1]
                                 # Top Right
                                      x2 , y2 = int(float((polygon.find('x2').text))) , int(float((polygon.find('y2').text)))
                                      top_right_point = [x2,y2]
                                 # Bottom Right
                                      x3 , y3 = int(float((polygon.find('x3').text))) , int(float((polygon.find('y3').text)))
                                      bottom_right_point = [x3,y3]
                                 # Bottom Left
                                      x4 , y4 = int(float((polygon.find('x4').text))) , int(float((polygon.find('y4').text)))
                                      bottom_left_point = [x4,y4]

                                      input_points = np.float32([top_left_point,top_right_point,bottom_right_point,bottom_left_point])
                                      output_points = np.float32([
                                    [0,0],
                                    [224,0],
                                    [224,224],
                                    [0,224]]
                                 )
                                 
                                 #Compute perspective  transform M
                                      M = cv2.getPerspectiveTransform(input_points,output_points)

                                      warped_image = cv2.warpPerspective(img,M,(224,224),flags=cv2.INTER_LINEAR)


                          # Crop ID card before canny and counting pixels
                          
                                #  crop = img[ymin:ymax, xmin:xmax]
                                #  crop = cv2.resize(crop, (224, 224))       
                          # Cnvert to grayscale image
                                      crop_img_gray = cv2.cvtColor(warped_image, cv2.COLOR_BGR2GRAY)
            
                          # Aply Gaussian Blur
                                      crop_img_blur = cv2.GaussianBlur(crop_img_gray, (3,3), 0)
                                
                                      t = filters.threshold_otsu(crop_img_gray)
                                      mask = crop_img_gray > t
                                      mask = np.logical_not(mask)
                                      mask = np.array(mask,dtype=np.float64) + 0
            
                          # Dtect the edges
                                      edges = cv2.Canny(image=crop_img_blur, threshold1 = 85, threshold2= 255,apertureSize= 3,L2gradient=1)
                                      final = edges * mask
                          # Gt width and height of image
                                      height, width  =  final.shape
            
                          #Ge number of egde/white pixelS
                                      number_of_white_pix = np.sum( final == 255)
                                 # print("Number of white pixels :", number_of_white_pix)
                                 # print("Ratio : ", "{:.2f}".format(number_of_white_pix/(height*width)))
                                      if 'motion' in image:
                                      # return(f"{image}"," ", number_of_white_pix," ", height," ", width," ", ("{:.2f}".format(number_of_white_pix/(height*width))), " ", 1)
                                             lst.append(([f"{image}_{i}",f"{name}",number_of_white_pix,height,width,(number_of_white_pix/(height*width)),1]))
                              
                                      else:
                                             pass
df = DataFrame(lst)
df.to_csv('poly_sub_for_blur_eval.csv')