import cv2
import numpy as np
import os.path
import xml.etree.ElementTree as ET
import csv
import os
# import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
import sklearn.metrics
from skimage import filters
plt.style.use('seaborn')

lst = []
header = ["file_name","Object Type","Number of edge pixels","height","width","ratio","Blurred?"]


def canny_result(lines,config,args):
     for line in lines:
          image, label = (line.strip().split(' '))
          if "poisson" not in image:
               if(os.path.exists(f'dataset/sk500/images/{image}')):
                   
                    img = cv2.imread(f'dataset/sk500/images/{image}')

                    if(os.path.exists(f'dataset/sk500/labels/{label}')): 
                         doc_tree = ET.parse(f'dataset/sk500/labels/{label}')
                         root = doc_tree.getroot()
                         objects = root.findall('object')
                          
                         for i, o in enumerate(list(objects)):
                                 
                              name = o.find('name').text
                              if name == "national_id_0":
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
                                   M = cv2.getPerspectiveTransform(input_points,output_points)
                                   warped_image = cv2.warpPerspective(img,M,(224,224),flags=cv2.INTER_LINEAR)


                                   crop_img_gray = cv2.cvtColor(warped_image, cv2.COLOR_BGR2GRAY)
            
                          
                                   crop_img_blur = cv2.GaussianBlur(crop_img_gray, (tuple(config["gausian_filter"])), 0)

                                   # Background subtraction
                                   t = filters.threshold_otsu(crop_img_gray)
                                   mask = crop_img_gray > t
                                   mask = np.logical_not(mask)
                                   mask = np.array(mask,dtype=np.float64) + 0
            
                                   edges = cv2.Canny(image=crop_img_blur, threshold1 = config["threshold_1"], threshold2= config["threshold_2"],apertureSize= config["aperture_size"],L2gradient=config["l2_gradient"])
                                   final = edges * mask
                 
                                   height, width  =  final.shape
            

                                   number_of_white_pix = np.sum( final == 255)
                           
                                   if 'motion' in image:
                                             lst.append(([f"{image}_{i}",f"{name}",number_of_white_pix,height,width,(number_of_white_pix/(height*width)),1]))
                              
                                   else:
                                           lst.append(([f"{image}_{i}",f"{name}",number_of_white_pix,height,width,(number_of_white_pix/(height*width)),0]))
     df = DataFrame(lst,columns=header)
     evaluate_canny(df,args,config)


def evaluate_canny(dataframe,args,config):

       set_ratio = 0.01

       result_lst = []
       result_headers = ["Set Ratio","Recall","Precision","Accuracy","F1_Score"]


       for i in range(41):
           pred_list = []
           for value in dataframe["ratio"]:
              if  value <= set_ratio:
                   pred_list.append(1)
              else:
                   pred_list.append(0)
           dataframe2 = dataframe.assign(prediction = pred_list)


           recall = sklearn.metrics.recall_score(dataframe2["Blurred?"], dataframe2["prediction"])
           precision = sklearn.metrics.precision_score(dataframe2["Blurred?"], dataframe2["prediction"])
           accuracy = sklearn.metrics.accuracy_score(dataframe2["Blurred?"], dataframe2["prediction"])
           f1_score = sklearn.metrics.f1_score(dataframe2["Blurred?"], dataframe2["prediction"])


           result_lst.append([set_ratio,recall,precision,accuracy,f1_score])

      
           set_ratio +=0.0025
           set_ratio = round(set_ratio,4)
           print("\n")

       df3 = DataFrame(result_lst,columns=result_headers)
       temp = config["model_type"]
       thres1 = config["threshold_1"]
       thres2 = config["threshold_2"]
       gaus = config["gausian_filter"][0]
       aperture = config["aperture_size"]
       l2 = config["l2_gradient"]

       df3.to_csv(f'poly_back_sub_{temp}_{thres1}_{thres2}_{gaus}_{aperture}_{l2}.csv',index=True, columns=result_headers)
