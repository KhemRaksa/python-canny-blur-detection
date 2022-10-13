from logging.handlers import DatagramHandler
from unittest import result
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
plt.style.use('seaborn')

lst = []
header = ["file_name","Object Type","Number of edge pixels","height","width","ratio","Blurred?"]


def canny_result(lines,config,args):
       # print("Hi")
       for line in lines:
            # print(line.strip())
            image, label = (line.strip().split(' '))
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
                                 bndbox = o.find('bndbox') # reading bound box
                                 xmin = int(float((bndbox.find('xmin').text)))
                                 ymin = int(float((bndbox.find('ymin').text)))
                                 xmax = int(float((bndbox.find('xmax').text)))
                                 ymax = int(float((bndbox.find('ymax').text)))
                          
                          # Crop ID card before canny and counting pixels
                          
                                 crop = img[ymin:ymax, xmin:xmax]
                                 crop = cv2.resize(crop, (224, 224))       
                          # Cnvert to grayscale image
                                 crop_img_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            
                          # Aply Gaussian Blur
                                 crop_img_blur = cv2.GaussianBlur(crop_img_gray, (tuple(config["gausian_filter"])), 0)
            
                          # Dtect the edges
                                 edges = cv2.Canny(image=crop_img_blur, threshold1 = config["threshold_1"], threshold2= config["threshold_2"],apertureSize= config["aperture_size"],L2gradient=config["l2_gradient"])
                              
                          # Gt width and height of image
                                 height, width  =  edges.shape
            
                          #Ge number of egde/white pixelS
                                 number_of_white_pix = np.sum(edges == 255)
                                 # print("Number of white pixels :", number_of_white_pix)
                                 # print("Ratio : ", "{:.2f}".format(number_of_white_pix/(height*width)))
                                 if 'motion' in image:
                                      # return(f"{image}"," ", number_of_white_pix," ", height," ", width," ", ("{:.2f}".format(number_of_white_pix/(height*width))), " ", 1)
                                       lst.append(([f"{image}_{i}",f"{name}",number_of_white_pix,height,width,(number_of_white_pix/(height*width)),1]))
                              
                                 else:
                                       lst.append(([f"{image}_{i}",f"{name}",number_of_white_pix,height,width,(number_of_white_pix/(height*width)),0]))
       df = DataFrame(lst,columns=header)
       evaluate_canny(df,args,config)


def evaluate_canny(dataframe,args,config):

       set_ratio = 0.01

       result_lst = []
       result_headers = ["Set Ratio","Recall","Precision","Accuracy","F1_Score"]


       for i in range(11):
           pred_list = []
           for value in dataframe["ratio"]:
              if  value <= set_ratio:
                   pred_list.append(1)
              else:
                   pred_list.append(0)
           dataframe2 = dataframe.assign(prediction = pred_list)

       
       #Total Rows 
       #     TOTAL_ROWS = dataframe2.shape[0]
       #     print(set_ratio)
           recall = sklearn.metrics.recall_score(dataframe2["Blurred?"], dataframe2["prediction"])
           precision = sklearn.metrics.precision_score(dataframe2["Blurred?"], dataframe2["prediction"])
           accuracy = sklearn.metrics.accuracy_score(dataframe2["Blurred?"], dataframe2["prediction"])
           f1_score = sklearn.metrics.f1_score(dataframe2["Blurred?"], dataframe2["prediction"])


           result_lst.append([set_ratio,recall,precision,accuracy,f1_score])

      
           set_ratio +=0.028
           set_ratio = round(set_ratio,3)
           print("\n")

       df3 = DataFrame(result_lst,columns=result_headers)
       temp = config["model_type"]
       thres1 = config["threshold_1"]
       thres2 = config["threshold_2"]
       gaus = config["gausian_filter"][0]
       aperture = config["aperture_size"]
       l2 = config["l2_gradient"]

       output_path = args.output_dir

       # os.path.join(output_path,f'{temp}/{thres1}/{thres2}/{gaus}/{aperture}/{l2}/eval.csv')

       df3.to_csv(f'{temp}_{thres1}_{thres2}_{gaus}_{aperture}_{l2}.csv',index=True, columns=result_headers)
