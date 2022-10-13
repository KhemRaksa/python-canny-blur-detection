from pkgutil import walk_packages
import cv2
import matplotlib.pyplot as plt
import json
import numpy as np
from skimage import data, io, filters
backSub_KNN = cv2.createBackgroundSubtractorKNN(detectShadows=False)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
backSub_MOG2 = cv2.createBackgroundSubtractorMOG2(detectShadows=False)


set_ratio = 0.2




# img = cv2.imread('C:/Users/user/Desktop/test_images/non_blur1.jpg')
# img = cv2.imread('C:/Users/user/Desktop/test_images/blurred_4.png')
# img = cv2.imread('C:/Users/user/Desktop/test_images/blurred_2.jpg')
# img = cv2.imread('C:/Users/user/Desktop/test_images/blurred_1.jpg')

# Create mask to only select black




cv2.imshow("Original image : ",img)

# f = open('C:/Users/user/Desktop/test_images/non_blur1.json')
# f = open('C:/Users/user/Desktop/test_images/blurred_4.json')
# f = open('C:/Users/user/Desktop/test_images/blurred_2.json')
# f = open('C:/Users/user/Desktop/test_images/blurred_1.json')


data = json.load(f)
f.close()

input_points = data['shapes'][0]['points']
print(input_points)
# print(type(input_points))

# input_points = np.float32([top_left_point,top_right_point,bottom_right_point,bottom_left_point])
input_points = np.float32(input_points)
output_points = np.float32([
                                [0,0],
                                    [224,0],
                                    [224,224],
                                    [0,224]]
                                 )
M = cv2.getPerspectiveTransform(input_points,output_points)
# print(M)

warped_image = cv2.warpPerspective(img,M,(224,224),flags=cv2.INTER_LINEAR)
cv2.imshow('Warped Image', warped_image)

crop_img_blur = cv2.GaussianBlur(warped_image, (5,5),0)


crop_img_gray = cv2.cvtColor(crop_img_blur, cv2.COLOR_BGR2GRAY)


# Automatic Thresholding
t = filters.threshold_otsu(crop_img_gray)
print(t)
mask = crop_img_gray > t
mask = np.logical_not(mask)

mask = np.array(mask,dtype=np.float64) + 0
print(mask)
print(mask.shape)
# mask = backSub_MOG2.apply(warped_image)
# print(np.sum(mask != 255))
# mask = 255 - mask
cv2.imshow("Mask", mask)

# Dtect the edges
edges = cv2.Canny(image=crop_img_gray, threshold1 = 100, threshold2= 200,apertureSize= 7,L2gradient=1)
# print(edges.shape)
cv2.imshow('Edge Image ', edges)

final = edges * mask
cv2.imshow("Final ", final)

# Gt width and height of image
height, width  =  final.shape
            
#Ge number of egde/white pixelS
number_of_white_pix = np.sum(final!=0)

ratio = round(number_of_white_pix / (height * width),2)
print(ratio)

if ratio <= set_ratio:
    print("Blurred")
else: 
    print("Not Blurred")


cv2.waitKey(0)
cv2.destroyAllWindows()