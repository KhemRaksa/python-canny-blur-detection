import cv2
font = cv2.FONT_HERSHEY_SIMPLEX
import numpy as np


set_ratio = 0.206

cap = cv2.VideoCapture(0)
cap.set(3,1280)

# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam")

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    crop_img = frame[80:270, 180:480]

    # img = cv2.imread(Path to your image)
    cv2.rectangle(frame, (180, 80), (480,270), (255,0,0), thickness=5)

    # Crop Image from the rectangle mok dak jol canny
    crop_img = cv2.resize(crop_img, (224, 224))       

    # Canny Algorithm
    crop_img_blur = cv2.GaussianBlur(crop_img, (3,3), 0)

    crop_img_gray = cv2.cvtColor(crop_img_blur, cv2.COLOR_BGR2GRAY)
    # Aply Gaussian Blur
    edges = cv2.Canny(crop_img_gray, threshold1 = 85, threshold2= 255,apertureSize= 7,L2gradient=1)

    height, width = edges.shape
    
    number_of_white_pix = np.sum(edges != 0)
    ratio = (number_of_white_pix / (height * width)) 
    # ratio = 0.3

    print(ratio)

    if ratio < set_ratio:
        cv2.putText(frame, 
                'Blurred', 
                (250, 50), 
                font, 1, 
                (0, 0, 255), 
                2, 
                cv2.LINE_4)
    else:
        cv2.putText(frame, 
                'OKAY', 
                (250, 50), 
                font, 1, 
                (0, 255, 255), 
                2, 
                cv2.LINE_4)
    # Evaluate
    cv2.imshow('Canny Blur Detection', frame)
    cv2.imshow('Crop', crop_img)
    cv2.imshow('Edges', edges)
    c = cv2.waitKey(1)
    # Escape Key
    if c == 27:
        break

cap.release()
cv2.destroyAllWindows()