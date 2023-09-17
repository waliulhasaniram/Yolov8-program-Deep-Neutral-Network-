from ultralytics import YOLO
import cv2 as cv 

model = YOLO('yolov8x.pt')

img = cv.imread('E:\YOLO_WORKS\images\dogge2.PNG')
 
resize = cv.resize(img, (500,350), interpolation=cv.INTER_AREA)

result = model(resize, show=True)

cv.waitKey(0)
