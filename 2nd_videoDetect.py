from ultralytics import YOLO
import cv2 as cv 
import cvzone
import math

classes = ['person', 'bicycle', 'car', 'motorbike', 'aeroplane', 'bus', 'train,' 'truck', 'boat',
           'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
           'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
           'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
            'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl','banana', 'apple',' sandwich', 
            'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
            'chair', 'sofa', 'pottedplant', 'bed', 'diningtable', 'toilet', 'tvmonitor', 'laptop', 'mouse', 
            'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
            'scissors', 'teddy bear', 'hair drier', 'toothbrush']

capture = cv.VideoCapture(r'videos\WqOnIZUB.mp4')

capture.set(3, 1280)
capture.set(4, 720)

model = YOLO('yolov8x.pt')

while True:
    isTrue, imge = capture.read()
    
    results = model(imge, stream=True)
    
    for r in results:
        boxes = r.boxes
        for box in boxes:

            #rerctangle
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            print(x1,y1,x2,y2) 
            cv.rectangle(imge, (x2,y2), (x1,y1), (0,0,255),3)

            #confidence
            conf = math.ceil(box.conf[0]*100)/100

            #class name
            cls = int(box.cls[0])

            cvzone.putTextRect(imge, f'{classes[cls]}  {conf} %', (max(0, x1), max(35, y1)))

    if isTrue:
        cv.imshow('video', imge)
        if cv.waitKey(20) & 0xff==ord('d'):
            break
    else:
        break
capture.release()
cv.destroyAllWindows()

