import cv2
import torch
from tracker import *
import numpy as np
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

cap=cv2.VideoCapture('cctv.mp4')

def POINTS(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE :  
        colorsBGR = [x, y]
        print(colorsBGR)
        

cv2.namedWindow('FRAME')
cv2.setMouseCallback('FRAME', POINTS)

area_1 = [(369, 317), (411, 370), (530,335), (478,287)]



cy1=225
offset=6
counter=0
tracker = Tracker()
area1=set()
while True:
    ret,frame=cap.read()
    frame=cv2.resize(frame,(1020,500))
    cv2.polylines(frame, [np.array(area_1, np.int32)], True, (15,220,10), 3)

    results=model(frame)
    detections=[]
    results.pandas().xyxy[0]
    for index,row in results.pandas().xyxy[0].iterrows():
         x1 = int(row['xmin'])
         y1 = int(row['ymin'])
         x2 = int(row['xmax'])
         y2 = int(row['ymax'])
         b=str(row['name'])
         if "person" in b:
             detections.append([x1, y1, x2, y2])
    boxes_ids = tracker.update(detections)
    for box_id in boxes_ids:
        x, y, w, h, id = box_id
        cv2.putText(frame, str(id), (x, y - 15), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0,255), 2)
        cv2.rectangle(frame, (x, y), (w, h), (255, 255, 0), 2)
        result = cv2.pointPolygonTest(np.array(area_1, np.int32), (int(w), int(h)), False)
        if result >= 0:
           area1.add(id)
           cv2.polylines(frame, [np.array(area_1, np.int32)], True, (255,0,255), 3)
    print(area1)       
    cv2.imshow('FRAME',frame)
    if cv2.waitKey(1)&0xFF==27:
        break
cap.release()
cv2.destroyAllWindows()
    
    