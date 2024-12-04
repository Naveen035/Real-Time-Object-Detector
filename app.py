import cv2
import numpy as np

thres = 0.45  
nms_threshold = 0.5  
cap = cv2.VideoCapture(0)  

cap.set(3, 1280)  #Width
cap.set(4, 720)  #Height
cap.set(10, 150)  #Brightness

classNames = []
classFile = r'C:\Users\jayas\OneDrive\Desktop\New folder\object_detector\coco.names'  # Path to coco.names

with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

configPath = r'C:\Users\jayas\OneDrive\Desktop\New folder\object_detector\ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'  # Path to config file
weightsPath = r'C:\Users\jayas\OneDrive\Desktop\New folder\object_detector\frozen_inference_graph.pb'  # Path to weights file

net = cv2.dnn_DetectionModel(weightsPath, configPath) #model

net.setInputSize(320, 320) 
net.setInputScale(1.0 / 127.5)  
net.setInputMean((127.5, 127.5, 127.5))  
net.setInputSwapRB(True)  #BGR to RGB

while True:
    success, image = cap.read()

    classIds, confs, bbox = net.detect(image, confThreshold=thres)

    bbox = list(bbox) 
    confs = list(np.array(confs).reshape(1, -1)[0])  
    confs = list(map(float, confs))  

    indicies = cv2.dnn.NMSBoxes(bbox, confs, thres, nms_threshold)

    for i in indicies:
        if isinstance(i, np.ndarray):
            i = i[0]  
        box = bbox[i]
        x, y, w, h = box[0], box[1], box[2], box[3]

        class_id = classIds[i]
        cv2.rectangle(image, (x, y), (x + w, y + h), color=(0, 255, 0), thickness=2) #Bounding boxes
        cv2.putText(image, classNames[class_id - 1], (box[0] + 10, box[1] + 30),
                    cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Output", image) #Display Image

    key = cv2.waitKey(1)  
    if key == 27:
        print("ESC key pressed. Exiting...")
        break  

cap.release()
cv2.destroyAllWindows()
