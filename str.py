import cv2
import numpy as np
import streamlit as st

st.set_page_config(page_title="Real-Time Object Detection", page_icon="🕵️", layout="wide")

page_bg = """
<style>
body {
    background-color: #f0f0f5; 
    color: #2c3e50;
    font-family: Arial, sans-serif;
}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)


st.sidebar.title("🔧 Settings")
thres = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.45)
nms_threshold = st.sidebar.slider("NMS Threshold", 0.1, 1.0, 0.5)
display_detection_info = st.sidebar.checkbox("Show Detection Info", value=True)

st.sidebar.markdown("---")
st.sidebar.markdown("👨‍💻 **Developed by Naveen Kumar**")
st.sidebar.markdown("💻 Powered by OpenCV & Streamlit")

# Title and Description
st.title("🕵️ Real-Time Object Detection")
st.markdown("""
Welcome to the **Real-Time Object Detection** app!  
This application detects objects in a live webcam feed using a pre-trained SSD MobileNet model.  
Use the settings on the left to adjust detection parameters.
""")

# Load the Classes
classFile = r'C:\Users\jayas\OneDrive\Desktop\New folder\object_detector\coco.names'
with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

# Model Configurations
configPath = r'C:\Users\jayas\OneDrive\Desktop\New folder\object_detector\ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = r'C:\Users\jayas\OneDrive\Desktop\New folder\object_detector\frozen_inference_graph.pb'
net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

# Start Webcam Button
start_detection = st.button("🎥 Start Webcam Detection")

if start_detection:
    st.warning("Press 'ESC' in the webcam window to stop the detection.")

    # Open Webcam
    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)  
    cap.set(4, 720)   
    cap.set(10, 150)  

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


            cv2.rectangle(image, (x, y), (x + w, y + h), color=(0, 255, 0), thickness=2)
            cv2.putText(image, f"{classNames[class_id - 1]} {confs[i]*100:.1f}%", 
                        (box[0] + 10, box[1] + 30),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)


            if display_detection_info:
                st.markdown(f"🟢 **Detected:** {classNames[class_id - 1]} - {confs[i]*100:.1f}%")


        cv2.imshow("Real-Time Object Detector", image)


        key = cv2.waitKey(1)
        if key == 27:  
            st.success("Detection Stopped. Webcam released.")
            break

    cap.release()
    cv2.destroyAllWindows()
