# 🕵️ Real-Time Object Detection with Streamlit and OpenCV

This project implements a **real-time object detection** system using a pre-trained **SSD MobileNet** model and displays results through an interactive **Streamlit interface**. The app allows you to detect objects from a live webcam feed and customize detection parameters like confidence threshold and NMS threshold dynamically.

---

## 🚀 Features
- **Real-Time Object Detection**: Uses a pre-trained SSD MobileNet model to detect objects from the webcam feed.
- **Streamlit Interface**:
  - Adjust detection parameters (confidence and NMS thresholds).
  - Toggle display of detection information.
  - Emoji-powered and visually appealing UI.
- **Model**: SSD MobileNet V3 trained on COCO dataset.
- **Dynamic Settings**: Change thresholds without restarting the app.

---

## 🛠️ Installation

Follow the steps below to set up the project on your local machine:

### Prerequisites
- Python 3.7 or higher
- Libraries: `streamlit`, `opencv-python-headless`, `numpy`

### Installation Steps
1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/object-detection-app.git
   cd object-detection-app
2. Install the required libraries:
   ```bash
   pip install -r requirements.txt
3. Ensure the following files are in the `object_detector` folder:
   - `coco.names`: Class labels for the COCO dataset.
   - `ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt`: Model configuration file.
   - `frozen_inference_graph.pb`: Pre-trained weights file.

---

### ⚙️ Usage

1. Run the Streamlit app:
   ```bash
   streamlit run app.py
2. The app will open in your default web browser. You can:
   - **Start the webcam feed** by clicking **"Start Webcam Detection"**.
   - **Adjust thresholds** and toggle detection info display from the sidebar.
   - **Press the ESC key** in the webcam window to stop the detection.
### 📊 Results

- **Real-Time Detection**: Detects objects like persons, cars, and animals using your webcam.
- **Visual Output**: Displays bounding boxes, class names, and confidence scores on the detected objects.
  
### 🤝 Contributing

We welcome contributions! Feel free to:
- Submit issues to report bugs or suggest new features.
- Create pull requests to improve this project.

Your feedback and contributions are greatly appreciated!

### 👨‍💻 Author

Developed by **Naveen Kumar**  
💻 Powered by **OpenCV** and **Streamlit**  
🌟 If you found this project helpful, please give it a star on GitHub!

### ✨ Acknowledgments

- **COCO Dataset**: Pre-trained model used in this project.
- **Streamlit**: For providing an amazing framework for building web apps.

