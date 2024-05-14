import streamlit as st
import cv2
import numpy as np
import time
import tempfile

# Load pre-trained model
@st.cache_resource
def load_model():
    net = cv2.dnn_DetectionModel('ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt', 'frozen_inference_graph.pb')
    return net

@st.cache_data
def serialize_model(model):
    return pickle.dumps(model)

@st.cache_data
def deserialize_model(serialized_model):
    return pickle.loads(serialized_model)

def detect_objects(frame, net):
    class_names = []
    with open('coco.names', 'rt') as f:
        class_names = f.read().rstrip('\n').split('\n')

    net.setInputSize(320, 320)
    net.setInputScale(1.0 / 127.5)
    net.setInputMean((127.5, 127.5, 127.5))
    net.setInputSwapRB(True)

    class_ids, confidences, bbox = net.detect(frame, confThreshold=0.5)
    
    if len(class_ids) > 0:
        for class_id, confidence, box in zip(class_ids.flatten(), confidences.flatten(), bbox):
            cv2.rectangle(frame, box, color=(0,255,0), thickness=4)
            cv2.putText(frame, class_names[class_id-1].upper(), (box[0]+10, box[1]+30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    return frame

def main():
    st.title("Object Detection Web App")
    st.sidebar.title("Detection")

    video_option = st.sidebar.radio("Choose video source:", ("Upload a video", "Webcam"))

    if video_option == "Upload a video":
        uploaded_file = st.sidebar.file_uploader("Upload a video", type=["mp4"])

        if uploaded_file is not None:
            with tempfile.NamedTemporaryFile(delete=False) as tmpfile:
                tmpfile.write(uploaded_file.read())

            video_file_path = tmpfile.name
            st.sidebar.video(video_file_path)

            net = load_model()

            if st.sidebar.button("Detect"):
                st.markdown("## Processing Video...")
                vid = cv2.VideoCapture(video_file_path)
                stframe = st.empty()
                while vid.isOpened():
                    ret, frame = vid.read()
                    if not ret:
                        break
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    detected_frame = detect_objects(frame, net)
                    stframe.image(detected_frame, channels="RGB")
                    time.sleep(0)

                vid.release()
                os.unlink(video_file_path)  # Delete the temporary file after processing

        else:
            st.write("Please upload a video file.")

    elif video_option == "Webcam":
        st.sidebar.markdown("## Using Webcam")
        st.sidebar.write("Press 'Start' to begin detection.")

        if st.sidebar.button("Start"):
            net = load_model()
            st.markdown("## Processing Webcam Feed...")
            vid = cv2.VideoCapture(0)
            stframe = st.empty()
            while vid.isOpened():
                ret, frame = vid.read()
                if not ret:
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                detected_frame = detect_objects(frame, net)
                stframe.image(detected_frame, channels="RGB")
                time.sleep(0.1)

            vid.release()

if __name__ == "__main__":
    main()
