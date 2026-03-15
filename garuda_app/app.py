import streamlit as st
import cv2
import numpy as np
import pandas as pd
from model_utils import load_model, load_yolo, load_tracker, enhance_image

st.set_page_config(
    page_title="All Weather Image Enhancement",
    layout="wide"
)

st.title("All Weather Surveillance Enhancement System")

with st.spinner("Loading models..."):
    model = load_model()
    yolo_model = load_yolo()
    tracker = load_tracker()

option = st.sidebar.selectbox(
    "Select Input Source",
    ["Image Upload", "Video Upload", "Webcam"]
)

# IMAGE
if option == "Image Upload":

    file = st.file_uploader("Upload Image")

    if file:

        file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes,1)

        if img is None:
            st.error("Error: Could not read image file. Please upload a valid image (JPG, PNG, etc.)")
        else:
            counted_ids = set()
            original, enhanced, detected, detections, counted_ids, p_count, v_count = enhance_image(model,yolo_model,tracker,img,counted_ids)

            col1,col2,col3 = st.columns(3)

            col1.image(original,channels="BGR",caption="Original")
            col2.image(enhanced,channels="BGR",caption="Dehazed")
            col3.image(detected,channels="BGR",caption="Object Detection")
            if detections:
                df = pd.DataFrame(detections)
                df = df.sort_values(by="Confidence", ascending=False)
                st.subheader("Detected Objects")
                st.dataframe(df)
                st.write("Total objects detected:", len(df))
            else:
                st.write("No objects detected")

# VIDEO
elif option == "Video Upload":

    video = st.file_uploader("Upload Video")

    if video:

        tfile = open("temp.mp4","wb")
        tfile.write(video.read())

        cap = cv2.VideoCapture("temp.mp4")

        if not cap.isOpened():
            st.error("Error: Could not read video file. Please upload a valid video (MP4, AVI, etc.)")
        else:
            counted_ids = set()
            total_people = 0
            total_vehicles = 0
            frame_placeholder = st.empty()

            while cap.isOpened():

                ret,frame = cap.read()

                if not ret:
                    break

                original, enhanced, detected, detections, counted_ids, p_count, v_count = enhance_image(model,yolo_model,tracker,frame,counted_ids)

                total_people += p_count
                total_vehicles += v_count

                combined = np.hstack((original, enhanced, detected))

                frame_placeholder.image(
                    combined,
                    channels="BGR",
                    caption="Left: Original | Center: Dehazed | Right: Detection"
                )

            cap.release()
            
            st.subheader("Surveillance Analytics")
            col1,col2 = st.columns(2)
            col1.metric("Total People Detected", total_people)
            col2.metric("Total Vehicles Detected", total_vehicles)


# WEBCAM
elif option == "Webcam":

    run = st.checkbox("Start Webcam")

    frame_window = st.image([])

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        st.error("Error: Could not access webcam. Please check if your webcam is connected and permissions are granted.")
    else:
        counted_ids = set()
        total_people = 0
        total_vehicles = 0
        frame_window = st.image([])
        analytics_placeholder = st.empty()

        while run:

            ret,frame = cap.read()

            if not ret:
                break

            original, enhanced, detected, detections, counted_ids, p_count, v_count = enhance_image(model,yolo_model,tracker,frame,counted_ids)

            total_people += p_count
            total_vehicles += v_count

            combined = np.hstack((original, enhanced, detected))

            frame_window.image(
                combined,
                channels="BGR",
                caption="Left: Original | Center: Dehazed | Right: Detection"
            )
            
            with analytics_placeholder.container():
                col1,col2 = st.columns(2)
                col1.metric("Total People", total_people)
                col2.metric("Total Vehicles", total_vehicles)

        cap.release()
