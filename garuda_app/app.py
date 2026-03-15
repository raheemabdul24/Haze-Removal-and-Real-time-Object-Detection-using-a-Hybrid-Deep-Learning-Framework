import streamlit as st
import cv2
import numpy as np
from model_utils import load_model, enhance_image

st.set_page_config(
    page_title="All Weather Image Enhancement",
    layout="wide"
)

st.title("All Weather Surveillance Enhancement System")

with st.spinner("Loading Enhancement Model..."):
    model = load_model()

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
            original, enhanced = enhance_image(model,img)

            col1,col2 = st.columns(2)

            col1.image(original,channels="BGR",caption="Original Image")
            col2.image(enhanced,channels="BGR",caption="Enhanced Image")


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
            frame_placeholder = st.empty()

            while cap.isOpened():

                ret,frame = cap.read()

                if not ret:
                    break

                original, enhanced = enhance_image(model,frame)

                combined = np.hstack((original,enhanced))

                frame_placeholder.image(
                    combined,
                    channels="BGR",
                    caption="Left: Original | Right: Enhanced"
                )

            cap.release()


# WEBCAM
elif option == "Webcam":

    run = st.checkbox("Start Webcam")

    frame_window = st.image([])

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        st.error("Error: Could not access webcam. Please check if your webcam is connected and permissions are granted.")
    else:
        while run:

            ret,frame = cap.read()

            if not ret:
                break

            original, enhanced = enhance_image(model,frame)

            combined = np.hstack((original,enhanced))

            frame_window.image(
                combined,
                channels="BGR",
                caption="Left: Original | Right: Enhanced"
            )

        cap.release()
