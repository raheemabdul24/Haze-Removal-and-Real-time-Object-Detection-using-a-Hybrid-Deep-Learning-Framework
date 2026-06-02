import streamlit as st
import cv2
import numpy as np
import pandas as pd
import os
import subprocess
import sys
from model_utils import load_model, load_yolo, load_tracker, enhance_image


def _running_under_streamlit() -> bool:
    try:
        import streamlit.runtime as streamlit_runtime

        return streamlit_runtime.exists()
    except Exception:
        return False


if __name__ == "__main__" and not _running_under_streamlit():
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", os.path.abspath(__file__), *sys.argv[1:]],
        cwd=os.path.dirname(os.path.abspath(__file__)),
        check=False,
    )
    raise SystemExit(0)

st.set_page_config(
    page_title="All Weather Image Enhancement",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced styling
st.markdown("""
<style>
    /* Main title styling */
    h1 {
        color: #1f77b4;
        text-align: center;
        font-size: 2.8em;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Subtitle styling */
    .subtitle {
        text-align: center;
        color: #555;
        font-size: 1.1em;
        margin-bottom: 2rem;
    }
    
    /* Section headers */
    h2 {
        color: #1f77b4;
        border-bottom: 3px solid #1f77b4;
        padding-bottom: 10px;
    }
    
    h3 {
        color: #0d5f8f;
    }
    
    /* Metric styling */
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f5f5f5 0%, #e8e8e8 100%);
    }
    
    /* Dataframe styling */
    .dataframe {
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* File uploader styling */
    [data-testid="fileUploadWidget"] {
        border: 2px dashed #1f77b4;
        border-radius: 10px;
        padding: 20px;
    }
    
    /* Button styling */
    button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 8px;
        border: none;
        padding: 10px 20px;
        font-weight: bold;
        transition: transform 0.2s;
    }
    
    button:hover {
        transform: scale(1.05);
    }
</style>
""", unsafe_allow_html=True)

# Header Section
st.markdown("# 🎥 All Weather Surveillance Enhancement System", unsafe_allow_html=True)
st.markdown("""
<div class="subtitle">
    Advanced AI-Powered Real-Time Enhancement & Object Detection
</div>
""", unsafe_allow_html=True)

with st.spinner("🔄 Loading AI Models..."):
    model = load_model()
    yolo_model = load_yolo()
    tracker = load_tracker()

# Sidebar with better styling
st.sidebar.markdown("## ⚙️ Configuration")
option = st.sidebar.selectbox(
    "📊 Select Input Source",
    ["📷 Image Upload", "🎬 Video Upload", "🎥 Webcam"],
    index=0
)

# IMAGE
if "Image Upload" in option:
    st.markdown("## 📷 Image Enhancement & Analysis")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png", "bmp", "tiff"])
    
    with col2:
        st.info("✓ Supported formats: JPG, PNG, BMP, TIFF")

    if file:
        file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)

        if img is None:
            st.error("❌ Error: Could not read image file. Please upload a valid image (JPG, PNG, etc.)")
        else:
            with st.spinner("🔍 Processing image..."):
                counted_ids = set()
                original, enhanced, detected, detections, counted_ids, p_count, v_count = enhance_image(model, yolo_model, tracker, img, counted_ids)

            # Display results with better layout
            st.markdown("### 📊 Processing Results")
            
            col1, col2, col3 = st.columns(3)

            with col1:
                st.image(original, channels="BGR", use_column_width=True)
                st.markdown("**Original Image**")
            
            with col2:
                st.image(enhanced, channels="BGR", use_column_width=True)
                st.markdown("**Enhanced (Dehazing)**")
            
            with col3:
                st.image(detected, channels="BGR", use_column_width=True)
                st.markdown("**Object Detection**")
            
            # Statistics and detection results
            st.markdown("---")
            
            stat_col1, stat_col2 = st.columns(2)
            
            with stat_col1:
                st.metric("👥 People Detected", p_count, delta=None)
            
            with stat_col2:
                st.metric("🚗 Vehicles Detected", v_count, delta=None)
            
            if detections:
                st.markdown("### 📋 Detected Objects Details")
                df = pd.DataFrame(detections)
                df = df.sort_values(by="Confidence", ascending=False)
                
                # Style the dataframe
                st.dataframe(
                    df.style.format({"Confidence": "{:.2%}"}, subset=["Confidence"] if "Confidence" in df.columns else []),
                    use_container_width=True
                )
                
                st.success(f"✓ Total objects detected: **{len(df)}**")
            else:
                st.warning("⚠️ No objects detected in this image")

# VIDEO
elif "Video Upload" in option:
    st.markdown("## 🎬 Video Enhancement & Tracking Analysis")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        video = st.file_uploader("Upload Video", type=["mp4", "avi", "mov", "mkv"])
    
    with col2:
        st.info("✓ Supported formats: MP4, AVI, MOV, MKV")

    if video:
        tfile = open("temp.mp4", "wb")
        tfile.write(video.read())

        cap = cv2.VideoCapture("temp.mp4")

        if not cap.isOpened():
            st.error("❌ Error: Could not read video file. Please upload a valid video (MP4, AVI, etc.)")
        else:
            st.markdown("### 🔄 Processing Video...")
            
            counted_ids = set()
            total_people = 0
            total_vehicles = 0
            
            progress_bar = st.progress(0)
            frame_placeholder = st.empty()
            status_text = st.empty()
            
            frame_count = 0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            while cap.isOpened():
                ret, frame = cap.read()

                if not ret:
                    break

                original, enhanced, detected, detections, counted_ids, p_count, v_count = enhance_image(model, yolo_model, tracker, frame, counted_ids)

                total_people += p_count
                total_vehicles += v_count

                combined = np.hstack((original, enhanced, detected))

                frame_count += 1
                progress = min(frame_count / max(total_frames, 1), 1.0)
                progress_bar.progress(progress)
                status_text.text(f"Processing frame {frame_count} of {total_frames}...")

                frame_placeholder.image(
                    combined,
                    channels="BGR",
                    caption="Left: Original | Center: Enhanced | Right: Detection"
                )

            cap.release()
            progress_bar.empty()
            status_text.empty()
            
            st.success("✓ Video processing completed!")
            
            st.markdown("---")
            st.markdown("### 📊 Surveillance Analytics")
            
            anal_col1, anal_col2 = st.columns(2)
            
            with anal_col1:
                st.metric("👥 Total People Detected", total_people)
            
            with anal_col2:
                st.metric("🚗 Total Vehicles Detected", total_vehicles)
            
            # Display summary
            summary_col1, summary_col2, summary_col3 = st.columns(3)
            
            with summary_col1:
                st.info(f"**Total Frames:** {frame_count}")
            
            with summary_col2:
                st.info(f"**Unique Persons:** {len([id for id in counted_ids if 'person' in str(id).lower()])}")
            
            with summary_col3:
                if total_people > 0:
                    avg_per_frame = total_people / frame_count if frame_count > 0 else 0
                    st.info(f"**Avg People/Frame:** {avg_per_frame:.2f}")


# WEBCAM
elif "Webcam" in option:
    st.markdown("## 🎥 Real-Time Webcam Monitoring")
    
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        run = st.toggle("▶️ Start Live Feed", value=False)
    
    with col2:
        st.info("✓ Real-time processing enabled")
    
    with col3:
        st.warning("⚠️ Click when ready")

    frame_window = st.image([])

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        st.error("❌ Error: Could not access webcam. Please check if your webcam is connected and permissions are granted.")
    else:
        counted_ids = set()
        total_people = 0
        total_vehicles = 0
        frame_window = st.image([])
        
        analytics_placeholder = st.empty()
        frame_counter = st.empty()

        while run:
            ret, frame = cap.read()

            if not ret:
                st.warning("⚠️ Failed to read frame from webcam")
                break

            original, enhanced, detected, detections, counted_ids, p_count, v_count = enhance_image(model, yolo_model, tracker, frame, counted_ids)

            total_people += p_count
            total_vehicles += v_count

            combined = np.hstack((original, enhanced, detected))

            frame_window.image(
                combined,
                channels="BGR",
                caption="Live: Original | Enhanced | Detection"
            )
            
            with analytics_placeholder.container():
                st.markdown("### 📊 Live Statistics")
                
                metric_col1, metric_col2 = st.columns(2)
                
                with metric_col1:
                    st.metric("👥 People", total_people)
                
                with metric_col2:
                    st.metric("🚗 Vehicles", total_vehicles)
                
                frame_counter.markdown(f"**Frames Processed:** {total_people + total_vehicles}")

        cap.release()
        
        if not run:
            st.info("✓ Webcam feed stopped")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.9em; margin-top: 2rem;'>
    <p>🔬 <strong>All Weather Image Enhancement & Surveillance System</strong></p>
    <p>Powered by YOLOv8 | Deep Learning Enhancement | Real-time Tracking</p>
    <p><em>Advanced AI for weather-resilient surveillance</em></p>
</div>
""", unsafe_allow_html=True)
