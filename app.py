import base64
import os

import cv2
import numpy as np
import streamlit as st
from PIL import Image

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

class SkinDetector:
    def detect_skin(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_skin = np.array([0, 48, 80], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower_skin, upper_skin)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=2)
        return mask

class DeepfakeDetector:
    def __init__(self):
        self.skin_detector = SkinDetector()

    def detect_deepfake_image(self, file_data):
        file_bytes = np.asarray(bytearray(file_data.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if img is None:
            return None, False
        faces = face_cascade.detectMultiScale(img, 1.1, 5, minSize=(30, 30))
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        skin_mask = self.skin_detector.detect_skin(img)
        img_with_skin = cv2.bitwise_and(img, img, mask=skin_mask)
        return cv2.resize(img_with_skin, (100, 100)), len(faces) > 0

    def detect_deepfake_video(self, file_data, frames_per_second):
        is_real = True
        video_bytes = file_data.read()
        temp_filename = "temp_video.mp4"
        with open(temp_filename, "wb") as f:
            f.write(video_bytes)
        cap = cv2.VideoCapture(temp_filename)
        if not cap.isOpened():
            return [], False, 0, 0

        frame_rate = cap.get(cv2.CAP_PROP_FPS)
        skip_frames = max(1, int(frame_rate / frames_per_second))

        total_frames = 0
        detected_fake_frames = 0
        real_frames = 0
        fake_frames = 0
        processed_frames = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            total_frames += 1
            if total_frames % skip_frames != 0:
                continue

            faces = face_cascade.detectMultiScale(frame, 1.1, 5, minSize=(30, 30))
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            skin_mask = self.skin_detector.detect_skin(frame)
            frame_with_skin = cv2.bitwise_and(frame, frame, mask=skin_mask)

            if len(faces) == 0:
                detected_fake_frames += 1
                fake_frames += 1
            else:
                real_frames += 1

            processed_frames.append((cv2.resize(frame_with_skin, (100, 100)), len(faces) > 0))

        cap.release()
        os.remove(temp_filename)
        if detected_fake_frames >= total_frames // 3:
            is_real = False
        return processed_frames, is_real, real_frames, fake_frames

def main():
    detector = DeepfakeDetector()
    st.set_page_config(page_title="Deepfake Detection", layout="wide")
    st.title("🕵️‍♂️ NTN Deepfake Detection")
    st.sidebar.title("Upload Input")

    # Custom CSS to enlarge uploaders
    st.markdown("""
        <style>
        .stFileUploader label {
            font-size: 20px;
            font-weight: bold;
        }
        .stFileUploader {
            padding: 10px;
            font-size: 18px;
        }
        </style>
    """, unsafe_allow_html=True)

    detection_mode = st.sidebar.selectbox("Select Mode", ("Image", "Video"))

    if detection_mode == "Image":
        st.subheader("Upload an Image")
        image_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
        if image_file and st.button("Detect"):
            with st.spinner("Detecting..."):
                result, is_real = detector.detect_deepfake_image(image_file)
                if result is None:
                    st.error("Could not read the image file.")
                else:
                    st.image(result, caption="Processed Image", channels="BGR", use_container_width=True)
                    st.markdown(f"<h1 style='text-align: center; color: {'green' if is_real else 'red'};'>Final Prediction: {'Real' if is_real else 'Fake'}</h1>", unsafe_allow_html=True)

    else:
        st.subheader("Upload a Video")
        video_file = st.file_uploader("Choose a video", type=["mp4", "avi"])
        frames_per_second = st.sidebar.slider("Analysis FPS", 1, 30, 15)
        if video_file and st.button("Detect"):
            with st.spinner("Processing video..."):
                frames, is_real, real_frames, fake_frames = detector.detect_deepfake_video(video_file, frames_per_second)

                st.write(f"### Total Real Frames: {real_frames}")
                st.write(f"### Total Fake Frames: {fake_frames}")

                rows = st.columns(4)
                for i, (frame, is_frame_real) in enumerate(frames):
                    rows[i % 4].image(frame, caption=f"Frame {i+1}: {'Real' if is_frame_real else 'Fake'}", channels="BGR", use_container_width=True)

                final_color = "green" if is_real else "red"
                final_label = "Real" if is_real else "Fake"
                st.markdown(f"<h1 style='text-align: center; color: {final_color};'>Final Prediction: {final_label}</h1>", unsafe_allow_html=True)

    def sidebar_image(title, filename):
        with open(filename, "rb") as f:
            data = base64.b64encode(f.read()).decode("utf-8")
        st.sidebar.header(title)
        st.sidebar.markdown(
            f"""<div style="display:table;"><img src="data:image/png;base64,{data}" width="300" height="200"></div>""",
            unsafe_allow_html=True,
        )

    sidebar_image("EPOCH VS ACCURACY", "epoch_accuracy.png")
    sidebar_image("PERFORMANCE METRICS ACROSS FOLDS", "performance_metrics.png")

    def get_base64_of_file(path):
        with open(path, 'rb') as f:
            return base64.b64encode(f.read()).decode("utf-8")

    bg_img = get_base64_of_file("background.jpeg")
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{bg_img}");
            background-size: cover;
        }}
        .custom-footer {{
            position: fixed;
            right: 10px;
            bottom: 10px;
            z-index: 9999;
            background-color: rgba(10, 0, 90, 0.8);
            color: white;
            padding: 8px 14px;
            border: 5px solid red;
            border-radius: 15px;
            font-size: 20px;
            font-weight: bold;
        }}
        .custom-footer .ntn {{
            color: red;
            font-weight: bold;
        }}
        </style>

        <div class="custom-footer">
            Developed By <span class="ntn">NTN ❤️</span>
        </div>
        """, unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
