from io import BytesIO
import os
import cv2
import uuid
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from ultralytics import YOLO
from datetime import timedelta


with st.spinner("Loading YOLO v8n model..."):
    model = YOLO("yolov8n.pt")


def process_image(image):
    results = model(image)
    people = []
    for result in results:
        for box in result.boxes:
            if box.cls == 0 and box.conf > 0.5:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                people.append((x1, y1, x2 - x1, y2 - y1))

    # Draw bounding boxes with labels
    for i, (x, y, w, h) in enumerate(people):
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 5)
        cv2.putText(
            image,
            f"Person {i+1}",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (36, 255, 12),
            2,
        )

    return image, people


def delete_temp_files(temp_folder):
    if os.path.exists(temp_folder):
        for filename in os.listdir(temp_folder):
            file_path = os.path.join(temp_folder, filename)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
            except Exception as e:
                print(f"Error deleting file: {file_path} - {e}")
    else:
        print(f"Folder not found: {temp_folder}")


def process_video(video_bytes):
    # TODO: Use facial recognition to track people across frames/Make traciing  accurate
    #       Now the detected person is wrongly tracked across frames
    temp_folder = "temp/"
    if not os.path.exists(temp_folder):
        os.makedirs(temp_folder)

    input_buffer = BytesIO(video_bytes)
    input_buffer.seek(0)
    file_id = str(uuid.uuid4())
    tmp_input_file = f"{temp_folder}{file_id}_input_video.mp4"
    with open(tmp_input_file, "wb") as f:
        f.write(input_buffer.read())

    cap = cv2.VideoCapture(tmp_input_file)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*"avc1")  # H.264 codec
    tmp_output_file = f"{temp_folder}{file_id}_output_video.mp4"
    out = cv2.VideoWriter(tmp_output_file, fourcc, fps, (width, height))

    frame_number = 0
    frame_rate = int(fps)
    detection_data = {}

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_number += 1
        annotated_frame, people = process_image(frame)

        for i, (x, y, w, h) in enumerate(people):
            person_id = i + 1
            if person_id not in detection_data:
                detection_data[person_id] = {
                    "Person ID": person_id,
                    "Total Time Appeared (s)": 0,
                    "Frames Appeared": [],
                }
            detection_data[person_id]["Total Time Appeared (s)"] += 1 / frame_rate
            detection_data[person_id]["Frames Appeared"].append(frame_number)

        out.write(annotated_frame)

    cap.release()
    out.release()

    video_file = open(tmp_output_file, "rb")
    video_bytes = video_file.read()
    video_file.close()
    delete_temp_files(temp_folder)

    df = pd.DataFrame(detection_data.values())
    df["Total Time Appeared (s)"] = df["Total Time Appeared (s)"].round(2)

    return video_bytes, df


# Streamlit app
st.title("Human Detection 🕵️‍♂️")
st.write("Model: YOLOv8n")
st.write(
    "Annotated images and videos will be generated and you can saved to your local storage. For videos, detection data will be displayed in a table."
)

if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0
if "output" not in st.session_state:
    # [<file_type>, <annotated_image/video>, <detection_data>]
    st.session_state.output = [None, None, None]


def update_key():
    st.session_state.uploader_key += 1


file = st.file_uploader(
    "Upload Image/Video",
    type=["jpg", "jpeg", "png", "mp4"],
    key=f"uploader_{st.session_state.uploader_key}",
)


if file is not None:
    file_type = file.type.split("/")[0]

    if file_type == "image":
        img = Image.open(file)
        image = np.array(img.convert("RGB"))
        annotated_image, people = process_image(image)
        detection_data = [
            {"Person ID": i + 1, "Bounding Box": (x, y, w, h)}
            for i, (x, y, w, h) in enumerate(people)
        ]
        df = pd.DataFrame(detection_data)
        st.session_state.output = [file_type, annotated_image, df]

    elif file_type == "video":
        video_bytes = file.read()
        with st.spinner("Processing the video..."):
            annotated_video, df = process_video(video_bytes)
            st.session_state.output = [file_type, annotated_video, df]

# st.session_state.output[0] == file_type
if st.session_state.output[0] == "image":
    st.image(st.session_state.output[1], channels="RGB")

    # Download annotated image
    buffer = BytesIO()
    annotated_image = Image.fromarray(st.session_state.output[1])
    annotated_image.save(buffer, format="PNG")
    byte_img = buffer.getvalue()
    st.download_button(
        "Download Annotated Image",
        byte_img,
        file_name="annotated_image.jpg",
        on_click=update_key,
    )
    st.dataframe(st.session_state.output[2])

elif st.session_state.output[0] == "video":
    st.video(st.session_state.output[1])
    st.download_button(
        "Download Annotated Video",
        st.session_state.output[1],
        file_name="annotated_video.mp4",
        on_click=update_key,
    )
    st.dataframe(st.session_state.output[2])
