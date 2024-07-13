from io import BytesIO
import cv2
import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO


model = YOLO("yolov8n.pt")


def process_image(image):

    results = model(image)

    people = []
    for result in results:
        for box in result.boxes:
            if box.cls == 0 and box.conf > 0.5:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                people.append((x1, y1, x2 - x1, y2 - y1))

    # Draw bounding boxes
    for x, y, w, h in people:
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 5)

    return image


def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    output_path = "annotated_video.mp4"

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        annotated_frame = process_image(frame)

        out.write(annotated_frame)

    cap.release()
    out.release()

    return output_path


# Streamlit app
st.title("People Detection using YOLO v8n")

if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0
if "output" not in st.session_state:
    # [<file_type>, <annotated_image/video>]
    st.session_state.output = [None, None]


def update_key():
    st.session_state.uploader_key += 1


file = st.file_uploader(
    "Upload Image/Video",
    type=["jpg", "jpeg", "png", "mp4"],
    key=f"uploader_{st.session_state.uploader_key}",
)


def update_key():
    st.session_state.uploader_key += 1


if file is not None:
    file_type = file.type.split("/")[0]

    if file_type == "image":
        img = Image.open(file)
        image = np.array(img.convert("RGB"))
        st.session_state.output = [file_type, process_image(image)]

    elif file_type == "video":
        video_path = "./uploaded_video.mp4"
        with open(video_path, "wb") as f:
            f.write(file.read())

        with st.spinner("Processing the video..."):
            st.session_state.output = [file_type, process_video(video_path)]

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
elif st.session_state.output[0] == "video":
    st.video(st.session_state.output[1])
    with open(st.session_state.output[1], "rb") as f:
        st.download_button(
            "Download Annotated Video",
            f,
            file_name="annotated_video.mp4",
            on_click=update_key,
        )
