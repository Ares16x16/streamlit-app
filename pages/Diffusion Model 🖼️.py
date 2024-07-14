import streamlit as st
import torch
import cv2
from io import BytesIO
from PIL import Image
import numpy as np
from diffusers.utils import load_image
from diffusers import (
    StableDiffusionControlNetPipeline,
    ControlNetModel,
    UniPCMultistepScheduler,
    StableDiffusionPipeline,
)
from controlnet_aux import OpenposeDetector
from transformers import pipeline
from typing import Tuple, List, Union

st.set_page_config(
    page_title="Diffusion",
    page_icon="🖼️",
)


def calculate_grid_size(num_of_images: int) -> Tuple[int, int]:
    if num_of_images == 1:
        return 1, 1
    elif num_of_images == 2:
        return 1, 2
    else:
        factors = []
        for i in range(1, int(num_of_images**0.5) + 1):
            if num_of_images % i == 0:
                factors.append((i, num_of_images // i))
        closest = min(factors, key=lambda f: abs(f[0] - f[1]))
        return closest


def image_grid(imgs: List[Image.Image], rows: int, cols: int) -> Image.Image:
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


def canny_diffusion(
    prompt: str,
    input_image: Union[np.ndarray, Image.Image],
    low_threshold: int,
    high_threshold: int,
    seed: int,
    steps: int,
    output_image_path: str = "output.png",
) -> Image.Image:
    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/sd-controlnet-canny", torch_dtype=torch.float32
    )
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        controlnet=controlnet,
        torch_dtype=torch.float32,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()
    # pipe.enable_xformers_memory_efficient_attention()

    input_image = np.array(input_image)
    canny_image = cv2.Canny(input_image, low_threshold, high_threshold)
    zero_start = canny_image.shape[1] // 4
    zero_end = zero_start + canny_image.shape[1] // 2
    canny_image[:, zero_start:zero_end] = 0
    canny_image = np.stack([canny_image] * 3, axis=2)
    canny_image = Image.fromarray(canny_image)
    num_of_images = 1
    row, col = calculate_grid_size(num_of_images)
    generator = [torch.manual_seed(seed) for _ in range(num_of_images)]
    output = pipe(
        [prompt],
        canny_image,
        negative_prompt=["monochrome, lowres, bad anatomy, worst quality, low quality"]
        * num_of_images,
        num_inference_steps=steps,
        generator=generator,
    )
    grid = image_grid(output.images, row, col)
    # grid.save(output_image_path)
    # st.write(f"Canny Diffusion completed. Output saved to {output_image_path}.")
    return grid


def openpose_diffusion(
    prompt: str,
    input_image: Image.Image,
    seed: int,
    steps: int,
    output_image_path: str = "output.png",
) -> Image.Image:
    refImg = input_image
    model = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")
    pose = model(refImg)
    controlnet = ControlNetModel.from_pretrained(
        "fusing/stable-diffusion-v1-5-controlnet-openpose", torch_dtype=torch.float32
    )
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        controlnet=controlnet,
        torch_dtype=torch.float32,
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()
    output = pipe(
        prompt,
        pose,
        negative_prompt="monochrome, lowres, bad anatomy, worst quality, low quality",
        num_inference_steps=steps,
        generator=torch.manual_seed(seed),
    )
    grid = image_grid(output.images, 1, 1)
    # grid.save(output_image_path)
    # st.write(f"OpenPose Diffusion completed. Output saved to {output_image_path}.")
    return grid


def depth_diffusion(
    prompt: str,
    input_image: Image.Image,
    seed: int,
    steps: int,
    output_image_path: str = "output.png",
) -> Image.Image:
    depth_estimator = pipeline("depth-estimation")
    image = depth_estimator(input_image)["depth"]
    image = np.array(image)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    image = Image.fromarray(image)
    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/sd-controlnet-depth", torch_dtype=torch.float32
    )
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        controlnet=controlnet,
        safety_checker=None,
        torch_dtype=torch.float32,
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()
    output = pipe(
        prompt,
        image,
        negative_prompt="monochrome, lowres, bad anatomy, worst quality, low quality",
        num_inference_steps=steps,
        generator=torch.manual_seed(seed),
    ).images[0]
    # output.save(output_image_path)
    # st.write(f"Depth Diffusion completed. Output saved to {output_image_path}.")
    return output


def stable_diffusion(
    prompt: str,
    seed: int,
    steps: int,
    output_image_path: str = "output.png",
) -> Image.Image:
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float32,
        safety_checker=None,
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()
    pipe.enable_xformers_memory_efficient_attention()

    generator = torch.manual_seed(seed)
    output = pipe(
        prompt,
        num_inference_steps=steps,
        generator=generator,
    ).images[0]
    # output.save(output_image_path)
    # st.write(f"Stable Diffusion completed. Output saved to {output_image_path}.")
    return output


# Initialize output session state variable
if "output_image" not in st.session_state:
    st.session_state.output_image = None

st.title("Diffusion Model")
st.write(
    "Please select a diffusion model and image will be generated using the selected model."
)
model_options = [
    "Stable Diffusion",
    "Canny Diffusion",
    "OpenPose Diffusion",
    "Depth Diffusion",
]
selected_model = st.selectbox("Select a diffusion model", model_options)

if selected_model:
    prompt = st.text_area("Enter prompt", "A beautiful landscape")
    # output_image_path = st.text_input("Enter output image path", "output.png")
    seed = st.number_input("Enter seed", value=2, step=1)
    steps = st.number_input("Number of inference steps", value=5, step=1)

    if selected_model == "Stable Diffusion":
        if st.button("Run Diffusion"):
            with st.spinner("Running Stable Diffusion..."):
                st.session_state.output_image = stable_diffusion(prompt, seed, steps)
                # st.image(output_image, caption="Output Image", use_column_width=True)
    else:
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png"])
        if uploaded_file is not None:
            input_image = Image.open(uploaded_file)
            input_image = input_image.convert("RGB")
            left_holder, centre_holder, right_holder = st.columns([0.1, 0.8, 0.1])
            with centre_holder:
                st.image(input_image, caption="Input Image", width=500)

            if selected_model == "Canny Diffusion":
                low_threshold = st.slider("Canny low threshold", 0, 255, 100)
                high_threshold = st.slider("Canny high threshold", 0, 255, 200)
                if st.button("Run Diffusion"):
                    with st.spinner("Running Canny Diffusion..."):
                        st.session_state.output_image = canny_diffusion(
                            prompt,
                            input_image,
                            low_threshold,
                            high_threshold,
                            seed,
                            steps,
                        )

            elif selected_model == "OpenPose Diffusion":
                if st.button("Run Diffusion"):
                    with st.spinner("Running OpenPose Diffusion..."):
                        st.session_state.output_image = openpose_diffusion(
                            prompt, input_image, seed, steps
                        )

            elif selected_model == "Depth Diffusion":
                if st.button("Run Diffusion"):
                    with st.spinner("Running Depth Diffusion..."):
                        st.session_state.output_image = depth_diffusion(
                            prompt, input_image, seed, steps
                        )


if st.session_state.output_image is not None:
    st.write("---")
    st.image(
        st.session_state.output_image, caption="Output Image", use_column_width=True
    )
    # Download
    buffer = BytesIO()
    st.session_state.output_image.save(buffer, format="PNG")
    byte_img = buffer.getvalue()
    donwload_button = st.download_button(
        label="Download image",
        data=byte_img,
        file_name="output.png",
        mime="image/png",
    )
