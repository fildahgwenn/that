import streamlit as st
import requests
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration, SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
import os
import torch
import soundfile as sf
from datasets import load_dataset
import matplotlib.pyplot as plt
import numpy as np
import cv2
import torchvision.transforms as transforms
from models.yolo import YOLO

# Load the pre-trained YOLO model
model = YOLO()
model.load_state_dict(torch.load('models/yolo.pth'))
model.eval()

# Define the transforms to be applied to the image
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((416, 416)),
])

# Helper function for optimizing processing time
def optimize_processing_time(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    # Resize the image to reduce processing time
    resized = cv2.resize(gray, (500, 500))
    return Image.fromarray(resized)

# Streamlit app
def main():
    st.title("Image Captioning and Object Detection for the Visually Impaired")

    # Choose image source
    image_source = st.radio("Select Image Source:", ("Upload Image", "Open from URL", "Capture from Camera"))

    image = None

    if image_source == "Upload Image":
        # File uploader for image
        uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
        else:
            image = None

    elif image_source == "Open from URL":
        # Input box for image URL
        url = st.text_input("Enter the image URL:")
        if url:
            try:
                response = requests.get(url, stream=True)
                if response.status_code == 200:
                    image = Image.open(response.raw)
                else:
                    st.error("Error loading image from URL.")
                    image = None
            except requests.exceptions.RequestException as e:
                st.error(f"Error loading image from URL: {e}")
                image = None

    elif image_source == "Capture from Camera":
        # Capture image from camera
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        if ret:
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        else:
            image = None

    # Optimize processing time
    if image is not None:
        image = optimize_processing_time(image)

    # Display the image
    if image is not None:
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Image captioning
        if st.button("Generate Caption"):
            processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
            model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
            inputs = processor(images=image, return
