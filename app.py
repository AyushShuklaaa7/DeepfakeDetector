
import os
import time
import sys

import torch
import streamlit as st


# --------------------------------------------------
# Project paths
# --------------------------------------------------

PROJECT_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

SRC_DIR = os.path.join(
    PROJECT_DIR,
    "src"
)

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)


from src.model import load_model
from src.inference import create_yunet, predict_video



MODEL_PATH = os.path.join(
    PROJECT_DIR,
    "models",
    "best_deepfake_model.pth"
)

YUNET_PATH = os.path.join(
    PROJECT_DIR,
    "models",
    "yunet.onnx"
)


# --------------------------------------------------
# Streamlit page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Deepfake Video Detector",
    page_icon="🕵️",
    layout="centered"
)


# --------------------------------------------------
# Load models
# --------------------------------------------------

@st.cache_resource
def load_detector():

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model, checkpoint = load_model(
        MODEL_PATH,
        device
    )

    yunet = create_yunet(
        YUNET_PATH
    )

    return (
        model,
        yunet,
        device,
        checkpoint
    )


# --------------------------------------------------
# Application UI
# --------------------------------------------------

st.title("🕵️ Deepfake Video Detector")

st.write(
    """
Upload a video and the AI will analyze
sampled facial frames using:

- **YuNet** for face detection
- **EfficientNet-B0** for deepfake classification
- **5 evenly spaced frames** per video
"""
)


# --------------------------------------------------
# Load detector
# --------------------------------------------------

try:

    model, yunet, device, checkpoint = (
        load_detector()
    )

    st.success("AI model loaded successfully.")

except Exception as e:

    st.error(
        "Failed to load the detector."
    )

    st.exception(e)

    st.stop()


# --------------------------------------------------
# Model information
# --------------------------------------------------

with st.expander("Model Information"):

    st.write(
        "**Model:** EfficientNet-B0"
    )

    st.write(
        "**Face Detector:** YuNet"
    )

    st.write(
        "**Classes:** fake, real"
    )

    st.write(
        "**Validation Accuracy:** "
        f"{checkpoint['val_accuracy'] * 100:.2f}%"
    )

    st.write(
        "**Device:** "
        f"{device}"
    )


# --------------------------------------------------
# Video upload
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload an MP4 video",
    type=["mp4"]
)


# --------------------------------------------------
# Detection
# --------------------------------------------------

if uploaded_file is not None:

    st.video(uploaded_file)

    if st.button(
        "🔍 Analyze Video",
        type="primary"
    ):

        # Save uploaded video temporarily
        temp_video_path = os.path.join(
            PROJECT_DIR,
            "temp_uploaded_video.mp4"
        )

        with open(
            temp_video_path,
            "wb"
        ) as f:

            f.write(
                uploaded_file.getbuffer()
            )

        start_time = time.time()

        with st.spinner(
            "Analyzing video..."
        ):

            result = predict_video(
                temp_video_path,
                model,
                yunet,
                device,
                frames_to_sample=5
            )

        processing_time = (
            time.time() - start_time
        )

        # Remove temporary video
        try:
            os.remove(
                temp_video_path
            )
        except OSError:
            pass


        # --------------------------------------------------
        # No face detected
        # --------------------------------------------------

        if result is None:

            st.error(
                "❌ Could not detect a face "
                "in this video."
            )

        else:

            fake_probability = (
                result["fake_probability"]
            )

            real_probability = (
                result["real_probability"]
            )

            prediction = (
                result["prediction"]
            )

            frames_used = (
                result["frames_used"]
            )


            # --------------------------------------------------
            # Result
            # --------------------------------------------------

            st.subheader(
                "Detection Result"
            )

            if prediction == "FAKE":

                st.error(
                    "⚠️ FAKE VIDEO"
                )

            else:

                st.success(
                    "✅ REAL VIDEO"
                )


            # --------------------------------------------------
            # Probabilities
            # --------------------------------------------------

            col1, col2 = st.columns(2)

            with col1:

                st.metric(
                    "Fake Probability",
                    f"{fake_probability * 100:.2f}%"
                )

            with col2:

                st.metric(
                    "Real Probability",
                    f"{real_probability * 100:.2f}%"
                )


            # --------------------------------------------------
            # Analysis information
            # --------------------------------------------------

            st.info(
                f"Frames analyzed: {frames_used}\n\n"
                f"Processing time: "
                f"{processing_time:.2f} seconds"
            )


# --------------------------------------------------
# Footer
# --------------------------------------------------

st.markdown("---")

st.caption(
    "Deepfake Video Detector | "
    "YuNet + EfficientNet-B0"
)

