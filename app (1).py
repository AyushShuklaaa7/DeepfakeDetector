
import os
import time
import torch
import gradio as gr

from src.model import load_model
from src.inference import create_yunet, predict_video


# --------------------------------------------------
# Project paths
# --------------------------------------------------

PROJECT_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

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
# Device
# --------------------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

print("Device:", device)


# --------------------------------------------------
# Load EfficientNet model
# --------------------------------------------------

print("Loading EfficientNet-B0...")

model, checkpoint = load_model(
    MODEL_PATH,
    device
)

print("EfficientNet loaded!")

print(
    "Validation accuracy:",
    f"{checkpoint['val_accuracy'] * 100:.2f}%"
)

print(
    "Classes:",
    checkpoint["class_names"]
)


# --------------------------------------------------
# Load YuNet
# --------------------------------------------------

print("Loading YuNet...")

yunet = create_yunet(
    YUNET_PATH
)

print("YuNet loaded!")


# --------------------------------------------------
# Deepfake detection
# --------------------------------------------------

def detect_deepfake(file_path):

    if file_path is None:
        return (
            "⚠️ Please upload a video.",
            "—",
            "—",
            "—"
        )

    start_time = time.time()

    result = predict_video(
        file_path,
        model,
        yunet,
        device,
        frames_to_sample=5
    )

    if result is None:
        return (
            "❌ Could not detect a face in this video.",
            "—",
            "—",
            "—"
        )

    fake_probability = result[
        "fake_probability"
    ]

    real_probability = result[
        "real_probability"
    ]

    prediction = result[
        "prediction"
    ]

    frames_used = result[
        "frames_used"
    ]

    processing_time = (
        time.time() - start_time
    )

    if prediction == "FAKE":
        prediction_text = "⚠️ FAKE VIDEO"
    else:
        prediction_text = "✅ REAL VIDEO"

    return (
        prediction_text,
        f"{fake_probability * 100:.2f}%",
        f"{real_probability * 100:.2f}%",
        f"{frames_used} frames | "
        f"{processing_time:.2f} seconds"
    )


# --------------------------------------------------
# Gradio interface
# --------------------------------------------------

demo = gr.Interface(
    fn=detect_deepfake,

    inputs=gr.File(
        label="Upload MP4 Video",
        file_types=[".mp4"]
    ),

    outputs=[
        gr.Textbox(
            label="Prediction"
        ),

        gr.Textbox(
            label="Fake Probability"
        ),

        gr.Textbox(
            label="Real Probability"
        ),

        gr.Textbox(
            label="Analysis Information"
        )
    ],

    title="🕵️ Deepfake Video Detector",

    description=(
        "Upload an MP4 video. The AI will analyze "
        "sampled facial frames and predict whether "
        "the video is REAL or FAKE."
    )
)


# --------------------------------------------------
# Run application
# --------------------------------------------------

if __name__ == "__main__":
    demo.launch()
