import os
import cv2
import numpy as np
import torch

from PIL import Image
from torchvision import transforms

from model import create_model


# --------------------------------------------------
# Configuration
# --------------------------------------------------

FRAMES_TO_SAMPLE = 5

CLASS_NAMES = ["fake", "real"]

# Image preprocessing used during testing
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# --------------------------------------------------
# Load YuNet
# --------------------------------------------------

def create_yunet(model_path):
    """
    Load the YuNet face detector.
    """

    yunet = cv2.FaceDetectorYN.create(
        model_path,
        "",
        (320, 320),
        0.6,
        0.3,
        5000
    )

    return yunet


# --------------------------------------------------
# Load Deepfake Model
# --------------------------------------------------

def load_deepfake_model(model_path, device):
    """
    Load the trained EfficientNet-B0 deepfake model.
    """

    model = create_model(num_classes=2)

    checkpoint = torch.load(
        model_path,
        map_location=device
    )

    # Your saved model is a checkpoint dictionary
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

    else:
        # Fallback for a plain state_dict
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()

    return model


# --------------------------------------------------
# Predict Video
# --------------------------------------------------

def predict_video(
    video_path,
    model,
    yunet,
    device,
    frames_to_sample=FRAMES_TO_SAMPLE
):
    """
    Analyze a video using YuNet + EfficientNet-B0.

    Five evenly spaced frames are sampled.
    The largest detected face from each frame
    is classified as FAKE or REAL.

    The final prediction is based on the
    average fake probability.
    """

    cap = cv2.VideoCapture(video_path)

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    if total_frames <= 0:
        cap.release()
        return None

    # Select evenly spaced frames
    frame_indices = np.linspace(
        0,
        total_frames - 1,
        frames_to_sample
    ).astype(int)

    fake_probabilities = []

    model.eval()

    for frame_idx in frame_indices:

        cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            int(frame_idx)
        )

        ret, frame = cap.read()

        if not ret:
            continue

        height, width = frame.shape[:2]

        # Configure YuNet for current frame size
        yunet.setInputSize(
            (width, height)
        )

        _, faces = yunet.detect(frame)

        if faces is None:
            continue

        # Select largest detected face
        largest_face = max(
            faces,
            key=lambda f: f[2] * f[3]
        )

        x, y, w, h = largest_face[:4]

        x = max(0, int(x))
        y = max(0, int(y))
        w = int(w)
        h = int(h)

        x2 = min(
            width,
            x + w
        )

        y2 = min(
            height,
            y + h
        )

        if x2 <= x or y2 <= y:
            continue

        face = frame[
            y:y2,
            x:x2
        ]

        if face.size == 0:
            continue

        # OpenCV BGR → RGB
        face = cv2.cvtColor(
            face,
            cv2.COLOR_BGR2RGB
        )

        # NumPy → PIL
        face = Image.fromarray(face)

        # Apply the exact test preprocessing
        face_tensor = test_transform(face)

        # Add batch dimension and move to device
        face_tensor = (
            face_tensor
            .unsqueeze(0)
            .to(device)
        )

        # EfficientNet prediction
        with torch.no_grad():

            output = model(
                face_tensor
            )

            probability = torch.softmax(
                output,
                dim=1
            )

            # Class 0 = FAKE
            fake_probability = probability[
                0, 0
            ].item()

        fake_probabilities.append(
            fake_probability
        )

    cap.release()

    # No usable faces found
    if len(fake_probabilities) == 0:
        return None

    # Average prediction across frames
    average_fake_probability = np.mean(
        fake_probabilities
    )

    # Final classification
    prediction = (
        "FAKE"
        if average_fake_probability >= 0.5
        else "REAL"
    )

    return {
        "prediction": prediction,
        "fake_probability": float(
            average_fake_probability
        ),
        "real_probability": float(
            1.0 - average_fake_probability
        ),
        "frames_used": len(
            fake_probabilities
        )
    }
