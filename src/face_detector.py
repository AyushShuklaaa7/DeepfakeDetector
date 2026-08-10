import cv2


class YuNetFaceDetector:
    """
    YuNet face detector used for detecting faces
    before deepfake classification.
    """

    def __init__(
        self,
        model_path,
        score_threshold=0.6,
        nms_threshold=0.3,
        top_k=5000
    ):
        self.detector = cv2.FaceDetectorYN.create(
            model_path,
            "",
            (320, 320),
            score_threshold,
            nms_threshold,
            top_k
        )

    def detect_largest_face(self, frame):
        """
        Detect faces in a frame and return
        the largest detected face.
        """

        height, width = frame.shape[:2]

        # Configure YuNet for the current frame size
        self.detector.setInputSize((width, height))

        _, faces = self.detector.detect(frame)

        if faces is None:
            return None

        # Select the largest detected face
        largest_face = max(
            faces,
            key=lambda f: f[2] * f[3]
        )

        x, y, w, h = largest_face[:4]

        x = max(0, int(x))
        y = max(0, int(y))
        w = int(w)
        h = int(h)

        x2 = min(width, x + w)
        y2 = min(height, y + h)

        if x2 <= x or y2 <= y:
            return None

        return frame[y:y2, x:x2]
