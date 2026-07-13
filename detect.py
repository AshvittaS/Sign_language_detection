"""
Sign Language Detection - Real-time prediction using MediaPipe Tasks API + Keras model.

NOTE: This uses the NEW MediaPipe Tasks API (mediapipe.tasks.vision.HandLandmarker),
not the old `mp.solutions.hands` API, which Google removed from recent mediapipe
releases (0.10.31+).

Pipeline:
1. MediaPipe HandLandmarker detects the hand + 21 landmarks in each webcam frame.
2. A square bounding box is computed around the landmarks (with margin).
3. The crop is resized to 224x224 (ResNet50 input size) and fed to model.h5.
4. The predicted class index is mapped to a label using class_names.npy.

SETUP (one-time):
1. Install dependencies:
       pip install tensorflow keras opencv-python mediapipe numpy

2. Download the hand landmark model file (~ a few MB) and place it in the
   same folder as this script, named "hand_landmarker.task":
       https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
   (You can paste that URL directly into a browser to download it, or run:
       curl -L -o hand_landmarker.task "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
   )

Files expected in the same folder as this script:
    - model.h5
    - class_names.npy
    - hand_landmarker.task

Run:
    python predict.py
Press 'q' to quit.
"""

import time

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_PATH = "model (2).keras"
CLASS_NAMES_PATH = "class_names.npy"
HAND_LANDMARKER_PATH = "hand_landmarker.task"
IMG_SIZE = 224          # model expects 224x224x3 (ResNet50 backbone)
BOX_MARGIN = 40         # extra pixels around the detected hand bounding box
CONF_THRESHOLD = 0.60   # only show prediction if confidence is above this
CAMERA_INDEX = 0

# ---------------------------------------------------------------------------
# Load Keras model + class names
# ---------------------------------------------------------------------------
print("Loading model...")
# Prefer standalone `keras` (Keras 3) to load the model; fall back to tf.keras.
try:
    import keras
    model = keras.models.load_model(MODEL_PATH, compile=False)
except Exception as e:
    print(f"Standalone keras load failed ({e}), falling back to tf.keras...")
    import tensorflow as tf
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)

class_names = np.load(CLASS_NAMES_PATH, allow_pickle=True)
class_names = [str(c) for c in class_names]
print(f"Loaded {len(class_names)} classes: {class_names}")

# ---------------------------------------------------------------------------
# MediaPipe HandLandmarker setup (new Tasks API)
# ---------------------------------------------------------------------------
BaseOptions = mp_python.BaseOptions
HandLandmarker = mp_vision.HandLandmarker
HandLandmarkerOptions = mp_vision.HandLandmarkerOptions
VisionRunningMode = mp_vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=HAND_LANDMARKER_PATH),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.6,
    min_tracking_confidence=0.6,
)
landmarker = HandLandmarker.create_from_options(options)

# Hand connections for manual drawing (21 landmarks, same topology as before)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # index
    (5, 9), (9, 10), (10, 11), (11, 12),     # middle
    (9, 13), (13, 14), (14, 15), (15, 16),   # ring
    (13, 17), (17, 18), (18, 19), (19, 20),  # pinky
    (0, 17),
]


def draw_landmarks(frame, hand_landmarks, w, h):
    """Manually draw landmarks + connections (draw_landmarks util was removed)."""
    points = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]
    for start_idx, end_idx in HAND_CONNECTIONS:
        cv2.line(frame, points[start_idx], points[end_idx], (255, 255, 255), 2)
    for x, y in points:
        cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)


def get_hand_bbox(hand_landmarks, frame_w, frame_h, margin=BOX_MARGIN):
    """Compute a pixel bounding box around the hand landmarks."""
    xs = [lm.x * frame_w for lm in hand_landmarks]
    ys = [lm.y * frame_h for lm in hand_landmarks]

    x_min, x_max = int(min(xs)) - margin, int(max(xs)) + margin
    y_min, y_max = int(min(ys)) - margin, int(max(ys)) + margin

    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(frame_w, x_max)
    y_max = min(frame_h, y_max)

    return x_min, y_min, x_max, y_max


def preprocess_crop(crop):
    """Resize + preprocess a hand crop for the model."""
    img = cv2.resize(crop, (IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype("float32")
    # NOTE: model.h5's graph already contains GetItem/Stack/Add layers right after
    # the input -- this is the model doing its OWN internal channel-split +
    # mean-subtraction (like tf.keras.applications.resnet50.preprocess_input)
    # baked into the graph. So we do NOT normalize here -- feed raw 0-255 pixels
    # and let the model's internal layers handle preprocessing.
    img = np.expand_dims(img, axis=0)  # add batch dimension
    return img


def predict_sign(crop):
    """Run the model on a preprocessed crop and return (label, confidence)."""
    x = preprocess_crop(crop)
    preds = model.predict(x, verbose=0)[0]
    idx = int(np.argmax(preds))
    conf = float(preds[idx])
    label = class_names[idx] if idx < len(class_names) else str(idx)
    return label, conf


def main():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("Error: could not open webcam.")
        return

    print("Starting webcam. Press 'q' to quit.")
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: failed to read frame from webcam.")
            break

        frame = cv2.flip(frame, 1)  # mirror for natural interaction
        h, w, _ = frame.shape

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        timestamp_ms = int((time.time() - start_time) * 1000)

        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        label_text = "No hand detected"

        if result.hand_landmarks:
            for hand_landmarks in result.hand_landmarks:
                draw_landmarks(frame, hand_landmarks, w, h)

                x_min, y_min, x_max, y_max = get_hand_bbox(hand_landmarks, w, h)
                crop = frame[y_min:y_max, x_min:x_max]

                if crop.size == 0:
                    continue

                label, conf = predict_sign(crop)

                if conf >= CONF_THRESHOLD:
                    label_text = f"{label} ({conf * 100:.1f}%)"
                else:
                    label_text = f"Unsure ({label}: {conf * 100:.1f}%)"

                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

        cv2.putText(
            frame,
            label_text,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow("Sign Language Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()


if __name__ == "__main__":
    main()