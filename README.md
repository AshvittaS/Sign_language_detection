# Sign Language Detection (MediaPipe + Keras/ResNet50)

Real-time sign language digit detection using a webcam. MediaPipe locates and tracks
the hand, and a ResNet50-based Keras model classifies the cropped hand region into
one of 28 classes (digits `0`–`27`).

## How it works

1. **MediaPipe HandLandmarker** (Tasks API) detects the hand and its 21 landmarks
   in each webcam frame.
2. A bounding box is computed around the landmarks (with a margin) and the hand
   region is cropped from the frame.
3. The crop is resized to `224x224` and fed into `model.h5`.
4. The model's internal `preprocess_input` layers (baked into the graph during
   training) handle normalization — **raw 0–255 RGB pixels** are passed in directly.
5. The predicted class index is mapped to a label using `class_names.npy`.

## Model details

- Backbone: `ResNet50` (ImageNet weights, frozen), `GlobalAveragePooling2D` →
  `Dense(512, relu)` → `Dropout(0.5)` → `Dense(28, softmax)`
- Input: `224x224x3`, RGB, raw pixel values (preprocessing is inside the model graph
  via `tf.keras.applications.resnet.preprocess_input`)
- Classes: 28 total, labeled `'0'`–`'27'` (see `class_names.npy`)
- Trained for 10 epochs on a subset (up to 5,000 images) of the
  [American Sign Language Dataset](https://www.kaggle.com/datasets/madhavanair/american-sign-language-dataset)

## Files

| File | Description |
|---|---|
| `predict.py` | Main script — runs webcam, hand detection, and prediction |
| `model.h5` | Trained Keras model |
| `class_names.npy` | Array mapping class index → label |
| `hand_landmarker.task` | MediaPipe hand landmark model (downloaded separately, see below) |
| `requirements.txt` | Python dependencies |

## Setup

### 1. Create/activate an environment (recommended)

```bash
conda create -n dl python=3.11
conda activate dl
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Download the MediaPipe hand landmark model

`predict.py` uses the new MediaPipe **Tasks API** (`mp.tasks.vision.HandLandmarker`),
which requires a separate model file. Download it and place it in the same folder
as `predict.py`, named exactly `hand_landmarker.task`:

```bash
curl.exe -L -o hand_landmarker.task "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
```

Or via PowerShell:

```powershell
Invoke-WebRequest -Uri "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task" -OutFile "hand_landmarker.task"
```

Or paste the URL directly into your browser and save the file into the project folder.

Verify it downloaded correctly (should be several MB, not a few KB):

```bash
dir hand_landmarker.task
```

### 4. Make sure all required files are in the same folder

```
sign_language prediction/
├── predict.py
├── model.h5
├── class_names.npy
├── hand_landmarker.task
└── requirements.txt
```

## Usage

```bash
python predict.py
```

- Show a hand sign to your webcam.
- The predicted class and confidence appear on-screen, with the hand's bounding
  box and landmarks drawn.
- Press `q` to quit.

## Configuration

Key settings at the top of `predict.py`:

| Setting | Default | Description |
|---|---|---|
| `IMG_SIZE` | `224` | Must match the model's trained input size |
| `BOX_MARGIN` | `40` | Extra pixels padded around the hand bounding box |
| `CONF_THRESHOLD` | `0.60` | Minimum confidence to show a prediction as "confident" |
| `CAMERA_INDEX` | `0` | Webcam device index |

## Troubleshooting

**`AttributeError: module 'mediapipe' has no attribute 'solutions'`**
Google removed the legacy `mp.solutions` API in recent MediaPipe releases. This
project already uses the replacement **Tasks API** (`mp.tasks.vision.HandLandmarker`),
so make sure you're running the current `predict.py`, not an older version.

**`FileNotFoundError: Unable to open file at hand_landmarker.task`**
The `.task` model file is missing or not in the same folder as `predict.py`. See
step 3 in Setup above.

**`ValueError: Unknown layer: 'GetItem'` when loading `model.h5`**
This happens when an older/incompatible Keras version tries to load the model
through its legacy `.h5` loading path. Install the versions pinned in
`requirements.txt` (TensorFlow `2.21.0` / Keras `3.15.0`), which load the model
correctly.

**Prediction is always the same class regardless of the sign shown**
Usually a preprocessing mismatch. This model has `resnet.preprocess_input` baked
into its graph, so `predict.py` must feed **raw, unnormalized 0–255 RGB pixels** —
do not divide by 255 or apply extra normalization before feeding the model.

## Requirements

See `requirements.txt`. Core dependencies:

- `tensorflow==2.21.0`
- `keras==3.15.0`
- `mediapipe==0.10.35`
- `opencv-python`
- `numpy`
