from pathlib import Path
import sys
from ultralytics import YOLO


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path.cwd()


# ============================================================
# MODEL PATH
# ============================================================

MODEL_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "eye_detection"
    / "yolo11n_eye"
    / "weights"
    / "best.pt"
)


# ============================================================
# OUTPUT PATH
# ============================================================

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "eye_detection"
    / "predictions"
)


# ============================================================
# GET IMAGE PATH FROM COMMAND LINE
# ============================================================

if len(sys.argv) < 2:
    raise ValueError(
        "Please provide an image path.\n\n"
        "Example:\n"
        'python eye_detection/predict_eye.py '
        '"data/processed_images/test/cataract/image_246.png"'
    )

IMAGE_PATH = Path(sys.argv[1])

if not IMAGE_PATH.is_absolute():
    IMAGE_PATH = PROJECT_ROOT / IMAGE_PATH


# ============================================================
# CHECK FILES
# ============================================================

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found:\n{MODEL_PATH}"
    )

if not IMAGE_PATH.exists():
    raise FileNotFoundError(
        f"Image not found:\n{IMAGE_PATH}"
    )


OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("TRINAY - EYE DETECTION PREDICTION")
print("=" * 70)

print("\nModel:")
print(MODEL_PATH)

print("\nInput image:")
print(IMAGE_PATH)

print("\nOutput:")
print(OUTPUT_DIR)


# ============================================================
# LOAD YOLO MODEL
# ============================================================

model = YOLO(str(MODEL_PATH))


# ============================================================
# RUN PREDICTION
# ============================================================

print("\nRunning prediction...\n")

results = model.predict(
    source=str(IMAGE_PATH),
    imgsz=640,
    conf=0.50,
    device="mps",
    save=True,
    project=str(OUTPUT_DIR),
    name="single_prediction",
    exist_ok=True,
    verbose=True
)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 70)
print("DETECTIONS")
print("=" * 70)

for result in results:

    if result.boxes is None or len(result.boxes) == 0:
        print("\nNo eye detected.")
        continue

    for i, box in enumerate(result.boxes):

        class_id = int(box.cls[0].item())
        confidence = float(box.conf[0].item())

        x1, y1, x2, y2 = box.xyxy[0].tolist()

        class_name = model.names[class_id]

        print(f"\nDetection {i + 1}")
        print(f"Class       : {class_id} ({class_name})")
        print(f"Confidence  : {confidence:.4f}")

        print(
            f"Bounding box: "
            f"x1={x1:.1f}, "
            f"y1={y1:.1f}, "
            f"x2={x2:.1f}, "
            f"y2={y2:.1f}"
        )


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 70)
print("PREDICTION COMPLETE")
print("=" * 70)

print("\nPrediction saved to:")
print(OUTPUT_DIR / "single_prediction")