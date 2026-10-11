"""
TRINAY - Master Cataract Screening Pipeline

Pipeline:

Input Image
    |
    v
1. Image Quality Assessment
    |
    +---- BAD ----> Retake Image
    |
    v
2. Eye Detection - YOLO11n
    |
    v
Eye ROI(s)
    |
    +----------------------+
    |                      |
    v                      v
3. Iris Segmentation   4. Pupil Segmentation
    |                      |
    +----------+-----------+
               |
               v
        Anatomical Analysis
               |
               v
5. Cataract Classification
               |
               v
         Final Result

Important:
- Models are loaded ONCE.
- No subprocess calls.
- No model is loaded inside run().
- All processing happens in memory.
- Supports one or multiple detected eyes.
- Designed for macOS MPS, CUDA and CPU.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

# ---------------------------------------------------------------------
# Make project root importable
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------
# External dependencies
# ---------------------------------------------------------------------

try:
    from ultralytics import YOLO
except ImportError as exc:
    raise ImportError(
        "Ultralytics is required.\n"
        "Install with:\n"
        "pip install ultralytics"
    ) from exc


from torchvision import transforms
from torchvision.models import MobileNet_V3_Small_Weights

# Your existing models
from quality_model import QualityModel
from model import CataractModel
from preprocessing import test_transform

# Existing UNet implementations
from iris_pupil_segmentation.train_iris import UNet as IrisUNet
from iris_pupil_segmentation.train_pupil import UNet as PupilUNet


# =====================================================================
# CONFIGURATION
# =====================================================================

QUALITY_MODEL_PATH = (
    PROJECT_ROOT / "models" / "quality_model.pth"
)

EYE_MODEL_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "eye_detection"
    / "yolo11n_eye-2"
    / "weights"
    / "best.pt"
)

IRIS_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "iris_segmentation"
    / "best_iris_model_it_ec_finetuned.pth"
)

PUPIL_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "pupil_segmentation"
    / "best_pupil_model_inp_itec_finetuned.pth"
)

CATARACT_MODEL_PATH = (
    PROJECT_ROOT / "models" / "best_model.pth"
)


# ---------------------------------------------------------------------
# Runtime configuration
# ---------------------------------------------------------------------

QUALITY_IMAGE_SIZE = 224

SEGMENTATION_IMAGE_SIZE = 256

EYE_DETECTION_IMAGE_SIZE = 640

QUALITY_THRESHOLD = 0.50
DEBUG_BYPASS_QUALITY = True
CATARACT_THRESHOLD = 0.50

IRIS_THRESHOLD = 0.50

PUPIL_THRESHOLD = 0.50

EYE_CONFIDENCE_THRESHOLD = 0.50

# NMS is handled by YOLO.
EYE_IOU_THRESHOLD = 0.45

# Padding around detected eye.
EYE_PADDING = 0.08

# Minimum eye crop size.
MIN_EYE_SIZE = 32

# ---------------------------------------------------------------------
# IMPORTANT
#
# "eye_roi" is the safest default because the cataract classifier was
# trained on image data rather than specifically on segmentation-derived
# ROIs.
#
# After validating classifier performance on anatomical ROIs, this can
# be changed to:
#
# CLASSIFIER_INPUT_MODE = "anatomical_roi"
#
# ---------------------------------------------------------------------

CLASSIFIER_INPUT_MODE = "eye_roi"

# When using anatomical ROI:
ANATOMICAL_PADDING = 0.20

# Maximum number of eyes processed from one image.
MAX_EYES = 2

# Return visualization masks.
GENERATE_VISUALIZATION = True


# =====================================================================
# DEVICE
# =====================================================================

def get_device() -> torch.device:
    """
    Select the best available device.

    Priority:
        MPS -> CUDA -> CPU
    """

    if torch.backends.mps.is_available():
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


DEVICE = get_device()


# =====================================================================
# DATA CLASSES
# =====================================================================

@dataclass
class BoundingBox:
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    class_id: int

    @property
    def width(self) -> int:
        return max(0, self.x2 - self.x1)

    @property
    def height(self) -> int:
        return max(0, self.y2 - self.y1)

    @property
    def area(self) -> int:
        return self.width * self.height


@dataclass
class QualityResult:
    probability_good: float
    probability_bad: float
    label: str
    passed: bool


@dataclass
class SegmentationResult:
    mask: np.ndarray
    probability: np.ndarray
    detected: bool
    pixels: int
    bbox: Optional[Tuple[int, int, int, int]]


@dataclass
class CataractResult:
    probability: float
    label: str
    possible_cataract: bool


# =====================================================================
# UTILITY FUNCTIONS
# =====================================================================

def load_state_dict_safely(
    model: torch.nn.Module,
    checkpoint_path: Path,
    device: torch.device,
) -> torch.nn.Module:
    """
    Load a PyTorch state_dict safely.

    Handles common checkpoint formats:
        - raw state_dict
        - {"state_dict": ...}
        - {"model_state_dict": ...}
    """

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"\nModel checkpoint not found:\n"
            f"{checkpoint_path}\n"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]

        elif "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]

        else:
            state_dict = checkpoint

    else:
        state_dict = checkpoint

    # Remove DataParallel prefix if present.
    cleaned_state_dict = {}

    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[len("module."):]

        cleaned_state_dict[key] = value

    model.load_state_dict(
        cleaned_state_dict,
        strict=True,
    )

    return model


def load_image(
    image: Any,
) -> Image.Image:
    """
    Convert supported image input to RGB PIL image.

    Supports:
        - file path
        - PIL Image
        - numpy array
    """

    if isinstance(image, Image.Image):
        return image.convert("RGB")

    if isinstance(image, (str, Path)):
        path = Path(image)

        if not path.exists():
            raise FileNotFoundError(
                f"Input image not found:\n{path}"
            )

        return Image.open(path).convert("RGB")

    if isinstance(image, np.ndarray):
        array = image

        if array.dtype != np.uint8:
            if array.max() <= 1.0:
                array = array * 255.0

            array = np.clip(
                array,
                0,
                255,
            ).astype(np.uint8)

        if array.ndim == 2:
            return Image.fromarray(
                array,
                mode="L",
            ).convert("RGB")

        if array.ndim == 3:
            return Image.fromarray(
                array
            ).convert("RGB")

    raise TypeError(
        "Unsupported image type. "
        "Use a file path, PIL Image, or numpy array."
    )


def pil_to_numpy(
    image: Image.Image,
) -> np.ndarray:
    return np.asarray(
        image.convert("RGB")
    )


def clamp_box(
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    width: int,
    height: int,
) -> Tuple[int, int, int, int]:

    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))

    x2 = max(x1 + 1, min(x2, width))
    y2 = max(y1 + 1, min(y2, height))

    return x1, y1, x2, y2


def expand_box(
    box: BoundingBox,
    image_width: int,
    image_height: int,
    padding_ratio: float,
) -> BoundingBox:

    pad_x = int(box.width * padding_ratio)
    pad_y = int(box.height * padding_ratio)

    x1, y1, x2, y2 = clamp_box(
        box.x1 - pad_x,
        box.y1 - pad_y,
        box.x2 + pad_x,
        box.y2 + pad_y,
        image_width,
        image_height,
    )

    return BoundingBox(
        x1=x1,
        y1=y1,
        x2=x2,
        y2=y2,
        confidence=box.confidence,
        class_id=box.class_id,
    )


def crop_box(
    image: Image.Image,
    box: BoundingBox,
) -> Image.Image:

    return image.crop(
        (
            box.x1,
            box.y1,
            box.x2,
            box.y2,
        )
    )


def largest_connected_component(
    mask: np.ndarray,
) -> np.ndarray:
    """
    Keep the largest connected component.

    Uses scipy if available.
    If scipy is unavailable, returns the original mask.

    This is a post-processing safety step for segmentation.
    """

    try:
        from scipy import ndimage
    except ImportError:
        return mask

    binary = mask.astype(np.uint8)

    if binary.sum() == 0:
        return mask

    labeled, count = ndimage.label(binary)

    if count <= 1:
        return mask

    component_sizes = np.bincount(
        labeled.ravel()
    )

    component_sizes[0] = 0

    largest = np.argmax(
        component_sizes
    )

    return (
        labeled == largest
    )


def mask_bbox(
    mask: np.ndarray,
) -> Optional[Tuple[int, int, int, int]]:

    ys, xs = np.where(mask)

    if len(xs) == 0:
        return None

    return (
        int(xs.min()),
        int(ys.min()),
        int(xs.max() + 1),
        int(ys.max() + 1),
    )


# =====================================================================
# MASTER PIPELINE
# =====================================================================

class TrinayPipeline:
    """
    Complete TRINAY cataract screening pipeline.

    All models are loaded once during initialization.

    Usage:

        pipeline = TrinayPipeline()

        result = pipeline.run(
            "test_image.jpg"
        )

        print(result)
    """

    def __init__(
        self,
        device: Optional[torch.device] = None,
    ):

        self.device = (
            device
            if device is not None
            else DEVICE
        )

        print("=" * 70)
        print("TRINAY PIPELINE")
        print("=" * 70)

        print(
            f"Device: {self.device}"
        )

        print(
            "\nLoading models..."
        )

        self._load_quality_model()
        self._load_eye_detector()
        self._load_iris_model()
        self._load_pupil_model()
        self._load_cataract_model()

        print(
            "\nAll five models loaded successfully."
        )

        print("=" * 70)


    # =================================================================
    # MODEL 1 - QUALITY
    # =================================================================

    def _load_quality_model(self):

        print(
            "Loading Model 1 - Image Quality..."
        )

        self.quality_model = QualityModel()

        load_state_dict_safely(
            self.quality_model,
            QUALITY_MODEL_PATH,
            self.device,
        )

        self.quality_model.to(
            self.device
        )

        self.quality_model.eval()

        weights = (
            MobileNet_V3_Small_Weights.DEFAULT
        )

        self.quality_transform = transforms.Compose(
            [
                transforms.Resize(
                    (
                        QUALITY_IMAGE_SIZE,
                        QUALITY_IMAGE_SIZE,
                    )
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=weights.transforms().mean,
                    std=weights.transforms().std,
                ),
            ]
        )


    # =================================================================
    # MODEL 2 - EYE DETECTION
    # =================================================================

    def _load_eye_detector(self):

        print(
            "Loading Model 2 - Eye Detection..."
        )

        if not EYE_MODEL_PATH.exists():
            raise FileNotFoundError(
                "\nEye detector not found:\n"
                f"{EYE_MODEL_PATH}"
            )

        self.eye_detector = YOLO(
            str(EYE_MODEL_PATH)
        )


    # =================================================================
    # MODEL 3 - IRIS
    # =================================================================

    def _load_iris_model(self):

        print(
            "Loading Model 3 - Iris Segmentation..."
        )

        self.iris_model = IrisUNet()

        load_state_dict_safely(
            self.iris_model,
            IRIS_MODEL_PATH,
            self.device,
        )

        self.iris_model.to(
            self.device
        )

        self.iris_model.eval()

        self.segmentation_transform = transforms.Compose(
            [
                transforms.Resize(
                    (
                        SEGMENTATION_IMAGE_SIZE,
                        SEGMENTATION_IMAGE_SIZE,
                    )
                ),
                transforms.ToTensor(),
            ]
        )


    # =================================================================
    # MODEL 4 - PUPIL
    # =================================================================

    def _load_pupil_model(self):

        print(
            "Loading Model 4 - Pupil Segmentation..."
        )

        self.pupil_model = PupilUNet()

        load_state_dict_safely(
            self.pupil_model,
            PUPIL_MODEL_PATH,
            self.device,
        )

        self.pupil_model.to(
            self.device
        )

        self.pupil_model.eval()


    # =================================================================
    # MODEL 5 - CATARACT
    # =================================================================

    def _load_cataract_model(self):

        print(
            "Loading Model 5 - Cataract Classification..."
        )

        self.cataract_model = CataractModel()

        load_state_dict_safely(
            self.cataract_model,
            CATARACT_MODEL_PATH,
            self.device,
        )

        self.cataract_model.to(
            self.device
        )

        self.cataract_model.eval()


    # =================================================================
    # STEP 1 - QUALITY
    # =================================================================

    @torch.inference_mode()
    def check_quality(
        self,
        image: Image.Image,
    ) -> QualityResult:

        tensor = self.quality_transform(
            image
        )

        tensor = tensor.unsqueeze(0)

        tensor = tensor.to(
            self.device
        )

        logits = self.quality_model(
            tensor
        )

        probability_good = (
            torch.sigmoid(
                logits
            )
            .item()
        )

        probability_bad = (
            1.0 - probability_good
        )

        passed = (
            probability_good
            >= QUALITY_THRESHOLD
        )

        label = (
            "GOOD"
            if passed
            else "BAD"
        )

        return QualityResult(
            probability_good=float(
                probability_good
            ),
            probability_bad=float(
                probability_bad
            ),
            label=label,
            passed=passed,
        )


    # =================================================================
    # STEP 2 - EYE DETECTION
    # =================================================================

    def detect_eyes(
        self,
        image: Image.Image,
    ) -> List[BoundingBox]:

        results = self.eye_detector.predict(
            source=image,
            imgsz=EYE_DETECTION_IMAGE_SIZE,
            conf=EYE_CONFIDENCE_THRESHOLD,
            iou=EYE_IOU_THRESHOLD,
            device=self._yolo_device(),
            verbose=False,
        )

        if not results:
            return []

        result = results[0]

        if result.boxes is None:
            return []

        boxes: List[BoundingBox] = []

        image_width, image_height = (
            image.size
        )

        for box in result.boxes:

            xyxy = (
                box.xyxy[0]
                .detach()
                .cpu()
                .numpy()
            )

            confidence = float(
                box.conf[0]
                .detach()
                .cpu()
                .item()
            )

            class_id = int(
                box.cls[0]
                .detach()
                .cpu()
                .item()
            )

            x1, y1, x2, y2 = map(
                int,
                xyxy
            )

            x1, y1, x2, y2 = clamp_box(
                x1,
                y1,
                x2,
                y2,
                image_width,
                image_height,
            )

            candidate = BoundingBox(
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                confidence=confidence,
                class_id=class_id,
            )

            if (
                candidate.width
                < MIN_EYE_SIZE
                or candidate.height
                < MIN_EYE_SIZE
            ):
                continue

            boxes.append(
                candidate
            )

        # Highest confidence first.
        boxes.sort(
            key=lambda b: b.confidence,
            reverse=True,
        )

        return boxes[:MAX_EYES]


    def _yolo_device(self):
        """
        Ultralytics accepts:
            "mps"
            "cuda"
            "cpu"
        """

        return self.device.type


    # =================================================================
    # SEGMENTATION CORE
    # =================================================================

    @torch.inference_mode()
    def _segment(
        self,
        image: Image.Image,
        model: torch.nn.Module,
    ) -> SegmentationResult:

        original_width, original_height = (
            image.size
        )

        tensor = self.segmentation_transform(
            image
        )

        tensor = tensor.unsqueeze(0)

        tensor = tensor.to(
            self.device
        )

        output = model(
            tensor
        )

        # Handle common UNet output shapes.
        if output.ndim == 4:

            if output.shape[1] == 1:
                logits = output[:, 0]

            else:
                # If the network outputs multiple channels,
                # use the final channel only for the binary mask.
                logits = output[:, -1]

        elif output.ndim == 3:

            logits = output

        else:
            raise RuntimeError(
                "Unexpected segmentation output shape: "
                f"{tuple(output.shape)}"
            )

        probability = torch.sigmoid(
            logits
        )

        probability = F.interpolate(
            probability.unsqueeze(1),
            size=(
                original_height,
                original_width,
            ),
            mode="bilinear",
            align_corners=False,
        )[0, 0]

        probability_np = (
            probability
            .detach()
            .cpu()
            .numpy()
        )

        mask = (
            probability_np
            >= IRIS_THRESHOLD
        )

        mask = largest_connected_component(
            mask
        )

        mask = mask.astype(
            np.uint8
        )

        pixels = int(
            mask.sum()
        )

        detected = (
            pixels > 0
        )

        bbox = mask_bbox(
            mask
        )

        return SegmentationResult(
            mask=mask,
            probability=probability_np,
            detected=detected,
            pixels=pixels,
            bbox=bbox,
        )


    # =================================================================
    # STEP 3 - IRIS
    # =================================================================

    def segment_iris(
        self,
        eye_image: Image.Image,
    ) -> SegmentationResult:

        return self._segment(
            eye_image,
            self.iris_model,
        )


    # =================================================================
    # STEP 4 - PUPIL
    # =================================================================

    def segment_pupil(
        self,
        eye_image: Image.Image,
    ) -> SegmentationResult:

        return self._segment(
            eye_image,
            self.pupil_model,
        )


    # =================================================================
    # ANATOMICAL ROI
    # =================================================================

    def create_anatomical_roi(
        self,
        eye_image: Image.Image,
        iris_result: SegmentationResult,
        pupil_result: SegmentationResult,
    ) -> Image.Image:

        """
        Create an anatomical ROI using the iris mask.

        The iris is the primary anatomical region used here.
        Pupil information is retained separately for analysis.

        If no iris is detected, falls back to the complete eye ROI.
        """

        bbox = iris_result.bbox

        if bbox is None:

            return eye_image.copy()

        x1, y1, x2, y2 = bbox

        width = x2 - x1
        height = y2 - y1

        pad_x = int(
            width * ANATOMICAL_PADDING
        )

        pad_y = int(
            height * ANATOMICAL_PADDING
        )

        image_width, image_height = (
            eye_image.size
        )

        x1, y1, x2, y2 = clamp_box(
            x1 - pad_x,
            y1 - pad_y,
            x2 + pad_x,
            y2 + pad_y,
            image_width,
            image_height,
        )

        return eye_image.crop(
            (
                x1,
                y1,
                x2,
                y2,
            )
        )


    # =================================================================
    # STEP 5 - CATARACT CLASSIFICATION
    # =================================================================

    @torch.inference_mode()
    def classify_cataract(
        self,
        image: Image.Image,
    ) -> CataractResult:

        tensor = test_transform(
            image
        )

        tensor = tensor.unsqueeze(0)

        tensor = tensor.to(
            self.device
        )

        logits = self.cataract_model(
            tensor
        )

        probability = torch.sigmoid(
            logits
        ).item()

        possible_cataract = (
            probability
            >= CATARACT_THRESHOLD
        )

        label = (
            "POSSIBLE CATARACT"
            if possible_cataract
            else "NORMAL"
        )

        return CataractResult(
            probability=float(
                probability
            ),
            label=label,
            possible_cataract=possible_cataract,
        )


    # =================================================================
    # SINGLE EYE PIPELINE
    # =================================================================

    def process_eye(
        self,
        full_image: Image.Image,
        eye_box: BoundingBox,
        eye_index: int,
    ) -> Dict[str, Any]:

        # -------------------------------------------------------------
        # Expand eye bounding box slightly.
        # -------------------------------------------------------------

        expanded_box = expand_box(
            eye_box,
            full_image.width,
            full_image.height,
            EYE_PADDING,
        )

        eye_image = crop_box(
            full_image,
            expanded_box,
        )

        # -------------------------------------------------------------
        # Iris
        # -------------------------------------------------------------

        iris_result = self.segment_iris(
            eye_image
        )

        # -------------------------------------------------------------
        # Pupil
        # -------------------------------------------------------------

        pupil_result = self.segment_pupil(
            eye_image
        )

        # -------------------------------------------------------------
        # Anatomical ROI
        # -------------------------------------------------------------

        anatomical_roi = (
            self.create_anatomical_roi(
                eye_image,
                iris_result,
                pupil_result,
            )
        )

        # -------------------------------------------------------------
        # Select classifier input.
        # -------------------------------------------------------------

        if (
            CLASSIFIER_INPUT_MODE
            == "anatomical_roi"
        ):

            classifier_image = (
                anatomical_roi
            )

        else:

            classifier_image = (
                eye_image
            )

        # -------------------------------------------------------------
        # Cataract classification
        # -------------------------------------------------------------

        cataract_result = (
            self.classify_cataract(
                classifier_image
            )
        )

        # -------------------------------------------------------------
        # Prepare output
        # -------------------------------------------------------------

        return {
            "eye_index": eye_index,

            "detection": {
                "class_id": eye_box.class_id,
                "confidence": eye_box.confidence,
                "bbox": [
                    eye_box.x1,
                    eye_box.y1,
                    eye_box.x2,
                    eye_box.y2,
                ],
                "expanded_bbox": [
                    expanded_box.x1,
                    expanded_box.y1,
                    expanded_box.x2,
                    expanded_box.y2,
                ],
            },

            "iris": {
                "detected": iris_result.detected,
                "pixels": iris_result.pixels,
                "bbox": iris_result.bbox,
                "mask": iris_result.mask,
            },

            "pupil": {
                "detected": pupil_result.detected,
                "pixels": pupil_result.pixels,
                "bbox": pupil_result.bbox,
                "mask": pupil_result.mask,
            },

            "anatomical_roi": anatomical_roi,

            "classifier_input_mode": (
                CLASSIFIER_INPUT_MODE
            ),

            "cataract": {
                "probability": (
                    cataract_result.probability
                ),
                "label": (
                    cataract_result.label
                ),
                "possible_cataract": (
                    cataract_result.possible_cataract
                ),
            },

            # Keep images available internally.
            "eye_image": eye_image,
        }


    # =================================================================
    # COMPLETE PIPELINE
    # =================================================================

    def run(
        self,
        image: Any,
    ) -> Dict[str, Any]:

        # -------------------------------------------------------------
        # Load image
        # -------------------------------------------------------------

        original_image = load_image(
            image
        )

        # -------------------------------------------------------------
        # STEP 1
        # -------------------------------------------------------------

        quality = self.check_quality(
            original_image
        )

        # -------------------------------------------------------------
        # HARD STOP
        #
        # Never continue to detection/classification if image quality
        # fails.
        # -------------------------------------------------------------

        if not quality.passed and not DEBUG_BYPASS_QUALITY:

            return {
                "success": False,

                "status": "RETAKE_IMAGE",

                "message": (
                    "Image quality is insufficient. "
                    "Please capture another image."
                ),

                "quality": asdict(
                    quality
                ),

                "eyes": [],

                "num_eyes": 0,

                "device": self.device.type,
            }

        # -------------------------------------------------------------
        # STEP 2
        # -------------------------------------------------------------

        eye_boxes = self.detect_eyes(
            original_image
        )

        # -------------------------------------------------------------
        # No eyes detected
        # -------------------------------------------------------------

        if not eye_boxes:

            return {
                "success": False,

                "status": "NO_EYE_DETECTED",

                "message": (
                    "No eye was detected in the image."
                ),

                "quality": asdict(
                    quality
                ),

                "eyes": [],

                "num_eyes": 0,

                "device": self.device.type,
            }

        # -------------------------------------------------------------
        # STEP 3 + STEP 4 + STEP 5
        # -------------------------------------------------------------

        processed_eyes = []

        for index, eye_box in enumerate(
            eye_boxes,
            start=1,
        ):

            try:

                eye_result = self.process_eye(
                    original_image,
                    eye_box,
                    index,
                )

                processed_eyes.append(
                    eye_result
                )

            except Exception as exc:

                # Do not silently hide an eye-level failure.
                processed_eyes.append(
                    {
                        "eye_index": index,
                        "success": False,
                        "error": str(exc),
                        "detection": {
                            "class_id": eye_box.class_id,
                            "confidence": eye_box.confidence,
                            "bbox": [
                                eye_box.x1,
                                eye_box.y1,
                                eye_box.x2,
                                eye_box.y2,
                            ],
                        },
                    }
                )

        # -------------------------------------------------------------
        # Overall result
        # -------------------------------------------------------------

        successful_eyes = [
            eye
            for eye in processed_eyes
            if eye.get("success", True)
        ]

        cataract_positive = [
            eye
            for eye in successful_eyes
            if eye.get(
                "cataract",
                {}
            ).get(
                "possible_cataract",
                False,
            )
        ]

        if cataract_positive:

            overall_status = (
                "POSSIBLE_CATARACT"
            )

        else:

            overall_status = "NO_CATARACT_DETECTED"

        return {
            "success": True,

            "status": overall_status,

            "message": (
                "Screening completed. "
                "Clinical confirmation is required."
            ),

            "device": self.device.type,

            "quality": asdict(
                quality
            ),

            "num_eyes": len(
                processed_eyes
            ),

            "eyes": processed_eyes,
        }


    # =================================================================
    # VISUALIZATION
    # =================================================================

    def visualize(
        self,
        image: Any,
        result: Dict[str, Any],
    ) -> Image.Image:

        """
        Create a simple diagnostic visualization.

        This is for debugging/dashboard development.
        """

        pil_image = load_image(
            image
        ).copy()

        draw = ImageDraw.Draw(
            pil_image
        )

        # -------------------------------------------------------------
        # Quality
        # -------------------------------------------------------------

        quality = result.get(
            "quality",
            {}
        )

        quality_label = quality.get(
            "label",
            "UNKNOWN"
        )

        # -------------------------------------------------------------
        # Eyes
        # -------------------------------------------------------------

        for eye in result.get(
            "eyes",
            [],
        ):

            if "detection" not in eye:
                continue

            bbox = eye[
                "detection"
            ].get(
                "expanded_bbox"
            )

            if not bbox:
                continue

            x1, y1, x2, y2 = bbox

            cataract = eye.get(
                "cataract",
                {}
            )

            label = cataract.get(
                "label",
                "UNKNOWN"
            )

            confidence = cataract.get(
                "probability",
                0.0,
            )

            text = (
                f"{label} "
                f"{confidence:.2%}"
            )

            draw.rectangle(
                (
                    x1,
                    y1,
                    x2,
                    y2,
                ),
                outline="red",
                width=3,
            )

            draw.text(
                (
                    x1,
                    max(0, y1 - 18),
                ),
                text,
                fill="red",
            )

        return pil_image


# =====================================================================
# SIMPLE PUBLIC API
# =====================================================================

_PIPELINE: Optional[
    TrinayPipeline
] = None


def get_pipeline() -> TrinayPipeline:
    """
    Singleton pipeline.

    This is important for dashboard applications.

    The five models are loaded only once.
    """

    global _PIPELINE

    if _PIPELINE is None:

        _PIPELINE = TrinayPipeline()

    return _PIPELINE


def run_trinay(
    image: Any,
) -> Dict[str, Any]:

    pipeline = get_pipeline()

    return pipeline.run(
        image
    )


# =====================================================================
# COMMAND LINE TEST
# =====================================================================

def main():

    if len(sys.argv) < 2:

        print(
            "\nUsage:"
        )

        print(
            "python pipeline/trinay_pipeline.py "
            "path/to/image.jpg\n"
        )

        sys.exit(1)

    image_path = sys.argv[1]

    pipeline = TrinayPipeline()

    result = pipeline.run(
        image_path
    )

    print("\n")
    print("=" * 70)
    print("TRINAY SCREENING RESULT")
    print("=" * 70)

    print(
        f"\nStatus: {result['status']}"
    )

    print(
        f"Success: {result['success']}"
    )

    if "quality" in result:

        quality = result[
            "quality"
        ]

        print(
            "\nImage Quality:"
        )

        print(
            f"  Label: "
            f"{quality['label']}"
        )

        print(
            f"  Good probability: "
            f"{quality['probability_good']:.4f}"
        )

        print(
            f"  Bad probability: "
            f"{quality['probability_bad']:.4f}"
        )

    print(
        f"\nEyes detected: "
        f"{result.get('num_eyes', 0)}"
    )

    for eye in result.get(
        "eyes",
        [],
    ):

        print("\n" + "-" * 50)

        print(
            f"Eye {eye['eye_index']}"
        )

        detection = eye.get(
            "detection",
            {}
        )

        print(
            f"Detection confidence: "
            f"{detection.get('confidence', 0):.4f}"
        )

        iris = eye.get(
            "iris",
            {}
        )

        pupil = eye.get(
            "pupil",
            {}
        )

        cataract = eye.get(
            "cataract",
            {}
        )

        print(
            f"Iris detected: "
            f"{iris.get('detected', False)}"
        )

        print(
            f"Pupil detected: "
            f"{pupil.get('detected', False)}"
        )

        print(
            f"Cataract probability: "
            f"{cataract.get('probability', 0):.4f}"
        )

        print(
            f"Cataract result: "
            f"{cataract.get('label', 'UNKNOWN')}"
        )

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()