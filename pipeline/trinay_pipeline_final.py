from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms

from ultralytics import YOLO

from pipeline.anatomy_refinement import refine_anatomy


# ============================================================
# TRINAY FINAL PIPELINE
#
# Model 1 : Cataract-image quality / gradability
# Model 2 : Eye detection (YOLO11n)
# Model 3 : Iris segmentation (U-Net)
# Model 4 : Pupil segmentation (U-Net)
# Model 5 : Cataract classification (EfficientNet-V2-S + CBAM)
#
# Model 5 is run on the detected eye ROI because that is the
# representation on which the existing cataract classifier
# was trained/evaluated.
#
# Iris/pupil masks are used for anatomy localization and XAI.
# ============================================================


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# MODEL PATHS
# ============================================================

# ------------------------------------------------------------
# MODEL 1 — QUALITY
# ------------------------------------------------------------

QUALITY_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "final"
    / "quality_model.pth"
)


# ------------------------------------------------------------
# MODEL 2 — EYE DETECTOR
# ------------------------------------------------------------

EYE_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "final"
    / "eye_detector.pt"
)


# ------------------------------------------------------------
# MODEL 3 — IRIS
#
# UPDATED:
# Use the fine-tuned ITEC checkpoint.
# ------------------------------------------------------------

IRIS_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "iris_segmentation"
    / "best_iris_model_it_ec_finetuned.pth"
)


# ------------------------------------------------------------
# MODEL 4 — PUPIL
#
# UPDATED:
# Use the InP → ITEC fine-tuned checkpoint.
# ------------------------------------------------------------

PUPIL_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "pupil_segmentation"
    / "best_pupil_model_inp_itec_finetuned.pth"
)


# ------------------------------------------------------------
# MODEL 5 — CATARACT CLASSIFIER
# ------------------------------------------------------------

CATARACT_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "final"
    / "cataract_model.pth"
)


# ============================================================
# SETTINGS
# ============================================================

QUALITY_THRESHOLD = 0.50

CATARACT_THRESHOLD = 0.50

EYE_CONFIDENCE = 0.50

EYE_IOU = 0.45

MAX_EYES = 2


# ------------------------------------------------------------
# SEGMENTATION
# ------------------------------------------------------------

SEG_SIZE = 256

CLS_SIZE = 224

MASK_THRESHOLD = 0.50


# ============================================================
# DEBUG MODE
# ============================================================

# IMPORTANT:
#
# Your test.jpeg is currently being classified as BAD by
# Model 1. During segmentation debugging we still want to
# test Models 3 and 4.
#
# True  = continue even if Model 1 rejects the image
# False = normal production behavior
#
# Set this back to False for production.
# ============================================================

DEBUG_BYPASS_QUALITY = True


# ============================================================
# DEVICE
# ============================================================

if torch.backends.mps.is_available():

    DEVICE = torch.device("mps")

elif torch.cuda.is_available():

    DEVICE = torch.device("cuda")

else:

    DEVICE = torch.device("cpu")


print(
    f"Using device: {DEVICE}"
)


# ============================================================
# NORMALIZATION
# ============================================================

IMAGENET_MEAN = [
    0.485,
    0.456,
    0.406
]

IMAGENET_STD = [
    0.229,
    0.224,
    0.225
]


# ============================================================
# MODEL 3 + MODEL 4 PREPROCESSING
#
# IMPORTANT:
#
# This MUST match the segmentation training/fine-tuning
# preprocessing:
#
# Resize 256x256
#       ↓
# ToTensor
#       ↓
# ImageNet Normalize
#
# ============================================================

SEG_TRANSFORM = transforms.Compose([

    transforms.Resize(
        (
            SEG_SIZE,
            SEG_SIZE
        )
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        IMAGENET_MEAN,
        IMAGENET_STD
    ),

])


# ============================================================
# MODEL 5 CLASSIFIER PREPROCESSING
# ============================================================

CLS_TRANSFORM = transforms.Compose([

    transforms.Resize(
        (
            CLS_SIZE,
            CLS_SIZE
        )
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        IMAGENET_MEAN,
        IMAGENET_STD
    ),

])


# ============================================================
# MODEL 1 QUALITY PREPROCESSING
# ============================================================

QUALITY_TRANSFORM = transforms.Compose([

    transforms.Resize(
        (
            224,
            224
        )
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        IMAGENET_MEAN,
        IMAGENET_STD
    ),

])


# ============================================================
# HELPERS
# ============================================================

def check_required_files() -> None:

    paths = {

        "Model 1 quality":
            QUALITY_MODEL_PATH,

        "Model 2 eye detector":
            EYE_MODEL_PATH,

        "Model 3 iris":
            IRIS_MODEL_PATH,

        "Model 4 pupil":
            PUPIL_MODEL_PATH,

        "Model 5 cataract":
            CATARACT_MODEL_PATH,

    }


    missing = [

        f"{name}: {path}"

        for name, path in paths.items()

        if not path.exists()

    ]


    if missing:

        raise FileNotFoundError(

            "Missing TRINAY model files:\n\n"
            + "\n".join(missing)

        )


# ============================================================
# CHECKPOINT LOADER
# ============================================================

def load_checkpoint_state(
    path: Path
) -> Dict[str, Any]:

    checkpoint = torch.load(
        path,
        map_location="cpu"
    )


    if (

        isinstance(
            checkpoint,
            dict
        )

        and

        "model_state_dict"
        in checkpoint

    ):

        return checkpoint[
            "model_state_dict"
        ]


    return checkpoint


# ============================================================
# LARGEST CONNECTED COMPONENT
# ============================================================

def largest_component(
    mask: np.ndarray
) -> np.ndarray:

    mask_u8 = (
        mask > 0
    ).astype(
        np.uint8
    )


    if mask_u8.sum() == 0:

        return mask_u8.astype(
            bool
        )


    num_labels, labels, stats, _ = (

        cv2.connectedComponentsWithStats(

            mask_u8,

            connectivity=8

        )

    )


    if num_labels <= 1:

        return mask_u8.astype(
            bool
        )


    largest_label = (

        1
        +
        np.argmax(
            stats[
                1:,
                cv2.CC_STAT_AREA
            ]
        )

    )


    return (
        labels == largest_label
    )


# ============================================================
# CROP BOX
# ============================================================

def crop_box(
    image: np.ndarray,
    box: List[int]
) -> np.ndarray:

    x1, y1, x2, y2 = box


    h, w = image.shape[:2]


    x1 = max(
        0,
        min(
            x1,
            w - 1
        )
    )


    y1 = max(
        0,
        min(
            y1,
            h - 1
        )
    )


    x2 = max(
        x1 + 1,
        min(
            x2,
            w
        )
    )


    y2 = max(
        y1 + 1,
        min(
            y2,
            h
        )
    )


    return image[
        y1:y2,
        x1:x2
    ].copy()


# ============================================================
# MODEL 1
# ============================================================

class QualityFeatureModel(
    nn.Module
):

    """
    Matches the trained quality model.

    Frozen MobileNetV3-Small feature extractor +
    CatScreen-trained linear classifier.
    """

    def __init__(
        self,
        checkpoint: Dict[str, Any]
    ):

        super().__init__()


        self.backbone = (
            models.mobilenet_v3_small(
                weights=None
            )
        )


        self.backbone.classifier = (
            nn.Identity()
        )


        self.backbone.load_state_dict(

            checkpoint[
                "backbone_state_dict"
            ]

        )


        self.classifier_weight = (
            nn.Parameter(

                checkpoint[
                    "classifier_weight"
                ],

                requires_grad=False

            )
        )


        self.classifier_bias = (
            nn.Parameter(

                checkpoint[
                    "classifier_bias"
                ],

                requires_grad=False

            )
        )


    def forward(
        self,
        x
    ):

        features = (
            self.backbone(x)
        )


        return F.linear(

            features,

            self.classifier_weight,

            self.classifier_bias

        )


# ============================================================
# LOAD QUALITY MODEL
# ============================================================

def load_quality_model():

    checkpoint = torch.load(

        QUALITY_MODEL_PATH,

        map_location="cpu"

    )


    model = (
        QualityFeatureModel(
            checkpoint
        )
        .to(DEVICE)
    )


    model.eval()


    return model


# ============================================================
# MODEL 3 + MODEL 4
# ============================================================

def load_unet_from_module(
    module_name: str
) -> nn.Module:

    import importlib


    module = importlib.import_module(
        module_name
    )


    if not hasattr(
        module,
        "UNet"
    ):

        raise ImportError(

            f"{module_name} "
            "does not contain UNet."

        )


    UNet = getattr(
        module,
        "UNet"
    )


    return (
        UNet()
        .to(DEVICE)
    )


# ============================================================
# LOAD SEGMENTATION MODEL
# ============================================================

def load_segmentation_model(

    checkpoint_path: Path,

    module_name: str

) -> nn.Module:

    model = load_unet_from_module(
        module_name
    )


    checkpoint = torch.load(

        checkpoint_path,

        map_location="cpu"

    )


    if (

        isinstance(
            checkpoint,
            dict
        )

        and

        "model_state_dict"
        in checkpoint

    ):

        state_dict = (
            checkpoint[
                "model_state_dict"
            ]
        )

    else:

        state_dict = checkpoint


    model.load_state_dict(
        state_dict
    )


    model.to(DEVICE)

    model.eval()


    return model


# ============================================================
# MODEL 5
# ============================================================

def load_cataract_model() -> nn.Module:

    from model import CataractModel


    model = (
        CataractModel()
        .to(DEVICE)
    )


    state_dict = (
        load_checkpoint_state(
            CATARACT_MODEL_PATH
        )
    )


    model.load_state_dict(
        state_dict
    )


    model.eval()


    return model


# ============================================================
# TRINAY PIPELINE
# ============================================================

class TrinayPipeline:

    def __init__(self):

        check_required_files()


        print(
            f"[TRINAY] Device: {DEVICE}"
        )


        # ----------------------------------------------------
        # MODEL 1
        # ----------------------------------------------------

        print(
            "[TRINAY] Loading Model 1..."
        )

        self.quality_model = (
            load_quality_model()
        )


        # ----------------------------------------------------
        # MODEL 2
        # ----------------------------------------------------

        print(
            "[TRINAY] Loading Model 2..."
        )

        self.eye_detector = YOLO(
            str(EYE_MODEL_PATH)
        )


        # ----------------------------------------------------
        # MODEL 3
        # ----------------------------------------------------

        print(
            "[TRINAY] Loading Model 3..."
        )

        self.iris_model = (
            load_segmentation_model(

                IRIS_MODEL_PATH,

                "iris_pupil_segmentation.train_iris"

            )
        )


        # ----------------------------------------------------
        # MODEL 4
        # ----------------------------------------------------

        print(
            "[TRINAY] Loading Model 4..."
        )

        self.pupil_model = (
            load_segmentation_model(

                PUPIL_MODEL_PATH,

                "iris_pupil_segmentation.train_pupil"

            )
        )


        # ----------------------------------------------------
        # MODEL 5
        # ----------------------------------------------------

        print(
            "[TRINAY] Loading Model 5..."
        )

        self.cataract_model = (
            load_cataract_model()
        )


        print(
            "[TRINAY] All five models loaded."
        )


    # ========================================================
    # MODEL 1 — QUALITY
    # ========================================================

    def quality_gate(
        self,
        image_bgr: np.ndarray
    ) -> Dict[str, Any]:

        image_rgb = cv2.cvtColor(

            image_bgr,

            cv2.COLOR_BGR2RGB

        )


        pil = Image.fromarray(
            image_rgb
        )


        tensor = (
            QUALITY_TRANSFORM(
                pil
            )
            .unsqueeze(0)
            .to(DEVICE)
        )


        with torch.no_grad():

            logit = (
                self.quality_model(
                    tensor
                )
            )


            good_probability = (
                torch.sigmoid(
                    logit
                ).item()
            )


        bad_probability = (
            1.0
            -
            good_probability
        )


        is_good = (
            good_probability
            >=
            QUALITY_THRESHOLD
        )


        return {

            "status":
                "GOOD"
                if is_good
                else
                "RETAKE_IMAGE",

            "good_probability":
                float(
                    good_probability
                ),

            "bad_probability":
                float(
                    bad_probability
                ),

            "threshold":
                QUALITY_THRESHOLD,

        }


    # ========================================================
    # MODEL 2 — EYE DETECTION
    # ========================================================

    def detect_eyes(
        self,
        image_bgr: np.ndarray
    ) -> List[Dict[str, Any]]:

        results = (
            self.eye_detector.predict(

                source=image_bgr,

                imgsz=640,

                conf=EYE_CONFIDENCE,

                iou=EYE_IOU,

                max_det=MAX_EYES,

                device=str(DEVICE),

                verbose=False

            )
        )


        if not results:

            return []


        result = results[0]


        if result.boxes is None:

            return []


        boxes = (
            result
            .boxes
            .xyxy
            .detach()
            .cpu()
            .numpy()
        )


        confidences = (
            result
            .boxes
            .conf
            .detach()
            .cpu()
            .numpy()
        )


        class_ids = (

            result
            .boxes
            .cls
            .detach()
            .cpu()
            .numpy()
            .astype(int)

        )


        detections = []


        for box, conf, cls_id in zip(

            boxes,
            confidences,
            class_ids

        ):

            x1, y1, x2, y2 = (

                np.round(
                    box
                )
                .astype(int)
                .tolist()

            )


            detections.append({

                "box":
                    [
                        x1,
                        y1,
                        x2,
                        y2
                    ],

                "confidence":
                    float(conf),

                "class_id":
                    int(cls_id),

            })


        return detections


    # ========================================================
    # MODEL 3 + MODEL 4 — SEGMENTATION
    # ========================================================

    def segment(

        self,

        model: nn.Module,

        eye_bgr: np.ndarray

    ) -> np.ndarray:

        # ----------------------------------------------------
        # BGR -> RGB
        # ----------------------------------------------------

        rgb = cv2.cvtColor(

            eye_bgr,

            cv2.COLOR_BGR2RGB

        )


        pil = Image.fromarray(
            rgb
        )


        # ----------------------------------------------------
        # IMPORTANT:
        #
        # Resize 256x256
        # ToTensor
        # ImageNet Normalize
        #
        # This matches training/fine-tuning.
        # ----------------------------------------------------

        tensor = (

            SEG_TRANSFORM(
                pil
            )
            .unsqueeze(0)
            .to(DEVICE)

        )


        # ----------------------------------------------------
        # MODEL
        # ----------------------------------------------------

        with torch.no_grad():

            output = model(
                tensor
            )


            # Some U-Net implementations can
            # return tuple/list.

            if isinstance(

                output,

                (tuple, list)

            ):

                output = output[0]


            # Ensure [B, C, H, W]

            if output.ndim == 3:

                output = (
                    output
                    .unsqueeze(1)
                )


            probability = torch.sigmoid(

                output[
                    :,
                    0:1
                ]

            )


            # ------------------------------------------------
            # Resize prediction back to original eye ROI
            # ------------------------------------------------

            probability = (
                F.interpolate(

                    probability,

                    size=eye_bgr.shape[:2],

                    mode="bilinear",

                    align_corners=False

                )
            )


            mask = (

                probability[
                    0,
                    0
                ]
                .detach()
                .cpu()
                .numpy()

                >=

                MASK_THRESHOLD

            )


        # ----------------------------------------------------
        # Largest connected component
        # ----------------------------------------------------

        return largest_component(
            mask
        )


    # ========================================================
    # MODEL 5 — CATARACT CLASSIFICATION
    # ========================================================

    def classify_cataract(

        self,

        eye_bgr: np.ndarray

    ) -> Dict[str, Any]:

        rgb = cv2.cvtColor(

            eye_bgr,

            cv2.COLOR_BGR2RGB

        )


        pil = Image.fromarray(
            rgb
        )


        tensor = (

            CLS_TRANSFORM(
                pil
            )
            .unsqueeze(0)
            .to(DEVICE)

        )


        with torch.no_grad():

            output = (
                self.cataract_model(
                    tensor
                )
            )


            if isinstance(

                output,

                (tuple, list)

            ):

                output = output[0]


            logit = output.squeeze()


            cataract_probability = (
                torch.sigmoid(
                    logit
                ).item()
            )


        prediction = (

            "CATARACT"

            if
            cataract_probability
            >=
            CATARACT_THRESHOLD

            else

            "NORMAL"

        )


        return {

            "prediction":
                prediction,

            "cataract_probability":
                float(
                    cataract_probability
                ),

            "normal_probability":
                float(
                    1.0
                    -
                    cataract_probability
                ),

            "threshold":
                CATARACT_THRESHOLD,

        }


    # ========================================================
    # COMPLETE INFERENCE
    # ========================================================

    def predict(

        self,

        image_path: str | Path

    ) -> Dict[str, Any]:

        image_path = Path(
            image_path
        )


        if not image_path.exists():

            raise FileNotFoundError(
                image_path
            )


        image_bgr = cv2.imread(
            str(image_path)
        )


        if image_bgr is None:

            raise ValueError(

                f"Unable to read image: "
                f"{image_path}"

            )


        # ----------------------------------------------------
        # MODEL 1
        # ----------------------------------------------------

        quality = (
            self.quality_gate(
                image_bgr
            )
        )


        result = {

            "image":
                str(image_path),

            "device":
                str(DEVICE),

            "status":
                quality["status"],

            "quality":
                quality,

            "eyes":
                [],

        }


        # ----------------------------------------------------
        # QUALITY GATE
        #
        # Production:
        #   DEBUG_BYPASS_QUALITY = False
        #
        # Testing:
        #   DEBUG_BYPASS_QUALITY = True
        # ----------------------------------------------------

        if (

            quality["status"] != "GOOD"

            and

            not DEBUG_BYPASS_QUALITY

        ):

            result["message"] = (

                "Image quality is below "
                "the configured threshold. "
                "Retake the image."

            )

            return result


        if (

            quality["status"] != "GOOD"

            and

            DEBUG_BYPASS_QUALITY

        ):

            print(
                "[DEBUG] Quality gate bypassed."
            )


        # ----------------------------------------------------
        # MODEL 2
        # ----------------------------------------------------

        detections = (
            self.detect_eyes(
                image_bgr
            )
        )


        if not detections:

            result["status"] = (
                "NO_EYE_DETECTED"
            )

            result["message"] = (
                "No eye was detected."
            )

            return result


        h, w = (
            image_bgr.shape[:2]
        )


        # ----------------------------------------------------
        # PROCESS EACH EYE
        # ----------------------------------------------------

        for (

            eye_index,
            detection

        ) in enumerate(
            detections
        ):


            # ------------------------------------------------
            # YOLO BOX
            # ------------------------------------------------

            x1, y1, x2, y2 = (
                detection["box"]
            )


            # ------------------------------------------------
            # SMALL PADDING
            #
            # Keep the same 3% padding as your uploaded
            # pipeline.
            # ------------------------------------------------

            pad_x = int(
                0.03
                *
                (x2 - x1)
            )


            pad_y = int(
                0.03
                *
                (y2 - y1)
            )


            x1p = max(
                0,
                x1 - pad_x
            )


            y1p = max(
                0,
                y1 - pad_y
            )


            x2p = min(
                w,
                x2 + pad_x
            )


            y2p = min(
                h,
                y2 + pad_y
            )


            eye_box = [

                x1p,
                y1p,
                x2p,
                y2p

            ]


            # ------------------------------------------------
            # EYE ROI
            # ------------------------------------------------

            eye_roi = crop_box(

                image_bgr,

                eye_box

            )


            # ------------------------------------------------
            # MODEL 3 — RAW IRIS
            # ------------------------------------------------

            iris_model_mask = (
                self.segment(

                    self.iris_model,

                    eye_roi

                )
            )


            # ------------------------------------------------
            # MODEL 4 — RAW PUPIL
            # ------------------------------------------------

            pupil_model_mask = (
                self.segment(

                    self.pupil_model,

                    eye_roi

                )
            )


            # ------------------------------------------------
            # ANATOMICAL REFINEMENT
            #
            # The segmentation checkpoints are NOT changed.
            #
            # This stage validates the masks against basic
            # eye anatomy and creates final masks.
            # ------------------------------------------------

            anatomy = refine_anatomy(

                eye_roi,

                iris_model_mask,

                pupil_model_mask

            )


            iris_mask = (
                anatomy[
                    "iris_mask"
                ]
            )


            pupil_mask = (
                anatomy[
                    "pupil_mask"
                ]
            )


            # ------------------------------------------------
            # MODEL 5 — CATARACT
            #
            # IMPORTANT:
            # Classifier still receives the original detected
            # eye ROI.
            #
            # We are NOT changing classifier input based on
            # segmentation yet.
            # ------------------------------------------------

            cataract = (
                self.classify_cataract(
                    eye_roi
                )
            )


            # ------------------------------------------------
            # RESULT
            # ------------------------------------------------

            result["eyes"].append({

                "index":
                    eye_index,

                "detection":
                    detection,

                "roi_box":
                    eye_box,

                "roi_width":
                    int(
                        eye_roi.shape[1]
                    ),

                "roi_height":
                    int(
                        eye_roi.shape[0]
                    ),

                "iris_area":
                    int(
                        iris_mask.sum()
                    ),

                "pupil_area":
                    int(
                        pupil_mask.sum()
                    ),

                "iris_source":
                    anatomy[
                        "iris_source"
                    ],

                "pupil_source":
                    anatomy[
                        "pupil_source"
                    ],

                "cataract":
                    cataract,

                # ------------------------------------------------
                # RUNTIME ARRAYS
                # Used by visualization/XAI.
                # ------------------------------------------------

                "_roi_bgr":
                    eye_roi,

                "_iris_mask":
                    iris_mask,

                "_pupil_mask":
                    pupil_mask,

                # ------------------------------------------------
                # RAW U-NET OUTPUTS
                #
                # VERY IMPORTANT FOR DEBUGGING.
                # ------------------------------------------------

                "_iris_model_mask":
                    iris_model_mask,

                "_pupil_model_mask":
                    pupil_model_mask,

            })


        result["status"] = (
            "COMPLETE"
        )


        return result


# ============================================================
# JSON OUTPUT
# ============================================================

def json_safe_result(

    result: Dict[str, Any]

) -> Dict[str, Any]:

    clean = {}


    for key, value in result.items():


        if key.startswith("_"):

            continue


        if key == "eyes":

            clean["eyes"] = []


            for eye in value:

                eye_clean = {}


                for ek, ev in eye.items():

                    if ek.startswith("_"):

                        continue


                    eye_clean[ek] = ev


                clean[
                    "eyes"
                ].append(
                    eye_clean
                )


        else:

            clean[key] = value


    return clean


# ============================================================
# SAVE JSON
# ============================================================

def save_result_json(

    result: Dict[str, Any],

    output_path: str | Path

) -> None:

    output_path = Path(
        output_path
    )


    output_path.parent.mkdir(

        parents=True,

        exist_ok=True

    )


    with open(

        output_path,

        "w",

        encoding="utf-8"

    ) as f:


        json.dump(

            json_safe_result(
                result
            ),

            f,

            indent=2

        )


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":

    import argparse


    parser = argparse.ArgumentParser(

        description=(
            "TRINAY final cataract "
            "screening pipeline"
        )

    )


    parser.add_argument(

        "image",

        help=(
            "Path to input eye image"
        )

    )


    parser.add_argument(

        "--json",

        default=None,

        help=(
            "Optional output JSON path"
        )

    )


    args = parser.parse_args()


    pipeline = (
        TrinayPipeline()
    )


    result = pipeline.predict(
        args.image
    )


    print(
        "\n"
        +
        "=" * 80
    )

    print(
        "TRINAY RESULT"
    )

    print(
        "=" * 80
    )


    print(

        json.dumps(

            json_safe_result(
                result
            ),

            indent=2

        )

    )


    if args.json:

        save_result_json(

            result,

            args.json

        )


        print(
            f"\nSaved result: "
            f"{args.json}"
        )