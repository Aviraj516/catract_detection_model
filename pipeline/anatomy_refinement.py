from __future__ import annotations

import cv2
import numpy as np


# ============================================================
# BASIC HELPERS
# ============================================================

def clean_mask(mask: np.ndarray) -> np.ndarray:
    """Clean a binary segmentation mask."""
    mask = (mask > 0).astype(np.uint8)

    kernel = np.ones((5, 5), np.uint8)

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel,
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel,
    )

    return mask.astype(bool)


def largest_component(mask: np.ndarray) -> np.ndarray:
    mask_u8 = (mask.astype(np.uint8) * 255)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask_u8,
        connectivity=8,
    )

    if num_labels <= 1:
        return mask.astype(bool)

    largest = 1 + np.argmax(
        stats[1:, cv2.CC_STAT_AREA]
    )

    return labels == largest


def mask_geometry(mask: np.ndarray):
    ys, xs = np.where(mask)

    if len(xs) == 0:
        return None

    x_min = int(xs.min())
    x_max = int(xs.max())
    y_min = int(ys.min())
    y_max = int(ys.max())

    return {
        "area": int(len(xs)),
        "cx": float(xs.mean()),
        "cy": float(ys.mean()),
        "width": x_max - x_min + 1,
        "height": y_max - y_min + 1,
        "x_min": x_min,
        "x_max": x_max,
        "y_min": y_min,
        "y_max": y_max,
    }


# ============================================================
# PUPIL VALIDATION
# ============================================================

def pupil_prediction_is_valid(
    mask: np.ndarray,
    image_shape,
) -> bool:

    geometry = mask_geometry(mask)

    if geometry is None:
        return False

    h, w = image_shape[:2]

    area_ratio = geometry["area"] / float(h * w)

    # Reject empty/tiny or implausibly large pupil masks.
    if area_ratio < 0.003:
        return False

    if area_ratio > 0.18:
        return False

    aspect_ratio = (
        geometry["width"]
        / max(geometry["height"], 1)
    )

    if aspect_ratio < 0.35 or aspect_ratio > 2.8:
        return False

    # A pupil should normally be near the center of the eye ROI.
    center_x = w / 2.0
    center_y = h / 2.0

    distance = np.sqrt(
        (geometry["cx"] - center_x) ** 2
        + (geometry["cy"] - center_y) ** 2
    )

    max_distance = 0.28 * min(w, h)

    return distance <= max_distance


# ============================================================
# CLASSICAL PUPIL FALLBACK
# ============================================================

def detect_pupil_classical(
    eye_bgr: np.ndarray,
) -> np.ndarray:

    h, w = eye_bgr.shape[:2]

    gray = cv2.cvtColor(
        eye_bgr,
        cv2.COLOR_BGR2GRAY,
    )

    gray = cv2.GaussianBlur(
        gray,
        (9, 9),
        0,
    )

    center_x = w // 2
    center_y = h // 2

    search_radius_x = int(w * 0.38)
    search_radius_y = int(h * 0.38)

    x1 = max(0, center_x - search_radius_x)
    x2 = min(w, center_x + search_radius_x)
    y1 = max(0, center_y - search_radius_y)
    y2 = min(h, center_y + search_radius_y)

    roi = gray[y1:y2, x1:x2]

    threshold = np.percentile(roi, 28)

    binary = (roi < threshold).astype(np.uint8) * 255

    kernel = np.ones((7, 7), np.uint8)

    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        kernel,
    )

    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_CLOSE,
        kernel,
    )

    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    candidates = []

    for contour in contours:

        area = cv2.contourArea(contour)

        if area < 300:
            continue

        if area > 0.18 * h * w:
            continue

        x, y, cw, ch = cv2.boundingRect(contour)

        cx = x1 + x + cw / 2.0
        cy = y1 + y + ch / 2.0

        distance = np.sqrt(
            (cx - center_x) ** 2
            + (cy - center_y) ** 2
        )

        aspect = cw / max(ch, 1)

        if aspect < 0.35 or aspect > 2.8:
            continue

        candidates.append(
            (distance, contour)
        )

    if not candidates:
        return np.zeros(
            (h, w),
            dtype=bool,
        )

    candidates.sort(
        key=lambda item: item[0]
    )

    contour = candidates[0][1]

    full_mask = np.zeros(
        (h, w),
        dtype=np.uint8,
    )

    contour_shifted = (
        contour
        + np.array([[x1, y1]])
    )

    cv2.drawContours(
        full_mask,
        [contour_shifted],
        -1,
        1,
        -1,
    )

    return largest_component(
        full_mask.astype(bool)
    )


# ============================================================
# HOUGH PUPIL FALLBACK
# ============================================================

def detect_pupil_hough(
    eye_bgr: np.ndarray,
) -> np.ndarray:

    h, w = eye_bgr.shape[:2]

    gray = cv2.cvtColor(
        eye_bgr,
        cv2.COLOR_BGR2GRAY,
    )

    gray = cv2.GaussianBlur(
        gray,
        (9, 9),
        2,
    )

    min_dim = min(h, w)

    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=int(min_dim * 0.20),
        param1=100,
        param2=25,
        minRadius=int(min_dim * 0.07),
        maxRadius=int(min_dim * 0.24),
    )

    if circles is None:
        return np.zeros(
            (h, w),
            dtype=bool,
        )

    center_x = w / 2.0
    center_y = h / 2.0

    best = None
    best_score = float("inf")

    for circle in circles[0]:

        cx, cy, radius = circle

        distance = np.sqrt(
            (cx - center_x) ** 2
            + (cy - center_y) ** 2
        )

        if distance < best_score:
            best_score = distance
            best = circle

    if best is None:
        return np.zeros(
            (h, w),
            dtype=bool,
        )

    cx, cy, radius = best

    mask = np.zeros(
        (h, w),
        dtype=np.uint8,
    )

    cv2.circle(
        mask,
        (int(cx), int(cy)),
        int(radius),
        1,
        -1,
    )

    return mask.astype(bool)


# ============================================================
# PUPIL REFINEMENT
# ============================================================

def refine_pupil(
    eye_bgr: np.ndarray,
    model_mask: np.ndarray,
):

    model_mask = clean_mask(model_mask)

    # First trust the trained model if it is anatomically plausible.
    if pupil_prediction_is_valid(
        model_mask,
        eye_bgr.shape,
    ):
        return (
            largest_component(model_mask),
            "MODEL",
        )

    # Fallback 1: dark-pupil segmentation.
    classical = detect_pupil_classical(
        eye_bgr
    )

    if pupil_prediction_is_valid(
        classical,
        eye_bgr.shape,
    ):
        return (
            classical,
            "CLASSICAL",
        )

    # Fallback 2: circle detection.
    hough = detect_pupil_hough(
        eye_bgr
    )

    if pupil_prediction_is_valid(
        hough,
        eye_bgr.shape,
    ):
        return (
            hough,
            "HOUGH",
        )

    # No reliable alternative.
    return (
        model_mask,
        "MODEL_UNVERIFIED",
    )


# ============================================================
# IRIS ANATOMICAL FALLBACK
# ============================================================

def detect_iris_from_pupil(
    eye_bgr: np.ndarray,
    pupil_mask: np.ndarray,
):

    h, w = eye_bgr.shape[:2]

    geometry = mask_geometry(
        pupil_mask
    )

    if geometry is None:
        return np.zeros(
            (h, w),
            dtype=bool,
        )

    pupil_cx = geometry["cx"]
    pupil_cy = geometry["cy"]

    pupil_radius = max(
        geometry["width"],
        geometry["height"],
    ) / 2.0

    # Iris should be substantially larger than pupil.
    radius = pupil_radius * 2.05

    radius = np.clip(
        radius,
        0.18 * min(h, w),
        0.48 * min(h, w),
    )

    mask = np.zeros(
        (h, w),
        dtype=np.uint8,
    )

    cv2.circle(
        mask,
        (int(pupil_cx), int(pupil_cy)),
        int(radius),
        1,
        -1,
    )

    return mask.astype(bool)


def refine_iris(
    eye_bgr: np.ndarray,
    model_mask: np.ndarray,
    pupil_mask: np.ndarray,
):

    model_mask = clean_mask(model_mask)

    pupil_geometry = mask_geometry(
        pupil_mask
    )

    if pupil_geometry is None:
        return (
            model_mask,
            "MODEL_UNVERIFIED",
        )

    # A valid iris prediction should contain most of the pupil.
    overlap = (
        model_mask & pupil_mask
    ).sum()

    pupil_area = max(
        pupil_mask.sum(),
        1,
    )

    pupil_inside_ratio = (
        overlap / pupil_area
    )

    model_geometry = mask_geometry(
        model_mask
    )

    if (
        model_geometry is not None
        and pupil_inside_ratio > 0.50
    ):
        return (
            largest_component(model_mask),
            "MODEL",
        )

    # Otherwise construct a constrained anatomical iris around
    # the refined pupil.
    iris_mask = detect_iris_from_pupil(
        eye_bgr,
        pupil_mask,
    )

    if np.any(iris_mask):
        return (
            iris_mask,
            "ANATOMICAL",
        )

    return (
        model_mask,
        "MODEL_UNVERIFIED",
    )


# ============================================================
# COMPLETE ANATOMY REFINEMENT
# ============================================================

def refine_anatomy(
    eye_bgr: np.ndarray,
    iris_model_mask: np.ndarray,
    pupil_model_mask: np.ndarray,
):

    pupil_mask, pupil_source = refine_pupil(
        eye_bgr,
        pupil_model_mask,
    )

    iris_mask, iris_source = refine_iris(
        eye_bgr,
        iris_model_mask,
        pupil_mask,
    )

    return {
        "iris_mask": iris_mask,
        "pupil_mask": pupil_mask,
        "iris_source": iris_source,
        "pupil_source": pupil_source,
    }
