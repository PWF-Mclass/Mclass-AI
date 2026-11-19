from typing import Any
import cv2
import numpy as np

from app.config import Config


def assess_face_quality(
    image_bgr: np.ndarray,
    bbox: list[int],
) -> dict[str, Any]:
    """Menilai kualitas area wajah: ketajaman (blur), kecerahan, dan proporsi ukuran wajah."""
    img_h, img_w = image_bgr.shape[:2]
    x, y, w, h = bbox

    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(img_w, x + w)
    y2 = min(img_h, y + h)

    face_crop = image_bgr[y1:y2, x1:x2]
    if face_crop.size == 0:
        return {
            "passed": False,
            "score": 0.0,
            "blur_score": 0.0,
            "brightness": 0.0,
            "face_area_ratio": 0.0,
            "reasons": ["FACE_TOO_SMALL"],
        }

    gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(np.mean(gray))
    face_area_ratio = float((w * h) / max(1, img_w * img_h))

    reasons: list[str] = []
    if blur_score < Config.MIN_BLUR_SCORE:
        reasons.append("TOO_BLURRY")
    if brightness < Config.MIN_BRIGHTNESS:
        reasons.append("TOO_DARK")
    if brightness > Config.MAX_BRIGHTNESS:
        reasons.append("TOO_BRIGHT")
    if face_area_ratio < Config.MIN_FACE_AREA_RATIO:
        reasons.append("FACE_TOO_SMALL")

    # Skor ringkas 0..1 untuk kebutuhan logging/qualityScore di Express
    blur_norm = min(1.0, blur_score / max(1.0, Config.MIN_BLUR_SCORE * 2.5))
    area_norm = min(1.0, face_area_ratio / max(0.01, Config.MIN_FACE_AREA_RATIO * 3.0))
    bright_ok = 1.0 if Config.MIN_BRIGHTNESS <= brightness <= Config.MAX_BRIGHTNESS else 0.3
    score = round(0.45 * blur_norm + 0.35 * area_norm + 0.20 * bright_ok, 4)

    return {
        "passed": len(reasons) == 0,
        "score": score,
        "blur_score": round(blur_score, 2),
        "brightness": round(brightness, 2),
        "face_area_ratio": round(face_area_ratio, 4),
        "reasons": reasons,
    }
