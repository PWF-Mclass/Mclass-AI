import hmac
import json
from typing import Any, Optional, Tuple
from flask import Blueprint, jsonify, request

from app.config import Config
from app.face_engine import DetectedFace, face_engine
from app.quality import assess_face_quality
from app.schemas import error_response, model_info

api_bp = Blueprint("api", __name__)


def _check_internal_token() -> Optional[Tuple[Any, int]]:
    provided = request.headers.get("X-Internal-Token", "")
    if not provided or not hmac.compare_digest(provided, Config.INTERNAL_TOKEN):
        return jsonify(error_response("UNAUTHORIZED", "Token internal tidak valid.")), 401
    return None


def _extract_image_and_analyze() -> Tuple[Optional[Any], Optional[list[DetectedFace]], Optional[Any], int]:
    """Validasi token, baca file 'image' di memori, dan jalankan deteksi wajah."""
    auth_err = _check_internal_token()
    if auth_err is not None:
        return None, None, auth_err[0], auth_err[1]

    if not face_engine.ready:
        return None, None, jsonify(error_response("AI_UNAVAILABLE", "Model AI belum siap.")), 503

    file = request.files.get("image")
    if file is None:
        return None, None, jsonify(error_response("BAD_REQUEST", "Field 'image' wajib diisi.")), 400

    image_bytes = file.read()
    if not image_bytes:
        return None, None, jsonify(error_response("BAD_REQUEST", "File gambar kosong.")), 400

    image_bgr = face_engine.decode_image(image_bytes)
    if image_bgr is None:
        return None, None, jsonify(error_response("INVALID_IMAGE", "Format gambar tidak dapat dibaca.")), 400

    faces = face_engine.analyze(image_bgr)
    return image_bgr, faces, None, 200


def _validate_single_face_with_quality(
    image_bgr: Any,
    faces: list[DetectedFace],
) -> Tuple[Optional[DetectedFace], Optional[dict[str, Any]], Optional[dict[str, Any]]]:
    """Pastikan tepat 1 wajah terdeteksi dan lolos pengecekan kualitas."""
    count = len(faces)
    m_info = model_info(face_engine.model_name, face_engine.model_version)

    if count == 0:
        return None, None, {
            "count": 0,
            "model": m_info,
            **error_response("NO_FACE", "Tidak ada wajah yang terdeteksi."),
        }

    if count > 1:
        return None, None, {
            "count": count,
            "model": m_info,
            **error_response("MULTIPLE_FACES", "Terdeteksi lebih dari satu wajah."),
        }

    face = faces[0]
    quality = assess_face_quality(image_bgr, face.bbox)
    if not quality["passed"]:
        return None, quality, {
            "count": 1,
            "quality": quality,
            "model": m_info,
            **error_response("LOW_QUALITY", "Kualitas foto wajah tidak memenuhi syarat."),
        }

    return face, quality, None


@api_bp.get("/health")
def health():
    if not face_engine.ready:
        return jsonify({
            "status": "starting",
            "model": model_info(face_engine.model_name, face_engine.model_version),
        }), 503

    return jsonify({
        "status": "ok",
        "model": model_info(face_engine.model_name, face_engine.model_version),
    }), 200


@api_bp.post("/v1/detect")
def detect():
    image_bgr, faces, err_resp, status_code = _extract_image_and_analyze()
    if err_resp is not None or faces is None or image_bgr is None:
        return err_resp, status_code

    quality = assess_face_quality(image_bgr, faces[0].bbox) if len(faces) == 1 else None
    return jsonify({
        "count": len(faces),
        "faces": [{"bbox": f.bbox, "score": f.score} for f in faces],
        "quality": quality,
    }), 200


@api_bp.post("/v1/embed")
def embed():
    image_bgr, faces, err_resp, status_code = _extract_image_and_analyze()
    if err_resp is not None or faces is None or image_bgr is None:
        return err_resp, status_code

    face, quality, rejection = _validate_single_face_with_quality(image_bgr, faces)
    if rejection is not None or face is None or quality is None:
        return jsonify(rejection), 200

    return jsonify({
        "count": 1,
        "quality": quality,
        "embedding": face.embedding,
        "dim": len(face.embedding),
        "model": model_info(face_engine.model_name, face_engine.model_version),
    }), 200


@api_bp.post("/v1/verify")
def verify():
    image_bgr, faces, err_resp, status_code = _extract_image_and_analyze()
    if err_resp is not None or faces is None or image_bgr is None:
        return err_resp, status_code

    raw_refs = request.form.get("references")
    if not raw_refs:
        return jsonify(error_response("BAD_REQUEST", "Field 'references' wajib diisi.")), 400

    try:
        references = json.loads(raw_refs)
    except json.JSONDecodeError:
        return jsonify(error_response("BAD_REQUEST", "Field 'references' harus berupa JSON valid.")), 400

    if not isinstance(references, list) or len(references) == 0:
        return jsonify(error_response("BAD_REQUEST", "Field 'references' harus berupa array embedding non-kosong.")), 400

    face, quality, rejection = _validate_single_face_with_quality(image_bgr, faces)
    if rejection is not None or face is None or quality is None:
        return jsonify(rejection), 200

    similarities = [
        face_engine.cosine_similarity(face.embedding, [float(v) for v in ref])
        for ref in references
        if isinstance(ref, list)
    ]
    if not similarities:
        return jsonify(error_response("BAD_REQUEST", "Format vektor pada 'references' tidak valid.")), 400

    similarity_max = max(similarities)
    print(f"[AI /v1/verify] count=1 quality_passed=True similarity_max={similarity_max}", flush=True)

    return jsonify({
        "count": 1,
        "quality": quality,
        "similarities": similarities,
        "similarity_max": similarity_max,
        "model": model_info(face_engine.model_name, face_engine.model_version),
    }), 200
