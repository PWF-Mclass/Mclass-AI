import os


class Config:
    HOST: str = os.getenv("AI_HOST", "127.0.0.1")
    PORT: int = int(os.getenv("AI_PORT", "5000"))
    INTERNAL_TOKEN: str = os.getenv("AI_INTERNAL_TOKEN", "dev-internal-token")

    # Batas ukuran upload gambar (sesuai PHOTO_MAX_MB di Express)
    PHOTO_MAX_MB: int = int(os.getenv("PHOTO_MAX_MB", "5"))
    MAX_CONTENT_LENGTH: int = PHOTO_MAX_MB * 1024 * 1024

    # Konfigurasi model FaceNet (MTCNN + InceptionResnetV1 vggface2)
    MODEL_NAME: str = os.getenv("AI_MODEL_NAME", "facenet_vggface2")
    MODEL_VERSION: str = os.getenv("AI_MODEL_VERSION", "1.0")
    MIN_FACE_SIZE: int = int(os.getenv("AI_MIN_FACE_SIZE", "40"))
    DET_CONFIDENCE: float = float(os.getenv("AI_DET_CONFIDENCE", "0.90"))

    # Ambang batas kualitas gambar wajah
    MIN_BLUR_SCORE: float = float(os.getenv("MIN_BLUR_SCORE", "40.0"))
    MIN_BRIGHTNESS: float = float(os.getenv("MIN_BRIGHTNESS", "40.0"))
    MAX_BRIGHTNESS: float = float(os.getenv("MAX_BRIGHTNESS", "235.0"))
    MIN_FACE_AREA_RATIO: float = float(os.getenv("MIN_FACE_AREA_RATIO", "0.04"))
