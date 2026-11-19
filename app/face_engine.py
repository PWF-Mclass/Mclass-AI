from dataclasses import dataclass
from typing import Optional
import cv2
import numpy as np

from app.config import Config


@dataclass
class DetectedFace:
    bbox: list[int]  # [x, y, w, h]
    score: float
    embedding: list[float]


class FaceEngine:
    """Wrapper model MTCNN + InceptionResnetV1 (512-dim L2-normalized) yang dimuat sekali saat startup."""

    def __init__(self) -> None:
        self._mtcnn = None
        self._resnet = None
        self._torch = None
        self.model_name = Config.MODEL_NAME
        self.model_version = Config.MODEL_VERSION
        self.ready = False

    def load(self) -> None:
        import torch
        from facenet_pytorch import InceptionResnetV1, MTCNN

        self._torch = torch
        self._mtcnn = MTCNN(
            image_size=160,
            margin=20,
            min_face_size=Config.MIN_FACE_SIZE,
            keep_all=True,
            device="cpu",
        )
        self._resnet = InceptionResnetV1(pretrained="vggface2").eval()
        self.ready = True

    @staticmethod
    def decode_image(image_bytes: bytes) -> Optional[np.ndarray]:
        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        if arr.size == 0:
            return None
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)

    def analyze(self, image_bgr: np.ndarray) -> list[DetectedFace]:
        if not self.ready or self._mtcnn is None or self._resnet is None or self._torch is None:
            raise RuntimeError("FaceEngine is not initialized")

        from PIL import Image

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(image_rgb)

        boxes, probs = self._mtcnn.detect(pil_img)
        if boxes is None or probs is None:
            return []

        aligned_faces = self._mtcnn(pil_img)
        if aligned_faces is None:
            return []

        if aligned_faces.ndim == 3:
            aligned_faces = aligned_faces.unsqueeze(0)

        valid_indices = [
            i for i, p in enumerate(probs)
            if p is not None and float(p) >= Config.DET_CONFIDENCE and i < aligned_faces.shape[0]
        ]
        if not valid_indices:
            return []

        batch = aligned_faces[valid_indices]
        with self._torch.no_grad():
            embeddings = self._resnet(batch).cpu().numpy()

        results: list[DetectedFace] = []
        for idx, emb_raw in zip(valid_indices, embeddings):
            x1, y1, x2, y2 = [int(round(v)) for v in boxes[idx]]
            w = max(0, x2 - x1)
            h = max(0, y2 - y1)

            emb = np.asarray(emb_raw, dtype=np.float32)
            norm = float(np.linalg.norm(emb))
            if norm > 0:
                emb = emb / norm

            results.append(
                DetectedFace(
                    bbox=[x1, y1, w, h],
                    score=round(float(probs[idx]), 4),
                    embedding=[round(float(x), 6) for x in emb.tolist()],
                )
            )
        return results

    @staticmethod
    def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
        a = np.asarray(vec_a, dtype=np.float32)
        b = np.asarray(vec_b, dtype=np.float32)
        if a.shape != b.shape or a.size == 0:
            return 0.0
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        if denom == 0.0:
            return 0.0
        sim = float(np.dot(a, b) / denom)
        return round(max(-1.0, min(1.0, sim)), 5)


face_engine = FaceEngine()
