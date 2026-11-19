# mclass-ai (Flask Face Recognition Service)

Layanan AI internal untuk sistem absensi **M-Class** sesuai kontrak di `backend_mclass/requirements/04_TRD.md` (§7).

## Struktur
- `app/main.py` — Factory Flask & startup model (*warm*).
- `app/routes.py` — Endpoint `/health`, `/v1/detect`, `/v1/embed`, `/v1/verify`.
- `app/face_engine.py` — Wrapper **InsightFace (ArcFace `buffalo_l`)** + *cosine similarity*.
- `app/quality.py` — Penilaian blur, kecerahan, dan rasio luas wajah.
- `app/config.py` — Konfigurasi via environment variable.
- `gunicorn.conf.py` — Konfigurasi Gunicorn untuk Linux production.

## Setup Lokal (Windows / Linux)
```bash
cd mclass-ai
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
# source .venv/bin/activate

pip install -r requirements.txt
copy .env.example .env   # Linux: cp .env.example .env
python -m app.main
```

> Saat pertama kali dijalankan, InsightFace akan mengunduh pack model `buffalo_l` ke cache lokal (`~/.insightface`).

## Menjalankan Test
```bash
pytest
```

## Production (Linux + Gunicorn)
```bash
gunicorn -c gunicorn.conf.py "app.main:create_app()"
```
