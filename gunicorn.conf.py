import os

bind = f"{os.getenv('AI_HOST', '127.0.0.1')}:{os.getenv('AI_PORT', '5000')}"
workers = int(os.getenv("AI_WORKERS", "2"))
threads = int(os.getenv("AI_THREADS", "2"))
timeout = int(os.getenv("AI_GUNICORN_TIMEOUT", "30"))
preload_app = True
