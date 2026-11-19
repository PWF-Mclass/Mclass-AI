from dotenv import load_dotenv
from flask import Flask, jsonify

from app.config import Config
from app.face_engine import face_engine
from app.routes import api_bp
from app.schemas import error_response


def create_app(load_model: bool = True) -> Flask:
    load_dotenv()

    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = Config.MAX_CONTENT_LENGTH
    app.register_blueprint(api_bp)

    @app.errorhandler(413)
    def handle_too_large(_err):
        return jsonify(
            error_response("FILE_TOO_LARGE", f"Ukuran gambar melebihi batas {Config.PHOTO_MAX_MB} MB.")
        ), 413

    if load_model and not face_engine.ready:
        face_engine.load()

    return app


if __name__ == "__main__":
    application = create_app(load_model=True)
    application.run(host=Config.HOST, port=Config.PORT, debug=False)
