from typing import Any


def error_response(code: str, message: str) -> dict[str, Any]:
    return {
        "error": {
            "code": code,
            "message": message,
        }
    }


def model_info(name: str, version: str) -> dict[str, str]:
    return {
        "name": name,
        "version": version,
    }
