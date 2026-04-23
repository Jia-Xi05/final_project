from __future__ import annotations

import sys
from pathlib import Path

from flask import Flask, send_from_directory

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from api.routes.analyze import analyze_bp  # noqa: E402
from config.settings import API_HOST, API_PORT, DEBUG, FRONTEND_DIR  # noqa: E402


def create_app() -> Flask:
    app = Flask(__name__, static_folder=str(FRONTEND_DIR), static_url_path="")
    app.register_blueprint(analyze_bp)

    @app.route("/")
    def index():
        return send_from_directory(FRONTEND_DIR, "index.html")

    return app


app = create_app()

if __name__ == "__main__":
    app.run(host=API_HOST, port=API_PORT, debug=DEBUG)
