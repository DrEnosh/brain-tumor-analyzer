import logging
import os
import secrets
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
log = logging.getLogger("app")


@dataclass(frozen=True)
class Settings:
    env: str
    data_dir: Path
    database_url: str
    secret_key: str
    model_path: Path
    model_meta_path: Path
    max_upload_bytes: int
    https_only_cookies: bool


def load_settings() -> Settings:
    # Variables are prefixed so generic names like DATABASE_URL left over from other projects are not picked up.
    env = os.getenv("MRI_ENV", "development")
    data_dir = Path(os.getenv("MRI_DATA_DIR", PROJECT_ROOT / "app_data"))
    secret = os.getenv("MRI_SECRET_KEY")
    if not secret:
        if env == "production":
            raise RuntimeError("MRI_SECRET_KEY must be set when MRI_ENV=production")
        log.warning("MRI_SECRET_KEY not set; using a random key, so sessions end when the server restarts")
        secret = secrets.token_urlsafe(48)
    return Settings(
        env=env,
        data_dir=data_dir,
        database_url=os.getenv("MRI_DATABASE_URL", f"sqlite:///{(data_dir / 'app.db').as_posix()}"),
        secret_key=secret,
        model_path=Path(os.getenv("MRI_MODEL_PATH", PROJECT_ROOT / "artifacts" / "model.onnx")),
        model_meta_path=Path(os.getenv("MRI_MODEL_META_PATH", PROJECT_ROOT / "artifacts" / "model_meta.json")),
        max_upload_bytes=int(float(os.getenv("MRI_MAX_UPLOAD_MB", "10")) * 1024 * 1024),
        https_only_cookies=os.getenv("MRI_HTTPS_ONLY_COOKIES", "false").lower() == "true",
    )
