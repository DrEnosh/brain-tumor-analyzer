import io
import re

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app import security
from app.config import PROJECT_ROOT, Settings
from app.main import create_app
from ml.dataset import read_split

MODEL = PROJECT_ROOT / "artifacts" / "model.onnx"
META = PROJECT_ROOT / "artifacts" / "model_meta.json"
PASSWORD = "correct-horse-battery"


@pytest.fixture
def app(tmp_path, monkeypatch):
    if not MODEL.exists():
        pytest.skip("run `python -m ml.export_onnx` first")
    monkeypatch.setattr(security, "PBKDF2_ITERATIONS", 1_000)
    settings = Settings(env="test", data_dir=tmp_path, database_url=f"sqlite:///{(tmp_path / 'test.db').as_posix()}",
                        secret_key="test-secret", model_path=MODEL, model_meta_path=META,
                        max_upload_bytes=2 * 1024 * 1024, https_only_cookies=False)
    return create_app(settings)


@pytest.fixture
def client(app):
    return TestClient(app)


def csrf(client: TestClient, url: str) -> str:
    return re.search(r'name="csrf_token" value="([^"]+)"', client.get(url).text).group(1)


def signup(client: TestClient, email: str = "doctor@example.com", name: str = "Dr Test"):
    data = {"full_name": name, "email": email, "password": PASSWORD, "confirm_password": PASSWORD,
            "csrf_token": csrf(client, "/signup")}
    return client.post("/signup", data=data, follow_redirects=False)


def register_patient(client: TestClient, name: str = "Asha Rao", age: str = "54", sex: str = "Female"):
    data = {"full_name": name, "age": age, "sex": sex, "contact": "", "notes": "Headaches for 3 months",
            "csrf_token": csrf(client, "/patients/new")}
    return client.post("/patients", data=data, follow_redirects=False)


def mri_bytes(label: str = "pituitary") -> tuple[str, bytes]:
    frame = read_split("test")
    rel = frame[frame["label"] == label]["path"].iloc[0]
    return rel.rsplit("/", 1)[-1], (PROJECT_ROOT / rel).read_bytes()


def upload(client: TestClient, patient_id: int, filename: str, data: bytes, content_type: str = "image/jpeg"):
    token = csrf(client, f"/patients/{patient_id}")
    return client.post(f"/patients/{patient_id}/scans", data={"csrf_token": token},
                       files={"file": (filename, data, content_type)}, follow_redirects=False)


def test_health_reports_loaded_model(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["model_loaded"] is True and body["model_version"]


@pytest.mark.parametrize("path", ["/dashboard", "/patients/new", "/patients/1", "/scans/1", "/scans/1/image"])
def test_pages_require_login(client, path):
    response = client.get(path, follow_redirects=False)
    assert response.status_code == 303
    assert response.headers["location"].startswith("/login?next=")


def test_form_posts_without_csrf_token_are_rejected(client):
    response = client.post("/signup", data={"full_name": "X", "email": "x@example.com", "password": PASSWORD,
                                            "confirm_password": PASSWORD})
    assert response.status_code == 400
    assert "CSRF" in response.text


def test_signup_validation_and_duplicate_email(client):
    token = csrf(client, "/signup")
    bad = client.post("/signup", data={"full_name": "A", "email": "not-an-email", "password": "short",
                                       "confirm_password": "different", "csrf_token": token})
    assert bad.status_code == 400
    assert "valid email" in bad.text and "8 to 128" in bad.text and "do not match" in bad.text

    assert signup(client).status_code == 303
    other = TestClient(client.app)
    duplicate = signup(other, email="DOCTOR@example.com")
    assert duplicate.status_code == 400 and "already exists" in duplicate.text


def test_login_logout_and_open_redirect_protection(client):
    signup(client)
    client.post("/logout", data={"csrf_token": csrf(client, "/dashboard")})
    assert client.get("/dashboard", follow_redirects=False).status_code == 303

    token = csrf(client, "/login")
    wrong = client.post("/login", data={"email": "doctor@example.com", "password": "wrong-password",
                                        "next": "/dashboard", "csrf_token": token})
    assert wrong.status_code == 400 and "Invalid email or password" in wrong.text

    ok = client.post("/login", data={"email": "doctor@example.com", "password": PASSWORD,
                                     "next": "//evil.example.com", "csrf_token": csrf(client, "/login")},
                     follow_redirects=False)
    assert ok.status_code == 303 and ok.headers["location"] == "/dashboard"


def test_patient_validation(client):
    signup(client)
    response = register_patient(client, name="", age="200", sex="Unknown")
    assert response.status_code == 400
    assert "full name" in response.text and "between 0 and 120" in response.text and "Select the patient" in response.text


def test_end_to_end_register_upload_and_report(client, app):
    signup(client)
    created = register_patient(client)
    assert created.status_code == 303
    patient_id = int(created.headers["location"].rsplit("/", 1)[-1])
    assert "PT-" in client.get(f"/patients/{patient_id}").text

    filename, data = mri_bytes("pituitary")
    expected = app.state.classifier.analyze(data)
    response = upload(client, patient_id, filename, data)
    assert response.status_code == 303 and response.headers["location"].startswith("/scans/")
    scan_url = response.headers["location"]

    report = client.get(scan_url)
    assert report.status_code == 200
    assert "MRI analysis report" in report.text
    assert expected.display_name in report.text
    assert "Grad-CAM" in report.text and "Brain" in report.text
    assert expected.atypical_input is False and "Unusual input" not in report.text
    assert report.headers["cache-control"] == "no-store"
    for kind in ("image", "heatmap"):
        image = client.get(f"{scan_url}/{kind}")
        assert image.status_code == 200 and image.headers["content-type"] == "image/png"
        assert image.headers["cache-control"] == "no-store"
        assert Image.open(io.BytesIO(image.content)).size == (448, 448)

    dashboard = client.get("/dashboard").text
    assert "Asha Rao" in dashboard and expected.display_name in dashboard


def test_users_cannot_see_each_others_patients_or_scans(app):
    owner, intruder = TestClient(app), TestClient(app)
    signup(owner, email="owner@example.com")
    patient_id = int(register_patient(owner).headers["location"].rsplit("/", 1)[-1])
    filename, data = mri_bytes("glioma")
    scan_url = upload(owner, patient_id, filename, data).headers["location"]

    signup(intruder, email="intruder@example.com")
    assert intruder.get(f"/patients/{patient_id}").status_code == 404
    assert intruder.get(scan_url).status_code == 404
    assert intruder.get(f"{scan_url}/image").status_code == 404
    assert intruder.get(f"{scan_url}/heatmap").status_code == 404
    token = csrf(intruder, "/dashboard")
    attempt = intruder.post(f"/patients/{patient_id}/scans", data={"csrf_token": token},
                            files={"file": (filename, data, "image/jpeg")}, follow_redirects=False)
    assert attempt.status_code == 404
    assert "Asha Rao" not in intruder.get("/dashboard").text


def _gif() -> bytes:
    buffer = io.BytesIO()
    Image.new("L", (128, 128), 90).save(buffer, format="GIF")
    return buffer.getvalue()


INVALID_UPLOADS = {
    "not-an-image": ("scan.jpg", lambda: b"this is not an image", "image/jpeg", "not a readable image"),
    "gif": ("scan.gif", _gif, "image/gif", "Only JPEG and PNG"),
    "empty": ("scan.png", lambda: b"", "image/png", "empty"),
    "too-large": ("huge.jpg", lambda: b"\xff" * (2 * 1024 * 1024 + 10), "image/jpeg", "larger than"),
}


@pytest.mark.parametrize("case", list(INVALID_UPLOADS))
def test_invalid_uploads_are_rejected_without_creating_a_scan(client, case):
    filename, make_payload, content_type, message = INVALID_UPLOADS[case]
    payload = make_payload()
    signup(client)
    patient_id = int(register_patient(client).headers["location"].rsplit("/", 1)[-1])
    response = upload(client, patient_id, filename, payload, content_type)
    assert response.status_code == 303 and response.headers["location"] == f"/patients/{patient_id}"
    page = client.get(f"/patients/{patient_id}").text
    assert message in page
    assert "No scans analysed for this patient yet" in page


def test_colour_photo_gets_a_warning(client):
    signup(client)
    patient_id = int(register_patient(client).headers["location"].rsplit("/", 1)[-1])
    buffer = io.BytesIO()
    photo = Image.new("RGB", (256, 256), (30, 160, 60))
    photo.paste((220, 40, 40), (60, 60, 200, 200))
    photo.save(buffer, format="PNG")
    scan_url = upload(client, patient_id, "holiday.png", buffer.getvalue(), "image/png").headers["location"]
    assert "contains colour" in client.get(scan_url).text


def test_non_mri_image_is_marked_as_unusual_input(client):
    signup(client)
    patient_id = int(register_patient(client).headers["location"].rsplit("/", 1)[-1])
    rng = np.random.default_rng(7)
    buffer = io.BytesIO()
    Image.fromarray(rng.integers(0, 255, (256, 256), dtype="uint8")).save(buffer, format="PNG")
    scan_url = upload(client, patient_id, "noise.png", buffer.getvalue(), "image/png").headers["location"]
    assert "unlike the brain MRI slices the model was trained on" in client.get(scan_url).text
    assert "Unusual input" in client.get(f"/patients/{patient_id}").text


def test_api_predict_requires_login_and_returns_json(client):
    filename, data = mri_bytes("meningioma")
    assert client.post("/api/predict", files={"file": (filename, data, "image/jpeg")}).status_code == 401
    signup(client)
    body = client.post("/api/predict", files={"file": (filename, data, "image/jpeg")}).json()
    assert body["label"] in {"glioma", "meningioma", "notumor", "pituitary"}
    assert abs(sum(body["probabilities"].values()) - 1) < 1e-3
    assert body["attention"]["text"]
    assert body["atypical_input"] is False
    invalid = client.post("/api/predict", files={"file": ("x.jpg", b"nope", "image/jpeg")})
    assert invalid.status_code == 422


def test_password_hashing_roundtrip():
    stored = security.hash_password("s3cret-password")
    assert security.verify_password("s3cret-password", stored)
    assert not security.verify_password("wrong", stored)
    assert not security.verify_password("anything", "malformed-hash")
