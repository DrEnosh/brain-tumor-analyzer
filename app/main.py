import json
import logging
import re
import uuid
from urllib.parse import quote

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy import func, select
from sqlalchemy.orm import Session, joinedload, selectinload
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware.sessions import SessionMiddleware

from app import security
from app.config import PROJECT_ROOT, Settings, load_settings
from app.content import BODY_PART, CLASS_INFO, DISCLAIMER, IMAGING
from app.database import Patient, Scan, User, make_session_factory
from app.inference import LOW_CONFIDENCE_THRESHOLD, InvalidImage, TumorClassifier
from ml.config import DISPLAY_NAMES

log = logging.getLogger("app")
APP_DIR = PROJECT_ROOT / "app"
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
SEXES = ("Female", "Male", "Other")
templates = Jinja2Templates(directory=APP_DIR / "templates")


class LoginRequired(Exception):
    pass


def get_db(request: Request):
    db = request.app.state.SessionLocal()
    try:
        yield db
    finally:
        db.close()


def current_user(request: Request, db: Session = Depends(get_db)) -> User:
    user_id = request.session.get("user_id")
    user = db.get(User, user_id) if user_id else None
    if user is None:
        raise LoginRequired()
    return user


def api_user(request: Request, db: Session = Depends(get_db)) -> User:
    user_id = request.session.get("user_id")
    user = db.get(User, user_id) if user_id else None
    if user is None:
        raise HTTPException(status_code=401, detail="Log in first.")
    return user


def flash(request: Request, text: str, kind: str = "success") -> None:
    request.session.setdefault("flashes", []).append({"kind": kind, "text": text})


def render(request: Request, name: str, status_code: int = 200, **context) -> HTMLResponse:
    context.update(
        csrf_token=security.csrf_token(request),
        flashes=request.session.pop("flashes", []),
        signed_in=bool(request.session.get("user_id")),
        user_name=request.session.get("user_name"),
        disclaimer=DISCLAIMER,
    )
    return templates.TemplateResponse(request, name, context, status_code=status_code)


def redirect(url: str) -> RedirectResponse:
    return RedirectResponse(url, status_code=303)


def owned_patient(db: Session, user: User, patient_id: int) -> Patient:
    patient = db.get(Patient, patient_id)
    if patient is None or patient.owner_id != user.id:
        raise HTTPException(status_code=404, detail="Patient not found.")
    return patient


def owned_scan(db: Session, user: User, scan_id: int) -> Scan:
    scan = db.get(Scan, scan_id, options=[joinedload(Scan.patient)])
    if scan is None or scan.patient.owner_id != user.id:
        raise HTTPException(status_code=404, detail="Scan not found.")
    return scan


def start_session(request: Request, user: User) -> None:
    request.session.clear()
    request.session["user_id"] = user.id
    request.session["user_name"] = user.full_name


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or load_settings()
    (settings.data_dir / "scans").mkdir(parents=True, exist_ok=True)

    app = FastAPI(title="Brain MRI Tumor Analyzer", version="1.0")
    app.state.settings = settings
    app.state.SessionLocal = make_session_factory(settings.database_url)
    app.state.classifier = None
    if settings.model_path.exists() and settings.model_meta_path.exists():
        app.state.classifier = TumorClassifier(settings.model_path, settings.model_meta_path)
    else:
        log.error("model not found at %s; uploads will be rejected", settings.model_path)

    app.add_middleware(SessionMiddleware, secret_key=settings.secret_key, session_cookie="mri_session",
                       max_age=8 * 3600, same_site="lax", https_only=settings.https_only_cookies)
    app.mount("/static", StaticFiles(directory=APP_DIR / "static"), name="static")

    @app.middleware("http")
    async def limit_body_and_disable_caching(request: Request, call_next):
        length = request.headers.get("content-length")
        if length and length.isdigit() and int(length) > settings.max_upload_bytes + 64_000:
            return JSONResponse({"detail": "Upload too large."}, status_code=413)
        response = await call_next(request)
        # Patient pages and scans must never be served from a browser cache (shared workstations, reused IDs).
        if not request.url.path.startswith("/static/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    @app.exception_handler(LoginRequired)
    async def _login_required(request: Request, _exc: LoginRequired):
        return redirect(f"/login?next={quote(request.url.path)}")

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(request: Request, exc: StarletteHTTPException):
        if request.url.path.startswith(("/api/", "/health", "/static/")):
            return JSONResponse({"detail": exc.detail}, status_code=exc.status_code)
        return render(request, "error.html", status_code=exc.status_code, message=exc.detail)

    def read_upload(file: UploadFile) -> bytes:
        data = file.file.read(settings.max_upload_bytes + 1)
        if len(data) > settings.max_upload_bytes:
            raise InvalidImage(f"The file is larger than {settings.max_upload_bytes // (1024 * 1024)} MB.")
        if not data:
            raise InvalidImage("The uploaded file is empty.")
        return data

    @app.get("/health")
    def health(request: Request):
        classifier = request.app.state.classifier
        return {"status": "ok", "model_loaded": classifier is not None,
                "model_version": classifier.version if classifier else None}

    @app.get("/", response_class=HTMLResponse)
    def index(request: Request):
        if request.session.get("user_id"):
            return redirect("/dashboard")
        return render(request, "index.html")

    @app.get("/signup", response_class=HTMLResponse)
    def signup_form(request: Request):
        return render(request, "signup.html", form={}, errors=[])

    @app.post("/signup", response_class=HTMLResponse)
    def signup(request: Request, full_name: str = Form(""), email: str = Form(""), password: str = Form(""),
               confirm_password: str = Form(""), csrf_token: str = Form(""), db: Session = Depends(get_db)):
        security.check_csrf(request, csrf_token)
        full_name, email = full_name.strip(), email.strip().lower()
        errors = []
        if not 1 <= len(full_name) <= 120:
            errors.append("Enter your full name (up to 120 characters).")
        if len(email) > 254 or not EMAIL_RE.match(email):
            errors.append("Enter a valid email address.")
        if not 8 <= len(password) <= 128:
            errors.append("Password must be 8 to 128 characters.")
        if password != confirm_password:
            errors.append("Passwords do not match.")
        if not errors and db.scalar(select(User).where(User.email == email)):
            errors.append("An account with this email already exists.")
        if errors:
            return render(request, "signup.html", status_code=400, form={"full_name": full_name, "email": email},
                          errors=errors)
        user = User(full_name=full_name, email=email, password_hash=security.hash_password(password))
        db.add(user)
        db.commit()
        start_session(request, user)
        flash(request, f"Welcome, {user.full_name}. Register your first patient to start.")
        return redirect("/dashboard")

    @app.get("/login", response_class=HTMLResponse)
    def login_form(request: Request, next: str = ""):
        return render(request, "login.html", form={}, next=security.safe_next_url(next), errors=[])

    @app.post("/login", response_class=HTMLResponse)
    def login(request: Request, email: str = Form(""), password: str = Form(""), next: str = Form(""),
              csrf_token: str = Form(""), db: Session = Depends(get_db)):
        security.check_csrf(request, csrf_token)
        email = email.strip().lower()
        user = db.scalar(select(User).where(User.email == email))
        if user is None or not security.verify_password(password, user.password_hash):
            return render(request, "login.html", status_code=400, form={"email": email},
                          next=security.safe_next_url(next), errors=["Invalid email or password."])
        start_session(request, user)
        return redirect(security.safe_next_url(next))

    @app.post("/logout")
    def logout(request: Request, csrf_token: str = Form("")):
        security.check_csrf(request, csrf_token)
        request.session.clear()
        return redirect("/")

    @app.get("/dashboard", response_class=HTMLResponse)
    def dashboard(request: Request, user: User = Depends(current_user), db: Session = Depends(get_db)):
        patients = db.scalars(select(Patient).where(Patient.owner_id == user.id)
                              .options(selectinload(Patient.scans)).order_by(Patient.created_at.desc())).all()
        recent = db.scalars(select(Scan).join(Patient).where(Patient.owner_id == user.id)
                            .options(joinedload(Scan.patient)).order_by(Scan.created_at.desc()).limit(8)).all()
        scan_count = db.scalar(select(func.count(Scan.id)).join(Patient).where(Patient.owner_id == user.id))
        return render(request, "dashboard.html", patients=patients, recent=recent, scan_count=scan_count,
                      class_names=_display_names(request))

    @app.get("/patients/new", response_class=HTMLResponse)
    def patient_form(request: Request, user: User = Depends(current_user)):
        return render(request, "patient_form.html", form={}, errors=[], sexes=SEXES)

    @app.post("/patients", response_class=HTMLResponse)
    def create_patient(request: Request, full_name: str = Form(""), age: str = Form(""), sex: str = Form(""),
                       contact: str = Form(""), notes: str = Form(""), csrf_token: str = Form(""),
                       user: User = Depends(current_user), db: Session = Depends(get_db)):
        security.check_csrf(request, csrf_token)
        form = {"full_name": full_name.strip(), "age": age.strip(), "sex": sex, "contact": contact.strip(),
                "notes": notes.strip()}
        errors = []
        if not 1 <= len(form["full_name"]) <= 120:
            errors.append("Enter the patient's full name (up to 120 characters).")
        if not form["age"].isdigit() or not 0 <= int(form["age"]) <= 120:
            errors.append("Age must be a whole number between 0 and 120.")
        if sex not in SEXES:
            errors.append("Select the patient's sex.")
        if len(form["contact"]) > 120:
            errors.append("Contact details must be 120 characters or fewer.")
        if len(form["notes"]) > 2000:
            errors.append("Notes must be 2000 characters or fewer.")
        if errors:
            return render(request, "patient_form.html", status_code=400, form=form, errors=errors, sexes=SEXES)
        patient = Patient(owner_id=user.id, full_name=form["full_name"], age=int(form["age"]), sex=sex,
                          contact=form["contact"], notes=form["notes"])
        db.add(patient)
        db.commit()
        flash(request, f"Registered {patient.full_name} as {patient.code}.")
        return redirect(f"/patients/{patient.id}")

    @app.get("/patients/{patient_id}", response_class=HTMLResponse)
    def patient_detail(patient_id: int, request: Request, user: User = Depends(current_user),
                       db: Session = Depends(get_db)):
        patient = owned_patient(db, user, patient_id)
        return render(request, "patient_detail.html", patient=patient, class_names=_display_names(request),
                      model_ready=request.app.state.classifier is not None)

    @app.post("/patients/{patient_id}/scans")
    def upload_scan(patient_id: int, request: Request, file: UploadFile = File(...), csrf_token: str = Form(""),
                    user: User = Depends(current_user), db: Session = Depends(get_db)):
        security.check_csrf(request, csrf_token)
        patient = owned_patient(db, user, patient_id)
        classifier = request.app.state.classifier
        if classifier is None:
            flash(request, "The model is not available on this server.", "error")
            return redirect(f"/patients/{patient.id}")
        try:
            result = classifier.analyze(read_upload(file))
        except InvalidImage as exc:
            flash(request, str(exc), "error")
            return redirect(f"/patients/{patient.id}")

        stem = uuid.uuid4().hex
        image_file, heatmap_file = f"{stem}_mri.png", f"{stem}_heatmap.png"
        result.display_image.save(settings.data_dir / "scans" / image_file)
        result.overlay_image.save(settings.data_dir / "scans" / heatmap_file)
        scan = Scan(patient_id=patient.id, original_filename=(file.filename or "upload")[:255],
                    image_file=image_file, heatmap_file=heatmap_file, predicted_label=result.label,
                    confidence=result.confidence, probabilities_json=json.dumps(result.probabilities),
                    attention_json=json.dumps(result.attention), colour_warning=result.colour_warning,
                    atypical_input=result.atypical_input, ood_score=result.ood_score,
                    model_version=classifier.version, inference_ms=result.inference_ms)
        db.add(scan)
        db.commit()
        return redirect(f"/scans/{scan.id}")

    @app.get("/scans/{scan_id}", response_class=HTMLResponse)
    def report(scan_id: int, request: Request, user: User = Depends(current_user), db: Session = Depends(get_db)):
        scan = owned_scan(db, user, scan_id)
        names = _display_names(request)
        probabilities = sorted(json.loads(scan.probabilities_json).items(), key=lambda kv: -kv[1])
        return render(request, "report.html", scan=scan, patient=scan.patient, class_names=names,
                      probabilities=probabilities, attention=json.loads(scan.attention_json),
                      info=CLASS_INFO[scan.predicted_label], body_part=BODY_PART, imaging=IMAGING,
                      low_confidence=scan.confidence < LOW_CONFIDENCE_THRESHOLD)

    @app.get("/scans/{scan_id}/{kind}")
    def scan_image(scan_id: int, kind: str, user: User = Depends(current_user), db: Session = Depends(get_db)):
        if kind not in ("image", "heatmap"):
            raise HTTPException(status_code=404)
        scan = owned_scan(db, user, scan_id)
        name = scan.image_file if kind == "image" else scan.heatmap_file
        return FileResponse(settings.data_dir / "scans" / name, media_type="image/png")

    @app.post("/api/predict")
    def api_predict(request: Request, file: UploadFile = File(...), user: User = Depends(api_user)):
        classifier = request.app.state.classifier
        if classifier is None:
            raise HTTPException(status_code=503, detail="Model not available.")
        try:
            result = classifier.analyze(read_upload(file))
        except InvalidImage as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {"label": result.label, "display_name": result.display_name, "confidence": result.confidence,
                "probabilities": result.probabilities, "attention": result.attention,
                "colour_warning": result.colour_warning, "atypical_input": result.atypical_input,
                "ood_score": round(result.ood_score, 1), "inference_ms": round(result.inference_ms, 1),
                "model_version": classifier.version}

    return app


def _display_names(request: Request) -> dict:
    classifier = request.app.state.classifier
    return classifier.meta["display_names"] if classifier is not None else DISPLAY_NAMES
