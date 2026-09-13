# Brain MRI Tumor Analyzer

An end-to-end web application that classifies a brain MRI slice as **glioma, meningioma, pituitary tumor or no tumor**, shows a **Grad-CAM heat-map** of the region that drove the decision, and explains what the finding means anatomically. Clinicians sign up, register patients, upload scans and keep a report history per patient.

> Research prototype for education only. It is not a medical device and must not be used for diagnosis.

## Results (held-out test set, 1,009 images)

| Model | Test accuracy (95% CI) | Test macro-F1 | Params |
|---|---|---|---|
| Custom CNN, trained from scratch (baseline) | 97.8% (96.7–98.8) | 0.978 | 2.4 M |
| **EfficientNet-B0, ImageNet transfer (deployed)** | **98.3% (97.3–99.2)** | **0.983** | 4.0 M |
| ResNet50, ImageNet transfer | 98.7% (97.9–99.4) | 0.987 | 23.5 M |

EfficientNet-B0 and ResNet50 tied on the validation set; the smaller model was deployed because the app runs on a CPU. The full evaluation — per-class metrics, confusion matrices, ROC curves, Grad-CAM review, unusual-input detection, response times and limitations — is in [reports/EVALUATION.md](reports/EVALUATION.md).

## How it works

1. **Data preparation** (`ml/prepare_data.py`). The 7,023 images from the Kaggle *Brain Tumor MRI Dataset* contain 297 byte-identical copies and many resized or contrast-shifted copies, and 327 duplicate groups crossed the original Training/Testing folders. Exact copies are removed, near-duplicates (perceptual hash) are grouped, and whole groups are split **70% train / 15% validation / 15% test**, stratified by class, with a fixed seed.
2. **Training** (`ml/train.py`). Three models on 224 × 224 inputs (aspect ratio kept by padding) with rotation, flip, zoom and brightness/contrast augmentation, AdamW, cosine learning rate, label smoothing and early stopping on validation macro-F1.
3. **Evaluation** (`ml/evaluate.py`). Accuracy, precision, recall, F1, ROC-AUC, confusion matrices, bootstrap confidence intervals and Grad-CAM galleries. The model is selected on validation data only.
4. **Export** (`ml/export_onnx.py`). The selected model is exported to ONNX with two extra outputs computed inside the graph: the class activation maps (mathematically identical to Grad-CAM for this architecture, verified by tests) and a Mahalanobis out-of-distribution score that flags uploads unlike brain MRI.
5. **Web app** (`app/`). FastAPI + Jinja2 pages, SQLite through SQLAlchemy, ONNX Runtime on CPU (no PyTorch in production). Features: sign-up/login (PBKDF2 password hashing, CSRF protection, session cookies), patient registration, scan upload with validation, report page with probabilities, heat-map, anatomical explanation and warnings, per-patient history, and a JSON endpoint `POST /api/predict`.

## Project layout

```
Training/, Testing/     raw dataset folders (not modified)
ml/                     data split, training, evaluation, Grad-CAM, ONNX export, benchmark
app/                    FastAPI web application (templates, static CSS)
data/splits/            train/val/test CSVs, split summary, removed duplicates
artifacts/              model.onnx + model_meta.json (deployed), models/<arch>/ checkpoints and logs
reports/                EVALUATION.md, metrics.json, figures, latency and OOD results
tests/                  split/leakage, Grad-CAM/ONNX parity, preprocessing and app tests
Dockerfile              production image (app only, ~22 MB model, no PyTorch)
```

## Run the app locally

Requires Python 3.11 or newer and the exported model in `artifacts/`. Run the commands from the project folder.

```bash
pip install -r requirements-app.txt
```

```bash
python -m uvicorn app.main:create_app --factory --port 8000
```

Open http://127.0.0.1:8000, create an account, register a patient and upload a JPEG or PNG MRI slice (for example an image from `Testing/`). Data is stored in `app_data/`.

## Reproduce the machine-learning pipeline

Install PyTorch for your GPU first (see `requirements.txt`), then:

```bash
pip install -r requirements.txt
```

Run these from the project folder, in order. Timings are for an RTX 4060 laptop GPU.

```bash
python -m ml.prepare_data
```

```bash
python -m ml.train --arch efficientnet_b0
```

```bash
python -m ml.train --arch resnet50
```

```bash
python -m ml.train --arch baseline_cnn
```

```bash
python -m ml.evaluate
```

```bash
python -m ml.export_onnx
```

Each training run takes about 10 minutes. To measure response times, start the app and run `python -m ml.benchmark_app`, then run `python -m ml.evaluate` again so the report includes the new timings and unusual-input results.

## Tests

```bash
python -m pytest
```

41 tests cover: no image, duplicate or near-duplicate shared across splits; 70/15/15 proportions per class; deterministic splitting; exported heat-maps equal to autograd Grad-CAM; ONNX outputs matching PyTorch; the unusual-input threshold; and the web app end to end (login required, CSRF, input validation, one user cannot open another user's patients or scans, no caching of patient pages).

## Deploy on AWS (EC2 + Docker)

The app needs about 180 MB of memory, so a free-tier **t3.micro** is enough.

1. Launch an EC2 instance with **Amazon Linux 2023**. In its security group allow **HTTP (80)** from anywhere and **SSH (22)** from your IP only.
2. Copy the deployable files from your computer (replace `key.pem` and `EC2_IP`):

```bash
ssh -i key.pem ec2-user@EC2_IP "mkdir -p brain-mri/ml brain-mri/artifacts"
```

```bash
scp -i key.pem Dockerfile .dockerignore requirements-app.txt ec2-user@EC2_IP:brain-mri/
```

```bash
scp -i key.pem -r app ec2-user@EC2_IP:brain-mri/
```

```bash
scp -i key.pem ml/__init__.py ml/config.py ml/preprocessing.py ml/explain.py ec2-user@EC2_IP:brain-mri/ml/
```

```bash
scp -i key.pem artifacts/model.onnx artifacts/model_meta.json ec2-user@EC2_IP:brain-mri/artifacts/
```

3. On the instance (`ssh -i key.pem ec2-user@EC2_IP`), install Docker, build and start the container:

```bash
sudo dnf install -y docker && sudo systemctl enable --now docker
```

```bash
cd brain-mri && sudo docker build -t brain-mri-app .
```

```bash
sudo docker run -d --name brain-mri --restart unless-stopped -p 80:8000 -e MRI_SECRET_KEY="$(openssl rand -hex 32)" -v brain-mri-data:/data brain-mri-app
```

4. Open `http://EC2_IP`. Check `http://EC2_IP/health` if the page does not load.

The database and uploaded scans live in the `brain-mri-data` Docker volume, so they survive restarts. For anything beyond a demo, serve the app over HTTPS (for example an Application Load Balancer with an ACM certificate, or Caddy with a domain) and add `-e MRI_HTTPS_ONLY_COOKIES=true`.

### Configuration

| Variable | Default | Purpose |
|---|---|---|
| `MRI_SECRET_KEY` | random per start (dev only) | Signs session cookies; required when `MRI_ENV=production` |
| `MRI_ENV` | `development` (`production` in Docker) | Production refuses to start without a secret key |
| `MRI_DATA_DIR` | `app_data/` (`/data` in Docker) | SQLite database and stored scan images |
| `MRI_DATABASE_URL` | SQLite in the data directory | Any SQLAlchemy URL, e.g. PostgreSQL on Amazon RDS |
| `MRI_MAX_UPLOAD_MB` | `10` | Upload size limit |
| `MRI_HTTPS_ONLY_COOKIES` | `false` | Set to `true` when served over HTTPS |

## Limitations

- The public dataset has no patient IDs, so near-duplicate grouping approximates, but does not guarantee, patient-level separation.
- "No tumor" images come from different sources than tumour images (only ~1% are 512 × 512 against 85–100% for tumour classes), and Grad-CAM shows the model partly uses such image-level cues. Expect lower accuracy on scans from a single hospital; there is no external validation.
- Single 2D slices only; DICOM and 3D volumes are not supported.

## Dataset

Masoud Nickparvar, *Brain Tumor MRI Dataset*, Kaggle (2021), combining the Figshare, SARTAJ and Br35H collections. Place its `Training/` and `Testing/` folders in the project root.
