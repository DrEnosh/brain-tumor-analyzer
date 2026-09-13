"""Measure response times of a running app (local or deployed) through its real HTTP interface."""
import argparse
import json
import re
import secrets
import statistics
import time

import httpx
import numpy as np

from ml.config import PROJECT_ROOT, REPORTS_DIR
from ml.dataset import read_split


def stats(values: list[float]) -> dict:
    return {"mean": statistics.fmean(values), "median": statistics.median(values),
            "p95": float(np.percentile(values, 95)), "n": len(values)}


def csrf(client: httpx.Client, url: str) -> str:
    return re.search(r'name="csrf_token" value="([^"]+)"', client.get(url).text).group(1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--requests", type=int, default=40)
    parser.add_argument("--environment", default="local machine, CPU inference (onnxruntime)")
    args = parser.parse_args()

    images = read_split("test").sample(n=args.requests, random_state=0)["path"].tolist()
    with httpx.Client(base_url=args.base_url, timeout=60, follow_redirects=True) as client:
        password = secrets.token_urlsafe(16)
        email = f"benchmark-{secrets.token_hex(4)}@example.com"
        client.post("/signup", data={"full_name": "Latency Benchmark", "email": email, "password": password,
                                     "confirm_password": password, "csrf_token": csrf(client, "/signup")})
        created = client.post("/patients", data={"full_name": "Benchmark Patient", "age": "40", "sex": "Other",
                                                 "csrf_token": csrf(client, "/patients/new")})
        patient_url = str(created.url.path)
        if not patient_url.startswith("/patients/"):
            raise SystemExit(f"could not create a benchmark patient (landed on {patient_url})")

        client.post("/api/predict", files={"file": ("warmup.jpg", (PROJECT_ROOT / images[0]).read_bytes())})
        page_ms, api_ms, server_ms = [], [], []
        for rel in images:
            data = (PROJECT_ROOT / rel).read_bytes()
            token = csrf(client, patient_url)
            start = time.perf_counter()
            report = client.post(f"{patient_url}/scans", data={"csrf_token": token},
                                 files={"file": (rel.rsplit("/", 1)[-1], data, "image/jpeg")})
            page_ms.append((time.perf_counter() - start) * 1000)
            if "MRI analysis report" not in report.text:
                raise SystemExit(f"upload of {rel} did not produce a report")

            start = time.perf_counter()
            body = client.post("/api/predict", files={"file": ("scan.jpg", data, "image/jpeg")}).json()
            api_ms.append((time.perf_counter() - start) * 1000)
            server_ms.append(body["inference_ms"])

    result = {"base_url": args.base_url, "environment": args.environment, "requests": len(images),
              "upload_to_report_ms": stats(page_ms), "api_predict_ms": stats(api_ms),
              "server_inference_ms": stats(server_ms)}
    REPORTS_DIR.mkdir(exist_ok=True)
    (REPORTS_DIR / "latency.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
