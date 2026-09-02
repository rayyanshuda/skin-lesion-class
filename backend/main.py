import asyncio
import base64
import io
import os
import threading
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError

from backend.inference import process_image, load_model

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}

# Per-IP request cap to deter abuse: MAX_REQUESTS_PER_IP requests per
# RATE_LIMIT_WINDOW_SECONDS, per IP, on a rolling window. In-memory, so a
# window can reset early if the container restarts (redeploys, or
# scale-to-zero on inactivity) -- the real guarantee is "N per rolling window
# per container lifetime," not an absolute clock-based guarantee across
# restarts. That's the honest ceiling of what's achievable without external
# storage. IP address is used (not a cookie/localStorage flag) specifically
# because it survives incognito mode and cleared browser storage -- it's a
# property of the network connection, not client-side state.
MAX_REQUESTS_PER_IP = int(os.environ.get("MAX_REQUESTS_PER_IP", "3"))
RATE_LIMIT_WINDOW_SECONDS = int(os.environ.get("RATE_LIMIT_WINDOW_SECONDS", str(24 * 60 * 60)))
_request_log: dict[str, dict] = {}
_request_log_lock = threading.Lock()


def _client_ip(request: Request) -> str:
    # Cloud Run's load balancer sets X-Forwarded-For reliably; fall back to
    # the raw connection address for local dev, where there's no proxy.
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _usage(ip: str) -> dict:
    """Read-only: current remaining count and reset time for this IP."""
    now = time.time()
    with _request_log_lock:
        entry = _request_log.get(ip)
        if entry is None or (now - entry["window_start"]) >= RATE_LIMIT_WINDOW_SECONDS:
            return {
                "remaining": MAX_REQUESTS_PER_IP,
                "limit": MAX_REQUESTS_PER_IP,
                "reset_at": now + RATE_LIMIT_WINDOW_SECONDS,
            }
        return {
            "remaining": max(0, MAX_REQUESTS_PER_IP - entry["count"]),
            "limit": MAX_REQUESTS_PER_IP,
            "reset_at": entry["window_start"] + RATE_LIMIT_WINDOW_SECONDS,
        }


def _check_and_increment(ip: str) -> dict:
    """Mutates state: starts/continues this IP's window, returns whether this request is allowed."""
    now = time.time()
    with _request_log_lock:
        entry = _request_log.get(ip)
        if entry is None or (now - entry["window_start"]) >= RATE_LIMIT_WINDOW_SECONDS:
            entry = {"count": 0, "window_start": now}
        reset_at = entry["window_start"] + RATE_LIMIT_WINDOW_SECONDS
        if entry["count"] >= MAX_REQUESTS_PER_IP:
            _request_log[ip] = entry
            return {"allowed": False, "remaining": 0, "limit": MAX_REQUESTS_PER_IP, "reset_at": reset_at}
        entry["count"] += 1
        _request_log[ip] = entry
        return {
            "allowed": True,
            "remaining": MAX_REQUESTS_PER_IP - entry["count"],
            "limit": MAX_REQUESTS_PER_IP,
            "reset_at": reset_at,
        }


async def _load_model_in_background(app: FastAPI):
    loop = asyncio.get_event_loop()
    try:
        app.state.model = await loop.run_in_executor(None, load_model)
    except Exception as e:
        app.state.model_error = str(e)
    finally:
        app.state.model_loading = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Loading TensorFlow + ResNet50 can take long enough that a platform's
    # startup health check gives up and restarts the container before the
    # model finishes loading -- so start serving requests immediately and
    # load the model in the background instead of blocking startup on it.
    app.state.model = None
    app.state.model_error = None
    app.state.model_loading = True
    asyncio.create_task(_load_model_in_background(app))
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": app.state.model is not None,
        "model_loading": app.state.model_loading,
        **({"error": app.state.model_error} if app.state.model_error else {}),
    }


@app.get("/api/usage")
async def usage(request: Request):
    return _usage(_client_ip(request))


def _encode_image(img_array, quality: int = 85) -> str:
    buf = io.BytesIO()
    Image.fromarray(img_array).save(buf, format="JPEG", quality=quality, optimize=True)
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


@app.post("/api/predict")
async def predict(request: Request, file: UploadFile = File(...)):
    if app.state.model is None:
        if app.state.model_loading:
            raise HTTPException(status_code=503, detail="Model is still loading, please try again shortly.")
        raise HTTPException(status_code=503, detail=f"Model unavailable: {app.state.model_error}")

    extension = (file.filename or "").rsplit(".", 1)[-1].lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=415, detail="Unsupported file type. Please upload a JPG or PNG image.")

    data = await file.read()
    try:
        image = Image.open(io.BytesIO(data))
        image.load()
    except (UnidentifiedImageError, OSError):
        raise HTTPException(status_code=400, detail="Could not read image file.")

    ip = _client_ip(request)
    usage_info = _check_and_increment(ip)
    if not usage_info["allowed"]:
        return JSONResponse(
            status_code=429,
            content={
                "detail": f"You've reached the demo limit of {MAX_REQUESTS_PER_IP} images. Thanks for trying it out!",
                "remaining": usage_info["remaining"],
                "limit": usage_info["limit"],
                "reset_at": usage_info["reset_at"],
            },
        )

    try:
        pred_label, confidence, heatmap_colored, overlay, original = process_image(image, app.state.model)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {e}")

    certainty = abs(confidence - 0.5) * 2

    return {
        "prediction": pred_label,
        "confidence": confidence,
        "certainty": certainty,
        "images": {
            "original": _encode_image(original),
            "heatmap": _encode_image(heatmap_colored),
            "overlay": _encode_image(overlay),
        },
        "remaining": usage_info["remaining"],
        "limit": usage_info["limit"],
        "reset_at": usage_info["reset_at"],
    }


app.mount("/", StaticFiles(directory="static", html=True), name="static")
