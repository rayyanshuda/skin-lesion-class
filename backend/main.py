import base64
import io
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError

from backend.inference import process_image, load_model

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        app.state.model = load_model()
        app.state.model_error = None
    except Exception as e:
        app.state.model = None
        app.state.model_error = str(e)
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": app.state.model is not None,
        **({"error": app.state.model_error} if app.state.model_error else {}),
    }


def _encode_image(img_array, quality: int = 85) -> str:
    buf = io.BytesIO()
    Image.fromarray(img_array).save(buf, format="JPEG", quality=quality, optimize=True)
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    if app.state.model is None:
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
    }


app.mount("/", StaticFiles(directory="static", html=True), name="static")
