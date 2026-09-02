import asyncio
import base64
import io
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError

from backend.inference import process_image, load_model

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}


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


def _encode_image(img_array, quality: int = 85) -> str:
    buf = io.BytesIO()
    Image.fromarray(img_array).save(buf, format="JPEG", quality=quality, optimize=True)
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
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
