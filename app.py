"""Entrypoint for the Hugging Face Space (Gradio SDK, free tier).

HF's free tier only allows the Gradio/Static SDKs, not Docker. This mounts our
real FastAPI app (backend/main.py, completely unchanged) at "/", alongside a
throwaway Gradio interface at "/gradio-ui" that exists only to satisfy the
Gradio SDK requirement -- it is never used by the actual site.

Copy this file into the Space repo as `app.py` (HF's default entrypoint name).
"""
from contextlib import asynccontextmanager

import gradio as gr
from fastapi import FastAPI

from backend.main import app as skin_lesion_app, lifespan as skin_lesion_lifespan


@asynccontextmanager
async def combined_lifespan(_app: FastAPI):
    # Mounting a FastAPI app as a sub-app does NOT forward lifespan events to
    # it (confirmed locally: app.state.model was never set without this) --
    # drive the inner app's own lifespan manually instead.
    async with skin_lesion_lifespan(skin_lesion_app):
        yield


app = FastAPI(lifespan=combined_lifespan)

placeholder = gr.Interface(
    fn=lambda x: x,
    inputs=gr.Textbox(label="Not used -- the real app is served at /"),
    outputs=gr.Textbox(),
)
gr.mount_gradio_app(app, placeholder, path="/gradio-ui")

# Registered after the Gradio mount so Starlette matches "/gradio-ui" first;
# everything else falls through to our real app, mounted last as the catch-all.
app.mount("/", skin_lesion_app)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
