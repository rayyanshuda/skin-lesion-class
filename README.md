---
title: Skin Lesion Classifier
emoji: 🔬
colorFrom: yellow
colorTo: red
sdk: gradio
app_file: app.py
pinned: false
---

Backend for the skin lesion classifier at [skin-lesion-class.rayyanhuda.com](https://skin-lesion-class.rayyanhuda.com).

This Space runs a FastAPI app (mounted alongside a placeholder Gradio interface, which
is required by HF's free SDK tiers but otherwise unused) serving a ResNet50 + Grad-CAM
skin lesion classifier. Full project source: [github.com/rayyanshuda/skin-lesion-class](https://github.com/rayyanshuda/skin-lesion-class).

> ⚠️ MEDICAL DISCLAIMER: This is a research prototype only. NOT for medical diagnosis. Always consult healthcare professionals for medical advice.
