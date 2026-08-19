import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import io

# Configure page
st.set_page_config(
    page_title="Skin Lesion Classifier",
    page_icon="🔬",
    layout="wide"
)

def inject_theme():
    """Restyle Streamlit's default widgets to match rayyanhuda.com's design system
    (coffee-cream palette, Spectral/Spline Sans Mono typography, flat bordered UI)."""
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Spectral:ital,wght@0,300;0,400;0,500;0,600;1,300;1,400&family=Spline+Sans+Mono:wght@400;500&display=swap');

    :root {
        --bg: #f3ece2;
        --text: #3d2f24;
        --muted: #7a6454;
        --divider: #d5c3b0;
        --underline: #b19479;
        --underline-hover: #3d2f24;
        --hover-surface: #ebdfd1;
        --hover-border: #d6c1ad;
        --grid-dot: rgba(122, 100, 84, 0.28);
    }

    /* base canvas + dotted grid backdrop, same recipe as index.html */
    .stApp {
        background-color: var(--bg) !important;
        position: relative;
    }
    .stApp::before {
        content: "";
        position: fixed;
        inset: 0;
        z-index: 0;
        pointer-events: none;
        font-family: "Spline Sans Mono", monospace;
        font-size: 15px;
        background-image: radial-gradient(circle, var(--grid-dot) 0.75px, transparent 0.75px);
        background-size: 1ch 1ex;
        background-position: center top;
    }
    [data-testid="stAppViewContainer"],
    [data-testid="stHeader"],
    .main .block-container {
        position: relative;
        z-index: 1;
        background: transparent !important;
    }

    /* typography */
    .stApp, .stApp p, .stApp li, .stApp span, .stApp label, .stMarkdown {
        font-family: "Spectral", Georgia, serif !important;
        color: var(--text);
    }
    h1, h2, h3 {
        font-family: "Spectral", Georgia, serif !important;
        font-weight: 400 !important;
        letter-spacing: -0.01em;
        color: var(--text) !important;
    }
    h1 { font-size: 2.4rem !important; }
    h2, h3 {
        font-size: 1.3rem !important;
        padding-bottom: 6px;
        border-bottom: 1px solid var(--divider);
    }
    .stMarkdown p, .stMarkdown li, .stMarkdown ol, .stMarkdown ul {
        font-size: 18px !important;
        line-height: 1.7 !important;
    }
    .stApp a {
        color: var(--text) !important;
        text-decoration: underline;
        text-decoration-color: var(--underline);
        transition: text-decoration-color 180ms ease;
    }
    .stApp a:hover {
        text-decoration-color: var(--underline-hover);
    }

    /* buttons */
    .stButton > button,
    button[kind="secondary"],
    button[data-testid^="stBaseButton"] {
        border-radius: 0 !important;
        border: 1px solid var(--underline) !important;
        background: transparent !important;
        color: var(--text) !important;
        font-family: "Spectral", Georgia, serif !important;
        box-shadow: none !important;
        padding: 6px 18px !important;
        min-height: 2.2em !important;
        transition: border-color 180ms ease, background-color 180ms ease;
    }
    .stButton > button:hover,
    button[kind="secondary"]:hover,
    button[data-testid^="stBaseButton"]:hover {
        border-color: var(--underline-hover) !important;
        background-color: var(--hover-surface) !important;
        color: var(--text) !important;
    }

    /* file uploader */
    [data-testid="stFileUploaderDropzone"] {
        border-radius: 0 !important;
        border: 1px dashed var(--divider) !important;
        background: transparent !important;
        min-height: 96px !important;
        padding: 18px 22px !important;
        display: flex !important;
        align-items: center !important;
        gap: 16px !important;
        flex-wrap: wrap !important;
    }
    [data-testid="stFileUploaderDropzone"] * {
        color: var(--muted) !important;
        font-family: "Spline Sans Mono", monospace !important;
    }
    [data-testid="stFileUploaderDropzone"] svg {
        color: var(--muted) !important;
        fill: var(--muted) !important;
        flex-shrink: 0;
    }
    [data-testid="stFileUploaderDropzone"] section {
        width: auto !important;
        min-height: 0 !important;
    }
    /* the "Browse files" button's icon+label were overlapping once the shared
       button rule above gave it padding/min-height it wasn't laid out for;
       force it into a normal flex row so icon and label sit side by side */
    [data-testid="stFileUploaderDropzone"] button {
        position: relative !important;
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
        gap: 8px !important;
        padding: 6px 16px !important;
        min-height: 2.2em !important;
        white-space: nowrap !important;
    }
    [data-testid="stFileUploaderDropzone"] button * {
        position: static !important;
    }

    /* metrics */
    [data-testid="stMetric"] {
        background: transparent !important;
        border: 1px solid var(--divider);
        padding: 12px 14px;
        border-radius: 0;
    }
    [data-testid="stMetricLabel"] {
        font-family: "Spline Sans Mono", monospace !important;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-size: 11px !important;
        color: var(--muted) !important;
    }
    [data-testid="stMetricValue"] {
        font-family: "Spectral", Georgia, serif !important;
        color: var(--text) !important;
    }

    /* alerts (error/success/warning/info) */
    div[data-baseweb="notification"],
    [data-testid="stAlert"] {
        border-radius: 0 !important;
        border: 1px solid var(--divider) !important;
        background: #eecfc1 !important;
    }
    [data-testid="stAlert"] p, [data-testid="stAlert"] span {
        font-size: 18px !important;
    }
    [data-testid="stAlertContentError"] {
        background: #eecfc1 !important;
        border-left: 3px solid #a5493a !important;
    }
    [data-testid="stAlertContentSuccess"] {
        background: #eecfc1 !important;
        border-left: 3px solid #5f7a4f !important;
    }
    [data-testid="stAlertContentWarning"] {
        background: #eecfc1 !important;
        border-left: 3px solid #a67c2e !important;
    }
    [data-testid="stAlertContentInfo"] {
        background: #eecfc1 !important;
        border-left: 3px solid var(--underline) !important;
    }

    /* spinner + image captions */
    .stSpinner > div {
        font-family: "Spline Sans Mono", monospace !important;
        color: var(--muted) !important;
    }
    [data-testid="stImageCaption"] {
        font-family: "Spline Sans Mono", monospace !important;
        font-size: 11px !important;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        color: var(--muted) !important;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the trained model (cached to avoid reloading)"""
    # Load model from Google Drive
    model_path = "model/skin_lesion_model_final.keras"

    # Re-define focal loss for loading
    def focal_loss(alpha=0.25, gamma=2.0):
        def loss(y_true, y_pred):
            y_true = tf.cast(y_true, tf.float32)
            bce = tf.keras.backend.binary_crossentropy(y_true, y_pred)
            pt = tf.where(tf.equal(y_true, 1), y_pred, 1-y_pred)
            fl = alpha * tf.keras.backend.pow(1-pt, gamma) * bce
            return tf.keras.backend.mean(fl)
        return loss

    model = tf.keras.models.load_model(
        model_path,
        custom_objects={"loss": focal_loss(alpha=0.25, gamma=2.0)}
    )
    return model

def generate_gradcam(model, img_array):
    """Generate Grad-CAM heatmap"""
    img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)
    resnet = model.get_layer("resnet50")

    # Auto-detect layer names
    gap_layer_name = None
    dropout_layer_name = None
    dense_layer_name = None

    for layer in model.layers:
        if 'global_average_pooling2d' in layer.name:
            gap_layer_name = layer.name
        elif 'dropout' in layer.name:
            dropout_layer_name = layer.name
        elif 'dense' in layer.name:
            dense_layer_name = layer.name

    with tf.GradientTape() as tape:
        tape.watch(img_tensor)
        features = resnet(img_tensor, training=False)
        x = model.get_layer(gap_layer_name)(features)
        x = model.get_layer(dropout_layer_name)(x, training=False)
        predictions = model.get_layer(dense_layer_name)(x)
        class_output = predictions[0, 0]

    grads = tape.gradient(class_output, features)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    features = tf.squeeze(features)
    heatmap = tf.reduce_sum(features * pooled_grads, axis=-1)
    heatmap = tf.nn.relu(heatmap)
    heatmap = heatmap / tf.reduce_max(heatmap)

    return heatmap.numpy()

def process_image(uploaded_image, model):
    """Process uploaded image and generate prediction + Grad-CAM"""
    # Convert to PIL if needed
    if not isinstance(uploaded_image, Image.Image):
        uploaded_image = Image.open(uploaded_image)

    # Make sure image is RGB (fixes RGBA/grayscale uploads)
    if uploaded_image.mode != 'RGB':
        uploaded_image = uploaded_image.convert('RGB')

    # Resize and preprocess
    img_resized = uploaded_image.resize((224, 224))
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Get prediction
    prediction = model.predict(img_array, verbose=0)
    confidence = float(prediction[0][0])
    pred_label = "Malignant" if confidence > 0.5 else "Benign"

    # Generate Grad-CAM
    heatmap = generate_gradcam(model, img_array)

    # Create overlay
    img_for_overlay = np.array(img_resized)
    heatmap_resized = cv2.resize(heatmap, (224, 224))
    heatmap_colored = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_colored, cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(img_for_overlay, 0.6, heatmap_colored, 0.4, 0)

    return pred_label, confidence, heatmap_resized, overlay, img_for_overlay

# Main app
def main():
    inject_theme()

    st.title("Skin Lesion Classifier")
    st.markdown("### Upload a skin lesion image for AI analysis")

    # medical disclaimer
    st.error("""
    ⚠️ **MEDICAL DISCLAIMER**: This is a research prototype only.
    NOT for medical diagnosis. Always consult healthcare professionals for medical advice.
    """)

    # Load model
    try:
        model = load_model()
        st.success("Model loaded successfully!")
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.stop()

    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=['jpg', 'jpeg', 'png'],
        help="Upload a clear image of the skin lesion"
    )

    if uploaded_file is not None:
        try:
            # Process the image
            with st.spinner('Analyzing image...'):
                pred_label, confidence, heatmap, overlay, original = process_image(uploaded_file, model)

            # Display results
            col1, col2 = st.columns([1, 2])

            with col1:
                st.subheader("Prediction Results")

                # Prediction with color coding
                if pred_label == "Malignant":
                    st.error(f"**Prediction: {pred_label}**")
                else:
                    st.success(f"**Prediction: {pred_label}**")

                # Calculate certainty (distance from 50%)
                certainty = abs(confidence - 0.5) * 2  # Convert to 0-100% scale

                # Display both metrics with tooltips
                col1_left, col1_right = st.columns(2)

                with col1_left:
                    st.metric(
                        "Model Certainty",
                        f"{certainty:.1%}",
                        help="How confident the model is in its prediction. Higher values mean the model is more sure of its decision. Calculated as the distance from 50% uncertainty."
                    )

                with col1_right:
                    st.metric(
                        "Malignancy Probability",
                        f"{confidence:.1%}",
                        help="Raw probability that the lesion is malignant. Values near 0% suggest benign, values near 100% suggest malignant, values near 50% indicate uncertainty."
                    )

                # Certainty interpretation with clearer logic
                if certainty > 0.6:  # >80% or <20% malignancy prob
                    st.info("High certainty prediction")
                elif certainty > 0.2:  # 60-80% or 20-40% malignancy prob
                    st.warning("Moderate certainty")
                else:  # 40-60% malignancy prob
                    st.error("Low certainty - uncertain prediction")

            with col2:
                st.subheader("Visual Analysis")

                # Create three columns for images
                img_col1, img_col2, img_col3 = st.columns(3)

                with img_col1:
                    st.image(original, caption="Original Image", use_container_width=True)

                with img_col2:
                    # Colorize the same way the overlay does, so all three panels match in size/crop
                    heatmap_display = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
                    heatmap_display = cv2.cvtColor(heatmap_display, cv2.COLOR_BGR2RGB)
                    st.image(heatmap_display, caption="Grad-CAM Heatmap", use_container_width=True)

                with img_col3:
                    st.image(overlay, caption="Overlay", use_container_width=True)

            # Explanation
            st.subheader("What is the AI looking at?")
            st.markdown("""
            The **Grad-CAM heatmap** shows which parts of the image the AI focused on:
            - 🔴 **Red areas**: High importance for the prediction
            - 🟡 **Yellow areas**: Moderate importance
            - 🔵 **Blue areas**: Low importance (background)

            Good AI should focus on medically relevant features like:
            - Lesion borders and shape irregularities
            - Color variations within the lesion
            - Texture and surface characteristics
            """)

        except Exception as e:
            st.error(f"Error processing image: {str(e)}")

if __name__ == "__main__":
    main()
