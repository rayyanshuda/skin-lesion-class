"""Model loading and inference logic, ported from the original Streamlit app.py."""

import numpy as np
import tensorflow as tf
import cv2
from PIL import Image

MODEL_PATH = "model/skin_lesion_model_final.keras"


def focal_loss(alpha=0.25, gamma=2.0):
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        bce = tf.keras.backend.binary_crossentropy(y_true, y_pred)
        pt = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
        fl = alpha * tf.keras.backend.pow(1 - pt, gamma) * bce
        return tf.keras.backend.mean(fl)
    return loss


def load_model():
    """Load the trained model. Called once at app startup."""
    model = tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={"loss": focal_loss(alpha=0.25, gamma=2.0)}
    )
    return model


def generate_gradcam(model, img_array):
    """Generate Grad-CAM heatmap."""
    img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)
    resnet = model.get_layer("resnet50")

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
    """Process an uploaded image and return prediction + Grad-CAM visualizations.

    Returns (pred_label, confidence, heatmap_colored, overlay, original) where
    heatmap_colored/overlay/original are all 224x224 uint8 RGB numpy arrays.
    """
    if not isinstance(uploaded_image, Image.Image):
        uploaded_image = Image.open(uploaded_image)

    if uploaded_image.mode != 'RGB':
        uploaded_image = uploaded_image.convert('RGB')

    img_resized = uploaded_image.resize((224, 224))
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array, verbose=0)
    confidence = float(prediction[0][0])
    pred_label = "Malignant" if confidence > 0.5 else "Benign"

    heatmap = generate_gradcam(model, img_array)

    img_for_overlay = np.array(img_resized)
    heatmap_resized = cv2.resize(heatmap, (224, 224))
    heatmap_colored = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_colored, cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(img_for_overlay, 0.6, heatmap_colored, 0.4, 0)

    return pred_label, confidence, heatmap_colored, overlay, img_for_overlay
