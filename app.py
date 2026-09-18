"""
Mastery Image Intelligence Dashboard
------------------------------------
Final mastery project: professional image recognition with pre-trained
ImageNet models, ensemble inference, Top-K predictions, confidence
visualization, prediction diagnostics, and Grad-CAM explainability.

Run:
    streamlit run app.py
"""

from __future__ import annotations

import hashlib
import io
import time
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
from tensorflow.keras.applications import (
    EfficientNetV2B0,
    MobileNetV3Large,
    efficientnet_v2,
    mobilenet_v3,
)
from tensorflow.keras.applications.imagenet_utils import decode_predictions


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

APP_TITLE = "AI Vision Intelligence"
APP_SUBTITLE = "Advanced Image Recognition & Explainability Dashboard"

MODEL_A_NAME = "EfficientNetV2B0"
MODEL_B_NAME = "MobileNetV3Large"
IMAGE_SIZE = (224, 224)
TOP_K_MAX = 5

# Ensemble weights. EfficientNetV2B0 receives the larger contribution because
# it is the primary model for this dashboard.
WEIGHT_A = 0.65
WEIGHT_B = 0.35


@dataclass
class PredictionResult:
    label: str
    class_id: int
    confidence: float
    top_predictions: List[Tuple[str, int, float]]
    latency_ms: float


# ---------------------------------------------------------------------------
# Page setup / styling
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="AI",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1450px;
    }

    .hero {
        padding: 1.6rem 1.8rem;
        border: 1px solid rgba(128,128,128,.22);
        border-radius: 18px;
        margin-bottom: 1.2rem;
        background: linear-gradient(135deg, rgba(80,80,120,.12), rgba(30,30,60,.04));
    }

    .hero h1 {
        margin: 0;
        font-size: 2.35rem;
        letter-spacing: -0.04em;
    }

    .hero p {
        margin: .45rem 0 0;
        opacity: .72;
        font-size: 1.02rem;
    }

    .result-card {
        padding: 1.2rem 1.35rem;
        border: 1px solid rgba(128,128,128,.24);
        border-radius: 16px;
        margin-bottom: 1rem;
    }

    .prediction-name {
        font-size: 1.55rem;
        font-weight: 700;
        margin-bottom: .2rem;
    }

    .muted {
        opacity: .65;
        font-size: .88rem;
    }

    .confidence {
        font-size: 2.2rem;
        font-weight: 800;
        line-height: 1.05;
    }

    .section-label {
        font-size: .78rem;
        font-weight: 700;
        letter-spacing: .08em;
        text-transform: uppercase;
        opacity: .58;
        margin-bottom: .5rem;
    }

    .notice {
        padding: .9rem 1rem;
        border-radius: 12px;
        border: 1px solid rgba(128,128,128,.22);
        background: rgba(128,128,128,.06);
        font-size: .92rem;
    }

    footer {
        visibility: hidden;
    }
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    f"""
<div class="hero">
    <h1>{APP_TITLE}</h1>
    <p>{APP_SUBTITLE} &nbsp;•&nbsp; Pre-trained deep learning &nbsp;•&nbsp;
    Ensemble inference &nbsp;•&nbsp; Grad-CAM explainability</p>
</div>
""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading pre-trained vision models...")
def load_models():
    """
    Load both ImageNet-pretrained models once per Streamlit process.

    The models are used for inference only; no training is performed.
    """
    model_a = EfficientNetV2B0(
        include_top=True,
        weights="imagenet",
        input_shape=(224, 224, 3),
    )
    model_b = MobileNetV3Large(
        include_top=True,
        weights="imagenet",
        input_shape=(224, 224, 3),
    )
    return model_a, model_b


# ---------------------------------------------------------------------------
# Image handling
# ---------------------------------------------------------------------------

def read_uploaded_image(uploaded_file) -> Image.Image:
    """Validate and safely decode an uploaded image."""
    if uploaded_file is None:
        raise ValueError("Please upload an image first.")

    raw = uploaded_file.getvalue()
    if not raw:
        raise ValueError("The uploaded file is empty.")

    if len(raw) > 20 * 1024 * 1024:
        raise ValueError("The image is larger than the 20 MB safety limit.")

    try:
        image = Image.open(io.BytesIO(raw))
        image.verify()
        image = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as exc:
        raise ValueError(
            "Unable to read this file as a valid image. "
            "Please upload a JPG, JPEG, or PNG image."
        ) from exc

    # EXIF-aware orientation prevents sideways/rotated photos from hurting
    # recognition quality.
    image = ImageOps.exif_transpose(image).convert("RGB")
    return image


def image_to_tensor(image: Image.Image) -> np.ndarray:
    """Resize and convert a PIL image to a float32 batch."""
    resized = image.resize(IMAGE_SIZE, Image.Resampling.LANCZOS)
    array = np.asarray(resized, dtype=np.float32)
    return np.expand_dims(array, axis=0)


def stable_image_id(image: Image.Image) -> str:
    """Create a stable identifier for caching/display state."""
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return hashlib.sha256(buffer.getvalue()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def preprocess_for_model(
    batch: np.ndarray,
    model_name: str,
) -> np.ndarray:
    """
    Apply the official preprocessing function for the selected Keras model.
    """
    if model_name == MODEL_A_NAME:
        return efficientnet_v2.preprocess_input(batch.copy())
    if model_name == MODEL_B_NAME:
        return mobilenet_v3.preprocess_input(batch.copy())
    raise ValueError(f"Unsupported model: {model_name}")


def predict_ensemble(
    image: Image.Image,
    model_a: tf.keras.Model,
    model_b: tf.keras.Model,
    top_k: int,
) -> PredictionResult:
    """
    Run both pretrained classifiers and combine their ImageNet probability
    vectors with a weighted soft-voting ensemble.
    """
    batch = image_to_tensor(image)

    start = time.perf_counter()

    x_a = preprocess_for_model(batch, MODEL_A_NAME)
    x_b = preprocess_for_model(batch, MODEL_B_NAME)

    probs_a = model_a.predict(x_a, verbose=0)[0].astype(np.float64)
    probs_b = model_b.predict(x_b, verbose=0)[0].astype(np.float64)

    # Numerical safety: normalize each output before blending.
    probs_a = probs_a / max(probs_a.sum(), 1e-12)
    probs_b = probs_b / max(probs_b.sum(), 1e-12)

    ensemble = WEIGHT_A * probs_a + WEIGHT_B * probs_b
    ensemble = ensemble / max(ensemble.sum(), 1e-12)

    top_indices = np.argsort(ensemble)[::-1][:top_k]
    decoded = decode_predictions(
        ensemble[np.newaxis, :],
        top=top_k,
    )[0]

    # decode_predictions returns ImageNet class id, human-readable label,
    # and probability. Keep class ids from the decoded result where possible.
    top_predictions = [
        (label.replace("_", " ").title(), class_id, float(score))
        for class_id, label, score in decoded
    ]

    top_label = top_predictions[0][0]
    top_score = top_predictions[0][2]

    elapsed = (time.perf_counter() - start) * 1000

    # The index is useful for Grad-CAM because model output indices map to
    # ImageNet classes.
    class_index = int(top_indices[0])

    return PredictionResult(
        label=top_label,
        class_id=class_index,
        confidence=top_score,
        top_predictions=top_predictions,
        latency_ms=elapsed,
    )


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def confidence_band(confidence: float) -> Tuple[str, str]:
    """
    Describe the model score without pretending that softmax confidence is
    the same thing as guaranteed real-world correctness.
    """
    if confidence >= 0.80:
        return "High model confidence", "The leading class has a strong model score."
    if confidence >= 0.50:
        return "Moderate model confidence", "The model has a leading candidate, but alternatives remain relevant."
    return "Low model confidence", "The model is uncertain; review the Top-5 candidates and image quality."


def build_prediction_table(result: PredictionResult) -> pd.DataFrame:
    rows = []
    for rank, (label, _, score) in enumerate(result.top_predictions, start=1):
        rows.append(
            {
                "Rank": rank,
                "Prediction": label,
                "Model Score": f"{score * 100:.2f}%",
                "Score": score,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Grad-CAM
# ---------------------------------------------------------------------------

def find_last_4d_layer(model: tf.keras.Model) -> tf.keras.layers.Layer:
    """
    Find the last layer whose output is a spatial 4-D tensor.
    This avoids hard-coding a fragile internal layer name.
    """
    for layer in reversed(model.layers):
        try:
            shape = layer.output.shape
            if len(shape) == 4:
                return layer
        except Exception:
            continue
    raise ValueError("No suitable convolutional feature layer was found.")


def generate_gradcam(
    image: Image.Image,
    model: tf.keras.Model,
    model_name: str,
    class_index: int,
) -> np.ndarray:
    """
    Generate a Grad-CAM heatmap for the selected ImageNet class.

    Grad-CAM uses the final spatial feature representation and the gradient
    of the selected class score to identify image regions that contributed
    most strongly to the prediction.
    """
    last_conv_layer = find_last_4d_layer(model)

    batch = image_to_tensor(image)
    batch = preprocess_for_model(batch, model_name)

    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[last_conv_layer.output, model.output],
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(batch, training=False)
        class_score = predictions[:, class_index]

    grads = tape.gradient(class_score, conv_outputs)

    if grads is None:
        raise RuntimeError("Gradient computation returned no gradients.")

    conv_outputs = conv_outputs[0]
    grads = grads[0]

    # Global-average-pool gradients to obtain channel importance weights.
    weights = tf.reduce_mean(grads, axis=(0, 1))
    cam = tf.reduce_sum(conv_outputs * weights, axis=-1)

    cam = tf.maximum(cam, 0)
    max_value = tf.reduce_max(cam)
    cam = tf.where(max_value > 0, cam / max_value, cam)

    cam = cam.numpy()
    cam = tf.image.resize(cam[..., np.newaxis], IMAGE_SIZE).numpy()[..., 0]
    return np.clip(cam, 0.0, 1.0)


def make_gradcam_overlay(
    image: Image.Image,
    heatmap: np.ndarray,
    alpha: float = 0.42,
) -> Image.Image:
    """Create a clean heatmap overlay without requiring OpenCV."""
    import matplotlib.cm as cm

    original = image.resize(IMAGE_SIZE, Image.Resampling.LANCZOS)
    original_array = np.asarray(original, dtype=np.float32) / 255.0

    colormap = cm.get_cmap("jet")
    heat_rgb = colormap(heatmap)[..., :3].astype(np.float32)

    overlay = (1 - alpha) * original_array + alpha * heat_rgb
    overlay = np.clip(overlay * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(overlay)


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("### Control Center")
    top_k = st.slider(
        "Predictions to display",
        min_value=3,
        max_value=TOP_K_MAX,
        value=5,
        help="Shows the strongest ImageNet candidates returned by the ensemble.",
    )

    st.markdown("---")
    st.markdown("### Recognition Engine")
    st.write(f"**Primary:** {MODEL_A_NAME}")
    st.write(f"**Secondary:** {MODEL_B_NAME}")
    st.write(f"**Ensemble:** {WEIGHT_A:.0%} / {WEIGHT_B:.0%}")

    st.markdown("---")
    st.markdown("### Explainability")
    gradcam_enabled = st.toggle(
        "Generate Grad-CAM",
        value=True,
        help="Highlights spatial regions that contributed to the selected prediction.",
    )

    st.markdown("---")
    st.caption(
        "ImageNet recognition • 1,000 supported classes • "
        "Inference only • No user image is uploaded to an external API."
    )

try:
    model_a, model_b = load_models()
except Exception as exc:
    st.error(
        "The pre-trained vision models could not be loaded. "
        "Check your TensorFlow installation and internet access for the first "
        "ImageNet weight download."
    )
    with st.expander("Technical details"):
        st.code(str(exc))
    st.stop()

uploaded = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"],
    help="Use a clear image with a recognizable primary subject for best results.",
)

if uploaded is None:
    st.markdown(
        """
        <div class="notice">
        <strong>Ready for inference.</strong><br>
        Upload a JPG, JPEG, or PNG image. The system will validate it, correct
        EXIF orientation, preprocess it using the official model pipeline,
        run an ensemble prediction, show Top-K candidates, and optionally
        generate a Grad-CAM explanation.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

try:
    image = read_uploaded_image(uploaded)
except ValueError as exc:
    st.error(str(exc))
    st.stop()

image_id = stable_image_id(image)

col_input, col_output = st.columns([1.05, 1.25], gap="large")

with col_input:
    st.markdown('<div class="section-label">Input Image</div>', unsafe_allow_html=True)
    st.image(image, use_container_width=True)
    st.caption(f"Image ID: {image_id} • {image.width} × {image.height}px")

    recognize = st.button(
        "Run AI Recognition",
        type="primary",
        use_container_width=True,
    )

with col_output:
    st.markdown('<div class="section-label">Recognition Result</div>', unsafe_allow_html=True)

    if recognize or st.session_state.get("last_image_id") == image_id:
        try:
            result = predict_ensemble(
                image=image,
                model_a=model_a,
                model_b=model_b,
                top_k=top_k,
            )
            st.session_state["last_result"] = result
            st.session_state["last_image_id"] = image_id
        except Exception as exc:
            st.error("The image could not be processed. Please try another image.")
            with st.expander("Technical details"):
                st.code(str(exc))
            st.stop()

        band_title, band_detail = confidence_band(result.confidence)

        st.markdown(
            f"""
            <div class="result-card">
                <div class="muted">PRIMARY PREDICTION</div>
                <div class="prediction-name">{result.label}</div>
                <div class="muted">{band_title}</div>
                <div class="confidence">{result.confidence * 100:.2f}%</div>
                <div class="muted">{band_detail}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        m1, m2, m3 = st.columns(3)
        m1.metric("Top-1 Score", f"{result.confidence * 100:.2f}%")
        m2.metric("Top-5 Coverage", f"{sum(x[2] for x in result.top_predictions) * 100:.2f}%")
        m3.metric("Inference", f"{result.latency_ms:.0f} ms")

        st.markdown("#### Top Predictions")
        prediction_df = build_prediction_table(result)
        st.dataframe(
            prediction_df[["Rank", "Prediction", "Model Score"]],
            use_container_width=True,
            hide_index=True,
        )

        chart_df = prediction_df.set_index("Prediction")[["Score"]]
        st.bar_chart(chart_df, horizontal=True)

        if result.confidence < 0.50:
            st.warning(
                "The model is uncertain. For a reliable recognition result, "
                "use a well-lit image where the main object is centered and visible."
            )
        elif result.confidence < 0.80:
            st.info(
                "The leading prediction is plausible but not decisive. "
                "Compare the Top-5 results before treating it as the final class."
            )

    else:
        st.info("Click **Run AI Recognition** to start inference.")

# ---------------------------------------------------------------------------
# Explainability panel
# ---------------------------------------------------------------------------

if st.session_state.get("last_image_id") == image_id and st.session_state.get("last_result"):
    result = st.session_state["last_result"]

    st.markdown("---")
    st.markdown("## Model Explainability")
    st.caption(
        "Grad-CAM is an explanation of where the model's selected class was "
        "most strongly activated. It is not a pixel-perfect object boundary."
    )

    if gradcam_enabled:
        with st.spinner("Generating Grad-CAM explanation..."):
            try:
                heatmap = generate_gradcam(
                    image=image,
                    model=model_a,
                    model_name=MODEL_A_NAME,
                    class_index=result.class_id,
                )
                overlay = make_gradcam_overlay(image, heatmap)

                g1, g2 = st.columns(2, gap="large")
                with g1:
                    st.markdown("**Original image**")
                    st.image(image, use_container_width=True)
                with g2:
                    st.markdown(f"**Grad-CAM — {result.label}**")
                    st.image(overlay, use_container_width=True)

                st.success(
                    "Grad-CAM generated successfully. Warmer regions indicate "
                    "areas receiving stronger positive activation for the selected class."
                )
            except Exception as exc:
                st.warning(
                    "Grad-CAM could not be generated for this model/image "
                    "combination. The prediction above is unaffected."
                )
                with st.expander("Why this can happen"):
                    st.write(
                        "Grad-CAM depends on a compatible spatial feature layer "
                        "and a valid gradient path. The classifier prediction "
                        "does not depend on the Grad-CAM visualization."
                    )
                    st.code(str(exc))
    else:
        st.info("Grad-CAM is disabled. Enable it from the sidebar to inspect model attention.")

# ---------------------------------------------------------------------------
# Technical summary
# ---------------------------------------------------------------------------

st.markdown("---")
st.markdown("## System Architecture")

arch_cols = st.columns(7)
steps = [
    ("01", "Upload", "Validate image"),
    ("02", "Prepare", "Resize + preprocess"),
    ("03", "Infer", "Two pretrained models"),
    ("04", "Ensemble", "Weighted soft voting"),
    ("05", "Rank", "Top-K ImageNet classes"),
    ("06", "Explain", "Grad-CAM"),
    ("07", "Interpret", "Confidence + diagnostics"),
]

for col, (number, title, detail) in zip(arch_cols, steps):
    with col:
        st.markdown(f"**{number} — {title}**")
        st.caption(detail)

st.markdown(
    """
<div class="notice">
<strong>Important interpretation:</strong> a high softmax score is a model
confidence signal, not a guarantee that the image is correctly classified.
This application therefore exposes Top-K predictions and uncertainty guidance
instead of hiding alternative candidates.
</div>
""",
    unsafe_allow_html=True,
)
