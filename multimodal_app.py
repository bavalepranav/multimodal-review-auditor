"""
AI-Powered Multimodal Product Review Auditor
============================================
Advanced Multimodal Portfolio Project (Classical ML + TensorFlow Deep Learning)

Pipeline
--------
1. Synthetic text dataset (45 labelled reviews) generated in-script.
2. Scikit-Learn Pipeline: TfidfVectorizer(stop_words='english') + LogisticRegression.
3. TensorFlow/Keras MobileNetV2 (ImageNet weights) for product-image analysis.
4. Multimodal decision layer combining sentiment + image evidence.
5. Streamlit dashboard with a visual audit summary card.

Run (GitHub Codespaces):
    pip install -r requirements.txt
    streamlit run multimodal_app.py --server.enableCORS false --server.enableXsrfProtection false
"""

from __future__ import annotations

# --- CPU-only / low-noise TensorFlow configuration (must precede the TF import) ---
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")   # force CPU
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")    # silence verbose logs
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")   # avoid oneDNN log noise

import html
import io
import re
from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps, UnidentifiedImageError
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

try:
    import tensorflow as tf

    TF_IMPORT_ERROR = None
except Exception as exc:  # noqa: BLE001 - surfaced to the user in the UI
    tf = None
    TF_IMPORT_ERROR = exc

# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------
IMG_SIZE = (224, 224)
MAX_UPLOAD_BYTES = 8 * 1024 * 1024   # 8 MB
MIN_TEXT_CHARS = 10
MAX_TEXT_CHARS = 3000

SENTIMENT_ORDER = ["Negative", "Neutral", "Positive"]
SENTIMENT_EMOJI = {"Negative": "😡", "Neutral": "😐", "Positive": "😀"}

# Decision thresholds
RELEVANCE_MATCH_THRESHOLD = 0.15      # min top-5 probability mass on product-matching classes
CONFIDENT_MISMATCH_THRESHOLD = 0.35   # top-1 confidence needed to call an image "clearly something else"
NEGATIVE_REFUND_CONFIDENCE = 0.60     # negative-text confidence that escalates to a refund request
MIN_TEXT_CONFIDENCE = 0.40            # below this the text signal is too weak to trust

REFUND_PATTERN = re.compile(
    r"\b(refund|money back|return(?:ed|ing)?|replacement|defective|faulty|"
    r"broke|broken|damaged|scam|fake|counterfeit)\b",
    re.IGNORECASE,
)

# ImageNet label-token groups used to test whether a photo plausibly shows the
# product category being reviewed. Matching is done on whole tokens, not
# substrings, to avoid false hits (e.g. "cap" vs "capuchin").
CATEGORY_KEYWORDS: Dict[str, FrozenSet[str]] = {
    "Electronics & Gadgets": frozenset({
        "telephone", "phone", "ipod", "laptop", "notebook", "computer", "desktop",
        "monitor", "screen", "television", "tv", "mouse", "keyboard", "remote",
        "speaker", "loudspeaker", "headphone", "headphones", "earphone", "joystick",
        "modem", "camera", "projector", "radio", "printer", "microphone", "disc",
        "tape", "player", "charger", "plug", "switch", "cd",
    }),
    "Fashion & Footwear": frozenset({
        "shoe", "boot", "sandal", "loafer", "clog", "sock", "jersey", "jean",
        "sweatshirt", "cardigan", "coat", "suit", "miniskirt", "skirt", "hoopskirt",
        "overskirt", "kimono", "poncho", "pajama", "trunks", "bikini", "maillot",
        "sombrero", "hat", "bonnet", "cap", "tie", "gown", "apron", "cloak",
        "stole", "abaya", "robe", "vestment", "uniform", "fur", "sunglasses",
        "sunglass", "shirt", "brassiere", "bra",
    }),
    "Bags & Accessories": frozenset({
        "backpack", "purse", "wallet", "pouch", "bag", "mailbag", "handbag",
        "umbrella", "watch", "clock", "stopwatch", "lipstick", "perfume", "lotion",
        "sunscreen", "spray", "buckle", "carton", "packet", "crate", "envelope",
        "basket",
    }),
    "Home & Kitchen": frozenset({
        "coffeepot", "espresso", "teapot", "toaster", "microwave", "refrigerator",
        "dishwasher", "washer", "vacuum", "measuring", "cup", "mixing", "bowl",
        "pan", "wok", "oven", "spatula", "ladle", "mug", "pitcher", "jug", "plate",
        "pot", "crock", "waffle", "opener", "corkscrew", "chair", "couch", "table",
        "lamp", "lampshade", "pillow", "quilt", "towel", "curtain", "wardrobe",
        "desk", "bookcase", "bottle", "iron", "fan", "stove", "rotisserie",
        "strainer",
    }),
    "Sports & Outdoors": frozenset({
        "ski", "dumbbell", "barbell", "racket", "racquet", "ball", "tennis",
        "basketball", "volleyball", "soccer", "golf", "rugby", "punching", "helmet",
        "bicycle", "bike", "unicycle", "tent", "sleeping", "paddle", "oar", "canoe",
        "snorkel", "scuba", "balance", "beam", "goggles",
    }),
    "Toys, Books & Hobbies": frozenset({
        "teddy", "jigsaw", "puzzle", "rubik", "cube", "pinwheel", "toyshop", "toy",
        "book", "comic", "binder", "pencil", "pen", "ballpoint", "fountain",
        "eraser", "sharpener", "paintbrush", "crayon", "harmonica", "guitar",
        "piano", "violin", "cello", "drum", "flute", "saxophone", "trumpet",
    }),
}
ANY_CATEGORY = "Any product (auto-detect)"
ALL_PRODUCT_KEYWORDS: FrozenSet[str] = frozenset().union(*CATEGORY_KEYWORDS.values())


# ----------------------------------------------------------------------------
# Data structures
# ----------------------------------------------------------------------------
@dataclass
class SentimentResult:
    label: str
    confidence: float
    probabilities: Dict[str, float]


@dataclass
class ImagePrediction:
    class_id: str
    label: str
    probability: float


@dataclass
class AuditResult:
    status: str
    level: str          # "success" | "danger" | "warning"
    icon: str
    summary: str
    reasons: List[str]
    relevance: float
    matched: List[ImagePrediction]


# ----------------------------------------------------------------------------
# 1. Synthetic text dataset
# ----------------------------------------------------------------------------
def build_review_dataset() -> pd.DataFrame:
    """Return a balanced, hand-written dataset of 45 product reviews."""
    positive = [
        "Absolutely love this product! The quality is outstanding and it works perfectly right out of the box.",
        "Fantastic purchase. Great build quality, fast delivery, and it exceeded all my expectations.",
        "Excellent value for the money. I am very happy with it and would happily recommend it to friends.",
        "Amazing product, looks beautiful and feels premium. Five stars without any hesitation.",
        "Works flawlessly and the battery lasts all day. Really impressed with the performance.",
        "Super comfortable, stylish and durable. Best purchase I have made this year.",
        "Perfect fit and great material. Arrived early and packaged carefully. Very satisfied customer.",
        "The sound quality is superb and setup was effortless. Highly recommended, worth every penny.",
        "Brilliant design and sturdy construction. It looks even better in person than in the photos.",
        "Great product with wonderful features. Customer support was helpful and friendly too.",
        "I am delighted with this item. It is well made, easy to use, and performs beautifully.",
        "Top notch quality and lovely finish. My whole family enjoys using it every single day.",
        "Really pleased with this purchase. Solid, reliable, and exactly as described by the seller.",
        "Outstanding performance and excellent packaging. Will definitely buy from this brand again.",
        "Gorgeous colour, smooth finish and comfortable to wear all day. Love it, highly recommend!",
    ]
    neutral = [
        "The product is okay. It does the job but nothing special about it.",
        "Average quality for the price. Some features are useful while others are mediocre.",
        "It arrived on time and works as described. Neither impressed nor upset.",
        "Decent item overall. The design is acceptable and the performance is fine for basic use.",
        "It is fine for the price. Might be better, might be worse, depends on your expectations.",
        "Standard product with average build quality. It functions as expected, nothing more.",
        "The item is alright. Delivery was normal and the packaging was ordinary.",
        "Mixed feelings about this. Good colour but the material feels average and so-so.",
        "Works adequately for everyday use. Just middle of the road, neither outstanding nor bad.",
        "Reasonable purchase. The size is acceptable and the quality is moderate.",
        "It does what it says. I have used it for a week and my opinion is neutral so far.",
        "Fair product, fair price. Some minor pros and cons balance each other out.",
        "An ordinary item that gets the job done. I would rate it three out of five.",
        "Acceptable performance, typical design. Comparable to other products in this price range.",
        "Middling experience. Setup took a while, but once running it works adequately.",
    ]
    negative = [
        "Terrible quality. It broke after two days and the seller refuses to answer my messages.",
        "Worst purchase ever. Completely defective and a total waste of money. I want a refund.",
        "Very disappointed. The product arrived damaged and looks nothing like the pictures.",
        "Awful build quality, cheap plastic that cracked immediately. Do not buy this junk.",
        "Stopped working after one week. Horrible experience and useless customer service. Requesting a refund.",
        "Fake and counterfeit item, poor material, terrible stitching. Extremely disappointed and angry.",
        "Battery drains within an hour and the screen flickers constantly. Defective and frustrating.",
        "Horrible smell, faded colour, and the size is completely wrong. Returning this garbage.",
        "Overpriced rubbish. Broke on first use and the packaging was ripped and dirty.",
        "Extremely poor quality and slow shipping. The item was damaged, scratched, and unusable.",
        "Regret buying this. Cheaply made, unreliable, and it failed within days. Want my money back.",
        "Dreadful product. Loud, faulty, and overheats quickly. Complete disaster, avoid at all costs.",
        "Very bad experience. Wrong item delivered, then the replacement was broken too. Awful.",
        "Poor design, flimsy parts, and a useless manual. Total disappointment and a scam.",
        "Pathetic quality. Cracked screen out of the box and the seller ignored my refund request.",
    ]
    rows = (
        [(text, "Positive") for text in positive]
        + [(text, "Neutral") for text in neutral]
        + [(text, "Negative") for text in negative]
    )
    return pd.DataFrame(rows, columns=["review", "sentiment"])


# ----------------------------------------------------------------------------
# 2. Scikit-Learn text pipeline (cached)
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Training text sentiment pipeline...")
def train_text_pipeline() -> Tuple[Pipeline, float, int]:
    """
    Train TF-IDF + Logistic Regression on the synthetic dataset.

    Returns (fitted pipeline, 5-fold CV accuracy, number of training samples).
    Note: sklearn's English stop-word list also removes negators such as "not",
    so the dataset relies on strongly polarised vocabulary instead of negation.
    """
    df = build_review_dataset()
    pipeline = Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    stop_words="english",
                    ngram_range=(1, 2),
                    sublinear_tf=True,
                    lowercase=True,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    C=10.0,
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(pipeline, df["review"], df["sentiment"], cv=cv, scoring="accuracy")
    pipeline.fit(df["review"], df["sentiment"])
    return pipeline, float(scores.mean()), int(len(df))


def analyze_text(pipeline: Pipeline, text: str) -> SentimentResult:
    """Run the sentiment pipeline on a single review."""
    probabilities = pipeline.predict_proba([text])[0]
    prob_map = {str(cls): float(p) for cls, p in zip(pipeline.classes_, probabilities)}
    label = max(prob_map, key=prob_map.get)
    return SentimentResult(label=label, confidence=prob_map[label], probabilities=prob_map)


# ----------------------------------------------------------------------------
# 3. TensorFlow / Keras image pipeline
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def configure_tensorflow() -> str:
    """Apply conservative CPU settings once per server process."""
    if tf is None:
        return "unavailable"
    try:
        tf.config.set_visible_devices([], "GPU")
    except (RuntimeError, ValueError):
        pass
    try:
        tf.config.threading.set_intra_op_parallelism_threads(2)
        tf.config.threading.set_inter_op_parallelism_threads(2)
    except RuntimeError:
        pass  # already initialised; safe to ignore
    return "cpu"


@st.cache_resource(show_spinner="Loading MobileNetV2 (ImageNet weights)...")
def load_image_model():
    """Load the lightweight pre-trained MobileNetV2 classifier and warm it up."""
    if tf is None:
        raise RuntimeError(f"TensorFlow could not be imported: {TF_IMPORT_ERROR}")
    model = tf.keras.applications.MobileNetV2(
        weights="imagenet",
        input_shape=(224, 224, 3),
    )
    model.predict(np.zeros((1, 224, 224, 3), dtype=np.float32), verbose=0)  # warm-up
    return model


def load_pil_image(data: bytes) -> Image.Image:
    """Safely decode uploaded bytes into an RGB Pillow image."""
    if not data:
        raise ValueError("The uploaded file is empty.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise ValueError(f"Image is larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB. Please upload a smaller file.")
    try:
        with Image.open(io.BytesIO(data)) as probe:
            probe.verify()                       # integrity check
        image = Image.open(io.BytesIO(data))     # re-open: verify() invalidates the handle
        image.load()
    except (UnidentifiedImageError, OSError, SyntaxError, Image.DecompressionBombError) as exc:
        raise ValueError("The file could not be read as a valid JPG/PNG image.") from exc

    image = ImageOps.exif_transpose(image)       # respect phone-camera orientation
    if image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info):
        rgba = image.convert("RGBA")
        background = Image.new("RGB", rgba.size, (255, 255, 255))
        background.paste(rgba, mask=rgba.split()[3])
        return background
    return image.convert("RGB")


def preprocess_for_mobilenet(image: Image.Image) -> np.ndarray:
    """PIL image -> float array -> 224x224 resize -> MobileNetV2 preprocessing (batch of 1)."""
    array = np.asarray(image, dtype=np.float32)                                   # (H, W, 3)
    resized = tf.image.resize(array, IMG_SIZE, method="bilinear", antialias=True).numpy()
    batch = np.expand_dims(resized, axis=0)                                       # (1, 224, 224, 3)
    return tf.keras.applications.mobilenet_v2.preprocess_input(batch)             # scales to [-1, 1]


def classify_image(model, batch: np.ndarray, top: int = 5) -> List[ImagePrediction]:
    """Run inference and decode the top ImageNet classes."""
    predictions = model.predict(batch, verbose=0)
    decoded = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=top)[0]
    return [
        ImagePrediction(class_id=str(cid), label=str(label), probability=float(prob))
        for cid, label, prob in decoded
    ]


# ----------------------------------------------------------------------------
# 4. Multimodal decision layer
# ----------------------------------------------------------------------------
def label_tokens(label: str) -> set:
    """Split an ImageNet label like 'running_shoe' into lowercase tokens."""
    return {token for token in re.split(r"[^a-z0-9]+", label.lower()) if token}


def compute_product_relevance(
    predictions: List[ImagePrediction], category: str
) -> Tuple[float, List[ImagePrediction]]:
    """Probability mass of top-k classes that match the expected product category."""
    keywords = CATEGORY_KEYWORDS.get(category, ALL_PRODUCT_KEYWORDS)
    matched = [p for p in predictions if label_tokens(p.label) & keywords]
    relevance = min(sum(p.probability for p in matched), 1.0)
    return relevance, matched


def pretty_label(label: str) -> str:
    return label.replace("_", " ").title()


def make_audit_decision(
    sentiment: SentimentResult,
    review_text: str,
    predictions: List[ImagePrediction],
    category: str,
) -> AuditResult:
    """Fuse text sentiment and image evidence into a final auditor status."""
    relevance, matched = compute_product_relevance(predictions, category)
    top = predictions[0]
    has_refund_language = bool(REFUND_PATTERN.search(review_text))
    top_name = pretty_label(top.label)

    reasons = [
        f"Text sentiment: {sentiment.label} ({sentiment.confidence:.0%} confidence).",
        f"Top image class: {top_name} ({top.probability:.0%} confidence).",
        f"Product-category match ('{category}'): {relevance:.0%} of top-5 probability mass.",
        "Refund/defect language detected in text." if has_refund_language
        else "No explicit refund/defect language detected in text.",
    ]

    image_matches = relevance >= RELEVANCE_MATCH_THRESHOLD

    # Rule 1: image contradicts the review -> potential fraud
    if not image_matches and top.probability >= CONFIDENT_MISMATCH_THRESHOLD:
        return AuditResult(
            status="Mismatched Data / Potential Fraud",
            level="danger",
            icon="🚨",
            summary=(
                f"The image looks like '{top_name}', which does not fit the expected product "
                f"category, while the review is {sentiment.label.lower()}. This pattern is typical "
                "of fake reviews or fraudulent claims."
            ),
            reasons=reasons,
            relevance=relevance,
            matched=matched,
        )

    # Rule 2: image is unrelated but the model is not sure -> human review
    if not image_matches:
        return AuditResult(
            status="Needs Manual Review",
            level="warning",
            icon="🕵️",
            summary=(
                "The image could not be confidently linked to the expected product category. "
                "A human moderator should verify the photo."
            ),
            reasons=reasons,
            relevance=relevance,
            matched=matched,
        )

    # Rule 3: text signal too weak to trust
    if sentiment.confidence < MIN_TEXT_CONFIDENCE:
        return AuditResult(
            status="Needs Manual Review",
            level="warning",
            icon="🕵️",
            summary="The image matches the product, but the review text is too ambiguous for an automated call.",
            reasons=reasons,
            relevance=relevance,
            matched=matched,
        )

    # Rule 4: consistent evidence
    if sentiment.label == "Negative":
        if sentiment.confidence >= NEGATIVE_REFUND_CONFIDENCE or has_refund_language:
            return AuditResult(
                status="High-Priority Refund Request",
                level="danger",
                icon="💸",
                summary=(
                    "A genuine-looking negative review is backed by a matching product image. "
                    "Route to the refund/returns team with priority."
                ),
                reasons=reasons,
                relevance=relevance,
                matched=matched,
            )
        return AuditResult(
            status="Needs Manual Review",
            level="warning",
            icon="🕵️",
            summary="Mildly negative review with a matching image. Human follow-up is recommended.",
            reasons=reasons,
            relevance=relevance,
            matched=matched,
        )

    return AuditResult(
        status="Safe Review",
        level="success",
        icon="✅",
        summary=(
            f"The {sentiment.label.lower()} review is consistent with the uploaded product image. "
            "No action required."
        ),
        reasons=reasons,
        relevance=relevance,
        matched=matched,
    )


# ----------------------------------------------------------------------------
# 5. Streamlit UI
# ----------------------------------------------------------------------------
def inject_css() -> None:
    st.markdown(
        """
<style>
.hero {padding: 1.6rem 1.8rem; border-radius: 18px; color: #fff;
       background: linear-gradient(135deg, #4b3fd6 0%, #7b2ff7 50%, #f107a3 100%);
       box-shadow: 0 10px 30px rgba(75, 63, 214, 0.35); margin-bottom: 1.2rem;}
.hero h1 {margin: 0 0 .4rem 0; color: #fff; font-size: 2.1rem;}
.hero p {margin: 0; font-size: 1.02rem; opacity: .95;}
.badge {display:inline-block; padding: .2rem .7rem; margin: .6rem .4rem 0 0; border-radius: 999px;
        background: rgba(255,255,255,.18); font-size: .82rem; font-weight: 600;}
.audit-card {padding: 1.4rem 1.7rem; border-radius: 18px; color: #fff; margin: 1.1rem 0;
             box-shadow: 0 10px 28px rgba(0,0,0,.22);}
.audit-card h2 {margin: .1rem 0 .4rem 0; color: #fff; font-size: 1.8rem;}
.audit-card p {margin: 0; font-size: 1.02rem; line-height: 1.5;}
.audit-label {font-size: .75rem; letter-spacing: .12em; opacity: .85; font-weight: 700;}
.audit-card.success {background: linear-gradient(135deg, #0f9b6c, #16c79a);}
.audit-card.danger  {background: linear-gradient(135deg, #b93124, #ef5350);}
.audit-card.warning {background: linear-gradient(135deg, #c77d0a, #f5b041);}
</style>
        """,
        unsafe_allow_html=True,
    )


def render_header() -> None:
    st.markdown(
        '<div class="hero">'
        "<h1>🛡️ AI-Powered Multimodal Product Review Auditor</h1>"
        "<p>An Advanced Multimodal (ML + TensorFlow Deep Learning) Portfolio Project. "
        "It fuses a Scikit-Learn TF-IDF sentiment classifier with a TensorFlow MobileNetV2 "
        "vision model to detect genuine refund requests, safe reviews, and mismatched or "
        "potentially fraudulent submissions.</p>"
        '<span class="badge">Scikit-Learn</span>'
        '<span class="badge">TensorFlow / Keras</span>'
        '<span class="badge">MobileNetV2</span>'
        '<span class="badge">Streamlit</span>'
        "</div>",
        unsafe_allow_html=True,
    )


def render_sidebar(cv_accuracy: float, n_samples: int) -> None:
    with st.sidebar:
        st.header("⚙️ System Status")
        st.success("Text pipeline trained")
        st.metric("Training reviews", n_samples)
        st.metric("5-fold CV accuracy", f"{cv_accuracy:.0%}")
        st.info("Image model: MobileNetV2 (ImageNet, CPU)")
        with st.expander("How the auditor decides"):
            st.markdown(
                "- **Mismatched Data / Potential Fraud**: the photo is confidently something "
                "unrelated to the chosen product category.\n"
                "- **High-Priority Refund Request**: strongly negative text (or refund/defect "
                "wording) plus a matching product photo.\n"
                "- **Safe Review**: positive or neutral text plus a matching photo.\n"
                "- **Needs Manual Review**: weak or ambiguous signals from either modality.\n\n"
                "The vision model recognises the 1,000 ImageNet classes, so this is a "
                "heuristic demonstration rather than a certified fraud detector."
            )


def render_decision_card(result: AuditResult) -> None:
    st.markdown(
        f'<div class="audit-card {result.level}">'
        '<div class="audit-label">FINAL AUDITOR STATUS</div>'
        f"<h2>{result.icon} {html.escape(result.status)}</h2>"
        f"<p>{html.escape(result.summary)}</p>"
        "</div>",
        unsafe_allow_html=True,
    )


def render_results(
    sentiment: SentimentResult,
    predictions: List[ImagePrediction],
    result: AuditResult,
) -> None:
    render_decision_card(result)

    top = predictions[0]
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Sentiment", f"{SENTIMENT_EMOJI.get(sentiment.label, '')} {sentiment.label}")
    c2.metric("Text confidence", f"{sentiment.confidence:.0%}")
    c3.metric("Top image class", pretty_label(top.label))
    c4.metric("Image confidence", f"{top.probability:.0%}")
    c5.metric("Product match", f"{result.relevance:.0%}")

    left, right = st.columns(2, gap="large")
    with left:
        st.markdown("#### 📊 Sentiment probabilities")
        chart_df = pd.DataFrame(
            {"Probability (%)": [sentiment.probabilities.get(k, 0.0) * 100 for k in SENTIMENT_ORDER]},
            index=SENTIMENT_ORDER,
        )
        st.bar_chart(chart_df)
    with right:
        st.markdown("#### 🖼️ Top-5 image predictions")
        matched_ids = {p.class_id for p in result.matched}
        table = pd.DataFrame(
            {
                "Class": [pretty_label(p.label) for p in predictions],
                "Confidence": [p.probability * 100 for p in predictions],
                "Category match": ["✅" if p.class_id in matched_ids else "—" for p in predictions],
            }
        )
        st.dataframe(
            table,
            hide_index=True,
            use_container_width=True,
            column_config={
                "Confidence": st.column_config.ProgressColumn(
                    "Confidence", format="%.1f%%", min_value=0.0, max_value=100.0
                )
            },
        )

    with st.expander("🔎 Audit reasoning", expanded=True):
        for reason in result.reasons:
            st.markdown(f"- {reason}")


def run_audit(text_pipeline: Pipeline, review_text: str, uploaded_file, category: str) -> None:
    """Validate inputs, run both pipelines, and render the results."""
    cleaned = (review_text or "").strip()
    if len(cleaned) < MIN_TEXT_CHARS:
        st.warning(f"Please enter a review of at least {MIN_TEXT_CHARS} characters.")
        return
    if uploaded_file is None:
        st.warning("Please upload a JPG or PNG product image to run the multimodal audit.")
        return

    # Text branch
    try:
        sentiment = analyze_text(text_pipeline, cleaned)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Text analysis failed: {exc}")
        return

    # Image branch
    try:
        image = load_pil_image(uploaded_file.getvalue())
    except ValueError as exc:
        st.error(str(exc))
        return

    try:
        model = load_image_model()
    except Exception as exc:  # noqa: BLE001
        st.error(
            "Could not load MobileNetV2. The first run needs internet access to download the "
            f"ImageNet weights (~14 MB).\n\nDetails: {exc}"
        )
        return

    try:
        with st.spinner("Running TensorFlow inference..."):
            batch = preprocess_for_mobilenet(image)
            predictions = classify_image(model, batch, top=5)
    except Exception as exc:  # noqa: BLE001
        st.error(
            "Image analysis failed. This can happen if the ImageNet class index could not be "
            f"downloaded.\n\nDetails: {exc}"
        )
        return

    # Fusion layer
    result = make_audit_decision(sentiment, cleaned, predictions, category)
    render_results(sentiment, predictions, result)


def main() -> None:
    st.set_page_config(
        page_title="Multimodal Review Auditor",
        page_icon="🛡️",
        layout="wide",
    )
    inject_css()

    if tf is None:
        st.error(
            "TensorFlow failed to import. Run `pip install -r requirements.txt` and restart.\n\n"
            f"Details: {TF_IMPORT_ERROR}"
        )
        st.stop()

    configure_tensorflow()

    try:
        text_pipeline, cv_accuracy, n_samples = train_text_pipeline()
    except Exception as exc:  # noqa: BLE001
        st.error(f"Failed to train the text pipeline: {exc}")
        st.stop()

    render_sidebar(cv_accuracy, n_samples)
    render_header()

    left, right = st.columns([1.1, 1], gap="large")

    with left:
        st.subheader("📝 Review text")
        review_text = st.text_area(
            "Paste the customer review",
            height=190,
            max_chars=MAX_TEXT_CHARS,
            placeholder="e.g. Terrible quality. The sneakers fell apart after two days and I want a refund.",
        )
        category = st.selectbox(
            "Expected product category",
            options=[ANY_CATEGORY] + list(CATEGORY_KEYWORDS.keys()),
            help="The image is checked against this category to detect mismatched or fraudulent submissions.",
        )

    with right:
        st.subheader("🖼️ Product image")
        uploaded_file = st.file_uploader(
            "Upload a JPG or PNG product photo",
            type=["jpg", "jpeg", "png"],
        )
        if uploaded_file is not None:
            try:
                st.image(uploaded_file.getvalue(), caption=uploaded_file.name, use_container_width=True)
            except Exception:  # noqa: BLE001
                st.warning("Preview unavailable for this file; validation will run on audit.")

    audit_clicked = st.button("🔍 Audit Review", type="primary", use_container_width=True)
    if audit_clicked:
        run_audit(text_pipeline, review_text, uploaded_file, category)


if __name__ == "__main__":
    main()