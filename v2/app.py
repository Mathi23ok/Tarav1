from pathlib import Path
import tempfile

import numpy as np
import streamlit as st
import torch
from torchvision.io import read_image

from dataset import load_npz_data, normalize_features
from extract_features import (
    extract_cnn_embeddings,
    extract_fft_features_batch,
    extract_noise_features_batch,
    get_feature_extractor,
    preprocess_image,
)
from model import FusionNet


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_PATH = Path("checkpoints") / "fusionnet_best.pt"
TRAIN_FEATURES_PATH = Path("features") / "train.npz"
HIDDEN_DIM = 64
DROPOUT = 0.3
FEATURE_DIM = 2100


def load_uploaded_image(uploaded_file) -> torch.Tensor:
    suffix = Path(uploaded_file.name).suffix or ".png"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        tmp_file.write(uploaded_file.getbuffer())
        temp_path = Path(tmp_file.name)

    try:
        image = read_image(str(temp_path)).float()
    finally:
        temp_path.unlink(missing_ok=True)

    if image.size(0) == 4:
        image = image[:3]
    if image.size(0) == 1:
        image = image.repeat(3, 1, 1)
    return image


def build_feature_vector(image: torch.Tensor, cnn_extractor: torch.nn.Module) -> np.ndarray:
    processed = preprocess_image(image)
    gray = image.mean(dim=0, keepdim=True).numpy()

    cnn_embedding = extract_cnn_embeddings([processed], cnn_extractor)[0]
    fft_features, _ = extract_fft_features_batch([gray])
    noise_features, _ = extract_noise_features_batch([gray])

    feature_vector = np.empty((1, FEATURE_DIM), dtype=np.float32)
    feature_vector[0, :2048] = cnn_embedding
    feature_vector[0, 2048:2080] = fft_features[0]
    feature_vector[0, 2080:] = noise_features[0]
    return feature_vector


@st.cache_data(show_spinner=False)
def load_normalization_stats() -> tuple[np.ndarray, np.ndarray]:
    train_features, _, _ = load_npz_data(str(TRAIN_FEATURES_PATH))
    _, mean, std = normalize_features(train_features)
    return mean.astype(np.float32, copy=False), std.astype(np.float32, copy=False)


@st.cache_resource(show_spinner=False)
def load_models() -> tuple[torch.nn.Module, torch.nn.Module]:
    cnn_extractor = get_feature_extractor().to(DEVICE)
    fusion_model = FusionNet(input_dim=FEATURE_DIM, hidden_dim=HIDDEN_DIM, dropout=DROPOUT).to(DEVICE)
    state_dict = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    fusion_model.load_state_dict(state_dict)
    fusion_model.eval()
    return cnn_extractor, fusion_model


def predict_image(
    image: torch.Tensor,
    cnn_extractor: torch.nn.Module,
    fusion_model: torch.nn.Module,
    mean: np.ndarray,
    std: np.ndarray,
) -> tuple[str, float, float]:
    features = build_feature_vector(image, cnn_extractor)
    normalized_features, _, _ = normalize_features(features, mean, std)
    feature_tensor = torch.from_numpy(normalized_features).to(DEVICE)

    with torch.inference_mode():
        logits, score = fusion_model(feature_tensor)
        probability = torch.sigmoid(logits).item()
        manipulation_score = score.item()

    label = "REAL" if probability >= 0.5 else "FAKE"
    confidence = probability if label == "REAL" else 1.0 - probability
    return label, confidence, manipulation_score


def main() -> None:
    st.set_page_config(page_title="FusionNet Inference", page_icon="🖼️", layout="centered")
    st.title("FusionNet Image Forensics")
    st.write("Upload an image to classify it as REAL or FAKE using the trained FusionNet model.")

    if not CHECKPOINT_PATH.exists():
        st.error(f"Missing checkpoint: {CHECKPOINT_PATH}")
        return
    if not TRAIN_FEATURES_PATH.exists():
        st.error(f"Missing training features: {TRAIN_FEATURES_PATH}")
        return

    uploaded_file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])
    if uploaded_file is None:
        return

    st.image(uploaded_file, caption=uploaded_file.name, use_container_width=True)

    with st.spinner("Loading model and extracting features..."):
        cnn_extractor, fusion_model = load_models()
        mean, std = load_normalization_stats()
        image = load_uploaded_image(uploaded_file)
        label, confidence, manipulation_score = predict_image(
            image,
            cnn_extractor,
            fusion_model,
            mean,
            std,
        )

    st.subheader(f"Label: {label}")
    st.write(f"Confidence: {confidence * 100:.2f}%")
    st.write(f"Score: {manipulation_score:.4f}")


if __name__ == "__main__":
    main()
