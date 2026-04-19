import numpy as np
import cv2
from PIL import Image
import io


def compute_noise_score(image):
    """
    Error Level Analysis (ELA) score.
    Detects compression inconsistencies.
    """

    # ---------------------------------
    # 1. Standardize Resolution
    # ---------------------------------
    image = image.resize((512, 512))

    # ---------------------------------
    # 2. Save image at known JPEG quality
    # ---------------------------------
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=90)
    buffer.seek(0)

    recompressed = Image.open(buffer)

    # ---------------------------------
    # 3. Compute difference
    # ---------------------------------
    original_np = np.array(image, dtype=np.float32)
    recompressed_np = np.array(recompressed, dtype=np.float32)

    diff = np.abs(original_np - recompressed_np)

    # Convert to grayscale difference
    diff_gray = cv2.cvtColor(diff.astype(np.uint8), cv2.COLOR_BGR2GRAY)

    # ---------------------------------
    # 4. Compute ELA intensity
    # ---------------------------------
    mean_diff = np.mean(diff_gray)

    # Normalize (empirical scaling)
    normalized_score = float(min(mean_diff / 25.0, 1.0))

    return normalized_score
