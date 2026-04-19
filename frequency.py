import numpy as np
import cv2


def compute_fft_score(image):
    """
    Computes high-frequency artifact score using FFT.
    Standardized to fixed resolution for consistency.
    """

    # -----------------------------
    # 1. Force Standard Resolution
    # -----------------------------
    image = image.resize((512, 512))

    # Convert to grayscale
    img = np.array(image.convert("L"), dtype=np.float32)

    # -----------------------------
    # 2. Compute FFT
    # -----------------------------
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift)

    # Log scaling (stable)
    magnitude_spectrum = np.log1p(magnitude)

    # -----------------------------
    # 3. High Frequency Ring Mask
    # -----------------------------
    h, w = magnitude_spectrum.shape
    center_h, center_w = h // 2, w // 2

    y, x = np.ogrid[:h, :w]
    distance = np.sqrt((x - center_w) ** 2 + (y - center_h) ** 2)

    # Define ring: ignore low frequencies (center),
    # capture outer high-frequency band
    low_radius = 40
    high_radius = 200

    ring_mask = (distance > low_radius) & (distance < high_radius)

    high_freq_energy = np.mean(magnitude_spectrum[ring_mask])
    total_energy = np.mean(magnitude_spectrum)

    # -----------------------------
    # 4. Normalized Score
    # -----------------------------
    score = high_freq_energy / (total_energy + 1e-8)

    # Soft normalization
    normalized_score = float(min(score / 3.0, 1.0))

    return normalized_score
