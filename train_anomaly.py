import os
from PIL import Image
import numpy as np
from model import extract_embedding
from anomaly import train_anomaly_model

DATA_PATH = "real_images"

embeddings = []

for file in os.listdir(DATA_PATH):
    path = os.path.join(DATA_PATH, file)
    try:
        image = Image.open(path).convert("RGB")
        embedding = extract_embedding(image)
        embeddings.append(embedding)
    except:
        continue

embeddings = np.array(embeddings)

train_anomaly_model(embeddings)
