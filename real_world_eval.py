import os
from PIL import Image
from model import predict_ai_probability
from main import evaluate_image

# ===============================
# Folder Paths
# ===============================
CLEAN_REAL_FOLDER = "real_images"
MESSY_REAL_FOLDER = "real_image"
FAKE_FOLDER = "ai_images"


# ===============================
# Folder Testing Function
# ===============================
def test_folder(folder, label_name):

    correct = 0
    total = 0
    uncertain = 0

    print(f"\nTesting {label_name} images...")

    for file in os.listdir(folder):

        path = os.path.join(folder, file)

        try:
            image = Image.open(path).convert("RGB")

            result = evaluate_image(image)
            prediction = result["predicted_label"]
            confidence = result["confidence"]

            total += 1

            if prediction == "UNCERTAIN":
                uncertain += 1
            elif prediction == label_name:
                correct += 1

            print(f"{file} -> {prediction} | Confidence: {confidence}")

        except Exception as e:
            print(f"Skipped {file}: {e}")
            continue

    accuracy = correct / total if total > 0 else 0
    uncertain_rate = uncertain / total if total > 0 else 0

    print(f"\n{label_name} Accuracy: {accuracy * 100:.2f}%")
    print(f"{label_name} Uncertain Rate: {uncertain_rate * 100:.2f}%")

    return accuracy


# ===============================
# Main Evaluation
# ===============================
if __name__ == "__main__":

    print("\n==============================")
    print("CLEAN REAL TEST")
    print("==============================")

    clean_real_acc = test_folder(CLEAN_REAL_FOLDER, "REAL")

    print("\n==============================")
    print("MESSY REAL TEST")
    print("==============================")

    messy_real_acc = test_folder(MESSY_REAL_FOLDER, "REAL")

    print("\n==============================")
    print("AI TEST")
    print("==============================")

    fake_acc = test_folder(FAKE_FOLDER, "FAKE")

    balanced_clean = (clean_real_acc + fake_acc) / 2
    balanced_messy = (messy_real_acc + fake_acc) / 2

    print("\n======================================")
    print(f"Balanced Accuracy (Clean vs AI): {balanced_clean * 100:.2f}%")
    print(f"Balanced Accuracy (Messy vs AI): {balanced_messy * 100:.2f}%")
    print("======================================")
