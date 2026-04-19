import os
from PIL import Image
from main import evaluate_image

REAL_PATH = "real_images"
AI_PATH = "ai_images"

def test_folder(path, label):
    correct = 0
    total = 0

    for file in os.listdir(path):
        try:
            img = Image.open(os.path.join(path, file)).convert("RGB")
            result = evaluate_image(img)

            prediction = result["predicted_label"]

            if prediction == label:
                correct += 1

            total += 1

            print(f"{file} -> {prediction} | Confidence: {result['confidence']}")

        except:
            continue

    accuracy = (correct / total) * 100 if total > 0 else 0
    print(f"\n{label} Accuracy: {accuracy:.2f}%\n")
    return accuracy


print("Testing REAL images...")
real_acc = test_folder(REAL_PATH, "REAL")

print("Testing AI images...")
ai_acc = test_folder(AI_PATH, "FAKE")

overall = (real_acc + ai_acc) / 2
print(f"\nOverall Balanced Accuracy: {overall:.2f}%")
