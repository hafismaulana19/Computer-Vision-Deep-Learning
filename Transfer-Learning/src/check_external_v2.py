import tensorflow as tf
import numpy as np
from pathlib import Path
from tensorflow.keras.utils import load_img, img_to_array

# ============================================================
# PATH CONFIGURATION
# ============================================================

# Folder utama Transfer-Learning/
BASE_DIR = Path(__file__).resolve().parent.parent

# Model V2
MODEL_PATH = BASE_DIR / "models" / "mg90s_pca9685_model_v2.keras"

# Dataset external test V2
TEST_DIR = BASE_DIR / "data" / "external_test_v2"

# File nama kelas
CLASS_FILE = BASE_DIR / "class_names_v2.txt"

IMG_SIZE = (224, 224)

CLASSES = ["MG90S", "PCA9685"]

print("=" * 70)
print("EXTERNAL TEST - MODEL V2")
print("=" * 70)

# ============================================================
# VALIDASI PATH
# ============================================================

if not MODEL_PATH.is_file():
    raise FileNotFoundError(
        f"Model tidak ditemukan: {MODEL_PATH}"
    )

if not CLASS_FILE.is_file():
    raise FileNotFoundError(
        f"File kelas tidak ditemukan: {CLASS_FILE}"
    )

if not TEST_DIR.is_dir():
    raise FileNotFoundError(
        f"Folder external test tidak ditemukan: {TEST_DIR}"
    )

for class_name in CLASSES:
    class_dir = TEST_DIR / class_name

    if not class_dir.is_dir():
        raise FileNotFoundError(
            f"Folder kelas external test tidak ditemukan: {class_dir}"
        )

with open(CLASS_FILE, "r", encoding="utf-8") as f:
    class_names = [line.strip() for line in f if line.strip()]

if class_names != CLASSES:
    raise ValueError(
        f"Urutan kelas model {class_names} tidak sesuai dengan {CLASSES}"
    )

model = tf.keras.models.load_model(MODEL_PATH)

if model.output_shape[-1] != len(CLASSES):
    raise ValueError(
        "Jumlah output model tidak sesuai dengan jumlah kelas."
    )

total = 0
correct = 0

class_correct = {
    "MG90S": 0,
    "PCA9685": 0
}

class_total = {
    "MG90S": 0,
    "PCA9685": 0
}

for true_class in CLASSES:

    files = sorted(
        list((TEST_DIR / true_class).glob("*.jpg")) +
        list((TEST_DIR / true_class).glob("*.jpeg")) +
        list((TEST_DIR / true_class).glob("*.png"))
    )

    print("\n" + "-" * 70)
    print(f"TRUE CLASS: {true_class}")
    print("-" * 70)

    for path in files:

        img = load_img(path, target_size=IMG_SIZE)
        img = img_to_array(img)

        img = np.expand_dims(img, axis=0)

        prediction = model.predict(img, verbose=0)[0]

        predicted_index = np.argmax(prediction)

        predicted_class = CLASSES[predicted_index]

        mg90s = prediction[0] * 100
        pca9685 = prediction[1] * 100

        is_correct = predicted_class == true_class

        if is_correct:
            correct += 1
            class_correct[true_class] += 1

        total += 1
        class_total[true_class] += 1

        status = "BENAR" if is_correct else "SALAH"

        print(
            f"{path.name}"
            f" -> MG90S: {mg90s:6.2f}%"
            f" | PCA9685: {pca9685:6.2f}%"
            f" | Prediksi: {predicted_class}"
            f" | {status}"
        )

# ============================================================
# HASIL
# ============================================================

print("\n" + "=" * 70)
print("HASIL EXTERNAL TEST")
print("=" * 70)

for cls in CLASSES:

    if class_total[cls] == 0:
        print(f"{cls:8s}: Tidak ada gambar untuk diuji.")
        continue

    acc = (
        class_correct[cls] /
        class_total[cls] *
        100
    )

    print(
        f"{cls:8s}: "
        f"{class_correct[cls]}/{class_total[cls]} "
        f"= {acc:.2f}%"
    )

if total == 0:
    raise ValueError(
        f"Tidak ada gambar yang ditemukan di {TEST_DIR}"
    )

overall = correct / total * 100

print("-" * 70)
print(
    f"TOTAL    : "
    f"{correct}/{total} "
    f"= {overall:.2f}%"
)

print("=" * 70)