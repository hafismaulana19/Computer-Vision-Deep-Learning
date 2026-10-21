import tensorflow as tf
from pathlib import Path
import time
import csv

# ============================================================
# P2 TRANSFER LEARNING
# MOBILENETV2 - EXTERNAL TEST
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "mg90s_pca9685_model_v2.keras"
EXTERNAL_DIR = BASE_DIR / "data" / "external_test_v2"

IMG_SIZE = (224, 224)

CLASSES = ["MG90S", "PCA9685"]

print("=" * 70)
print("MOBILENETV2 V2 - EXTERNAL TEST")
print("=" * 70)

# ============================================================
# VALIDASI PATH DAN KELAS
# ============================================================

if not MODEL_PATH.is_file():
    raise FileNotFoundError(
        f"Model tidak ditemukan: {MODEL_PATH}"
    )

if not EXTERNAL_DIR.is_dir():
    raise FileNotFoundError(
        f"Folder external test tidak ditemukan: {EXTERNAL_DIR}"
    )

for class_name in CLASSES:
    class_dir = EXTERNAL_DIR / class_name

    if not class_dir.is_dir():
        raise FileNotFoundError(
            f"Folder kelas tidak ditemukan: {class_dir}"
        )

# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading model...")
print("Model:", MODEL_PATH)

model = tf.keras.models.load_model(MODEL_PATH)

if model.output_shape[-1] != len(CLASSES):
    raise ValueError(
        "Jumlah output model tidak sesuai dengan jumlah kelas."
    )

print("Model berhasil dimuat.")

# ============================================================
# CHECK GPU
# ============================================================

gpus = tf.config.list_physical_devices("GPU")

if gpus:
    print("\nDevice : GPU")
    print("GPU    :", gpus[0])
else:
    print("\nDevice : CPU")

# ============================================================
# LOAD EXTERNAL TEST
# ============================================================

print("\n" + "=" * 70)
print("EXTERNAL TEST DATASET")
print("=" * 70)

image_files = []

for class_index, class_name in enumerate(CLASSES):

    class_dir = EXTERNAL_DIR / class_name

    files = (
        list(class_dir.glob("*.jpg")) +
        list(class_dir.glob("*.jpeg")) +
        list(class_dir.glob("*.png"))
    )

    print(f"{class_name}: {len(files)} images")

    for file in files:
        image_files.append(
            (file, class_index, class_name)
        )

print("\nTotal images:", len(image_files))

# ============================================================
# EVALUATION
# ============================================================

correct = 0
total = 0

class_correct = {
    cls: 0 for cls in CLASSES
}

class_total = {
    cls: 0 for cls in CLASSES
}

latencies = []

results = []

print("\n" + "=" * 70)
print("TESTING")
print("=" * 70)

for image_path, true_index, true_class in image_files:

    # --------------------------------------------------------
    # LOAD IMAGE
    # --------------------------------------------------------

    img = tf.keras.utils.load_img(
        image_path,
        target_size=IMG_SIZE
    )

    img_array = tf.keras.utils.img_to_array(img)

    img_array = tf.expand_dims(
        img_array,
        axis=0
    )

    # --------------------------------------------------------
    # INFERENCE
    # --------------------------------------------------------

    start = time.perf_counter()

    prediction = model.predict(
        img_array,
        verbose=0
    )

    end = time.perf_counter()

    latency_ms = (end - start) * 1000

    latencies.append(latency_ms)

    predicted_index = int(
        tf.argmax(prediction[0])
    )

    predicted_class = CLASSES[predicted_index]

    confidence = float(
        prediction[0][predicted_index]
    ) * 100

    # --------------------------------------------------------
    # ACCURACY
    # --------------------------------------------------------

    total += 1

    class_total[true_class] += 1

    if predicted_index == true_index:

        correct += 1
        class_correct[true_class] += 1

        result = "CORRECT"

    else:

        result = "WRONG"

    results.append([
        image_path.name,
        true_class,
        predicted_class,
        confidence,
        latency_ms,
        result
    ])

# ============================================================
# FINAL RESULTS
# ============================================================

overall_accuracy = (
    correct / total * 100
    if total > 0
    else 0
)

average_latency = (
    sum(latencies) / len(latencies)
    if latencies
    else 0
)

print("\n" + "=" * 70)
print("HASIL EXTERNAL TEST MOBILENETV2 V2")
print("=" * 70)

print(f"\nTotal images       : {total}")
print(f"Correct prediction : {correct}")
print(f"Wrong prediction   : {total - correct}")

print(
    f"\nExternal Accuracy  : {overall_accuracy:.2f}%"
)

print("\nAccuracy per class:")

for cls in CLASSES:

    acc = (
        class_correct[cls]
        / class_total[cls]
        * 100
        if class_total[cls] > 0
        else 0
    )

    print(
        f"  {cls:<10}: "
        f"{acc:.2f}% "
        f"({class_correct[cls]}/{class_total[cls]})"
    )

print(
    f"\nLatency per image : "
    f"{average_latency:.3f} ms"
)

# ============================================================
# SAVE CSV
# ============================================================

results_dir = BASE_DIR / "results"
results_dir.mkdir(
    parents=True,
    exist_ok=True
)

csv_path = (
    results_dir
    / "mobilenetv2_v2_external_test_results.csv"
)

with open(
    csv_path,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "filename",
        "true_class",
        "predicted_class",
        "confidence_percent",
        "latency_ms",
        "result"
    ])

    writer.writerows(results)

# ============================================================
# SUMMARY
# ============================================================

print("\nHasil disimpan:")
print(csv_path)

print("\n" + "=" * 70)
print("SELESAI")
print("=" * 70)