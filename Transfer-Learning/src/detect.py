import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path

# ============================================================
# P2 TRANSFER LEARNING - CAMERA TEST
# ============================================================

# ============================================================
# PATH CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "mg90s_pca9685_model.keras"
CLASS_NAMES_PATH = BASE_DIR / "class_names.txt"

IMG_SIZE = (224, 224)
CONFIDENCE_THRESHOLD = 0.50

# ============================================================
# VALIDASI FILE
# ============================================================

if not MODEL_PATH.is_file():
    raise FileNotFoundError(
        f"Model tidak ditemukan: {MODEL_PATH}"
    )

if not CLASS_NAMES_PATH.is_file():
    raise FileNotFoundError(
        f"File nama kelas tidak ditemukan: {CLASS_NAMES_PATH}"
    )

# ------------------------------------------------------------
# Load model
# ------------------------------------------------------------
print("=" * 60)
print("P2 TRANSFER LEARNING - CAMERA TEST")
print("=" * 60)

print("\nLoading model...")
model = tf.keras.models.load_model(MODEL_PATH)

print("Model berhasil dimuat.")


# ------------------------------------------------------------
# Load class names
# ------------------------------------------------------------
with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
    class_names = [line.strip() for line in f if line.strip()]

if len(class_names) != model.output_shape[-1]:
    raise ValueError(
        "Jumlah nama kelas tidak sesuai dengan output model."
    )

print("Class names:")
for i, name in enumerate(class_names):
    print(f"  {i}: {name}")


# ------------------------------------------------------------
# Open camera
# ------------------------------------------------------------
print("\nMembuka kamera...")

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Kamera tidak dapat dibuka.")
    raise SystemExit


print("\nKamera aktif.")
print("Tekan Q untuk keluar.")
print("=" * 60)


# ------------------------------------------------------------
# Prediction loop
# ------------------------------------------------------------
while True:

    ret, frame = cap.read()

    if not ret:
        print("Gagal membaca frame kamera.")
        break

    # Resize untuk MobileNetV2
    img = cv2.resize(frame, IMG_SIZE)

    # BGR -> RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Convert ke float
    img = img.astype(np.float32)

    # Preprocessing MobileNetV2
    img = tf.keras.applications.mobilenet_v2.preprocess_input(img)

    # Tambahkan batch dimension
    img = np.expand_dims(img, axis=0)

    # Prediction
    prediction = model.predict(img, verbose=0)[0]

    # Validasi jumlah output prediksi
    if len(prediction) != len(class_names):
        raise ValueError(
            "Jumlah output prediksi tidak sesuai dengan nama kelas."
        )

    mg90s_conf = float(prediction[0])
    pca9685_conf = float(prediction[1])

    class_id = int(np.argmax(prediction))
    confidence = float(prediction[class_id])

    # Nama kelas
    class_name = class_names[class_id]

    print(
        f"MG90S: {mg90s_conf * 100:.2f}% | "
        f"PCA9685: {pca9685_conf * 100:.2f}%"
    )  

    # --------------------------------------------------------
    # Display
    # --------------------------------------------------------

    if confidence >= CONFIDENCE_THRESHOLD:
        text = f"{class_name} : {confidence * 100:.2f}%"
    else:
        text = f"UNKNOWN : {confidence * 100:.2f}%"

    cv2.putText(
        frame,
        text,
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 0),
        3
    )

    cv2.putText(
        frame,
        "Q = Quit",
        (20, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.imshow("P2 Transfer Learning - Object Detection", frame)

    # Q untuk keluar
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break


# ------------------------------------------------------------
# Cleanup
# ------------------------------------------------------------
cap.release()
cv2.destroyAllWindows()

print("\nProgram selesai.")