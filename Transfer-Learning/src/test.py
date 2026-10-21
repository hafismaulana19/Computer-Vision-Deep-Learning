import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path

# ============================================================
# P2 TRANSFER LEARNING V2
# CAMERA TEST - MOBILENETV2
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "mg90s_pca9685_model_v2.keras"
CLASS_FILE = BASE_DIR / "class_names_v2.txt"

IMG_SIZE = 224
CONFIDENCE_THRESHOLD = 0.60

# ============================================================
# VALIDASI FILE MODEL DAN KELAS
# ============================================================

if not MODEL_PATH.is_file():
    raise FileNotFoundError(
        f"File model tidak ditemukan: {MODEL_PATH}"
    )

if not CLASS_FILE.is_file():
    raise FileNotFoundError(
        f"File kelas tidak ditemukan: {CLASS_FILE}"
    )

# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 70)
print("P2 TRANSFER LEARNING V2 - CAMERA TEST")
print("=" * 70)

print("\nLoading model...")
print("Model:", MODEL_PATH)

model = tf.keras.models.load_model(MODEL_PATH)

with open(CLASS_FILE, "r", encoding="utf-8") as f:
    class_names = [
        line.strip()
        for line in f
        if line.strip()
    ]

if len(class_names) != model.output_shape[-1]:
    raise ValueError(
        "Jumlah kelas pada class_names_v2.txt tidak sesuai "
        "dengan jumlah output model."
    )

print("Model berhasil dimuat.")
print("Classes:", class_names)

# ============================================================
# CAMERA
# ============================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("\nERROR: Kamera tidak dapat dibuka.")
    exit()

print("\nKamera aktif.")
print("Letakkan objek MG90S atau PCA9685 di depan kamera.")
print("Tekan Q untuk keluar.")

# ============================================================
# CAMERA LOOP
# ============================================================

while True:

    ret, frame = cap.read()

    if not ret:
        print("ERROR: Tidak dapat membaca frame.")
        break

    # --------------------------------------------------------
    # PREPROCESS IMAGE
    # --------------------------------------------------------

    image = cv2.resize(
        frame,
        (IMG_SIZE, IMG_SIZE)
    )

    # OpenCV menggunakan BGR
    # TensorFlow model menggunakan RGB
    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    image = image.astype(
        np.float32
    )

    image = np.expand_dims(
        image,
        axis=0
    )

    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------
    # CATATAN:
    # MobileNetV2 preprocess_input SUDAH ADA
    # DI DALAM MODEL V2.
    # Jadi tidak dipanggil lagi di sini.

    prediction = model.predict(
        image,
        verbose=0
    )[0]

    class_id = int(
        np.argmax(prediction)
    )

    confidence = float(
        prediction[class_id]
    )

    if confidence >= CONFIDENCE_THRESHOLD:
        label = class_names[class_id]
    else:
        label = "Unknown"

    # --------------------------------------------------------
    # DISPLAY
    # --------------------------------------------------------

    text = (
        f"{label} : "
        f"{confidence * 100:.2f}%"
    )

    cv2.putText(
        frame,
        text,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow(
        "P2 Transfer Learning V2 - Camera Test",
        frame
    )

    # Q = keluar
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# ============================================================
# CLEANUP
# ============================================================

cap.release()
cv2.destroyAllWindows()

print("\nProgram selesai.")