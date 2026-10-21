import cv2
from pathlib import Path
from datetime import datetime

# =========================
# CONFIGURATION
# =========================

CAMERA_ID = 0

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset_new"

CLASSES = {
    "1": "MG90S",
    "2": "PCA9685",
}

# =========================
# PREPARE DIRECTORIES
# =========================

for class_name in CLASSES.values():
    (DATASET_DIR / class_name).mkdir(parents=True, exist_ok=True)

# =========================
# OPEN CAMERA
# =========================

cap = cv2.VideoCapture(CAMERA_ID)

if not cap.isOpened():
    print("ERROR: Kamera tidak dapat dibuka.")
    raise SystemExit(1)

print("=" * 60)
print("CAMERA DATASET CAPTURE - P2 TRANSFER LEARNING")
print("=" * 60)
print()
print("Kontrol:")
print("  [1] Pilih objek MG90S")
print("  [2] Pilih objek PCA9685")
print("  [S] Simpan gambar")
print("  [Q] Keluar")
print()
print("Objek awal: MG90S")
print("=" * 60)

current_class = "MG90S"

while True:

    ret, frame = cap.read()

    if not ret:
        print("ERROR: Gagal membaca frame dari kamera.")
        break

    # =========================
    # DISPLAY INFORMATION
    # =========================

    display = frame.copy()

    cv2.putText(
        display,
        f"Class: {current_class}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
    )

    cv2.putText(
        display,
        "[1] MG90S  [2] PCA9685  [S] Save  [Q] Quit",
        (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
    )

    # =========================
    # SHOW CAMERA
    # =========================

    cv2.imshow("P2 Dataset Capture", display)

    key = cv2.waitKey(1) & 0xFF

    # =========================
    # SELECT MG90S
    # =========================

    if key == ord("1"):
        current_class = "MG90S"
        print("Class selected: MG90S")

    # =========================
    # SELECT PCA9685
    # =========================

    elif key == ord("2"):
        current_class = "PCA9685"
        print("Class selected: PCA9685")

    # =========================
    # SAVE IMAGE
    # =========================

    elif key == ord("s"):

        class_dir = DATASET_DIR / current_class

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

        filename = f"{current_class}_{timestamp}.jpg"

        filepath = class_dir / filename

        cv2.imwrite(str(filepath), frame)

        print(f"[SAVED] {filepath}")

    # =========================
    # EXIT
    # =========================

    elif key == ord("q"):
        print("Menutup kamera...")
        break

# =========================
# CLEANUP
# =========================

cap.release()
cv2.destroyAllWindows()

print("Program selesai.")