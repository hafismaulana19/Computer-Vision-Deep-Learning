from pathlib import Path
import random
import shutil

# ==============================
# KONFIGURASI
# ==============================

# Folder utama Transfer-Learning/
BASE_DIR = Path(__file__).resolve().parent.parent

# Dataset tambahan
SOURCE_DIR = BASE_DIR / "data" / "dataset" / "additional"

# Hasil split khusus dataset tambahan
OUTPUT_DIR = BASE_DIR / "data" / "dataset_split_new"

CLASSES = ["MG90S", "PCA9685"]

TRAIN_RATIO = 0.8
SEED = 42

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# ==============================
# VALIDASI FOLDER DATASET
# ==============================

for class_name in CLASSES:
    source_class = SOURCE_DIR / class_name

    if not source_class.is_dir():
        raise FileNotFoundError(
            f"Folder kelas tidak ditemukan: {source_class}"
        )

# ==============================
# RANDOM SEED
# ==============================

random.seed(SEED)

# ==============================
# PERSIAPAN FOLDER OUTPUT
# ==============================

if OUTPUT_DIR.exists():
    shutil.rmtree(OUTPUT_DIR)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ==============================
# PROSES SPLIT
# ==============================

print("=" * 60)
print("DATASET SPLIT - P2 TRANSFER LEARNING")
print("=" * 60)

for class_name in CLASSES:

    source_class = SOURCE_DIR / class_name

    train_class = OUTPUT_DIR / "train" / class_name
    val_class = OUTPUT_DIR / "val" / class_name

    train_class.mkdir(parents=True, exist_ok=True)
    val_class.mkdir(parents=True, exist_ok=True)

    images = sorted(
        image for image in source_class.iterdir()
        if image.is_file() and image.suffix.lower() in IMAGE_EXTENSIONS
    )

    if len(images) == 0:
        raise ValueError(
            f"Tidak ada gambar yang valid di folder: {source_class}"
        )

    random.shuffle(images)

    train_count = int(len(images) * TRAIN_RATIO)

    train_images = images[:train_count]
    val_images = images[train_count:]

    print()
    print(f"Kelas       : {class_name}")
    print(f"Total       : {len(images)}")
    print(f"Train       : {len(train_images)}")
    print(f"Validation  : {len(val_images)}")

    # Copy train
    for image in train_images:
        shutil.copy2(image, train_class / image.name)

    # Copy validation
    for image in val_images:
        shutil.copy2(image, val_class / image.name)

print()
print("=" * 60)
print("SPLIT DATASET SELESAI")
print("=" * 60)
print(f"Output: {OUTPUT_DIR.resolve()}")