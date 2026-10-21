import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from pathlib import Path
import matplotlib.pyplot as plt
import shutil
import random

# ============================================================
# P2 TRANSFER LEARNING - MG90S vs PCA9685
# DATASET LAMA: 60 MG90S + 60 PCA9685
# SPLIT OTOMATIS 80:20
# ============================================================

# ============================================================
# PATH CONFIGURATION
# ============================================================

# Folder utama Transfer-Learning/
BASE_DIR = Path(__file__).resolve().parent.parent

# Dataset lama
DATASET_DIR = BASE_DIR / "data" / "dataset" / "original"

# Hasil split untuk training versi awal
SPLIT_DIR = BASE_DIR / "data" / "dataset_split"

# Lokasi model dan hasil training
MODEL_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

MODEL_PATH = MODEL_DIR / "mg90s_pca9685_model.keras"
CLASS_FILE = BASE_DIR / "class_names.txt"

IMG_SIZE = (224, 224)
BATCH_SIZE = 8
EPOCHS = 15
SEED = 42

random.seed(SEED)

print("=" * 60)
print("TRANSFER LEARNING - P2")
print("=" * 60)

CLASSES = ["MG90S", "PCA9685"]

for cls in CLASSES:
    class_dir = DATASET_DIR / cls

    if not class_dir.is_dir():
        raise FileNotFoundError(
            f"Folder dataset tidak ditemukan: {class_dir}"
        )

# ============================================================
# BUAT DATASET SPLIT
# ============================================================

if SPLIT_DIR.exists():
    shutil.rmtree(SPLIT_DIR)

for split in ["train", "val"]:
    for cls in ["MG90S", "PCA9685"]:
        (SPLIT_DIR / split / cls).mkdir(parents=True, exist_ok=True)

print("\nMembagi dataset lama menjadi 80% train dan 20% validation...")

for cls in ["MG90S", "PCA9685"]:

    source_dir = DATASET_DIR / cls

    images = list(source_dir.glob("*.jpg")) + \
             list(source_dir.glob("*.jpeg")) + \
             list(source_dir.glob("*.png"))

    random.shuffle(images)

    split_index = int(len(images) * 0.8)

    train_images = images[:split_index]
    val_images = images[split_index:]

    print(f"\n{cls}")
    print(f"  Total      : {len(images)}")
    print(f"  Train      : {len(train_images)}")
    print(f"  Validation : {len(val_images)}")

    for img in train_images:
        shutil.copy2(
            img,
            SPLIT_DIR / "train" / cls / img.name
        )

    for img in val_images:
        shutil.copy2(
            img,
            SPLIT_DIR / "val" / cls / img.name
        )

TRAIN_DIR = SPLIT_DIR / "train"
VAL_DIR = SPLIT_DIR / "val"

# ============================================================
# LOAD DATASET
# ============================================================

train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=SEED,
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

class_names = train_ds.class_names

print("\nClass:")
for i, name in enumerate(class_names):
    print(f"  {i}: {name}")

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)

# ============================================================
# DATA AUGMENTATION
# ============================================================

data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.10),
    layers.RandomZoom(0.10),
    layers.RandomContrast(0.10),
])

# ============================================================
# MOBILENETV2
# ============================================================

base_model = tf.keras.applications.MobileNetV2(
    input_shape=IMG_SIZE + (3,),
    include_top=False,
    weights="imagenet",
)

base_model.trainable = False

# ============================================================
# BUILD MODEL
# ============================================================

inputs = keras.Input(shape=IMG_SIZE + (3,))

x = data_augmentation(inputs)

x = tf.keras.applications.mobilenet_v2.preprocess_input(x)

x = base_model(x, training=False)

x = layers.GlobalAveragePooling2D()(x)

x = layers.Dropout(0.2)(x)

outputs = layers.Dense(
    len(class_names),
    activation="softmax"
)(x)

model = keras.Model(inputs, outputs)

# ============================================================
# COMPILE
# ============================================================

model.compile(
    optimizer=keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

print("\nModel summary:")
model.summary()

# ============================================================
# TRAINING
# ============================================================

print("\n" + "=" * 60)
print("START TRAINING")
print("=" * 60)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
)

# ============================================================
# SAVE MODEL
# ============================================================

MODEL_DIR.mkdir(parents=True, exist_ok=True)

model.save(MODEL_PATH)

print("\nModel berhasil disimpan:")
print(MODEL_PATH)

# ============================================================
# SAVE CLASS NAMES
# ============================================================

class_file = BASE_DIR / "class_names.txt"

with open(class_file, "w", encoding="utf-8") as f:
    for name in class_names:
        f.write(name + "\n")

print("Class names disimpan:")
print(class_file)

# ============================================================
# BUAT FOLDER HASIL TRAINING
# ============================================================

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# ACCURACY GRAPH
# ============================================================

plt.figure()

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.title("Training vs Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)

accuracy_path = RESULTS_DIR / "accuracy.png"

plt.savefig(
    accuracy_path,
    dpi=150
)

plt.close()

print("Grafik accuracy:")
print(accuracy_path)

# ============================================================
# LOSS GRAPH
# ============================================================

plt.figure()

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.title("Training vs Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

loss_path = RESULTS_DIR / "loss.png"

plt.savefig(
    loss_path,
    dpi=150
)

plt.close()

print("Grafik loss:")
print(loss_path)

print("\n" + "=" * 60)
print("TRAINING SELESAI")
print("=" * 60)