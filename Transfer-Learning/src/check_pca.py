import tensorflow as tf
import numpy as np
from pathlib import Path
from tensorflow.keras.utils import load_img, img_to_array

MODEL_PATH = "mg90s_pca9685_model.keras"
PCA_DIR = Path("dataset/PCA9685")

IMG_SIZE = (224, 224)

print("=" * 80)
print("CHECK 20 GAMBAR PCA9685 - DATASET LAMA")
print("=" * 80)

model = tf.keras.models.load_model(MODEL_PATH)

files = sorted(PCA_DIR.glob("*.jpg"))[:20]

correct = 0

for i, path in enumerate(files, 1):

    img = load_img(
        path,
        target_size=IMG_SIZE
    )

    img = img_to_array(img)

    img = np.expand_dims(img, axis=0)

    prediction = model.predict(
        img,
        verbose=0
    )[0]

    mg90s = prediction[0] * 100
    pca9685 = prediction[1] * 100

    predicted_class = (
        "MG90S"
        if prediction[0] > prediction[1]
        else "PCA9685"
    )

    if predicted_class == "PCA9685":
        correct += 1

    print(
        f"{i:02d}. {path.name}"
        f" -> MG90S: {mg90s:6.2f}%"
        f" | PCA9685: {pca9685:6.2f}%"
        f" | Prediksi: {predicted_class}"
    )

print()

print("=" * 80)

if len(files) > 0:
    accuracy = correct / len(files) * 100

    print(f"BENAR: {correct}/{len(files)}")
    print(
        f"AKURASI PADA {len(files)} GAMBAR PCA9685: "
        f"{accuracy:.2f}%"
    )
else:
    print("Tidak ada gambar PCA9685 ditemukan!")

print("=" * 80)