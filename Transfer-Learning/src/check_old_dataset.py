import tensorflow as tf
import numpy as np
from pathlib import Path
from tensorflow.keras.utils import load_img, img_to_array

MODEL_PATH = "mg90s_pca9685_model.keras"
DATASET_DIR = Path("dataset")

IMG_SIZE = (224, 224)

model = tf.keras.models.load_model(MODEL_PATH)

print("=" * 80)
print("TEST DATASET LAMA - MG90S vs PCA9685")
print("=" * 80)

for true_class in ["MG90S", "PCA9685"]:

    files = sorted((DATASET_DIR / true_class).glob("*.jpg"))

    correct = 0

    print()
    print("=" * 80)
    print(f"TEST {true_class} - {len(files)} GAMBAR")
    print("=" * 80)

    for i, path in enumerate(files, 1):

        img = load_img(path, target_size=IMG_SIZE)
        img = img_to_array(img)

        img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
        img = np.expand_dims(img, axis=0)

        prediction = model.predict(img, verbose=0)[0]

        mg90s = prediction[0] * 100
        pca9685 = prediction[1] * 100

        predicted_class = (
            "MG90S"
            if prediction[0] > prediction[1]
            else "PCA9685"
        )

        if predicted_class == true_class:
            correct += 1

        print(
            f"{i:02d}. {path.name}"
            f" -> MG90S: {mg90s:6.2f}%"
            f" | PCA9685: {pca9685:6.2f}%"
            f" | Prediksi: {predicted_class}"
        )

    accuracy = correct / len(files) * 100

    print()
    print(f"BENAR     : {correct}/{len(files)}")
    print(f"AKURASI   : {accuracy:.2f}%")

print()
print("=" * 80)
print("TEST SELESAI")
print("=" * 80)