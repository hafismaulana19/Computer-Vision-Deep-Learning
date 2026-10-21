import tensorflow as tf
import numpy as np

MODEL_PATH = "mg90s_pca9685_model.keras"
VAL_DIR = "dataset_split/val"

IMG_SIZE = (224, 224)
BATCH_SIZE = 8

model = tf.keras.models.load_model(MODEL_PATH)

ds = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

class_names = ds.class_names

print("=" * 70)
print("CHECK VALIDATION PER CLASS")
print("=" * 70)

correct = {
    "MG90S": 0,
    "PCA9685": 0
}

total = {
    "MG90S": 0,
    "PCA9685": 0
}

for images, labels in ds:

    predictions = model.predict(images, verbose=0)

    predicted_labels = np.argmax(predictions, axis=1)

    for true_label, predicted_label in zip(labels.numpy(), predicted_labels):

        true_class = class_names[true_label]
        predicted_class = class_names[predicted_label]

        total[true_class] += 1

        if true_label == predicted_label:
            correct[true_class] += 1

        print(
            f"True: {true_class:8s} | "
            f"Prediksi: {predicted_class:8s} | "
            f"{'BENAR' if true_label == predicted_label else 'SALAH'}"
        )

print()
print("=" * 70)
print("HASIL PER CLASS")
print("=" * 70)

for cls in class_names:

    accuracy = correct[cls] / total[cls] * 100

    print(
        f"{cls:8s}: "
        f"{correct[cls]}/{total[cls]} "
        f"= {accuracy:.2f}%"
    )

print("=" * 70)