import time
import csv
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models


# ============================================================
# EVALUASI EXTERNAL TEST - RESNET18 3 MODE
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

EXTERNAL_DIR = BASE_DIR / "data" / "external_test_v2"
RESULTS_DIR = BASE_DIR / "results"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE = 224
BATCH_SIZE = 8

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 70)
print("RESNET-18 EXTERNAL TEST")
print("=" * 70)

print(f"Device : {DEVICE}")

if torch.cuda.is_available():
    print(f"GPU    : {torch.cuda.get_device_name(0)}")

# ============================================================
# VALIDASI DATASET
# ============================================================

if not EXTERNAL_DIR.is_dir():
    raise FileNotFoundError(
        f"Folder external test tidak ditemukan: {EXTERNAL_DIR}"
    )

for class_name in ["MG90S", "PCA9685"]:
    class_dir = EXTERNAL_DIR / class_name

    if not class_dir.is_dir():
        raise FileNotFoundError(
            f"Folder kelas tidak ditemukan: {class_dir}"
        )

# ============================================================
# TRANSFORM
# ============================================================

test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# DATASET
# ============================================================

dataset = datasets.ImageFolder(
    EXTERNAL_DIR,
    transform=test_transform
)

if len(dataset) == 0:
    raise ValueError(
        f"Tidak ada gambar untuk diuji di {EXTERNAL_DIR}"
    )

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

class_names = dataset.classes
NUM_CLASSES = len(class_names)

EXPECTED_CLASSES = ["MG90S", "PCA9685"]

if class_names != EXPECTED_CLASSES:
    raise ValueError(
        f"Urutan kelas tidak sesuai. "
        f"Ditemukan: {class_names}, "
        f"seharusnya: {EXPECTED_CLASSES}"
    )

print("\nExternal Test:")
print(f"Classes : {class_names}")
print(f"Total   : {len(dataset)}")

for cls in class_names:
    count = sum(
        1 for _, label in dataset.samples
        if dataset.classes[label] == cls
    )
    print(f"  {cls}: {count}")


# ============================================================
# MODEL BUILDER
# ============================================================

def build_model(mode):

    if mode == "feature":

        model = models.resnet18(
            weights=None
        )

        model.fc = nn.Linear(
            model.fc.in_features,
            NUM_CLASSES
        )

    elif mode == "partial":

        model = models.resnet18(
            weights=None
        )

        model.fc = nn.Linear(
            model.fc.in_features,
            NUM_CLASSES
        )

    elif mode == "scratch":

        model = models.resnet18(
            weights=None
        )

        model.fc = nn.Linear(
            model.fc.in_features,
            NUM_CLASSES
        )

    else:
        raise ValueError("Mode tidak dikenal.")

    return model.to(DEVICE)


# ============================================================
# CHECKPOINT
# ============================================================

checkpoint_files = {
    "feature":
        RESULTS_DIR / "resnet18_feature_best.pth",

    "partial":
        RESULTS_DIR / "resnet18_partial_best.pth",

    "scratch":
        RESULTS_DIR / "resnet18_scratch_best.pth"
}

# ============================================================
# VALIDASI CHECKPOINT
# ============================================================

for mode, checkpoint in checkpoint_files.items():
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"Checkpoint {mode} tidak ditemukan: {checkpoint}"
        )

# ============================================================
# EVALUATION
# ============================================================

results = []


for mode in [
    "feature",
    "partial",
    "scratch"
]:

    print("\n" + "=" * 70)
    print(f"TEST MODEL : {mode.upper()}")
    print("=" * 70)

    checkpoint = checkpoint_files[mode]

    print(f"Checkpoint : {checkpoint}")

    model = build_model(mode)

    state_dict = torch.load(
        checkpoint,
        map_location=DEVICE
    )

    model.load_state_dict(state_dict)

    model.eval()

    correct = 0
    total = 0

    class_correct = [0] * NUM_CLASSES
    class_total = [0] * NUM_CLASSES

    inference_times = []

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            # ------------------------------------------------
            # GPU SYNCHRONIZATION
            # ------------------------------------------------

            if DEVICE.type == "cuda":
                torch.cuda.synchronize()

            start = time.perf_counter()

            outputs = model(images)

            if DEVICE.type == "cuda":
                torch.cuda.synchronize()

            end = time.perf_counter()

            inference_times.append(
                end - start
            )

            _, predicted = torch.max(
                outputs,
                1
            )

            total += labels.size(0)

            correct += (
                predicted == labels
            ).sum().item()

            for label, prediction in zip(
                labels,
                predicted
            ):

                label_index = label.item()

                class_total[label_index] += 1

                if prediction.item() == label_index:
                    class_correct[label_index] += 1

    # ========================================================
    # ACCURACY
    # ========================================================

    overall_accuracy = (
        100.0 * correct / total
    )

    print(
        f"\nExternal Accuracy : "
        f"{overall_accuracy:.2f}%"
    )

    # ========================================================
    # CLASS ACCURACY
    # ========================================================

    print("\nAccuracy per class:")

    for i, cls in enumerate(class_names):

        if class_total[i] > 0:

            acc = (
                100.0 *
                class_correct[i] /
                class_total[i]
            )

        else:
            acc = 0.0

        print(
            f"  {cls:10s}: "
            f"{acc:.2f}% "
            f"({class_correct[i]}/{class_total[i]})"
        )

    # ========================================================
    # LATENCY
    # ========================================================

    total_inference_time = sum(
        inference_times
    )

    num_images = total

    latency_per_image_ms = (
        total_inference_time /
        num_images
    ) * 1000.0

    print(
        f"\nLatency per image : "
        f"{latency_per_image_ms:.3f} ms"
    )

    results.append({
        "mode": mode,
        "accuracy": overall_accuracy,
        "correct": correct,
        "total": total,
        "latency_ms": latency_per_image_ms
    })


# ============================================================
# SAVE CSV
# ============================================================

csv_path = (
    RESULTS_DIR /
    "resnet18_external_test_results.csv"
)

with open(
    csv_path,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "Mode",
        "External Accuracy (%)",
        "Correct",
        "Total",
        "Latency per Image (ms)"
    ])

    for result in results:

        writer.writerow([
            result["mode"],
            f"{result['accuracy']:.2f}",
            result["correct"],
            result["total"],
            f"{result['latency_ms']:.3f}"
        ])


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("HASIL EXTERNAL TEST")
print("=" * 70)

for result in results:

    print(
        f"{result['mode']:10s} | "
        f"Accuracy: {result['accuracy']:.2f}% | "
        f"Latency: {result['latency_ms']:.3f} ms/image"
    )

print("\nHasil disimpan:")
print(csv_path)

print("\nSELESAI.")