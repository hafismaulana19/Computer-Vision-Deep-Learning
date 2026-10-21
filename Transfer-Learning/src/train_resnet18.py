import time
import csv
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.models import ResNet18_Weights
import matplotlib.pyplot as plt


# ============================================================
# P2 TRANSFER LEARNING
# RESNET-18 - 3 MODE
#
# MODE 1 : FEATURE EXTRACTION
# MODE 2 : PARTIAL FINE-TUNING
# MODE 3 : SCRATCH
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATASET_DIR = BASE_DIR / "data" / "dataset_split_v2"
RESULTS_DIR = BASE_DIR / "results"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_DIR = DATASET_DIR / "train"
VAL_DIR = DATASET_DIR / "val"

# ============================================================
# VALIDASI DATASET
# ============================================================

for folder in [TRAIN_DIR, VAL_DIR]:
    if not folder.is_dir():
        raise FileNotFoundError(
            f"Folder dataset tidak ditemukan: {folder}"
        )

IMG_SIZE = 224
BATCH_SIZE = 8
EPOCHS = 10
NUM_WORKERS = 0

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 70)
print("P2 TRANSFER LEARNING - RESNET18")
print("=" * 70)

print(f"\nDevice : {DEVICE}")

if torch.cuda.is_available():
    print(f"GPU    : {torch.cuda.get_device_name(0)}")


# ============================================================
# TRANSFORM
# SESUAI ARAHAN DOSEN
# RandomResizedCrop
# HorizontalFlip
# ColorJitter
# ============================================================

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        hue=0.05
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


val_transform = transforms.Compose([
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

print("\nLoading dataset...")

train_dataset = datasets.ImageFolder(
    TRAIN_DIR,
    transform=train_transform
)

val_dataset = datasets.ImageFolder(
    VAL_DIR,
    transform=val_transform
)

# ============================================================
# VALIDASI DATASET DAN URUTAN KELAS
# ============================================================

EXPECTED_CLASSES = ["MG90S", "PCA9685"]

if train_dataset.classes != EXPECTED_CLASSES:
    raise ValueError(
        f"Urutan kelas training tidak sesuai: "
        f"{train_dataset.classes}"
    )

if val_dataset.classes != EXPECTED_CLASSES:
    raise ValueError(
        f"Urutan kelas validasi tidak sesuai: "
        f"{val_dataset.classes}"
    )

if len(train_dataset) == 0 or len(val_dataset) == 0:
    raise ValueError(
        "Dataset training atau validasi tidak memiliki gambar."
    )

# ============================================================
# DATALOADER
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS
)

class_names = train_dataset.classes
NUM_CLASSES = len(class_names)

print(f"Classes       : {class_names}")
print(f"Train images  : {len(train_dataset)}")
print(f"Val images    : {len(val_dataset)}")

for dataset_name, dataset in [
    ("Training", train_dataset),
    ("Validasi", val_dataset)
]:
    for class_name in EXPECTED_CLASSES:
        class_index = dataset.class_to_idx[class_name]
        image_count = sum(
            1 for _, label in dataset.samples
            if label == class_index
        )

        if image_count == 0:
            raise ValueError(
                f"Tidak ada gambar kelas {class_name} "
                f"pada dataset {dataset_name}."
            )

# ============================================================
# MODEL BUILDER
# ============================================================

def build_model(mode):

    if mode == "feature":

        print("\nLoading ResNet18 ImageNet weights...")

        model = models.resnet18(
            weights=ResNet18_Weights.DEFAULT
        )

        # Freeze seluruh backbone
        for param in model.parameters():
            param.requires_grad = False

        # Ganti classifier
        model.fc = nn.Linear(
            model.fc.in_features,
            NUM_CLASSES
        )

    elif mode == "partial":

        print("\nLoading ResNet18 ImageNet weights...")

        model = models.resnet18(
            weights=ResNet18_Weights.DEFAULT
        )

        # Freeze semua layer terlebih dahulu
        for param in model.parameters():
            param.requires_grad = False

        # Buka layer4
        for param in model.layer4.parameters():
            param.requires_grad = True

        # Ganti classifier
        model.fc = nn.Linear(
            model.fc.in_features,
            NUM_CLASSES
        )

    elif mode == "scratch":

        print("\nCreating ResNet18 from scratch...")

        model = models.resnet18(
            weights=None
        )

        # Semua parameter trainable
        for param in model.parameters():
            param.requires_grad = True

        # Ganti classifier
        model.fc = nn.Linear(
            model.fc.in_features,
            NUM_CLASSES
        )

    else:
        raise ValueError("Mode tidak dikenal.")

    return model.to(DEVICE)


# ============================================================
# TRAIN ONE MODEL
# ============================================================

def train_model(mode):

    print("\n" + "=" * 70)
    print(f"TRAINING MODE : {mode.upper()}")
    print("=" * 70)

    model = build_model(mode)

    # --------------------------------------------------------
    # LEARNING RATE
    # --------------------------------------------------------

    if mode == "feature":

        optimizer = torch.optim.Adam(
            model.fc.parameters(),
            lr=1e-3
        )

    elif mode == "partial":

        optimizer = torch.optim.Adam([
            {
                "params": model.layer4.parameters(),
                "lr": 1e-4
            },
            {
                "params": model.fc.parameters(),
                "lr": 1e-3
            }
        ])

    else:

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=1e-3
        )

    # --------------------------------------------------------
    # COSINE ANNEALING LR
    # --------------------------------------------------------

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS
    )

    criterion = nn.CrossEntropyLoss()

    best_val_acc = -1.0
    epoch_90 = None

    train_acc_history = []
    val_acc_history = []

    start_time = time.perf_counter()

    # ========================================================
    # EPOCH LOOP
    # ========================================================

    for epoch in range(EPOCHS):

        # ----------------------------------------------------
        # TRAIN
        # ----------------------------------------------------

        model.train()

        correct = 0
        total = 0

        for images, labels in train_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            loss.backward()

            optimizer.step()

            _, predicted = torch.max(
                outputs,
                1
            )

            total += labels.size(0)

            correct += (
                predicted == labels
            ).sum().item()

        train_acc = 100.0 * correct / total

        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        model.eval()

        correct = 0
        total = 0

        with torch.no_grad():

            for images, labels in val_loader:

                images = images.to(DEVICE)
                labels = labels.to(DEVICE)

                outputs = model(images)

                _, predicted = torch.max(
                    outputs,
                    1
                )

                total += labels.size(0)

                correct += (
                    predicted == labels
                ).sum().item()

        val_acc = 100.0 * correct / total

        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)

        if val_acc > best_val_acc:

            best_val_acc = val_acc

            torch.save(
                model.state_dict(),
                RESULTS_DIR /
                f"resnet18_{mode}_best.pth"
            )

        if epoch_90 is None and val_acc >= 90.0:

            epoch_90 = epoch + 1

        scheduler.step()

        print(
            f"Epoch {epoch + 1:02d}/{EPOCHS} | "
            f"Train Acc: {train_acc:.2f}% | "
            f"Val Acc: {val_acc:.2f}%"
        )

    training_time = time.perf_counter() - start_time

    print("\nHasil:")
    print(f"Best Val Accuracy : {best_val_acc:.2f}%")
    print(f"Training Time     : {training_time:.2f} detik")

    if epoch_90 is not None:
        print(f"Epoch >= 90%      : {epoch_90}")
    else:
        print("Epoch >= 90%      : Tidak tercapai")

    return {
        "mode": mode,
        "best_val_accuracy": best_val_acc,
        "training_time_sec": training_time,
        "epoch_90": epoch_90,
        "train_accuracy": train_acc_history,
        "val_accuracy": val_acc_history
    }


# ============================================================
# RUN 3 MODES
# ============================================================

results = []

for mode in [
    "feature",
    "partial",
    "scratch"
]:

    result = train_model(mode)

    results.append(result)


# ============================================================
# SAVE RESULT TABLE
# ============================================================

csv_path = RESULTS_DIR / "resnet18_3_modes_results.csv"

with open(
    csv_path,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "Mode",
        "Best Validation Accuracy (%)",
        "Training Time (sec)",
        "Epoch >= 90%"
    ])

    for result in results:

        writer.writerow([
            result["mode"],
            f"{result['best_val_accuracy']:.2f}",
            f"{result['training_time_sec']:.2f}",
            result["epoch_90"]
            if result["epoch_90"] is not None
            else "-"
        ])

print("\nTabel hasil disimpan:")
print(csv_path)


# ============================================================
# GRAPH ACCURACY 3 MODE
# ============================================================

plt.figure(figsize=(10, 6))

for result in results:

    plt.plot(
        range(1, EPOCHS + 1),
        result["val_accuracy"],
        marker="o",
        label=result["mode"]
    )

plt.title(
    "ResNet-18 Validation Accuracy - 3 Modes"
)

plt.xlabel("Epoch")
plt.ylabel("Validation Accuracy (%)")

plt.legend()
plt.grid(True)

graph_path = RESULTS_DIR / "resnet18_accuracy_3_modes.png"

plt.savefig(
    graph_path,
    dpi=150,
    bbox_inches="tight"
)

plt.close()

print("\nGrafik accuracy disimpan:")
print(graph_path)


# ============================================================
# GRAPH TRAINING TIME
# ============================================================

plt.figure(figsize=(8, 5))

modes = [
    result["mode"]
    for result in results
]

times = [
    result["training_time_sec"]
    for result in results
]

plt.bar(
    modes,
    times
)

plt.title(
    "ResNet-18 Training Time - 3 Modes"
)

plt.xlabel("Mode")
plt.ylabel("Training Time (seconds)")

plt.grid(
    axis="y"
)

time_graph_path = RESULTS_DIR / "resnet18_training_time.png"

plt.savefig(
    time_graph_path,
    dpi=150,
    bbox_inches="tight"
)

plt.close()

print("Grafik training time disimpan:")
print(time_graph_path)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("RINGKASAN RESNET-18")
print("=" * 70)

for result in results:

    epoch90_text = (
        str(result["epoch_90"])
        if result["epoch_90"] is not None
        else "-"
    )

    print(
        f"{result['mode']:10s} | "
        f"Best Val: {result['best_val_accuracy']:.2f}% | "
        f"Time: {result['training_time_sec']:.2f}s | "
        f"Epoch >=90%: {epoch90_text}"
    )

print("\nSELESAI.")