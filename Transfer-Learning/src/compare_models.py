import csv
from pathlib import Path
import matplotlib.pyplot as plt

# ============================================================
# FINAL MODEL COMPARISON
# MobileNetV2 vs ResNet18
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"

print("=" * 70)
print("FINAL MODEL COMPARISON")
print("=" * 70)

# ============================================================
# DATA HASIL PENGUJIAN
# ============================================================

models = {
    "MobileNetV2": {
        "accuracy": 100.00,
        "latency": 85.225
    },
    "ResNet18 Feature": {
        "accuracy": 97.50,
        "latency": 5.658
    },
    "ResNet18 Partial": {
        "accuracy": 97.50,
        "latency": 0.673
    },
    "ResNet18 Scratch": {
        "accuracy": 75.00,
        "latency": 0.638
    }
}

# ============================================================
# TAMPILKAN HASIL
# ============================================================

print("\nHASIL PERBANDINGAN:")
print("-" * 70)

for name, data in models.items():

    print(
        f"{name:<22} | "
        f"Accuracy: {data['accuracy']:>6.2f}% | "
        f"Latency: {data['latency']:>7.3f} ms"
    )

# ============================================================
# SIMPAN CSV
# ============================================================

csv_path = RESULTS_DIR / "final_model_comparison.csv"

with open(
    csv_path,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "model",
        "external_accuracy_percent",
        "latency_ms_per_image"
    ])

    for name, data in models.items():

        writer.writerow([
            name,
            data["accuracy"],
            data["latency"]
        ])

print("\nCSV disimpan:")
print(csv_path)

# ============================================================
# GRAPH ACCURACY
# ============================================================

names = list(models.keys())
accuracies = [
    models[name]["accuracy"]
    for name in names
]

plt.figure(figsize=(10, 6))

plt.bar(
    names,
    accuracies
)

plt.title("External Test Accuracy Comparison")
plt.xlabel("Model")
plt.ylabel("Accuracy (%)")
plt.ylim(0, 110)

plt.xticks(
    rotation=20,
    ha="right"
)

plt.tight_layout()

accuracy_path = (
    RESULTS_DIR
    / "final_accuracy_comparison.png"
)

plt.savefig(
    accuracy_path,
    dpi=150
)

plt.close()

print("Grafik accuracy disimpan:")
print(accuracy_path)

# ============================================================
# GRAPH LATENCY
# ============================================================

latencies = [
    models[name]["latency"]
    for name in names
]

plt.figure(figsize=(10, 6))

plt.bar(
    names,
    latencies
)

plt.title("Inference Latency Comparison")
plt.xlabel("Model")
plt.ylabel("Latency (ms/image)")

plt.xticks(
    rotation=20,
    ha="right"
)

plt.tight_layout()

latency_path = (
    RESULTS_DIR
    / "final_latency_comparison.png"
)

plt.savefig(
    latency_path,
    dpi=150
)

plt.close()

print("Grafik latency disimpan:")
print(latency_path)

# ============================================================
# RINGKASAN
# ============================================================

best_accuracy = max(
    models.items(),
    key=lambda x: x[1]["accuracy"]
)

fastest = min(
    models.items(),
    key=lambda x: x[1]["latency"]
)

print("\n" + "=" * 70)
print("RINGKASAN")
print("=" * 70)

print(
    f"Best Accuracy : "
    f"{best_accuracy[0]} "
    f"({best_accuracy[1]['accuracy']:.2f}%)"
)

print(
    f"Fastest Model : "
    f"{fastest[0]} "
    f"({fastest[1]['latency']:.3f} ms/image)"
)

print("\nSELESAI.")