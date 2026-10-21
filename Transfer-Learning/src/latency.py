import torch
import torchvision
import time
import csv
from pathlib import Path

from torchvision.models import (
    resnet18,
    ResNet18_Weights,
    mobilenet_v3_small,
    MobileNet_V3_Small_Weights
)

# ============================================================
# P2 TRANSFER LEARNING
# LATENCY: RESNET-18 vs MOBILENETV3-SMALL
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMG_SIZE = 224
WARMUP = 20
ITERATIONS = 100

print("=" * 70)
print("P2 TRANSFER LEARNING - LATENCY TEST")
print("=" * 70)

print(f"\nDevice : {DEVICE}")

if torch.cuda.is_available():
    print(f"GPU    : {torch.cuda.get_device_name(0)}")

# ============================================================
# DUMMY INPUT
# ============================================================

input_tensor = torch.randn(
    1,
    3,
    IMG_SIZE,
    IMG_SIZE,
    device=DEVICE
)

# ============================================================
# LOAD MODELS
# ============================================================

print("\nLoading models...")

models = {}

print("Loading ResNet18...")
models["ResNet18"] = resnet18(
    weights=ResNet18_Weights.DEFAULT
).to(DEVICE)

print("Loading MobileNetV3-Small...")
models["MobileNetV3-Small"] = mobilenet_v3_small(
    weights=MobileNet_V3_Small_Weights.DEFAULT
).to(DEVICE)

# ============================================================
# EVALUATION MODE
# ============================================================

for model in models.values():
    model.eval()

# ============================================================
# LATENCY FUNCTION
# ============================================================

def measure_latency(model):

    # --------------------------------------------------------
    # WARMUP
    # --------------------------------------------------------

    with torch.inference_mode():

        for _ in range(WARMUP):

            _ = model(input_tensor)

        if DEVICE.type == "cuda":
            torch.cuda.synchronize()

    # --------------------------------------------------------
    # BENCHMARK
    # --------------------------------------------------------

    latencies = []

    with torch.inference_mode():

        for _ in range(ITERATIONS):

            if DEVICE.type == "cuda":
                torch.cuda.synchronize()

            start = time.perf_counter()

            _ = model(input_tensor)

            if DEVICE.type == "cuda":
                torch.cuda.synchronize()

            end = time.perf_counter()

            latency_ms = (end - start) * 1000

            latencies.append(latency_ms)

    average = sum(latencies) / len(latencies)
    minimum = min(latencies)
    maximum = max(latencies)

    fps = 1000 / average

    return average, minimum, maximum, fps


# ============================================================
# RUN TEST
# ============================================================

results = []

print("\n" + "=" * 70)
print("LATENCY TEST")
print("=" * 70)

for name, model in models.items():

    print(f"\nTesting {name}...")

    average, minimum, maximum, fps = measure_latency(model)

    print(f"Average latency : {average:.3f} ms")
    print(f"Minimum latency : {minimum:.3f} ms")
    print(f"Maximum latency : {maximum:.3f} ms")
    print(f"Estimated FPS   : {fps:.2f}")

    results.append([
        name,
        average,
        minimum,
        maximum,
        fps
    ])

# ============================================================
# SAVE CSV
# ============================================================

csv_path = RESULTS_DIR / "latency_comparison.csv"

with open(
    csv_path,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "model",
        "average_latency_ms",
        "minimum_latency_ms",
        "maximum_latency_ms",
        "estimated_fps"
    ])

    writer.writerows(results)

# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("HASIL LATENCY")
print("=" * 70)

for row in results:

    name = row[0]
    average = row[1]
    minimum = row[2]
    maximum = row[3]
    fps = row[4]

    print(
        f"{name:<20} | "
        f"Avg: {average:>8.3f} ms | "
        f"Min: {minimum:>8.3f} ms | "
        f"Max: {maximum:>8.3f} ms | "
        f"FPS: {fps:>7.2f}"
    )

print("\nHasil disimpan:")
print(csv_path)

print("\n" + "=" * 70)
print("SELESAI")
print("=" * 70)