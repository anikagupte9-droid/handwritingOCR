from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch
import time
from pathlib import Path


MODEL_PATH = "model_fp16"
DATA_DIR = Path("test_data")
GROUND_TRUTH = "ground_truth.txt"

print("Loading processor...")
processor = TrOCRProcessor.from_pretrained(MODEL_PATH)

print("Loading model...")
model = VisionEncoderDecoderModel.from_pretrained(MODEL_PATH)
model.eval()


# -----------------------------
# CER calculation
# -----------------------------
def edit_distance(reference, hypothesis):
    m = len(reference)
    n = len(hypothesis)

    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i

    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if reference[i - 1] == hypothesis[j - 1]:
                cost = 0
            else:
                cost = 1

            dp[i][j] = min(
                dp[i - 1][j] + 1,       # deletion
                dp[i][j - 1] + 1,       # insertion
                dp[i - 1][j - 1] + cost # substitution
            )

    return dp[m][n]


def calculate_cer(reference, hypothesis):
    reference = reference.strip()
    hypothesis = hypothesis.strip()

    if len(reference) == 0:
        return 0.0 if len(hypothesis) == 0 else 1.0

    distance = edit_distance(reference, hypothesis)

    return distance / len(reference)


# -----------------------------
# Load ground truth
# -----------------------------
ground_truth = {}

with open(GROUND_TRUTH, "r", encoding="utf-8") as f:
    for line in f:
        image_id, text = line.rstrip("\n").split("|", 1)
        ground_truth[image_id] = text


# -----------------------------
# Benchmark
# -----------------------------
results = []
total_time = 0

print("\nStarting FP32 benchmark...")
print("=" * 60)

for i in range(1, 101):

    image_id = f"{i:03d}"
    image_path = DATA_DIR / f"{image_id}.png"

    image = Image.open(image_path).convert("RGB")

    pixel_values = processor(
        images=image,
        return_tensors="pt"
    ).pixel_values

    start = time.perf_counter()

    with torch.no_grad():
        generated_ids = model.generate(
            pixel_values,
            max_new_tokens=64
        )

    elapsed = (time.perf_counter() - start) * 1000
    total_time += elapsed

    prediction = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True
    )[0]

    reference = ground_truth[image_id]

    cer = calculate_cer(reference, prediction)

    results.append({
        "id": image_id,
        "reference": reference,
        "prediction": prediction,
        "cer": cer,
        "latency_ms": elapsed
    })

    print(
        f"{image_id} | "
        f"CER: {cer * 100:6.2f}% | "
        f"Latency: {elapsed:7.2f} ms"
    )


# -----------------------------
# Final results
# -----------------------------
average_cer = sum(r["cer"] for r in results) / len(results)
average_latency = total_time / len(results)

print("\n")
print("=" * 60)
print("TR OCR FP16 BENCHMARK")
print("=" * 60)
print(f"Images tested       : {len(results)}")
print(f"Average CER         : {average_cer * 100:.2f}%")
print(f"Average latency     : {average_latency:.2f} ms")
print(f"Total inference     : {total_time / 1000:.2f} seconds")
print("=" * 60)