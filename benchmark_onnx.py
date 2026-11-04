from transformers import TrOCRProcessor
from optimum.onnxruntime import ORTModelForVision2Seq
from PIL import Image
import os
import time
import re

MODEL_PATH = "onnx_int8"
DATASET_PATH = "test_data"
GROUND_TRUTH = os.path.join(DATASET_PATH, "ground_truth.txt")


def normalize(text):
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def cer(reference, hypothesis):
    reference = normalize(reference)
    hypothesis = normalize(hypothesis)

    m = len(reference)
    n = len(hypothesis)

    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i

    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if reference[i - 1] == hypothesis[j - 1] else 1

            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost
            )

    return dp[m][n] / max(m, 1)


print("Loading processor...")
processor = TrOCRProcessor.from_pretrained(MODEL_PATH)

print("Loading ONNX model...")
model = ORTModelForVision2Seq.from_pretrained(
    MODEL_PATH,
    provider="CPUExecutionProvider",
    use_cache=False
)

# Load ground truth
ground_truth = {}

with open(GROUND_TRUTH, "r", encoding="utf-8") as f:
    for line in f:
        line = line.rstrip("\n")

        if not line.strip():
            continue

        filename, text = line.split("|", 1)
        filename = filename.strip() + ".png"
        ground_truth[filename] = text.strip()

image_files = sorted(
    f for f in os.listdir(DATASET_PATH)
    if f.lower().endswith((".png", ".jpg", ".jpeg"))
)

print(f"\nImages found: {len(image_files)}")
print("Starting ONNX FP32 benchmark...\n")

total_cer = 0
total_time = 0

for index, filename in enumerate(image_files, start=1):

    image_path = os.path.join(DATASET_PATH, filename)
    image = Image.open(image_path).convert("RGB")

    pixel_values = processor(
        images=image,
        return_tensors="pt"
    ).pixel_values

    start = time.perf_counter()

    generated_ids = model.generate(pixel_values)

    elapsed = (time.perf_counter() - start) * 1000

    prediction = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True
    )[0]

    reference = ground_truth[filename]

    image_cer = cer(reference, prediction)

    total_cer += image_cer
    total_time += elapsed

    print(
        f"[{index:03d}/{len(image_files)}] "
        f"{filename} | "
        f"CER: {image_cer * 100:.2f}% | "
        f"Latency: {elapsed:.2f} ms"
    )


average_cer = total_cer / len(image_files)
average_latency = total_time / len(image_files)

print("\n===================================")
print("ONNX FP32 BENCHMARK")
print("===================================")
print(f"Images tested:       {len(image_files)}")
print(f"Average CER:         {average_cer * 100:.2f}%")
print(f"Average latency:     {average_latency:.2f} ms")
print(f"Total inference:     {total_time / 1000:.2f} s")
print("===================================")