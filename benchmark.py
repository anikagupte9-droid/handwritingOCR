from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import time
import os

MODEL_PATH = "model_fp16"
IMAGE_PATH = "test.png"
RUNS = 5

print("Loading processor...")
processor = TrOCRProcessor.from_pretrained(MODEL_PATH)

print("Loading model...")
model = VisionEncoderDecoderModel.from_pretrained(MODEL_PATH)
model.eval()

print("Loading image...")
image = Image.open(IMAGE_PATH).convert("RGB")

pixel_values = processor(
    images=image,
    return_tensors="pt"
).pixel_values

# Warm-up
print("Warming up...")
with __import__("torch").no_grad():
    model.generate(
        pixel_values,
        max_new_tokens=64
    )

print(f"Running benchmark ({RUNS} runs)...")

times = []
text = ""

for i in range(RUNS):
    start = time.perf_counter()

    with __import__("torch").no_grad():
        generated_ids = model.generate(
            pixel_values,
            max_new_tokens=64
        )

    elapsed = (time.perf_counter() - start) * 1000
    times.append(elapsed)

    text = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True
    )[0]

    print(f"Run {i + 1}: {elapsed:.2f} ms")

model_size_mb = os.path.getsize("pytorch_model.bin") / (1024 * 1024)

print("\n==============================")
print("TR OCR FP32 BASELINE")
print("==============================")
print(f"Recognized text : {text}")
print(f"Model size      : {model_size_mb:.2f} MB")
print(f"Average latency : {sum(times) / len(times):.2f} ms")
print(f"Minimum latency : {min(times):.2f} ms")
print(f"Maximum latency : {max(times):.2f} ms")
print("==============================")