from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import time

MODEL_PATH = "."

print("Loading processor...")
processor = TrOCRProcessor.from_pretrained(MODEL_PATH)

print("Loading model...")
model = VisionEncoderDecoderModel.from_pretrained(MODEL_PATH)

model.eval()

print("Loading image...")
image = Image.open("test.png").convert("RGB")

print("Running inference...")

start = time.perf_counter()

pixel_values = processor(
    images=image,
    return_tensors="pt"
).pixel_values

generated_ids = model.generate(
    pixel_values,
    max_new_tokens=64
)

text = processor.batch_decode(
    generated_ids,
    skip_special_tokens=True
)[0]

elapsed = (time.perf_counter() - start) * 1000

print("\n==============================")
print("Recognized text:")
print(text)
print("==============================")
print(f"Inference time: {elapsed:.2f} ms")