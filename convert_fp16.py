from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import torch
from pathlib import Path

FP32_PATH = "."
FP16_PATH = "model_fp16"

print("Loading FP32 model...")

model = VisionEncoderDecoderModel.from_pretrained(FP32_PATH)
processor = TrOCRProcessor.from_pretrained(FP32_PATH)

print("Converting model to FP16...")

model = model.half()

output_dir = Path(FP16_PATH)
output_dir.mkdir(exist_ok=True)

print("Saving FP16 model...")

model.save_pretrained(
    output_dir,
    safe_serialization=False
)

processor.save_pretrained(output_dir)

print("\n==============================")
print("FP16 conversion complete")
print("==============================")
print(f"Saved to: {FP16_PATH}")
print("==============================")