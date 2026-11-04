import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from pathlib import Path

MODEL_PATH = "."
OUTPUT_PATH = "model_int8"

print("Loading FP32 model...")

processor = TrOCRProcessor.from_pretrained(MODEL_PATH)
model = VisionEncoderDecoderModel.from_pretrained(MODEL_PATH)

model.eval()

print("Applying dynamic INT8 quantization...")

model_int8 = torch.ao.quantization.quantize_dynamic(
    model,
    {torch.nn.Linear},
    dtype=torch.qint8
)

output_dir = Path(OUTPUT_PATH)
output_dir.mkdir(exist_ok=True)

print("Saving INT8 model state...")

torch.save(
    model_int8.state_dict(),
    output_dir / "model_int8_state_dict.pt"
)

processor.save_pretrained(output_dir)

print("\n==============================")
print("INT8 quantization complete")
print("==============================")
print(f"Saved to: {OUTPUT_PATH}")
print("==============================")