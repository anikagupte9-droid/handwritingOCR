from pathlib import Path
from onnxconverter_common import float16
import onnx

INPUT_DIR = Path("onnx_fp32")
OUTPUT_DIR = Path("onnx_fp16")

OUTPUT_DIR.mkdir(exist_ok=True)

for filename in ["encoder_model.onnx", "decoder_model.onnx"]:
    input_path = INPUT_DIR / filename
    output_path = OUTPUT_DIR / filename

    print(f"Converting {filename}...")

    model = onnx.load(str(input_path))

    model_fp16 = float16.convert_float_to_float16(
        model,
        keep_io_types=True
    )

    onnx.save(model_fp16, str(output_path))

    print(f"Saved: {output_path}")

# Copy the non-model files needed by the loader
for filename in [
    "config.json",
    "generation_config.json",
    "preprocessor_config.json",
    "sentencepiece.bpe.model",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
]:
    source = INPUT_DIR / filename
    destination = OUTPUT_DIR / filename

    if source.exists():
        destination.write_bytes(source.read_bytes())

print("\nONNX FP16 conversion complete.")