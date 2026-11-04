from pathlib import Path
from onnxruntime.quantization import quantize_dynamic, QuantType

INPUT_DIR = Path("onnx_fp32")
OUTPUT_DIR = Path("onnx_int8")
OUTPUT_DIR.mkdir(exist_ok=True)

for filename in ["encoder_model.onnx", "decoder_model.onnx"]:
    input_path = INPUT_DIR / filename
    output_path = OUTPUT_DIR / filename

    print(f"\nQuantizing {filename}...")

    quantize_dynamic(
        model_input=str(input_path),
        model_output=str(output_path),
        weight_type=QuantType.QInt8
    )

    print(f"Saved: {output_path}")

# Copy supporting files
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

print("\nONNX INT8 conversion complete.")