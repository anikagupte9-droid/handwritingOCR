from pathlib import Path
import onnx
from onnxruntime.transformers.onnx_model import OnnxModel

INPUT_DIR = Path("onnx_fp32")
OUTPUT_DIR = Path("onnx_fp16")
OUTPUT_DIR.mkdir(exist_ok=True)

for filename in ["encoder_model.onnx", "decoder_model.onnx"]:
    input_path = INPUT_DIR / filename
    output_path = OUTPUT_DIR / filename

    print(f"\nConverting {filename}...")

    model = OnnxModel(onnx.load(str(input_path)))

    model.convert_float_to_float16(
        keep_io_types=True,
        use_symbolic_shape_infer=False
    )

    model.save_model_to_file(str(output_path))

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

print("\nONNX Runtime FP16 conversion complete.")