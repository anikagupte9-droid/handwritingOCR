from transformers import TrOCRProcessor
from PIL import Image

processor = TrOCRProcessor.from_pretrained("onnx_int8")

image = Image.open("test_data/001.png").convert("RGB")

pixel_values = processor(
    images=image,
    return_tensors="pt"
).pixel_values

# Convert normalized tensor back to image
img = pixel_values[0].permute(1, 2, 0)
img = ((img * 0.5) + 0.5) * 255
img = img.clamp(0, 255).byte().numpy()

Image.fromarray(img).save("001_preprocessed.png")

print(pixel_values.shape)
print(pixel_values.min())
print(pixel_values.max())
print(pixel_values.mean())
print(pixel_values.std())