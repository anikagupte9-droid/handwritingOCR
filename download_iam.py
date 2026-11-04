from datasets import load_dataset
from pathlib import Path

OUTPUT_DIR = Path("test_data")
OUTPUT_DIR.mkdir(exist_ok=True)

print("Downloading IAM-line test split...")

dataset = load_dataset(
    "Teklia/IAM-line",
    split="test"
)

print(dataset)
print("Number of samples:", len(dataset))

# Take the first 100 samples for our benchmark
for i in range(100):
    sample = dataset[i]

    image = sample["image"]
    text = sample["text"]

    image_path = OUTPUT_DIR / f"{i+1:03d}.png"
    image.save(image_path)

    print(f"{i+1:03d}: {text}")

# Save ground truth
with open("ground_truth.txt", "w", encoding="utf-8") as f:
    for i in range(100):
        text = dataset[i]["text"]
        f.write(f"{i+1:03d}|{text}\n")

print("\n==============================")
print("Dataset preparation complete!")
print("==============================")
print("Images saved to:", OUTPUT_DIR)
print("Ground truth saved to: ground_truth.txt")
