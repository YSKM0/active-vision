import os
import torch
import pickle
from tqdm import tqdm
from PIL import Image
from transformers import BlipProcessor, BlipModel

# Set your image directory and output file path
image_dir = "/local/home/hanwliu/wheelbarrow1/nerfstudio/images"
output_file = (
    "/local/home/hanwliu/wheelbarrow1/vlm_embeddings/blip_ViTB16_embeddings.pkl"
)

# Load the BLIP model and processor
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipModel.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

# Prepare list of image paths
image_paths = [
    os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(".png")
]

# Dictionary to store embeddings
image_embeddings = {}

# Process images
with torch.no_grad():
    for img_path in tqdm(image_paths, desc="Embedding images"):
        try:
            image = Image.open(img_path).convert("RGB")
            inputs = processor(images=image, return_tensors="pt").to(device)
            embedding = model.get_image_features(**inputs).squeeze().cpu()
            image_embeddings[img_path] = embedding
        except Exception as e:
            print(f"Failed to process {img_path}: {e}")

# Save the embeddings to disk using pickle
with open(output_file, "wb") as f:
    pickle.dump(image_embeddings, f)

print(f"\n Embeddings saved to: {output_file}")
