import os
import torch
import pickle
from PIL import Image
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel

# Configuration
image_dir = "/local/home/hanwliu/wheelbarrow1/nerfstudio/images"
output_file = "/local/home/hanwliu/wheelbarrow1/vlm_embeddings/dinov2_embeddings.pkl"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the DINOv2 model and image processor
model_name = "facebook/dinov2-base"  # Alternatives: "facebook/dinov2-large", "facebook/dinov2-giant"
processor = AutoImageProcessor.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name).to(device)
model.eval()

# Gather image paths
image_paths = [
    os.path.join(image_dir, fname)
    for fname in os.listdir(image_dir)
    if fname.endswith(".png")
]

# Dictionary to store embeddings
image_embeddings = {}

# Process images and extract embeddings
with torch.no_grad():
    for img_path in tqdm(image_paths, desc="Extracting DINOv2 embeddings"):
        try:
            # Load and preprocess image
            image = Image.open(img_path).convert("RGB")
            inputs = processor(images=image, return_tensors="pt").to(device)

            # Forward pass through the model
            outputs = model(**inputs)

            # Extract the [CLS] token embedding
            embedding = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()

            # Store the embedding
            image_embeddings[img_path] = embedding
        except Exception as e:
            print(f"Error processing {img_path}: {e}")

# Save embeddings to a file
with open(output_file, "wb") as f:
    pickle.dump(image_embeddings, f)
print(f"\nEmbeddings saved to: {output_file}")

# Example: Access and display an embedding
sample_image = list(image_embeddings.keys())[0]
print(f"Embedding for {sample_image}:")
print(image_embeddings[sample_image])
