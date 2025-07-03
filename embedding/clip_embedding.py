import os
import clip
import torch
import pickle
from PIL import Image
from tqdm import tqdm


image_dir = "/local/home/hanwliu/table/nerfstudio/images"  
output_file = "/local/home/hanwliu/table/vlm_embedding/clip_ViTL14_336px_embeddings.pkl"         # clip_ViTL14_embeddings.pkl" clip_ViTL14_336px_embeddings.pkl"

# Load the CLIP model and preprocessing
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-L/14@336px", device=device) # "ViT-L/14" "ViT-B/32" "ViT-L/14@336px"

# Prepare list of image paths
image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(".png")]

# Dictionary to store embeddings
image_embeddings = {}

# Process images
with torch.no_grad():
    for img_path in tqdm(image_paths, desc="Embedding images"):
        try:
            image = preprocess(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)
            embedding = model.encode_image(image).squeeze().cpu()
            image_embeddings[img_path] = embedding
        except Exception as e:
            print(f"Failed to process {img_path}: {e}")

# Save the embeddings
with open(output_file, "wb") as f:
    pickle.dump(image_embeddings, f)
print(f"\n Embeddings saved to: {output_file}")

# Example: Accessing the embedding for a specific image
sample_image = list(image_embeddings.keys())[0]
print(f"Embedding for {sample_image}:")
print(image_embeddings[sample_image])
