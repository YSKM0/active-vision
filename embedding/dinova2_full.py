import os
import torch
import pickle
from PIL import Image
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel

# ——— Configuration ———
image_dir = "/local/home/hanwliu/table/nerfstudio/images"
output_file = (
    "/local/home/hanwliu/table/vlm_embedding/dinov2_large_fullres_embeddings.pkl"
)
model_name = "facebook/dinov2-large"

# ——— Device Setup ———
device = "cuda" if torch.cuda.is_available() else "cpu"

# ——— Load DINOv2 Processor & Model ———
processor = AutoImageProcessor.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name).to(device)
model.eval()

# ——— Gather Image Paths ———
image_paths = [
    os.path.join(image_dir, fn)
    for fn in os.listdir(image_dir)
    if fn.lower().endswith(".png")
]

# ——— Embed Images & Save ———
image_embeddings = {}

with torch.no_grad():
    for img_path in tqdm(image_paths, desc="Embedding with DINOv2"):
        try:
            # Load full-resolution image
            img = Image.open(img_path).convert("RGB")

            # Preprocess without resizing or cropping
            inputs = processor(
                images=img, return_tensors="pt", do_resize=False, do_center_crop=False
            ).to(device)

            # Forward pass
            outputs = model(**inputs)

            # Extract the [CLS] token as global embedding
            cls_emb = outputs.last_hidden_state[:, 0, :].squeeze().cpu()
            image_embeddings[img_path] = cls_emb

        except Exception as e:
            print(f"⚠️ Failed to process {img_path}: {e}")

# Save all embeddings to a pickle file
with open(output_file, "wb") as f:
    pickle.dump(image_embeddings, f)

print(f"Saved full-resolution embeddings to: {output_file}")
