import os
import torch
import torchvision.transforms as T
from transformers import VideoMAEFeatureExtractor, VideoMAEModel
from PIL import Image
from tqdm import tqdm

# Paths and device configuration
image_dir = "/local/home/hanwliu/lab_record/nerfstudio/images_4"
output_file = "/local/home/hanwliu/lab_record/vlm_embedding/videomae_embeddings2.pkl"
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load VideoMAE model and feature extractor
feature_extractor = VideoMAEFeatureExtractor.from_pretrained("MCG-NJU/videomae-base")
model = VideoMAEModel.from_pretrained("MCG-NJU/videomae-base").to(device)

transform = T.Compose(
    [
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# Prepare list of image paths (sorted to maintain continuity)
image_paths = sorted(
    [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(".png")]
)

# Dictionary to store embeddings
video_embeddings = {}
failed_images = []

# Batch settings
batch_size = 16
batch = []
index_counter = 0

with torch.no_grad():
    for img_path in tqdm(image_paths, desc="Embedding images"):
        try:
            image = Image.open(img_path).convert("RGB")
            image_tensor = transform(image).unsqueeze(0)
            batch.append(image_tensor)

            # If the batch is full or at the end, process it
            if len(batch) == batch_size or img_path == image_paths[-1]:
                video_tensor = torch.cat(batch).unsqueeze(0).to(device)

                # Try-catch for model inference
                try:
                    outputs = model(video_tensor)
                    embedding = outputs.last_hidden_state.squeeze().cpu()
                except Exception as e:
                    print(f"Model failed to process batch: {e}")
                    failed_images.extend(
                        image_paths[index_counter : index_counter + len(batch)]
                    )
                    batch = []
                    continue

                # Store embeddings for each image in the batch
                for i in range(len(batch)):
                    frame_path = image_paths[index_counter]
                    if i < len(embedding):  # Ensure the index is valid
                        video_embeddings[frame_path] = embedding[i]
                    index_counter += 1

                # Clear batch
                batch = []

        except Exception as e:
            print(f"Failed to process {img_path}: {e}")
            failed_images.append(img_path)
            index_counter += 1
