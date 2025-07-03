import os
import torch
from PIL import Image
from tqdm import tqdm
import pickle
from transformers import AutoModelForCausalLM
from torchvision import transforms

# Set your image directory and output file path
image_dir = "/local/home/hanwliu/lab_record/nerfstudio/images_4"
output_file = "/local/home/hanwliu/lab_record/vlm_embedding/cogvlm_image_representations.pkl"

# Load the CogVLM model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModelForCausalLM.from_pretrained(
    "THUDM/cogvlm-base-224-hf", 
    trust_remote_code=True,
    device_map="auto",
    offload_folder="/tmp/model_offload"
)
model.eval()

# Custom preprocessing function
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Prepare list of image paths
image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(".png")]

# Dictionary to store image representations
image_representations = {}

# Process and obtain image representations
with torch.no_grad():
    for img_path in tqdm(image_paths, desc="Processing images"):
        try:
            image = Image.open(img_path).convert("RGB")
            input_tensor = preprocess(image).unsqueeze(0).to(device).long()

            
            # Directly use the model for embedding extraction
            outputs = model(input_tensor, output_hidden_states=True)
            
            # Extract the last hidden state as the embedding
            embedding = outputs.hidden_states[-1].mean(dim=1).squeeze().cpu()
            image_representations[img_path] = embedding
        except Exception as e:
            print(f"Failed to process {img_path}: {e}")

# Save the image representations to disk using pickle
with open(output_file, "wb") as f:
    pickle.dump(image_representations, f)

print(f"\n Image representations saved to: {output_file}")

