"""
Runs the trained model on ANY regular image file (jpg/png).
Usage: python run_custom_image.py path/to/your/image.png
"""

import sys
import os
import numpy as np
from PIL import Image
import torch
from model import NoiseAwareRestorationNet

CHECKPOINT_PATH = "../outputs/model_noise_aware_LOCAL_v1.pt"

if len(sys.argv) < 2:
    print("Usage: python run_custom_image.py path/to/image.png")
    sys.exit(1)

input_path = sys.argv[1]
base_name = os.path.splitext(os.path.basename(input_path))[0]  # e.g. "lenna" or "mandrill"

device = "cuda" if torch.cuda.is_available() else "cpu"

img = Image.open(input_path).convert("L")
img = img.resize((128, 128))
arr = np.array(img).astype(np.float32) / 255.0

model = NoiseAwareRestorationNet()
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
model.to(device)
model.eval()

x = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).to(device)
with torch.no_grad():
    pred = torch.clamp(model(x), 0, 1).squeeze(0).squeeze(0).cpu().numpy()

Image.fromarray((arr * 255).astype(np.uint8)).save(f"../outputs/{base_name}_input_128.png")
Image.fromarray((pred * 255).astype(np.uint8)).save(f"../outputs/{base_name}_output_256.png")

print(f"Saved outputs/{base_name}_input_128.png and outputs/{base_name}_output_256.png")