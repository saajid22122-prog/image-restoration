import numpy as np
from PIL import Image
import torch
from model import NoiseAwareRestorationNet

device = "cuda" if torch.cuda.is_available() else "cpu"
model = NoiseAwareRestorationNet()
model.load_state_dict(torch.load("../outputs/model_noise_aware_TV_v1.pt", map_location=device))
model.to(device)
model.eval()

noisy = np.load('../data/test/NoisyLR/000000.npy').astype(np.float32)
x = torch.from_numpy(noisy).unsqueeze(0).unsqueeze(0).to(device)
with torch.no_grad():
    pred = torch.clamp(model(x), 0, 1).squeeze(0).squeeze(0).cpu().numpy()

pred_img = Image.fromarray((pred * 255).astype(np.uint8))
pred_img.save('../outputs/check_pred_tv.png')
print("saved outputs/check_pred_tv.png")