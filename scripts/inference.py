import torch
from model import LensCNN
import numpy as np

# Load model
model = LensCNN()
model.load_state_dict(torch.load('../models/lens_cnn.pth'))
model.eval()

# Load image
image = np.load('path/to/image.npy')  # Shape: (3, 64, 64)
image = (image - np.mean(image)) / (np.std(image) + 1e-8)  # Normalize
image_tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0)  # Add batch dimension

# Make prediction
with torch.no_grad():
    output = model(image_tensor)
    probabilities = torch.softmax(output, dim=1)
    is_lens = probabilities[0][1].item()  # Probability of being a lens

print(f"Probability of gravitational lens: {is_lens:.2%}")