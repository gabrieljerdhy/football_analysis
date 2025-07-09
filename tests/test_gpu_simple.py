import torch
from ultralytics import YOLO
import numpy as np

print("CUDA Status:")
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"GPU Memory before: {torch.cuda.memory_allocated(0) / 1024**2:.1f} MB")

# Load model
print("Loading YOLO model...")
model = YOLO('data/models/best_player_detect.pt')
print(f"GPU Memory after model load: {torch.cuda.memory_allocated(0) / 1024**2:.1f} MB")

# Create dummy frame
dummy_frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)

# Test inference with device parameter
print("Running inference with device=cuda...")
results = model.predict(dummy_frame, device='cuda', verbose=False)
print(f"GPU Memory after inference: {torch.cuda.memory_allocated(0) / 1024**2:.1f} MB")

print("Test completed!")
