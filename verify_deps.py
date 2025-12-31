import torch
try:
    import numpy as np
    print(f"Numpy version: {np.__version__}")
except ImportError as e:
    print(f"Numpy import failed: {e}")

try:
    import facenet_pytorch
    print(f"Facenet-pytorch version: {facenet_pytorch.__version__}")
except ImportError as e:
    print(f"Facenet-pytorch import failed: {e}")

try:
    import gradio as gr
    print(f"Gradio version: {gr.__version__}")
except ImportError as e:
    print(f"Gradio import failed: {e}")

print("Imports successful!")
