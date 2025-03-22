# Deep Image Duplicates

A Python tool for finding duplicate or similar images using neural networks.

## Overview

Deep Image Duplicates is a powerful tool that uses deep learning models to detect visually similar images in a directory. It helps you find and manage duplicate or near-duplicate images by:

1. Scanning directories recursively for images
2. Computing image embeddings using neural networks
3. Finding similar images based on embedding similarity
4. Providing a graphical interface to review and delete duplicates

## Features

- **Multiple neural network models** to choose from (ResNet, EfficientNet, ViT, etc.)
- **CUDA acceleration** for GPU-enabled systems
- **Interactive preview** of duplicate image groups
- **Bulk or selective deletion** of duplicate images
- **Adjustable similarity threshold** to control detection sensitivity

## Requirements

All required packages are listed in `reqs.txt`. The main dependencies are:

```
torch
torchvision
pillow
numpy
tqdm
tk
```

## Installation

1. Clone the repository:
```bash
git https://github.com/ZeroDifa/Deep-Image-Duplicates.git
cd cd Deep-Image-Duplicates
```

2. Install the requirements:
```bash
pip install -r reqs.txt
```

3. Setting up CUDA (for GPU acceleration):
   - Use Python 3.10
   - Ensure you have a CUDA-compatible NVIDIA GPU
   - Install NVIDIA CUDA Toolkit from https://developer.nvidia.com/cuda-downloads
   - Install the PyTorch version with CUDA support:
   ```bash
   # For CUDA 11.8
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   
   # For CUDA 12.1
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
   ```
   - Verify CUDA is working by running:
   ```python
   import torch
   print("CUDA available:", torch.cuda.is_available())
   print("CUDA device count:", torch.cuda.device_count())
   print("CUDA device name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No CUDA device")
   ```

## Usage

Run the main script:
```bash
python main.py
```

### Step-by-step guide:

1. **Select directory**: Enter the path to scan for duplicate images
2. **Choose AI model**: Select from various models, balancing speed vs accuracy
3. **Set similarity threshold**: Adjust how sensitive the duplicate detection should be
   - Lower threshold (e.g., 0.7): Finds more potential duplicates but with more false positives
   - Higher threshold (e.g., 0.95): Finds only very close matches
4. **Review duplicates**: Use the file viewer interface to browse through groups of similar images
5. **Delete duplicates**: Choose which images to keep or delete

## Interface Controls

The file viewer provides the following controls:

- **Next/Prev**: Navigate between files in a group
- **Delete**: Remove the currently viewed file
- **Delete group**: Remove all files in the current group
- **Close**: Close the current group and move to the next one
- **STOP VIEWING**: Exit the viewer and keep all remaining files

## How it works

1. The program extracts feature vectors (embeddings) from each image using the selected neural network
2. It computes similarity between images by calculating the cosine similarity of their embeddings
3. Images with similarity above the specified threshold are grouped together
4. The user interface helps you review and manage these groups

## Models

Different models offer trade-offs between speed and accuracy:

- **ResNet18**: Very fast, less accurate
- **ResNet50**: Fast, accurate (good default choice)
- **EfficientNet B0**: Fast, accurate
- **ViT Base**: Very accurate but slower
- **DenseNet121**: Balanced performance
- **Inception V3**: Good for varied image types
- **ResNet152**: Very accurate but slower

## Tips

- For large directories, use faster models like ResNet18 or EfficientNet B0
- Start with a high similarity threshold (0.9) and decrease if needed
- GPU acceleration significantly improves performance

## License

[MIT License](LICENSE)
