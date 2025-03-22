import torch
from torchvision import models, transforms
from PIL import Image
import os
import numpy as np
import concurrent.futures
from tqdm import tqdm

available_models = {
    '1': ('ResNet50 | fast, accurate', models.resnet50, 224),
    '2': ('ResNet18 | very fast, less accurate', models.resnet18, 224),
    '3': ('EfficientNet B0 | fast, accurate', models.efficientnet_b0, 224),
    '4': ('ViT Base | very accurate, slow', models.vit_b_16, 224),
    '5': ('DenseNet121', models.densenet121, 224),
    '6': ('Inception V3', models.inception_v3, 299),
    '7': ('ResNet152', models.resnet152, 224),
}

model = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(model_choice):
    """Loads the selected model and returns it along with the input image size"""
    if model_choice not in available_models:
        print(f"Invalid model selection. Using default model (ResNet50).")
        model_choice = '1'
    
    model_name, model_func, image_size = available_models[model_choice]
    print(f"Loading {model_name} model...")
    print(f"Using device: {device}")
    
    model = model_func(pretrained=True)
    model.to(device)
    model.eval()
    
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    return model, transform

def get_image_embedding(image_path, model, transform):
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = model(image)
    return embedding.cpu().squeeze().numpy()

def process_single_image(args):
    """Processes a single image and returns its embedding"""
    image_path, model, transform = args
    try:
        embedding = get_image_embedding(image_path, model, transform)
        return image_path, embedding
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return image_path, None

def get_all_images(directory):
    """Recursively finds all images in the directory and subdirectories"""
    all_images = []
    
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp')):
                all_images.append(os.path.join(root, file))
    
    return all_images

def calculate_embeddings(directory, model, transform):
    """Calculates embeddings for all images recursively in the specified directory"""
    images = get_all_images(directory)
    embeddings = {}

    print(f"Processing {len(images)} images in multi-threaded mode...")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        tasks = [(image_path, model, transform) for image_path in images]
        
        results = list(tqdm(
            executor.map(process_single_image, tasks),
            total=len(images),
            desc="Computing embeddings"
        ))
    
    for image_path, embedding in results:
        if embedding is not None:
            embeddings[image_path] = embedding
    
    print(f"Successfully processed {len(embeddings)} out of {len(images)} images")
    return embeddings

def find_similar_groups(embeddings, similarity_threshold):
    """Finds groups of similar images based on the given similarity threshold"""
    similarity_graph = {}
    for path in embeddings:
        similarity_graph[path] = []
    
    print("Comparing images and finding similar ones...")
    
    similar_count = 0
    total_comparisons = len(embeddings) * (len(embeddings) - 1) // 2
    
    comparison_pairs = []
    for i, (image_path1, embedding1) in enumerate(embeddings.items()):
        for j, (image_path2, embedding2) in enumerate(list(embeddings.items())[i+1:]):
            comparison_pairs.append((image_path1, embedding1, image_path2, embedding2))
    
    for image_path1, embedding1, image_path2, embedding2 in tqdm(
        comparison_pairs, 
        desc="Comparing images", 
        unit="comparisons"
    ):
        try:
            similarity = np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
            if similarity > similarity_threshold:
                similarity_graph[image_path1].append(image_path2)
                similarity_graph[image_path2].append(image_path1)
                similar_count += 1
        except Exception as e:
            print(f"Error comparing {image_path1} and {image_path2}: {e}")
    
    print(f"Total comparisons completed: {len(comparison_pairs)}, found similar pairs: {similar_count}")

    visited = set()
    groups = []
    
    def dfs(node, component):
        """Depth-first search to find connected components"""
        visited.add(node)
        component.append(node)
        for neighbor in similarity_graph[node]:
            if neighbor not in visited:
                dfs(neighbor, component)
    
    for node in similarity_graph:
        if node not in visited:
            component = []
            dfs(node, component)
            if len(component) > 1:  # Add group only if it has more than 1 image
                groups.append(component)

    groups.sort(key=len, reverse=True)
    
    return groups

def view_similar_groups(groups):
    """Displays groups of similar images in FileViewer"""
    from file_viewer import FileViewer
    
    print(f"Found {len(groups)} groups of similar images")
    
    for i, group in enumerate(groups):
        print(f"Group {i+1} of {len(groups)}: {len(group)} images")


        
        viewer = FileViewer(
            [image_path for image_path in group if os.path.exists(image_path)], 
            keep_one=False
        )
        remaining_files, stop_browsing = viewer.run()
        
        if hasattr(viewer, 'skip_batch') and viewer.skip_batch:
            print(f"Group {i+1} skipped by user")
            continue
            
        if stop_browsing:
            print("Browsing stopped by user")
            break
    
    print("Processing of all groups completed")

def select_model():
    """Allows user to select a model for image processing"""
    print("\nAvailable models for image processing:")
    for key, (name, _, _) in available_models.items():
        print(f"{key}. {name}")
    
    while True:
        choice = input("Select model number (or press Enter for ResNet50): ")
        if choice == "":
            return "1"  # Default value - ResNet50
        if choice in available_models:
            return choice
        print("Invalid selection. Please enter a number from the list.")

def interactive_mode(directory):
    """Interactive mode for finding similar images"""
    model_choice = select_model()
    model, transform = load_model(model_choice)
    
    print(f"Recursively scanning directory: {directory}")
    print("Computing image embeddings...")
    embeddings = calculate_embeddings(directory, model, transform)
    
    while True:
        user_input = input("\nEnter similarity threshold (0 to 1) or 'exit' to quit: ")
        
        if user_input.lower() == 'exit':
            print("Exiting program.")
            break
        
        try:
            threshold = float(user_input)
            if threshold <= 0 or threshold > 1:
                print("Threshold value must be between 0 and 1.")
                continue
                
            print(f"\nSearching for groups of similar images with threshold {threshold}...")
            groups = find_similar_groups(embeddings, threshold)
            
            if not groups:
                print("No similar image groups found with this threshold.")
                continue
                
            view_similar_groups(groups)
            
        except ValueError:
            print("Error: enter a numeric value between 0 and 1 or 'exit'.")

def get_directory():
    """Asks user for a directory to scan"""
    default_dir = './'
    print(f"Default directory: {default_dir}")
    
    user_input = input(f"Enter path to directory to scan: ")
    
    if user_input.strip():
        if os.path.isdir(user_input):
            return user_input
        else:
            print("The specified directory does not exist. Using default directory.")
            return default_dir
    return default_dir

if __name__ == "__main__":
    directory = get_directory()
    interactive_mode(directory)
