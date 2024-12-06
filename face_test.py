# import onnxruntime
# import numpy as np
# from PIL import Image

# # Preprocessing function for input images
# def preprocess_image(image_path):
#     # Open the image, convert to RGB, and resize to 112x112
#     image = Image.open(image_path).convert("RGB").resize((112, 112))
#     # Convert image to numpy array and normalize to [-1, 1]
#     img_np = np.array(image).astype(np.float32) / 127.5 - 1
#     # Change shape to (1, 3, 112, 112) to match ONNX model input
#     img_np = img_np.transpose(2, 0, 1)  # Convert to channel-first format
#     return np.expand_dims(img_np, axis=0)  # Add batch dimension

# # Load ONNX model
# def load_onnx_model(model_path):
#     # Load the ONNX model with GPU as the preferred execution provider
#     session = onnxruntime.InferenceSession(
#         model_path,
#         providers=['CUDAExecutionProvider', 'CPUExecutionProvider']  # CUDA prioritized over CPU
#     )

#     # Check the current execution provider
#     current_provider = session.get_providers()
#     print(f"Using Execution Provider: {current_provider}")

#     return session


# # Extract embeddings from the ONNX model
# def extract_embedding(model, image_tensor):
#     input_name = model.get_inputs()[0].name  # Get model input name
#     embedding = model.run(None, {input_name: image_tensor})[0]  # Run inference
#     return embedding

# # Compute cosine similarity
# # Compute cosine similarity
# def cosine_similarity(embedding1, embedding2):
#     # Flatten embeddings to 1D
#     embedding1 = embedding1.flatten()
#     embedding2 = embedding2.flatten()
#     # Normalize the embeddings
#     embedding1 = embedding1 / np.linalg.norm(embedding1)
#     embedding2 = embedding2 / np.linalg.norm(embedding2)
#     # Compute dot product
#     return np.dot(embedding1, embedding2)


# # Main function to compare two face images
# def compare_faces(onnx_model_path, image1_path, image2_path):
#     # Load the ONNX model
#     model = load_onnx_model(onnx_model_path)
    
#     # Preprocess the images
#     img1_tensor = preprocess_image(image1_path)
#     img2_tensor = preprocess_image(image2_path)
    
#     # Extract embeddings
#     embedding1 = extract_embedding(model, img1_tensor)
#     embedding2 = extract_embedding(model, img2_tensor)
    
#     # Compute cosine similarity
#     similarity = cosine_similarity(embedding1, embedding2)
#     print(f"Cosine Similarity: {similarity}")
    
#     # Threshold for similarity
#     threshold = 0.3  # Adjust based on your use case
#     if similarity > threshold:
#         print("The faces match (likely the same person).")
#     else:
#         print("The faces do not match (likely different persons).")

# # Example usage
# #onnx_model_path1 = "w600k_r50.onnx"  # Path to the ONNX model
# onnx_model_path = "glintr100.onnx"
# image1_path = "19.jpg"  # Replace with the path to the first face image
# image2_path = "20.jpg"  # Replace with the path to the second face image

# compare_faces(onnx_model_path, image1_path, image2_path)


import os
import shutil
import onnxruntime
import numpy as np
import pickle
from PIL import Image
from tqdm import tqdm

# Preprocessing function for input images
def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB").resize((112, 112))
    img_np = np.array(image).astype(np.float32) / 127.5 - 1
    img_np = img_np.transpose(2, 0, 1)  # Convert to channel-first format
    return np.expand_dims(img_np, axis=0)  # Add batch dimension

# Load ONNX model
def load_onnx_model(model_path):
    session = onnxruntime.InferenceSession(
        model_path,
        providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
    )
    print(f"Using Execution Provider: {session.get_providers()}")
    return session

# Extract embeddings from the ONNX model
def extract_embedding(model, image_tensor):
    input_name = model.get_inputs()[0].name
    embedding = model.run(None, {input_name: image_tensor})[0]
    return embedding.flatten()

# Precompute embeddings for all images in a folder
def precompute_embeddings(model, folder, save_path=None):
    embeddings = {}
    print(f"Precomputing embeddings for images in {folder}...")
    for image_file in tqdm(os.listdir(folder), desc=f"Processing {folder}"):
        image_path = os.path.join(folder, image_file)
        if os.path.isfile(image_path):
            img_tensor = preprocess_image(image_path)
            embeddings[image_file] = extract_embedding(model, img_tensor)
    
    # Save embeddings if a save path is provided
    if save_path:
        with open(save_path, 'wb') as f:
            pickle.dump(embeddings, f)
        print(f"Embeddings saved to {save_path}.")
    
    return embeddings

# Load embeddings from a file
def load_embeddings(file_path):
    with open(file_path, 'rb') as f:
        embeddings = pickle.load(f)
    print(f"Loaded embeddings from {file_path}.")
    return embeddings

# Compute cosine similarity
def cosine_similarity(embedding1, embedding2):
    embedding1 = embedding1 / np.linalg.norm(embedding1)
    embedding2 = embedding2 / np.linalg.norm(embedding2)
    return np.dot(embedding1, embedding2)

# Compare precomputed embeddings from folder_1 with those in folder_2
def find_top_matches(embeddings1, embeddings2, folder2, folder3, top_k=5, threshold=0.3):
    os.makedirs(folder3, exist_ok=True)
    match_dict = {}
    
    print("\nComparing embeddings...")
    for image1, embedding1 in tqdm(embeddings1.items(), desc="Comparing Folder 1 Embeddings"):
        matches = []
        for image2, embedding2 in embeddings2.items():
            similarity = cosine_similarity(embedding1, embedding2)
            if similarity > threshold:
                matches.append((image2, similarity))
        
        # Sort matches by similarity (highest first) and select top_k matches
        matches.sort(key=lambda x: x[1], reverse=True)
        top_matches = matches[:top_k] if matches else []

        match_dict[image1] = top_matches  # Store the top matches
        
        # Copy top matches to folder3
        for idx, (match_image, similarity) in enumerate(top_matches):
            src_path = os.path.join(folder2, match_image)
            dst_path = os.path.join(folder3, f"{os.path.splitext(image1)[0]}_match{idx + 1}_{match_image}")
            shutil.copy(src_path, dst_path)
            
            print(f"Match {idx + 1} for {image1}: {match_image} (Similarity: {similarity:.2f})")
    
    return match_dict

# Main function to process folders
def process_folders_top_matches(onnx_model_path, folder1, folder2, folder3, embeddings_file, top_k=5, threshold=0.3):
    model = load_onnx_model(onnx_model_path)
    
    # Precompute or load embeddings for folder2
    if os.path.exists(embeddings_file):
        embeddings2 = load_embeddings(embeddings_file)
    else:
        embeddings2 = precompute_embeddings(model, folder2, save_path=embeddings_file)
    
    # Precompute embeddings for folder1 (no need to save)
    embeddings1 = precompute_embeddings(model, folder1)
    
    # Find top matches
    match_dict = find_top_matches(embeddings1, embeddings2, folder2, folder3, top_k=top_k, threshold=threshold)
    
    print("\nMatching complete. Results saved in folder_3.")
    print("\nTop Match dictionary:")
    for key, value in match_dict.items():
        print(f"{key}: {value}")

# Example usage
onnx_model_path = "glintr100.onnx"  # Path to the ONNX model
folder1 = "New folder"  # Path to folder_1
folder2 = "Staff_Faces"  # Path to folder_2
folder3 = "folder_3"  # Path to save matches
embeddings_file = "embeddings_staffs.pkl"  # Path to save/load embeddings

process_folders_top_matches(onnx_model_path, folder1, folder2, folder3, embeddings_file, top_k=1, threshold=0.4)
