import os
import cv2
import logging
import numpy as np
import pickle
import tkinter as tk
from tkinter import filedialog
from PIL import Image
import onnxruntime
from tqdm import tqdm
import csv
from ultralight import UltraLightDetector

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Face Detector Initialization
detector = UltraLightDetector()

# Face Cropping
def enlarge_box(box, image, scale=2):
    """Enlarge bounding box by a scale factor."""
    x1, y1, x2, y2 = box
    width = x2 - x1
    height = y2 - y1
    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2

    new_width = int(width * scale)
    new_height = int(height * scale)

    # Calculate new coordinates
    new_x1 = max(center_x - new_width // 2, 0)
    new_y1 = max(center_y - new_height // 2, 0)
    new_x2 = min(center_x + new_width // 2, image.shape[1])
    new_y2 = min(center_y + new_height // 2, image.shape[0])

    return (new_x1, new_y1, new_x2, new_y2)

def detect_and_crop_faces(image_path, scale=1.5, min_confidence=0.5):
    """Detect and crop a face from the image."""
    image = cv2.imread(image_path)
    if image is None:
        logging.error(f"Error: Could not read the image {image_path}.")
        return None

    boxes, scores = detector.detect_one(image)
    logging.info(f'Found {len(boxes)} face(s) in {image_path}')

    # Filter boxes and scores based on confidence
    valid_faces = [(box, score) for box, score in zip(boxes, scores) if score >= min_confidence]

    if not valid_faces:
        logging.warning(f"No faces with confidence >= {min_confidence} found in {image_path}.")
        return None

    # Select the face with the highest confidence
    best_face = max(valid_faces, key=lambda item: item[1])
    box, _ = best_face

    # Enlarge the bounding box
    enlarged_box = enlarge_box(box, image, scale=scale)
    x1, y1, x2, y2 = enlarged_box
    face = image[y1:y2, x1:x2]

    # Skip if face size is too small
    if face.size == 0 or face.shape[0] < 10 or face.shape[1] < 10:
        logging.warning(f'Skipping small face detected in {image_path}.')
        return None

    return face

# Preprocessing for Embedding Generation
def preprocess_image(image):
    """Preprocess cropped face for embedding generation."""
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    image = Image.fromarray(image).resize((112, 112))  # Resize to 112x112
    img_np = np.array(image).astype(np.float32) / 127.5 - 1
    img_np = img_np.transpose(2, 0, 1)  # Convert to channel-first format
    return np.expand_dims(img_np, axis=0)  # Add batch dimension

# Load ONNX Model
def load_onnx_model(model_path):
    session = onnxruntime.InferenceSession(
        model_path,
        providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
    )
    print(f"Using Execution Provider: {session.get_providers()}")
    return session

# Generate or Update Embeddings
def generate_or_update_embeddings(model, input_folder, output_file, log_file):
    if os.path.exists(output_file):
        with open(output_file, 'rb') as f:
            embeddings = pickle.load(f)
        print(f"Loaded existing embeddings from {output_file}.")
    else:
        embeddings = {}

    existing_files = set()
    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            reader = csv.reader(f)
            existing_files = {row[0] for row in reader}

    updated_count = 0
    print(f"Processing images in {input_folder}...")

    for image_file in tqdm(os.listdir(input_folder), desc="Processing images"):
        if image_file in existing_files:  # Skip images that already have embeddings
            continue

        image_path = os.path.join(input_folder, image_file)
        cropped_face = detect_and_crop_faces(image_path)
        if cropped_face is None:
            continue

        # Generate embedding
        img_tensor = preprocess_image(cropped_face)
        embeddings[image_file] = model.run(
            None, {model.get_inputs()[0].name: img_tensor}
        )[0].flatten()
        updated_count += 1

    with open(output_file, 'wb') as f:
        pickle.dump(embeddings, f)

    with open(log_file, 'a', newline='') as f:
        writer = csv.writer(f)
        for filename in embeddings.keys():
            if filename not in existing_files:
                writer.writerow([filename])

    print(f"Updated {updated_count} embeddings. Total embeddings saved to {output_file}.")

# Main Function
def main():
    onnx_model_path = "glintr100.onnx"
    output_file = "image_embeddings.pkl"
    log_file = "processed_files_log.csv"

    root = tk.Tk()
    root.withdraw()
    input_folder = filedialog.askdirectory(title="Select Input Folder")
    if not input_folder:
        print("No folder selected. Exiting.")
        return

    model = load_onnx_model(onnx_model_path)
    generate_or_update_embeddings(model, input_folder, output_file, log_file)

if __name__ == "__main__":
    main()
