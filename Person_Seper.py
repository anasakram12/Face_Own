import os
import face_recognition
import cv2
import numpy as np
from typing import List, Tuple
import shutil
from datetime import datetime

# GPU Acceleration Imports
import torch
import dlib

class FaceProcessor:
    def __init__(self, input_folder: str, output_folder: str, similarity_threshold: float = 0.6, use_gpu: bool = True):
        """
        Initialize the FaceProcessor with optional GPU acceleration.
        
        Args:
            input_folder (str): Path to folder containing input face images
            output_folder (str): Path to store processed results
            similarity_threshold (float): Threshold for face similarity
            use_gpu (bool): Whether to use GPU acceleration
        """
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.similarity_threshold = similarity_threshold
        self.use_gpu = use_gpu and torch.cuda.is_available()
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        
        # Configure GPU processing
        if self.use_gpu:
            print(f"GPU Acceleration Enabled: {torch.cuda.get_device_name(0)}")
            # Configure dlib to use CUDA
            dlib.DLIB_USE_CUDA = True
        else:
            print("GPU Acceleration Not Available. Falling back to CPU.")

    def process_faces(self) -> List[List[str]]:
        """
        Process faces and group similar faces together with optional GPU acceleration.
        """
        # Get all image files
        image_files = [
            os.path.join(self.input_folder, f) for f in os.listdir(self.input_folder)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]
        
        if not image_files:
            print("No images found in the input folder!")
            return []
        
        # Initialize variables
        current_group = []
        all_groups = []
        current_encoding = None
        
        print("\nProcessing images...")
        for i, image_path in enumerate(image_files, 1):
            try:
                print(f"Processing image {i}/{len(image_files)}: {os.path.basename(image_path)}")
                
                # Load image with optional GPU acceleration
                if self.use_gpu:
                    # Use GPU-enabled image loading if possible
                    import torch
                    import torchvision.transforms as transforms
                    from PIL import Image
                    
                    # Load image using PyTorch
                    transform = transforms.ToTensor()
                    pil_image = Image.open(image_path)
                    image = np.array(pil_image)
                else:
                    # Standard face_recognition image loading
                    image = face_recognition.load_image_file(image_path)
                
                # Detect face encodings
                face_encodings = face_recognition.face_encodings(image)
                
                if not face_encodings:
                    print(f"Warning: No face detected in {os.path.basename(image_path)}")
                    continue
                
                new_encoding = face_encodings[0]
                
                # If this is the first face, start new group
                if current_encoding is None:
                    current_encoding = new_encoding
                    current_group = [image_path]
                    continue
                
                # Compare with current encoding
                # Use GPU-accelerated computation if available
                if self.use_gpu and torch.cuda.is_available():
                    current_encoding_tensor = torch.tensor(current_encoding)
                    new_encoding_tensor = torch.tensor(new_encoding)
                    
                    # Compute Euclidean distance on GPU
                    distance = torch.norm(current_encoding_tensor - new_encoding_tensor).item()
                else:
                    # Fallback to standard face_recognition distance calculation
                    distance = face_recognition.face_distance([current_encoding], new_encoding)[0]
                
                if distance <= self.similarity_threshold:
                    # Similar face found, add to current group
                    current_group.append(image_path)
                else:
                    # Different face found
                    if current_group:
                        all_groups.append(current_group)
                    # Start new group with current face
                    current_group = [image_path]
                    current_encoding = new_encoding
                    
            except Exception as e:
                print(f"Error processing {os.path.basename(image_path)}: {str(e)}")
                continue
        
        # Add the last group if it exists
        if current_group:
            all_groups.append(current_group)
        
        return all_groups

    def organize_results(self, groups: List[List[str]]):
        """
        Organize results by copying images to separate folders for each person.
        """
        # Create timestamp for unique folder names
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        print("\nOrganizing results...")
        for i, group in enumerate(groups, 1):
            # Create person directory
            person_dir = os.path.join(self.output_folder, f"person_{i}_{timestamp}")
            os.makedirs(person_dir, exist_ok=True)
            
            # Copy images to person directory
            for image_path in group:
                filename = os.path.basename(image_path)
                destination = os.path.join(person_dir, filename)
                shutil.copy2(image_path, destination)
            
            print(f"Person {i}: {len(group)} images -> {person_dir}")

    def display_results(self, groups: List[List[str]]):
        """
        Display the grouped results in a readable format.
        """
        print("\nFace Groups Found:")
        for i, group in enumerate(groups, 1):
            print(f"\nPerson {i} - {len(group)} images:")
            for image_path in group:
                print(f"  - {os.path.basename(image_path)}")

def main():
    # Configuration
    INPUT_FOLDER = "test"    # Put your input images here
    OUTPUT_FOLDER = "output_faces"   # Results will be saved here
    SIMILARITY_THRESHOLD = 0.6       # Adjust this value to control sensitivity
    USE_GPU = True                   # Enable/disable GPU acceleration
    
    # Create processor instance
    processor = FaceProcessor(
        input_folder=INPUT_FOLDER,
        output_folder=OUTPUT_FOLDER,
        similarity_threshold=SIMILARITY_THRESHOLD,
        use_gpu=USE_GPU
    )
    
    # Process faces
    print(f"Starting face processing...")
    print(f"Input folder: {INPUT_FOLDER}")
    print(f"Output folder: {OUTPUT_FOLDER}")
    print(f"Similarity threshold: {SIMILARITY_THRESHOLD}")
    
    # Process and group faces
    groups = processor.process_faces()
    
    if not groups:
        print("No face groups were created. Please check your input images.")
        return
    
    # Display results
    processor.display_results(groups)
    
    # Organize results into folders
    processor.organize_results(groups)
    
    print("\nProcessing complete!")
    print(f"Results have been saved to: {OUTPUT_FOLDER}")

if __name__ == "__main__":
    main()