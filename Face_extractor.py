import os
import cv2
import logging
from ultralight import UltraLightDetector

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Directory setup
input_folder = "D:\Dataset\Guest"  # Folder with input images
output_folder = "Guest_Faces"  # Folder to save cropped faces

# Create output directory if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Initialize UltraLightDetector
detector = UltraLightDetector()

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

def process_image(image_path):
    """Detect, enlarge, crop, and save faces from an image."""
    image = cv2.imread(image_path)
    if image is None:
        logging.error(f"Error: Could not read the image {image_path}.")
        return

    boxes, scores = detector.detect_one(image)
    logging.info(f'Found {len(boxes)} face(s) in {image_path}')

    MIN_CONFIDENCE = 0.5  # Threshold for face confidence
    filename = os.path.splitext(os.path.basename(image_path))[0]

    for i, (box, score) in enumerate(zip(boxes, scores)):
        if score < MIN_CONFIDENCE:
            continue

        # Enlarge the bounding box
        enlarged_box = enlarge_box(box, image, scale=1.5)
        x1, y1, x2, y2 = enlarged_box
        face = image[y1:y2, x1:x2]

        # Skip if face size is too small
        if face.size == 0 or face.shape[0] < 10 or face.shape[1] < 10:
            logging.warning(f'Skipping small face detected in {image_path}.')
            continue

        # Save the cropped face to the output folder
        face_filename = f"{filename}_face_{i+1}.png"
        face_path = os.path.join(output_folder, face_filename)
        cv2.imwrite(face_path, face)
        logging.info(f"Saved cropped face to {face_path}")

# Process all images in the input folder
for image_file in os.listdir(input_folder):
    image_path = os.path.join(input_folder, image_file)
    process_image(image_path)

logging.info("Face detection and saving completed.")
