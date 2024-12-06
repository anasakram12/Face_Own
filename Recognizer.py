import os
import time
import pickle
import csv
import threading
import numpy as np
from PIL import Image, ImageTk
import onnxruntime
from tkinter import Tk, Label, Button, filedialog, Toplevel
from tkinter.scrolledtext import ScrolledText
from tqdm import tqdm
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler


# Preprocess images for the ONNX model
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


# Extract embedding for a single image
def extract_embedding(model, image_tensor):
    input_name = model.get_inputs()[0].name
    embedding = model.run(None, {input_name: image_tensor})[0]
    return embedding.flatten()


# Compute cosine similarity between two embeddings
def cosine_similarity(embedding1, embedding2):
    embedding1 = embedding1 / np.linalg.norm(embedding1)
    embedding2 = embedding2 / np.linalg.norm(embedding2)
    return np.dot(embedding1, embedding2)


# Read last update timestamp from CSV
def read_last_update(csv_file):
    if os.path.exists(csv_file):
        with open(csv_file, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if row and row[0] == "last_update":
                    return float(row[1])
    return 0  # Default: process all files initially


# Write last update timestamp to CSV
def write_last_update(csv_file, timestamp):
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["last_update", timestamp])


# Update embeddings using timestamp-based method
def update_embeddings_with_timestamp(model, database_folder, embeddings_file, timestamp_csv="last_update.csv"):
    # Load existing embeddings if the file exists
    if os.path.exists(embeddings_file):
        with open(embeddings_file, 'rb') as f:
            embeddings = pickle.load(f)
        print(f"Loaded existing embeddings from {embeddings_file}.")
    else:
        embeddings = {}

    # Read last update timestamp
    last_update = read_last_update(timestamp_csv)
    updated_count = 0
    current_time = time.time()

    # Process new or modified images
    for image_file in tqdm(os.listdir(database_folder), desc="Updating embeddings"):
        image_path = os.path.join(database_folder, image_file)
        if not os.path.isfile(image_path):
            continue

        file_mod_time = os.path.getmtime(image_path)  # Get file modification time
        if file_mod_time <= last_update:
            continue  # Skip files that were not modified since the last update

        # Preprocess and generate embedding
        img_tensor = preprocess_image(image_path)
        embeddings[image_file] = extract_embedding(model, img_tensor)
        updated_count += 1

    # Save updated embeddings to file
    with open(embeddings_file, 'wb') as f:
        pickle.dump(embeddings, f)

    # Write the current timestamp to the CSV file
    write_last_update(timestamp_csv, current_time)

    print(f"Added {updated_count} new embeddings to {embeddings_file}.")
    return embeddings


# Recognize a new image and display results
def recognize_and_display(image_path, embeddings_file, model, threshold=0.5, top_k=5):
    # Load existing embeddings
    if not os.path.exists(embeddings_file):
        print(f"Embeddings file {embeddings_file} not found. Please generate it first.")
        return

    with open(embeddings_file, 'rb') as f:
        embeddings = pickle.load(f)

    # Preprocess and extract embedding for the new image
    img_tensor = preprocess_image(image_path)
    input_embedding = extract_embedding(model, img_tensor)

    # Compare with database embeddings
    similarities = []
    for db_image, db_embedding in embeddings.items():
        similarity = cosine_similarity(input_embedding, db_embedding)
        if similarity >= threshold:
            similarities.append((db_image, similarity))

    # Sort matches by similarity and get top_k
    similarities.sort(key=lambda x: x[1], reverse=True)
    top_matches = similarities[:top_k]

    # Display results
    display_results(image_path, top_matches)


# Display the selected image and results in a new window
def display_results(image_path, matches):
    # Create a new window
    result_window = Toplevel()
    result_window.title("Recognition Results")
    result_window.geometry("600x600")

    # Display the new image
    img = Image.open(image_path)
    img.thumbnail((300, 300))
    img = ImageTk.PhotoImage(img)
    img_label = Label(result_window, image=img)
    img_label.image = img  # Keep a reference to avoid garbage collection
    img_label.pack(pady=10)

    # Display the top matches
    result_text = ScrolledText(result_window, wrap="word", width=70, height=20)
    result_text.pack(pady=10)
    result_text.insert("1.0", f"Recognition Results for {os.path.basename(image_path)}:\n\n")
    for rank, (match_image, similarity) in enumerate(matches, start=1):
        result_text.insert("end", f"{rank}. {match_image} (Similarity: {similarity:.2f})\n")
    result_text.config(state="disabled")


# Watchdog event handler for input folder monitoring
class InputFolderEventHandler(FileSystemEventHandler):
    def __init__(self, model, embeddings_file):
        self.model = model
        self.embeddings_file = embeddings_file

    def on_created(self, event):
        if not event.is_directory and event.src_path.lower().endswith(('.jpg', '.jpeg', '.png')):
            print(f"New image detected: {event.src_path}")
            recognize_and_display(event.src_path, self.embeddings_file, self.model)


# Start monitoring the input folder
def start_monitoring(model, input_folder, embeddings_file):
    event_handler = InputFolderEventHandler(model, embeddings_file)
    observer = Observer()
    observer.schedule(event_handler, path=input_folder, recursive=False)
    observer.start()
    print(f"Started monitoring {input_folder} for new images.")
    return observer


# Main function with GUI
def main():
    # Load the ONNX model
    onnx_model_path = "Sources/glintr100.onnx"  # Path to the ONNX model
    embeddings_file = "Sources/image_embeddings.pkl"  # Path to embeddings file
    database_folder = "Sources/Database_Images"  # Folder containing database images
    timestamp_csv = "Sources/last_update.csv"  # CSV file to store the last update timestamp
    os.makedirs(database_folder, exist_ok=True)  # Ensure database folder exists
    model = load_onnx_model(onnx_model_path)

    # Create the main window
    root = Tk()
    root.title("Image Recognition")
    root.geometry("400x250")

    # Variable to store selected input folder
    input_folder_var = {"path": None}

    # Function to select input folder
    def select_input_folder():
        folder_selected = filedialog.askdirectory(title="Select Input Folder")
        if folder_selected:
            input_folder_var["path"] = folder_selected
            Label(root, text=f"Monitoring Folder: {folder_selected}", font=("Arial", 10)).pack(pady=10)
            # Start monitoring the selected folder
            start_monitoring(model, folder_selected, embeddings_file)

    # Add a label
    Label(root, text="Select an input folder to monitor for new images:", font=("Arial", 12)).pack(pady=20)

    # Add a button to select input folder
    Button(
        root,
        text="Select Folder",
        font=("Arial", 12),
        command=select_input_folder
    ).pack(pady=10)

    # Run the GUI
    root.mainloop()


if __name__ == "__main__":
    main()
