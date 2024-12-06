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


# Periodic update function
def periodic_update(model, database_folder, embeddings_file, timestamp_csv, interval=600):
    while True:
        print("Running periodic embedding update...")
        update_embeddings_with_timestamp(model, database_folder, embeddings_file, timestamp_csv)
        print(f"Next update in {interval // 60} minutes.")
        time.sleep(interval)  # Wait for the specified interval


# Compute cosine similarity between two embeddings
def cosine_similarity(embedding1, embedding2):
    embedding1 = embedding1 / np.linalg.norm(embedding1)
    embedding2 = embedding2 / np.linalg.norm(embedding2)
    return np.dot(embedding1, embedding2)


# Match selected image to database embeddings and find top 5 matches
def match_image(image_path, embeddings, model, threshold=0.5, top_k=5):
    # Preprocess and extract embedding for the selected image
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

    return top_matches


# Display the selected image and results in a new window
def display_results(image_path, matches):
    # Create a new window
    result_window = Toplevel()
    result_window.title("Matching Results")
    result_window.geometry("600x600")

    # Display the selected image
    img = Image.open(image_path)
    img.thumbnail((300, 300))
    img = ImageTk.PhotoImage(img)
    img_label = Label(result_window, image=img)
    img_label.image = img  # Keep a reference to avoid garbage collection
    img_label.pack(pady=10)

    # Display the top matches
    result_text = ScrolledText(result_window, wrap="word", width=70, height=20)
    result_text.pack(pady=10)
    result_text.insert("1.0", f"Top matches for {os.path.basename(image_path)}:\n\n")
    for rank, (match_image, similarity) in enumerate(matches, start=1):
        result_text.insert("end", f"{rank}. {match_image} (Similarity: {similarity:.2f})\n")
    result_text.config(state="disabled")


# Select an image and find its matches
def select_and_match_image(model, embeddings_file, database_folder, timestamp_csv):
    # Update embeddings with new images in the database folder
    embeddings = update_embeddings_with_timestamp(model, database_folder, embeddings_file, timestamp_csv)

    # Open file dialog to select an image
    file_path = filedialog.askopenfilename(
        title="Select an Image",
        filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")]
    )
    if not file_path:
        return

    # Match the selected image
    matches = match_image(file_path, embeddings, model)
    if matches:
        display_results(file_path, matches)
    else:
        print("No matches found.")


# Main function with GUI
def main():
    # Load the ONNX model
    onnx_model_path = "glintr100.onnx"  # Path to the ONNX model
    embeddings_file = "image_embeddings.pkl"  # Existing embeddings file
    database_folder = "Database_Images"  # Folder containing images for the database
    timestamp_csv = "last_update.csv"  # CSV file to store the last update timestamp
    interval = 600  # Time interval in seconds (10 minutes)
    model = load_onnx_model(onnx_model_path)

    # Start periodic updates in a separate thread
    update_thread = threading.Thread(
        target=periodic_update,
        args=(model, database_folder, embeddings_file, timestamp_csv, interval),
        daemon=True
    )
    update_thread.start()

    # Create the main window
    root = Tk()
    root.title("Image Matching")
    root.geometry("400x200")

    # Add a label
    Label(root, text="Select an image to find matches", font=("Arial", 14)).pack(pady=20)

    # Add a button to select an image
    Button(
        root,
        text="Select Image",
        font=("Arial", 12),
        command=lambda: select_and_match_image(model, embeddings_file, database_folder, timestamp_csv)
    ).pack(pady=20)

    # Run the GUI
    root.mainloop()


if __name__ == "__main__":
    main()
