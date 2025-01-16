


import os
import pickle
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed


dataset_root = "C:/Dataset/raw/tuh_eeg_abnormal/v3.0.1/edf/processed/"


# Function to load and process a single file
def process_file(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
        X = data["X"]  # Shape: (samples, 16, 2000)
        Y = data["y"]  # List of labels
    return X, Y

# Function to combine files into separate X and y pickle files using multi-threading
def combine_x_y_pickle_files(input_folder, output_x_file, output_y_file, max_workers=8):
    combined_X = []
    combined_Y = []

    # List all files in the folder
    files = os.listdir(input_folder)
    file_paths = [os.path.join(input_folder, file) for file in files]

    # Use ThreadPoolExecutor to load and process files concurrently
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {executor.submit(process_file, file_path): file_path for file_path in file_paths}
        for future in tqdm(as_completed(future_to_file), desc=f"Processing {input_folder}", total=len(file_paths)):
            try:
                X, Y = future.result()
                combined_X.append(X)  # Append each sample
                combined_Y.append(Y)  # Append each label
            except Exception as e:
                print(f"Error processing file {future_to_file[future]}: {e}")

    # Convert combined_X to a numpy array
    combined_X = np.array(combined_X)
    combined_Y = np.array(combined_Y)
    # Save the combined X and Y to separate pickle files
    with open(output_x_file, 'wb') as f:
        pickle.dump(combined_X, f)

    with open(output_y_file, 'wb') as f:
        pickle.dump(combined_Y, f)

# Paths to train, val, and test folders
train_folder = os.path.join(dataset_root, "train")
val_folder = os.path.join(dataset_root, "val")
test_folder = os.path.join(dataset_root, "test")

# Output pickle files
train_x_file = os.path.join(dataset_root, "train_x.pkl")
train_y_file = os.path.join(dataset_root, "train_y.pkl")
val_x_file = os.path.join(dataset_root, "val_x.pkl")
val_y_file = os.path.join(dataset_root, "val_y.pkl")
test_x_file = os.path.join(dataset_root, "test_x.pkl")
test_y_file = os.path.join(dataset_root, "test_y.pkl")

# Combine the files into X and Y pickle files
combine_x_y_pickle_files(train_folder, train_x_file, train_y_file, max_workers=8)
combine_x_y_pickle_files(val_folder, val_x_file, val_y_file, max_workers=8)
combine_x_y_pickle_files(test_folder, test_x_file, test_y_file, max_workers=8)

print("Pickle files for X and Y created successfully!")

