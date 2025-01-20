import os
import pickle
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

def process_sample_with_index(args):
    """Processes a single sample and returns its index, X, and Y."""
    index, processed_train_dir, file, sampling_rate = args
    file_path = os.path.join(processed_train_dir, file)

    # Load the file
    with open(file_path, "rb") as f:
        sample = pickle.load(f)

    # Example processing logic (adjust based on your requirements)
    X = sample["X"]
    Y = sample["y"]
    return index, X, Y

if __name__ == '__main__':
    # Define paths and parameters
    processed_root = 'C:/Dataset/raw/tuh_eeg_abnormal/v3.0.1/edf/processed'
    processed_train_dir = os.path.join(processed_root, 'train')
    sampling_rate = 100  # Example parameter, adjust as needed

    # List all the files in the directory
    train_files = os.listdir(processed_train_dir)

    # Prepare arguments for parallel processing
    args = [(i, processed_train_dir, file, sampling_rate) for i, file in enumerate(train_files)]

    # Initialize lists to store data
    train_X = [None] * len(train_files)
    train_Y = [None] * len(train_files)

    # Process files with multiprocessing
    with ProcessPoolExecutor(max_workers=6) as executor:
        # Use tqdm with executor.map for progress tracking
        for index, X, Y in tqdm(executor.map(process_sample_with_index, args), total=len(train_files), desc="Processing train files"):
            train_X[index] = X
            train_Y[index] = Y

    print("Processing complete!")
