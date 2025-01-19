# import pickle
# import numpy as np
# import os
# from scipy.signal import resample
# from tqdm import tqdm
#
# from datasets.IIIC.process import patient_ids, train_mask
#
#
# # Function to process each sample
# def process_sample(root, file, sampling_rate):
#     sample = pickle.load(open(os.path.join(root, file), "rb"))
#     X = sample["signal"]
#     # Resample the signal
#     X = resample(X, 5 * sampling_rate, axis=-1)
#     # Normalize the signal
#     X = X / (
#         np.quantile(np.abs(X), q=0.95, method="linear", axis=-1, keepdims=True)
#         + 1e-8
#     )
#     # Get the label
#     Y = int(sample["label"][0] - 1)
#     return X, Y
#
# # Define paths
# processed_root = 'C:/Dataset/raw/tuh_eeg_events/v2.0.1/edf'
# processed_train_dir = os.path.join(processed_root, 'processed_train')
# processed_eval_dir = os.path.join(processed_root, 'processed_eval')
#
# # List all the files in the directory
# train_files = os.listdir(processed_train_dir)
# # Get the file names without the extension
# train_keys = [file.split('.')[0] for file in train_files]
#
# patient_ids = np.array([key.split('_')[0] for key in train_keys])
#
# unique_patient_ids = np.unique(patient_ids)
#
#
# # Sampling rate
# sampling_rate = 200
#
#
#
# # Initialize lists to store data
# train_X = []
# train_Y = []
#
# # Loop through files with a progress bar
# for file in tqdm(train_files, desc="Processing train files"):
#     X, Y = process_sample(processed_train_dir, file, sampling_rate)
#     train_X.append(X)
#     train_Y.append(Y)

import pickle
import numpy as np
import os
from scipy.signal import resample
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

# Function to process each sample
def process_sample_with_index(args):
    index, root, file, sampling_rate = args
    sample = pickle.load(open(os.path.join(root, file), "rb"))
    X = sample["signal"]
    # Resample the signal
    X = resample(X, 5 * sampling_rate, axis=-1)
    # Normalize the signal
    X = X / (
        np.quantile(np.abs(X), q=0.95, method="linear", axis=-1, keepdims=True)
        + 1e-8
    )
    # Get the label
    Y = int(sample["label"][0] - 1)
    return index, X, Y

if __name__ == '__main__':

    # Define paths
    processed_root = 'C:/Dataset/raw/tuh_eeg_events/v2.0.1/edf'
    processed_train_dir = os.path.join(processed_root, 'processed_train')

    # List all the files in the directory
    train_files = os.listdir(processed_train_dir)

    # Sampling rate
    sampling_rate = 200

    # Prepare arguments for parallel processing
    args = [(i, processed_train_dir, file, sampling_rate) for i, file in enumerate(train_files)]

    # Initialize lists to store data
    train_X = [None] * len(train_files)
    train_Y = [None] * len(train_files)

    # Process files with multiprocessing
    with ProcessPoolExecutor(max_workers=12) as executor:
        for index, X, Y in tqdm(executor.map(process_sample_with_index, args), total=len(train_files), desc="Processing train files"):
            train_X[index] = X
            train_Y[index] = Y

    # Verify results
    print(f"Processed {len(train_X)} samples.")

    train_keys = [file.split('.')[0] for file in train_files]
    patient_ids = np.array([key.split('_')[0] for key in train_keys])
    unique_patient_ids = np.unique(patient_ids)

    # Shuffle patient IDs
    np.random.seed(42)  # For reproducibility
    np.random.shuffle(unique_patient_ids)



    num_patients = len(unique_patient_ids)
    print('Number of unique patients:', num_patients)

    train_size = int(0.8 * num_patients)
    print('Train size:', train_size)

    train_ids = unique_patient_ids[:train_size]
    val_ids = unique_patient_ids[train_size:]

    train_mask = np.isin(patient_ids, train_ids)
    val_mask = np.isin(patient_ids, val_ids)

    x_train = np.array(train_X)[train_mask]
    y_train = np.array(train_Y)[train_mask]
    x_val = np.array(train_X)[val_mask]
    y_val = np.array(train_Y)[val_mask]

    # print the shapes
    print('Train set:', x_train.shape, y_train.shape)
    print('Validation set:', x_val.shape, y_val.shape)

    # Save the data
    # make a new folder called 'processed'
    dir_name = os.path.join(processed_root, 'processed')
    os.makedirs(dir_name, exist_ok=True)
    # save the data
    np.save(os.path.join(dir_name, 'train_X.npy'), x_train)
    np.save(os.path.join(dir_name, 'train_Y.npy'), y_train)
    np.save(os.path.join(dir_name, 'val_X.npy'), x_val)
    np.save(os.path.join(dir_name, 'val_Y.npy'), y_val)


    # do it for the test set

    processed_test_dir = os.path.join(processed_root, 'processed_eval')

    test_files = os.listdir(processed_test_dir)

    args = [(i, processed_test_dir, file, sampling_rate) for i, file in enumerate(test_files)]

    test_X = [None] * len(test_files)
    test_Y = [None] * len(test_files)

    with ProcessPoolExecutor(max_workers=12) as executor:
        for index, X, Y in tqdm(executor.map(process_sample_with_index, args), total=len(test_files), desc="Processing test files"):
            test_X[index] = X
            test_Y[index] = Y


    print(f"Processed {len(test_X)} samples.")

    test_X = np.array(test_X)
    test_Y = np.array(test_Y)

    print('Test set:', test_X.shape, test_Y.shape)

    # save the test data
    np.save(os.path.join(dir_name, 'test_X.npy'), test_X)
    np.save(os.path.join(dir_name, 'test_Y.npy'), test_Y)


