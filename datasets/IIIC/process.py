import os
import time
import numpy as np


root = 'C:/Dataset/raw/IIIC'

# use the all dataset for data split. The split will be based on the patient_id, following the
# since the authors did not provide the test set, we will split the data into train, val, and test set.
# Note: the 6 y labels are ['

x = np.load(os.path.join(root, 'all_train_X.npy'))
y = np.load(os.path.join(root, 'all_train_Y2_hard.npy'))
keys = np.load(os.path.join(root, 'all_train_key.npy'))

# preprocess the x in a way for each x,
percentiles = np.quantile(np.abs(x), q=0.95, axis=-1, method="linear", keepdims=True)
x = x / (percentiles + 1e-6)

# y was in one hot encoding, convert it to integer
y = np.argmax(y, axis=1)

# print their shape:
print('x shape:', x.shape)
print('y shape:', y.shape)
print('keys shape:', keys.shape)

# Extract unique patient IDs
patient_ids = np.array([key.split('_')[0] for key in keys])
unique_patient_ids = np.unique(patient_ids)

# Shuffle patient IDs
np.random.seed(42)  # For reproducibility
np.random.shuffle(unique_patient_ids)

# Calculate the split sizes
num_patients = len(unique_patient_ids)
train_size = int(0.6 * num_patients)
val_size = int(0.2 * num_patients)

# Split patient IDs
train_ids = unique_patient_ids[:train_size]
val_ids = unique_patient_ids[train_size:train_size + val_size]
test_ids = unique_patient_ids[train_size + val_size:]

# Create masks for each split
train_mask = np.isin(patient_ids, train_ids)
val_mask = np.isin(patient_ids, val_ids)
test_mask = np.isin(patient_ids, test_ids)

# Split the data
x_train, y_train = x[train_mask], y[train_mask]
x_val, y_val = x[val_mask], y[val_mask]
x_test, y_test = x[test_mask], y[test_mask]


# # Step 1: Count occurrences of each class
# unique_classes, class_counts = np.unique(y_train, return_counts=True)
#
# # Step 2: Calculate class weights (inverse of class frequencies)
# class_weights = 1.0 / class_counts
#
# # Step 3: Normalize the weights (optional, but recommended for stability)
# class_weights = class_weights / class_weights.sum()



# Print the shapes
print('Train set:', x_train.shape, y_train.shape)
print('Validation set:', x_val.shape, y_val.shape)
print('Test set:', x_test.shape, y_test.shape)


# save the data into the processed folder
processed_root = os.path.join(root, 'processed')



# make a new directory
os.makedirs(processed_root, exist_ok=True)
# save the data
np.save(os.path.join(processed_root, 'train_X.npy'), x_train)
np.save(os.path.join(processed_root, 'train_Y.npy'), y_train)
np.save(os.path.join(processed_root, 'val_X.npy'), x_val)
np.save(os.path.join(processed_root, 'val_Y.npy'), y_val)
np.save(os.path.join(processed_root, 'test_X.npy'), x_test)
np.save(os.path.join(processed_root, 'test_Y.npy'), y_test)


