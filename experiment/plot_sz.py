# import numpy as np
# import matplotlib.pyplot as plt
# import os
#
# dataset_root = "C:/Dataset/raw/IIIC/processed/"
#
# x = np.load(os.path.join(dataset_root, "train_X.npy"))
# y = np.load(os.path.join(dataset_root, "train_Y.npy"))
#
# print(x.shape)
# print(y.shape)
#
# # select x where y is 1
# x = x[y == 1]
# eeg_data = x[7000]#[:,500:1000]
# n, m= eeg_data.shape
# time = np.arange(m)  # Adjust if you have a specific sampling rate
#
#
#
# # Create a high-resolution plot
# plt.figure(figsize=(15, 8), dpi=700)  # Increased figure size and resolution
#
# offset = 5  # Offset for better visualization
# y_ticks = np.arange(-offset, n * offset, 5)  # Define y-axis tick positions
# x_ticks = np.arange(0, m, 50)  # Define x-axis tick positions
#
# for i in range(n):
#     plt.plot(time, eeg_data[i] + i * offset, label=f'Channel {i+1}', linewidth=0.8)  # Thinner lines for clarity
#
# plt.xlabel('Time', fontsize=14)
# plt.ylabel('Amplitude', fontsize=14)
# plt.title('EEG Signal', fontsize=16)
# plt.xticks(x_ticks)  # Set x-axis tick positions
# plt.yticks(y_ticks)  # Set y-axis tick positions
# plt.grid(True, linestyle='--', linewidth=0.5)
#
# plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1.0), fontsize=10)
# plt.show()
#
#





import numpy as np
import matplotlib.pyplot as plt
import os

dataset_root = "C:/Dataset/raw/IIIC/"

x = np.load(os.path.join(dataset_root, "10_train_X.npy"))
y = np.load(os.path.join(dataset_root, "10_train_Y2_hard.npy"))
y_vote = np.load(os.path.join(dataset_root, "10_train_Y.npy"))
y = np.argmax(y, axis=1)
print(x.shape)
print(y.shape)

# select x where y is 1
# x = x[y == 1]

index = 1076
print(y[index])
print(y_vote[index])

eeg_data = x[index]#[:,500:1000]
n, m= eeg_data.shape
time = np.arange(m)  # Adjust if you have a specific sampling rate



# Create a high-resolution plot
plt.figure(figsize=(15, 8), dpi=700)  # Increased figure size and resolution

offset = 200  # Offset for better visualization
y_ticks = np.arange(-offset, n * offset, 200)  # Define y-axis tick positions
x_ticks = np.arange(0, m, 50)  # Define x-axis tick positions

for i in range(n):
    plt.plot(time, eeg_data[i] + i * offset, label=f'Channel {i+1}', linewidth=0.8)  # Thinner lines for clarity

plt.xlabel('Time', fontsize=14)
plt.ylabel('Amplitude', fontsize=14)
plt.title('EEG Signal', fontsize=16)
plt.xticks(x_ticks)  # Set x-axis tick positions
plt.yticks(y_ticks)  # Set y-axis tick positions
plt.grid(True, linestyle='--', linewidth=0.5)

plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1.0), fontsize=10)
plt.show()







