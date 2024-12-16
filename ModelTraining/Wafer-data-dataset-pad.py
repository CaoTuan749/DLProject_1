import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import RandomOverSampler
import matplotlib.pyplot as plt
import random

# Path to .pkl file
file_path = 'D:/Waffer Data/WM811K.pkl'

# Load the dataset
df = pd.read_pickle(file_path)

# Function to replace [0, 0] with 'Unknown'
def replace_zero_zero(x):
    if isinstance(x, (list, np.ndarray)) and np.array_equal(x, [0, 0]):
        return 'Unknown'
    return x

# Create a copy of the DataFrame
df_modified = df.copy()

# Apply the function to 'failureType' and 'trainTestLabel' columns
df_modified['failureType'] = df_modified['failureType'].apply(replace_zero_zero)
df_modified['trainTestLabel'] = df_modified['trainTestLabel'].apply(replace_zero_zero)

# Remove entries with 'No pattern' or 'Unknown'
df_modified_labelled = df_modified[
    ~df_modified['failureType'].isin(['none', 'Unknown'])
].reset_index(drop=True)

# Find the maximum dimensions across all wafer maps
def find_dim(x):
    return x.shape

# Apply the function to get dimensions
dimensions = df_modified_labelled['waferMap'].apply(find_dim)

# Get maximum height and width
max_height = max([dim[0] for dim in dimensions])
max_width = max([dim[1] for dim in dimensions])
max_dim = (max_height, max_width)

print(f"Maximum dimensions: Height={max_height}, Width={max_width}")

# Function to center wafer map in a box of maximum dimensions
def pad_wafer_map(wmap, max_dim):
    height, width = wmap.shape
    max_height, max_width = max_dim
   
    # Calculate padding sizes
    pad_height_top = (max_height - height) // 2
    pad_height_bottom = max_height - height - pad_height_top
    pad_width_left = (max_width - width) // 2
    pad_width_right = max_width - width - pad_width_left
   
    # Pad the wafer map with zeros
    padded_wmap = np.pad(
        wmap,
        ((pad_height_top, pad_height_bottom), (pad_width_left, pad_width_right)),
        mode='constant',
        constant_values=0
    )
    return padded_wmap

# Apply padding to wafer maps
df_modified_labelled['waferMap_padded'] = df_modified_labelled['waferMap'].apply(lambda x: pad_wafer_map(x, max_dim))

# Flatten the wafer maps
df_modified_labelled['waferMap_flat'] = df_modified_labelled['waferMap_padded'].apply(lambda x: x.flatten())

# Reset index to ensure alignment
df_modified_labelled.reset_index(drop=True, inplace=True)

# Use 'trainTestLabel' column to split the data
# Split 'trainTestLabel' values base on 'Training' and 'Test' columnns
df_train = df_modified_labelled[df_modified_labelled['trainTestLabel'] == 'Training'].reset_index(drop=True)
df_test = df_modified_labelled[df_modified_labelled['trainTestLabel'] == 'Test'].reset_index(drop=True)

# Prepare data for oversampling on the training set
X_train = np.stack(df_train['waferMap_flat'].values).astype('float32')
y_train = df_train['failureType'].values

# Oversample the training data to balance the dataset
ros = RandomOverSampler(random_state=42)
X_train_resampled, y_train_resampled = ros.fit_resample(X_train, y_train)

# Prepare the test data without oversampling
X_test = np.stack(df_test['waferMap_flat'].values).astype('float32')
y_test = df_test['failureType'].values

# Encode labels using LabelEncoder fitted on training labels
encoder = LabelEncoder()
encoder.fit(y_train_resampled)  # Fit on training labels

y_train_enc = encoder.transform(y_train_resampled)
y_test_enc = encoder.transform(y_test)  # Apply the same encoder to test labels

num_classes = len(encoder.classes_)

# Create DataFrames with resampled training data and processed test data
df_train_resampled = pd.DataFrame({
    'waferMap_flat': list(X_train_resampled),
    'failureType_enc': y_train_enc
})

df_test_processed = pd.DataFrame({
    'waferMap_flat': list(X_test),
    'failureType_enc': y_test_enc
})

# Custom Dataset class
class WaferMapDataset(Dataset):
    def __init__(self, df, max_dim):
        self.X = np.stack(df['waferMap_flat'].values).astype('float32')
        self.y = df['failureType_enc'].values.astype('int64')
        self.num_samples = len(df)
        self.image_size = max_dim  # (max_height, max_width)
       
    def __len__(self):
        return self.num_samples
       
    def __getitem__(self, idx):
        # Get wafer map and label
        wafer_map = self.X[idx]
        label = self.y[idx]
       
        # Convert wafer map to tensor and reshape
        wafer_map = torch.from_numpy(wafer_map)
        wafer_map = wafer_map.view(1, *self.image_size)  # Assuming grayscale image
       
        # Convert label to tensor
        label = torch.tensor(label, dtype=torch.long)
       
        return wafer_map, label

# Create Dataset instances
train_dataset = WaferMapDataset(df_train_resampled, max_dim)
test_dataset = WaferMapDataset(df_test_processed, max_dim)

# Create DataLoaders
batch_size = 64  
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

