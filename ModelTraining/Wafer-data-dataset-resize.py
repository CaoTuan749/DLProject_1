import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from PIL import Image

# Path to .pkl file
file_path = 'D:/Waffer Data/WM811K.pkl'

# Load the dataset
df = pd.read_pickle(file_path)

# Function to find dimensions
def find_dim(x):
    dim0 = np.size(x, axis=0)
    dim1 = np.size(x, axis=1)
    return dim0, dim1

# Apply the function to the waferMap column
df['waferMapDim'] = df['waferMap'].apply(find_dim)

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

# Define the target dimension for resizing
target_dim = (64, 64)

def resize_wafer_map(wmap):
    # Convert numpy array to PIL Image
    img = Image.fromarray(wmap.astype('uint8'))
    # Resize the image
    img_resized = img.resize(target_dim, Image.Resampling.LANCZOS)
    # Convert back to numpy array
    return np.array(img_resized)

# Apply resizing to wafer maps
df_modified_labelled['waferMap_resized'] = df_modified_labelled['waferMap'].apply(resize_wafer_map)

# Flatten the wafer maps
df_modified_labelled['waferMap_flat'] = df_modified_labelled['waferMap_resized'].apply(lambda x: x.flatten())

# Reset index to ensure alignment
df_modified_labelled.reset_index(drop=True, inplace=True)
df_modified_labelled['original_index'] = df_modified_labelled.index

# Prepare data for oversampling
X = np.stack(df_modified_labelled['waferMap_flat'].values).astype('float32')
y = df_modified_labelled['failureType']
original_indices = df_modified_labelled['original_index'].values

# Perform oversampling to balance the dataset
ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X, y)

# Get sample indices from oversampling
sample_indices = ros.sample_indices_

# Map back to original indices
resampled_original_indices = original_indices[sample_indices]

# Encode labels
encoder = LabelEncoder()
y_resampled_enc = encoder.fit_transform(y_resampled)
num_classes = len(encoder.classes_)

# Create a DataFrame with resampled data
df_resampled = pd.DataFrame({
    'waferMap_flat': list(X_resampled),
    'failureType_enc': y_resampled_enc,
    'original_index': resampled_original_indices
})

# Split the data into training and testing sets
df_train, df_test = train_test_split(
    df_resampled,
    test_size=0.2,
    random_state=42,
    stratify=df_resampled['failureType_enc']
)

# Dataset class
class WaferMapDataset(Dataset):
    def __init__(self, df):
        self.X = np.stack(df['waferMap_flat'].values).astype('float32')
        self.y = df['failureType_enc'].values.astype('int64')
        self.num_samples = len(df)
        self.image_size = target_dim

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Get wafer map and label
        wafer_map = self.X[idx]
        label = self.y[idx]

        # Convert wafer map to tensor and reshape
        wafer_map = torch.from_numpy(wafer_map)
        wafer_map = wafer_map.view(1, *self.image_size)  

        # Convert label to tensor
        label = torch.tensor(label, dtype=torch.long)

        return wafer_map, label

# Create Dataset instances
train_dataset = WaferMapDataset(df_train)
test_dataset = WaferMapDataset(df_test)

# Create DataLoaders
batch_size = 64 
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
