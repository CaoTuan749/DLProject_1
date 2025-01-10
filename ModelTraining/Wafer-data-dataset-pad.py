import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import RandomOverSampler

class WaferDataset(Dataset):
    def __init__(self, file_path, split='train', oversample=False):
        """
        Args:
            file_path (str): Path to the .pkl file containing the wafer dataset.
            split (str): 'train' or 'test' to specify dataset type.
            oversample (bool): Whether to apply oversampling (only for training split).
        """
        self.file_path = file_path
        self.split = split.lower()
        self.oversample = oversample
        self.process_data()

    def process_data(self):
        # Load dataset
        df = pd.read_pickle(self.file_path)

        # Helper functions
        def replace_zero_zero(x):
            if isinstance(x, (list, np.ndarray)) and np.array_equal(x, [0, 0]):
                return 'Unknown'
            return x

        def find_dim(x):
            return x.shape

        def pad_wafer_map(wmap, max_dim):
            height, width = wmap.shape
            max_height, max_width = max_dim
            pad_height_top = (max_height - height) // 2
            pad_height_bottom = max_height - height - pad_height_top
            pad_width_left = (max_width - width) // 2
            pad_width_right = max_width - width - pad_width_left
            return np.pad(
                wmap,
                ((pad_height_top, pad_height_bottom), (pad_width_left, pad_width_right)),
                mode='constant',
                constant_values=0
            )

        # Preprocess dataset
        df['failureType'] = df['failureType'].apply(replace_zero_zero)
        df['trainTestLabel'] = df['trainTestLabel'].apply(replace_zero_zero)
        df_labelled = df[~df['failureType'].isin(['none', 'Unknown'])].reset_index(drop=True)

        # Determine maximum dimensions for padding
        dimensions = df_labelled['waferMap'].apply(find_dim)
        max_height = max(dim[0] for dim in dimensions)
        max_width = max(dim[1] for dim in dimensions)
        self.image_size = (max_height, max_width)

        # Pad and flatten wafer maps
        df_labelled['waferMap_padded'] = df_labelled['waferMap'].apply(lambda x: pad_wafer_map(x, self.image_size))
        df_labelled['waferMap_flat'] = df_labelled['waferMap_padded'].apply(lambda x: x.flatten())
        df_labelled.reset_index(drop=True, inplace=True)

        # Split data into training and testing sets
        df_train = df_labelled[df_labelled['trainTestLabel'] == 'Training'].reset_index(drop=True)
        df_test = df_labelled[df_labelled['trainTestLabel'] == 'Test'].reset_index(drop=True)

        # Prepare features and labels for both splits
        X_train = np.stack(df_train['waferMap_flat'].values).astype('float32')
        y_train = df_train['failureType'].values
        X_test = np.stack(df_test['waferMap_flat'].values).astype('float32')
        y_test = df_test['failureType'].values

        # Apply oversampling for training set if requested
        if self.oversample and self.split == 'train':
            ros = RandomOverSampler(random_state=42)
            X_resampled, y_resampled = ros.fit_resample(X_train, y_train)
            X_train, y_train = X_resampled, y_resampled

        # Encode labels using LabelEncoder fitted on training labels
        encoder = LabelEncoder()
        encoder.fit(y_train)
        y_train_enc = encoder.transform(y_train)
        y_test_enc = encoder.transform(y_test)

        # Select data based on split
        if self.split == 'train':
            self.X = X_train
            self.y = y_train_enc
        else:
            self.X = X_test
            self.y = y_test_enc

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        wafer_map_flat = self.X[idx]
        label = self.y[idx]
        wafer_map_tensor = torch.from_numpy(wafer_map_flat).float()
        wafer_map_tensor = wafer_map_tensor.view(1, *self.image_size)
        label_tensor = torch.tensor(label, dtype=torch.long)
        return wafer_map_tensor, label_tensor


from torch.utils.data import DataLoader

file_path = 'D:/Waffer Data/WM811K.pkl'
batch_size = 64

# Create dataset instances
train_dataset = WaferDataset(file_path=file_path, split='train', oversample=True)
test_dataset = WaferDataset(file_path=file_path, split='test', oversample=False)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
