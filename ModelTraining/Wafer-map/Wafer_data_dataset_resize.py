import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import RandomOverSampler
from PIL import Image

class WaferMapDataset(Dataset):
    def __init__(self,
                 file_path,
                 split='train',       # 'train' or 'test'
                 oversample=False,    # oversample if split='train'
                 target_dim=(224, 224),
                 task_classes=None    # List of integer labels to keep for the current task (if any)
                ):
        """
        Args:
            file_path (str): Path to .pkl file.
            split (str): 'train' or 'test'.
            oversample (bool): Apply oversampling on training split.
            target_dim (tuple): Dimensions to resize wafer maps.
            task_classes (list or None): If provided, only samples with these class indices are kept.
        """
        self.file_path = file_path
        self.split = split.lower()
        self.oversample = oversample
        self.target_dim = target_dim
        self.task_classes = task_classes

        # 1) Load DataFrame
        df = pd.read_pickle(self.file_path)

        # 2) Replace [0,0] in 'failureType' / 'trainTestLabel' if they exist
        def replace_zero_zero(x):
            if isinstance(x, (list, np.ndarray)) and np.array_equal(x, [0, 0]):
                return 'Unknown'
            return x

        if 'failureType' not in df.columns:
            raise KeyError("Missing 'failureType' column.")
        df['failureType'] = df['failureType'].apply(replace_zero_zero)

        if 'trainTestLabel' not in df.columns:
            raise KeyError("Missing 'trainTestLabel' column.")
        df['trainTestLabel'] = df['trainTestLabel'].apply(replace_zero_zero)

        # 3) Filter out 'none' or 'Unknown'
        valid_mask = ~df['failureType'].isin(['none', 'Unknown'])
        df = df[valid_mask].reset_index(drop=True)

        # 4) Resize wafer maps
        def resize_wafer_map(wmap):
            img = Image.fromarray(wmap.astype('uint8'))
            img_resized = img.resize(self.target_dim, Image.Resampling.LANCZOS)
            return np.array(img_resized)
        df['waferMap_resized'] = df['waferMap'].apply(resize_wafer_map)

        # 5) Flatten wafer maps
        df['waferMap_flat'] = df['waferMap_resized'].apply(lambda x: x.flatten())

        # 6) Separate train & test
        df_train = df[df['trainTestLabel'] == 'Training'].reset_index(drop=True)
        df_test  = df[df['trainTestLabel'] == 'Test'].reset_index(drop=True)
        if len(df_train) == 0:
            raise ValueError("No training samples with 'trainTestLabel' == 'Training'!")

        # 7) Fit LabelEncoder on the training set
        encoder = LabelEncoder()
        encoder.fit(df_train['failureType'].values)
        self.encoder = encoder

        # 8) If task_classes is provided, filter both training and test sets.
        if self.task_classes is not None:
            df_train['encoded'] = encoder.transform(df_train['failureType'].values)
            df_test['encoded'] = encoder.transform(df_test['failureType'].values)
            df_train = df_train[df_train['encoded'].isin(self.task_classes)].reset_index(drop=True)
            df_test = df_test[df_test['encoded'].isin(self.task_classes)].reset_index(drop=True)
            new_encoder = LabelEncoder()
            new_encoder.fit(df_train['failureType'].values)
            self.encoder = new_encoder

        # 9) Oversample if in training split and oversample is True.
        if self.oversample and self.split == 'train':
            X_train = np.stack(df_train['waferMap_flat'].values).astype('float32')
            y_train = df_train['failureType'].values
            ros = RandomOverSampler(random_state=42)
            X_res, y_res = ros.fit_resample(X_train, y_train)
            df_train_res = pd.DataFrame({
                'waferMap_flat': list(X_res),
                'failureType': y_res
            }).reset_index(drop=True)
            df_train = df_train_res

        # 10) Choose the final split
        df_split = df_train if self.split == 'train' else df_test

        # 11) Encode labels
        X_data = np.stack(df_split['waferMap_flat'].values).astype('float32')
        y_data = df_split['failureType'].values
        y_enc  = self.encoder.transform(y_data)

        # 12) Store arrays
        self.X = X_data
        self.y = y_enc
        self.num_samples = len(self.X)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        wafer_map_flat = self.X[idx]
        label = self.y[idx]
        wafer_map_tensor = torch.from_numpy(wafer_map_flat).float()
        wafer_map_tensor = wafer_map_tensor.view(1, self.target_dim[0], self.target_dim[1]).repeat(3, 1, 1)
        label_tensor = torch.tensor(label, dtype=torch.long)
        return wafer_map_tensor, label_tensor
