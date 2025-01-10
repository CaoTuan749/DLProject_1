import os
import glob
import numpy as np
import pandas as pd
import torch
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import Dataset, DataLoader

class SMARTDataset(Dataset):
    def __init__(
        self,
        data_directory,
        models_to_include=None,
        days_before_failure=30,
        sequence_length=30,
        smart_attribute_numbers=[5, 187, 197, 198],
        include_raw=True,
        include_normalized=True,
        scaler=None,  
        model_label_encoder=None,  # To ensure consistent model encoding
    ):
        self.data_directory = data_directory
        self.models_to_include = models_to_include  # List of models to include
        self.days_before_failure = days_before_failure
        self.sequence_length = sequence_length
        self.smart_attribute_numbers = smart_attribute_numbers
        self.include_raw = include_raw
        self.include_normalized = include_normalized
        self.scaler = scaler  
        self.model_label_encoder = model_label_encoder  # Use existing encoder if provided

        # Initialize lists for features and labels
        self.X = []
        self.y = []

        # Process the data to populate self.X and self.y
        self.process_data()

    def process_data(self):
        # Generate the list of SMART attribute column names
        smart_attributes = []
        for num in self.smart_attribute_numbers:
            if self.include_raw:
                smart_attributes.append(f'smart_{num}_raw')
            if self.include_normalized:
                smart_attributes.append(f'smart_{num}_normalized')
        self.smart_attributes = smart_attributes

        # Get a list of all CSV files in the directory
        all_files = glob.glob(os.path.join(self.data_directory, '*.csv'))
        all_files.sort()  # Ensure files are processed in order

        # Initialize dictionary to hold failure date for each serial number
        failure_date_dict = {}

        # Iterate over each file to collect failure records and build failure_date_dict
        for filename in all_files:
            df = pd.read_csv(filename, usecols=['date', 'serial_number', 'failure'])
            df['date'] = pd.to_datetime(df['date'])
            failures = df[df['failure'] == 1]
            if not failures.empty:
                for _, row in failures.iterrows():
                    serial_number = row['serial_number']
                    failure_date = row['date']
                    # Since a drive fails only once, we store its failure date
                    failure_date_dict[serial_number] = failure_date

        # Dictionary to hold data for each failed drive
        failed_drives_data = defaultdict(list)

        # Columns to read from each CSV file
        columns_to_read = ['date', 'serial_number', 'model'] + self.smart_attributes + ['failure']

        # Iterate over each file to collect data for failed drives
        for filename in all_files:
            df = pd.read_csv(filename, usecols=columns_to_read)
            df['date'] = pd.to_datetime(df['date'])
            df_failed = df[df['serial_number'].isin(failure_date_dict.keys())]
            if df_failed.empty:
                continue  # Skip if no failed drives are present in this file
            for serial_number, group in df_failed.groupby('serial_number'):
                failed_drives_data[serial_number].append(group)

        # Initialize an empty list to store filtered data
        filtered_data_list = []

        for serial_number, data_list in failed_drives_data.items():
            drive_data = pd.concat(data_list, ignore_index=True)
            # Normalize failure_date to remove time component (hours, minutes, etc.)
            failure_date = failure_date_dict[serial_number].normalize()
            drive_data['date'] = drive_data['date'].dt.normalize()
            start_date = failure_date - pd.Timedelta(days=self.days_before_failure)
            drive_data = drive_data[(drive_data['date'] >= start_date) & (drive_data['date'] <= failure_date)]
            
            # Check if the drive has at least half of the days' data in the period
            if drive_data['date'].nunique() < (self.days_before_failure / 2):
                continue  # Skip this drive if it doesn't meet the data requirement
            
            drive_data['days_until_failure'] = (failure_date - drive_data['date']).dt.days
            drive_data = drive_data[
                (drive_data['days_until_failure'] >= 0) & (drive_data['days_until_failure'] <= self.days_before_failure)
            ]
            filtered_data_list.append(drive_data)

        # Concatenate all filtered data
        filtered_data = pd.concat(filtered_data_list, ignore_index=True)

        # Encode the model types
        if self.model_label_encoder is None:
            le_model = LabelEncoder()
            filtered_data['model_encoded'] = le_model.fit_transform(filtered_data['model'])
            self.model_label_encoder = le_model
        else:
            # Use existing label encoder
            filtered_data['model_encoded'] = self.model_label_encoder.transform(filtered_data['model'])

        self.model_mapping = dict(zip(self.model_label_encoder.classes_, 
                                      self.model_label_encoder.transform(self.model_label_encoder.classes_)))

        # If models_to_include is provided, filter the data
        if self.models_to_include is not None:
            filtered_data = filtered_data[filtered_data['model_encoded'].isin(self.models_to_include)].reset_index(drop=True)

        # Initialize lists for features and labels
        X = []
        y = []

        # Group data by serial_number
        grouped = filtered_data.groupby('serial_number')

        for name, group in grouped:
            group = group.sort_values(by='date').reset_index(drop=True)
            date_to_data = group.set_index('date').to_dict('index')
            dates = group['date'].unique()
            failure_date = failure_date_dict[name].normalize()

            for current_date in dates:
                days_until_failure = (failure_date - current_date).days
                if days_until_failure < 0 or days_until_failure > self.days_before_failure:
                    continue

                # Generate sequence_dates for the sequence_length window ending at current_date
                sequence_dates = [
                    current_date - pd.Timedelta(days=i) for i in range(self.sequence_length - 1, -1, -1)
                ]
                sequence_records = []
                for seq_date in sequence_dates:
                    if seq_date in date_to_data:
                        seq_record = date_to_data[seq_date]
                        smart_values = {attr: seq_record.get(attr, np.nan) for attr in self.smart_attributes}
                    else:
                        smart_values = {attr: np.nan for attr in self.smart_attributes}
                    sequence_records.append(smart_values)

                sequence_df = pd.DataFrame(sequence_records)
                missing_count = sequence_df[self.smart_attributes].isna().sum().sum()
                total_values = self.sequence_length * len(self.smart_attributes)
                if missing_count > total_values / 2:
                    continue  # Discard this data point if too many values are missing

                # Always perform forward fill followed by backward fill, then fill remaining NaNs with 0
                sequence_df[self.smart_attributes] = sequence_df[self.smart_attributes].ffill()
                sequence_df[self.smart_attributes] = sequence_df[self.smart_attributes].bfill()
                sequence_df[self.smart_attributes] = sequence_df[self.smart_attributes].fillna(0)

                sequence_data = sequence_df[self.smart_attributes].values.flatten()
                model_encoded = group['model_encoded'].iloc[0]
                features = [model_encoded] + sequence_data.tolist()
                X.append(features)
                y.append(days_until_failure)

        # Convert features and labels to NumPy arrays
        X = np.array(X)
        y = np.array(y)

        # Normalize SMART attributes (exclude model_encoded)
        if self.scaler is None:
            scaler = StandardScaler()
            X[:, 1:] = scaler.fit_transform(X[:, 1:])
            self.scaler = scaler  # Save scaler for future use
        else:
            X[:, 1:] = self.scaler.transform(X[:, 1:])  # Use existing scaler

        # Save features and labels
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        features = self.X[idx]
        label = self.y[idx]
        features_tensor = torch.tensor(features, dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.float32).unsqueeze(0)  # For regression
        return features_tensor, label_tensor


# Parameters
data_directory = 'D:/Backblaze_Data/Training-Q1-data/'
days_before_failure = 30
sequence_length = 30
smart_attribute_numbers = [5, 187, 197, 198]
include_raw = True
include_normalized = True

# First, create a full dataset to get models and their sample counts
full_dataset = SMARTDataset(
    data_directory=data_directory,
    days_before_failure=days_before_failure,
    sequence_length=sequence_length,
    smart_attribute_numbers=smart_attribute_numbers,
    include_raw=include_raw,
    include_normalized=include_normalized,
    scaler=None,
    model_label_encoder=None,  
)

# Extract models and their sample counts
model_encoded_list = full_dataset.X[:, 0].astype(int)
model_counts = pd.Series(model_encoded_list).value_counts().sort_index()
model_indices = model_counts.index.tolist()
model_sample_counts = model_counts.values.tolist()

# Map encoded model indices back to model names
model_names = [full_dataset.model_label_encoder.inverse_transform([idx])[0] for idx in model_indices]

# Create a DataFrame for models and sample counts
model_info_df = pd.DataFrame({
    'model_encoded': model_indices,
    'model_name': model_names,
    'sample_count': model_sample_counts,
})

print("Model sample counts:")
print(model_info_df)

# Now, split the models into train and test sets
from sklearn.model_selection import train_test_split

# Ensure models with more samples are evenly distributed
train_models_encoded, test_models_encoded = train_test_split(
    model_info_df['model_encoded'],
    test_size=0.2,
    random_state=42,
    shuffle=True,
    stratify=model_info_df['sample_count']
)

print(f"Train models (encoded): {train_models_encoded.tolist()}")
print(f"Test models (encoded): {test_models_encoded.tolist()}")

# Create training dataset with models in train_models_encoded
train_dataset = SMARTDataset(
    data_directory=data_directory,
    models_to_include=train_models_encoded.tolist(),
    days_before_failure=days_before_failure,
    sequence_length=sequence_length,
    smart_attribute_numbers=smart_attribute_numbers,
    include_raw=include_raw,
    include_normalized=include_normalized,
    scaler=None,  # Scaler will be created using training data
    model_label_encoder=full_dataset.model_label_encoder,  # Use the same encoder
)

# Create test dataset with models in test_models_encoded, using the same scaler as training data
test_dataset = SMARTDataset(
    data_directory=data_directory,
    models_to_include=test_models_encoded.tolist(),
    days_before_failure=days_before_failure,
    sequence_length=sequence_length,
    smart_attribute_numbers=smart_attribute_numbers,
    include_raw=include_raw,
    include_normalized=include_normalized,
    scaler=train_dataset.scaler,  # Use scaler from training data
    model_label_encoder=full_dataset.model_label_encoder,  # Use the same encoder
)

# Create DataLoaders
batch_size = 64  
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
