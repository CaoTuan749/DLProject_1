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
        days_before_failure=30,
        sequence_length=30,
        smart_attribute_numbers=[5, 187, 197, 198],
        include_raw=True,
        include_normalized=True,
        remove_models_with_few_samples=True,
        min_samples_per_model=60,
        enable_bfill=True,
    ):
        self.data_directory = data_directory
        self.days_before_failure = days_before_failure
        self.sequence_length = sequence_length
        self.smart_attribute_numbers = smart_attribute_numbers
        self.include_raw = include_raw
        self.include_normalized = include_normalized
        self.remove_models_with_few_samples = remove_models_with_few_samples
        self.min_samples_per_model = min_samples_per_model
        self.enable_bfill = enable_bfill

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

        # Initialize an empty list to store failure records
        failure_records = []

        # Get a list of all CSV files in the directory
        all_files = glob.glob(os.path.join(self.data_directory, '*.csv'))
        all_files.sort()  # Ensure files are processed in order

        # Iterate over each file to collect failure records
        for filename in all_files:
            df = pd.read_csv(filename, usecols=['date', 'serial_number', 'failure'])
            df['date'] = pd.to_datetime(df['date'])
            failures = df[df['failure'] == 1]
            if not failures.empty:
                failure_records.append(failures)

        # Concatenate all failure records into a single DataFrame
        failure_data = pd.concat(failure_records, ignore_index=True)

        # Get the list of failed drives and their failure dates
        failed_drives_info = failure_data.groupby('serial_number')['date'].max().reset_index()
        failed_drives_info.rename(columns={'date': 'failure_date'}, inplace=True)

        # Dictionary to hold data for each failed drive
        failed_drives_data = defaultdict(list)

        # Convert failed_drives_info to a dictionary for quick access
        failure_date_dict = failed_drives_info.set_index('serial_number')['failure_date'].to_dict()

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
            failure_date = failure_date_dict[serial_number].normalize()
            drive_data['date'] = drive_data['date'].dt.normalize()
            start_date = failure_date - pd.Timedelta(days=self.days_before_failure)
            drive_data = drive_data[(drive_data['date'] >= start_date) & (drive_data['date'] <= failure_date)]
            drive_data['days_until_failure'] = (failure_date - drive_data['date']).dt.days
            drive_data = drive_data[
                (drive_data['days_until_failure'] >= 0) & (drive_data['days_until_failure'] <= self.days_before_failure)
            ]
            filtered_data_list.append(drive_data)

        # Concatenate all filtered data
        filtered_data = pd.concat(filtered_data_list, ignore_index=True)

        # Encode the model types
        le = LabelEncoder()
        filtered_data['model_encoded'] = le.fit_transform(filtered_data['model'])

        # Optionally remove models with few samples
        if self.remove_models_with_few_samples:
            model_counts = filtered_data['model'].value_counts()
            models_to_keep = model_counts[model_counts >= self.min_samples_per_model].index.tolist()
            filtered_data = filtered_data[filtered_data['model'].isin(models_to_keep)].reset_index(drop=True)
            le = LabelEncoder()
            filtered_data['model_encoded'] = le.fit_transform(filtered_data['model'])

        # Save label encoder and model mapping
        self.label_encoder = le
        self.model_mapping = dict(zip(le.classes_, le.transform(le.classes_)))

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
                    continue  # Discard this data point
                sequence_df[self.smart_attributes] = sequence_df[self.smart_attributes].ffill()
                if self.enable_bfill:
                    sequence_df[self.smart_attributes] = sequence_df[self.smart_attributes].bfill()
                sequence_df[self.smart_attributes] = sequence_df[self.smart_attributes].fillna(0)
                sequence_data = sequence_df[self.smart_attributes].values.flatten()
                model_encoded = date_to_data[current_date]['model_encoded']
                features = [model_encoded] + sequence_data.tolist()
                X.append(features)
                y.append(days_until_failure)

        # Convert features and labels to NumPy arrays
        X = np.array(X)
        y = np.array(y)

        # Normalize SMART attributes (exclude model_encoded)
        scaler = StandardScaler()
        X[:, 1:] = scaler.fit_transform(X[:, 1:])
        self.scaler = scaler  # Save scaler if needed

        # Save features and labels
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        features = self.X[idx]
        label = self.y[idx]
        features_tensor = torch.tensor(features, dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.float32).unsqueeze(0)
        return features_tensor, label_tensor

# Parameters
data_directory = 'D:/Backblaze_Data/data_Q2_2024/Training-Q1-data/'
days_before_failure = 30
sequence_length = 30
smart_attribute_numbers = [5, 187, 197, 198]
include_raw = True
include_normalized = True
remove_models_with_few_samples = True
min_samples_per_model = 60
enable_bfill = False

# Create the custom dataset
dataset = SMARTDataset(
    data_directory=data_directory,
    days_before_failure=days_before_failure,
    sequence_length=sequence_length,
    smart_attribute_numbers=smart_attribute_numbers,
    include_raw=include_raw,
    include_normalized=include_normalized,
    remove_models_with_few_samples=remove_models_with_few_samples,
    min_samples_per_model=min_samples_per_model,
    enable_bfill=enable_bfill,
)

# Create DataLoader
batch_size = 64  
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

