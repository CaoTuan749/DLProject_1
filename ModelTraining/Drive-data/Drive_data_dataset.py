import os
import glob
import numpy as np
import pandas as pd
import torch
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import Dataset, DataLoader

class SMARTDataset(Dataset):
    """
    PyTorch custom Dataset for SMART data.

    This dataset processes SMART attribute data from CSV files, prepares sequences of
    SMART attributes leading up to a failure event, and encodes model types for use
    in machine learning models.

    Args:
        data_directory (str): 
            Path to the directory of SMART CSV data files.
        
        models_to_include (list, optional): 
            List of encoded model identifiers to include in the dataset. 
            If `None`, all models are included.
        
        days_before_failure (int, default=30): 
            Number of days before a failure event to include in the dataset.
            Only data within this window leading up to the failure is considered.
        
        sequence_length (int, default=10): 
            Number of consecutive days to include in each input sequence.
            Each sample will consist of `sequence_length` days of SMART attributes.
        
        smart_attribute_numbers (list, default=[5, 187, 197, 198]): 
            List of SMART attribute numbers to include as features. 
            These correspond to specific SMART metrics monitored by the drives.
        
        include_raw (bool, default=True): 
            Whether to include raw SMART attribute values as features.
        
        include_normalized (bool, default=True): 
            Whether to include normalized SMART attribute values as features.
        
        scaler (sklearn.preprocessing.StandardScaler, optional): 
            Pre-fitted scaler for normalizing SMART attributes. 
            If `None`, a new scaler is fitted on the training data.
        
        model_label_encoder (sklearn.preprocessing.LabelEncoder, optional): 
            Pre-fitted label encoder for encoding model types. 
            Ensures consistent encoding across training and test datasets.
    """
    def __init__(
        self,
        data_directory,
        models_to_include=None,
        days_before_failure=30,
        sequence_length=10,
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
        print(f"Found {len(all_files)} CSV files in '{self.data_directory}'.")

        # Initialize dictionary to hold failure date for each serial number
        failure_date_dict = {}

        # Iterate over each file to collect failure records and build failure_date_dict
        for filename in all_files:
            try:
                df = pd.read_csv(filename, usecols=['date', 'serial_number', 'failure'])
            except Exception as e:
                print(f"Error reading {filename}: {e}")
                continue
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            # Convert failure to numeric (if stored as string) and then filter for 1
            df['failure'] = pd.to_numeric(df['failure'], errors='coerce')
            failures = df[df['failure'] == 1]
            if not failures.empty:
                for _, row in failures.iterrows():
                    serial_number = row['serial_number']
                    failure_date = row['date']
                    # Since a drive fails only once, store its failure date
                    failure_date_dict[serial_number] = failure_date

        print(f"Number of drives with failure events: {len(failure_date_dict)}")
        if len(failure_date_dict) == 0:
            print("Warning: No drives with failure events found. Check the 'failure' column values.")

        # Dictionary to hold data for each failed drive
        failed_drives_data = defaultdict(list)

        # Columns to read from each CSV file
        columns_to_read = ['date', 'serial_number', 'model'] + self.smart_attributes + ['failure']

        # Iterate over each file to collect data for failed drives
        for filename in all_files:
            try:
                df = pd.read_csv(filename, usecols=columns_to_read)
            except Exception as e:
                print(f"Error reading {filename}: {e}")
                continue
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            df_failed = df[df['serial_number'].isin(failure_date_dict.keys())]
            if df_failed.empty:
                continue  # Skip if no failed drives are present in this file
            for serial_number, group in df_failed.groupby('serial_number'):
                failed_drives_data[serial_number].append(group)

        # Initialize an empty list to store filtered data
        filtered_data_list = []

        # Relax the minimum unique days requirement to 1/4 of the window instead of 1/2
        min_unique_days = max(1, self.days_before_failure // 4)
        print(f"Minimum unique dates required per drive: {min_unique_days}")

        for serial_number, data_list in failed_drives_data.items():
            drive_data = pd.concat(data_list, ignore_index=True)
            # Normalize failure_date to remove time component
            failure_date = failure_date_dict[serial_number].normalize()
            drive_data['date'] = drive_data['date'].dt.normalize()
            start_date = failure_date - pd.Timedelta(days=self.days_before_failure)
            drive_data = drive_data[(drive_data['date'] >= start_date) & (drive_data['date'] <= failure_date)]
            
            num_unique_dates = drive_data['date'].nunique()
            # Check if the drive has at least min_unique_days of data in the period
            if num_unique_dates < min_unique_days:
                print(f"Drive {serial_number} skipped: only {num_unique_dates} unique dates (min required: {min_unique_days})")
                continue  # Skip this drive if it doesn't meet the data requirement
            
            drive_data['days_until_failure'] = (failure_date - drive_data['date']).dt.days
            drive_data = drive_data[
                (drive_data['days_until_failure'] >= 0) & (drive_data['days_until_failure'] <= self.days_before_failure)
            ]
            filtered_data_list.append(drive_data)

        if not filtered_data_list:
            raise ValueError("No drive data passed the filtering criteria. Check filtering thresholds and data content.")

        # Concatenate all filtered data
        filtered_data = pd.concat(filtered_data_list, ignore_index=True)

        # Encode the model types
        if self.model_label_encoder is None:
            le_model = LabelEncoder()
            filtered_data['model_encoded'] = le_model.fit_transform(filtered_data['model'])
            self.model_label_encoder = le_model
        else:
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
                sequence_dates = [current_date - pd.Timedelta(days=i) for i in range(self.sequence_length - 1, -1, -1)]
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

                # Fill missing values: forward fill, then backward fill, then zeros
                sequence_df[self.smart_attributes] = sequence_df[self.smart_attributes].ffill().bfill().fillna(0)
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
            self.scaler = scaler
        else:
            X[:, 1:] = self.scaler.transform(X[:, 1:])

        # Save features and labels
        self.X = X
        self.y = y
        print(f"Total samples after filtering: {len(self.y)}")

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.y)

    def __getitem__(self, idx):
        """
        Retrieves the features and label for a given index.

        Returns:
            tuple: (features_tensor, label_tensor)
        """
        features = self.X[idx]
        label = self.y[idx]
        features_tensor = torch.tensor(features, dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.float32).unsqueeze(0)
        return features_tensor, label_tensor

# ------------------------------------------------------------
# Testing the dataset creation and splitting (for debugging)
# ------------------------------------------------------------
if __name__ == "__main__":
    # Define parameters for testing
    data_directory = '~/tmp/Dataset/Harddrive-dataset/data_Q1_2024'
    days_before_failure = 30
    sequence_length = 30
    smart_attribute_numbers = [5, 187, 197, 198]
    include_raw = True
    include_normalized = True

    # Create a full dataset to get models and their sample counts
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

    # Extract model encoded values and sample counts
    model_encoded_list = full_dataset.X[:, 0].astype(int)
    model_counts = pd.Series(model_encoded_list).value_counts().sort_index()
    model_indices = model_counts.index.tolist()
    model_sample_counts = model_counts.values.tolist()
    model_names = [full_dataset.model_label_encoder.inverse_transform([idx])[0] for idx in model_indices]

    # Create a DataFrame for models and sample counts
    model_info_df = pd.DataFrame({
        'model_encoded': model_indices,
        'model_name': model_names,
        'sample_count': model_sample_counts,
    })

    print("Model sample counts:")
    print(model_info_df)

    # Split the models into train and test sets
    from sklearn.model_selection import train_test_split
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
        scaler=None,  # Scaler will be fitted on training data
        model_label_encoder=full_dataset.model_label_encoder,
    )

    # Create test dataset with models in test_models_encoded (using scaler from training)
    test_dataset = SMARTDataset(
        data_directory=data_directory,
        models_to_include=test_models_encoded.tolist(),
        days_before_failure=days_before_failure,
        sequence_length=sequence_length,
        smart_attribute_numbers=smart_attribute_numbers,
        include_raw=include_raw,
        include_normalized=include_normalized,
        scaler=train_dataset.scaler,
        model_label_encoder=full_dataset.model_label_encoder,
    )

    # Create DataLoaders for testing
    batch_size = 64  
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"Train dataset samples: {len(train_dataset)}")
    print(f"Test dataset samples: {len(test_dataset)}")
