#!/usr/bin/env python
import os
import glob
import pandas as pd

def main():
    # Define the path to your dataset (expand the tilde if needed)
    data_dir = os.path.expanduser('~/tmp/Dataset/Harddrive-dataset/data_Q1_2024')
    
    # Use glob to list all CSV files in the directory, sorted by name
    csv_files = sorted(glob.glob(os.path.join(data_dir, '*.csv')))
    
    # Print the total number of CSV files found
    print(f"Found {len(csv_files)} CSV files in '{data_dir}'.")
    
    # Optionally, list the filenames
    for file_path in csv_files:
        print(os.path.basename(file_path))
    
    # Read a sample file (for example, the first file) if available
    if csv_files:
        sample_file = csv_files[0]
        print(f"\nReading sample file: {sample_file}")
        
        # Read the CSV file into a pandas DataFrame
        df = pd.read_csv(sample_file)
        
        # Display the column names and first 5 rows
        print("Columns:", df.columns.tolist())
        print("First 5 rows:")
        print(df.head())
    else:
        print("No CSV files found.")

if __name__ == '__main__':
    main()
