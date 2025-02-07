import pandas as pd

# Define the path to your pickle file
pkl_file = '~/tmp/Dataset/Wafermap-dataset/WM811K.pkl'

# Load the pickle file into a pandas DataFrame
df = pd.read_pickle(pkl_file)

# Print the shape of the DataFrame (rows, columns)
print("Shape of the DataFrame:", df.shape)

# Print the names of all columns
print("\nColumn names:")
for col in df.columns:
    print("  -", col)

# Show the number of columns
print("\nNumber of columns:", len(df.columns))

# Optionally, preview the first few rows of the DataFrame
print("\nFirst 5 rows of the DataFrame:")
print(df.head())

# (Optional) Display column data types and more info
print("\nDataFrame info:")
print(df.info())
