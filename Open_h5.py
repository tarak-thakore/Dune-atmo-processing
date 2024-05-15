import pandas as pd

# Replace 'your_file.h5' with the path to your HDF5 file
file_path = 'output.h5'

# Replace 'dataframe_key' with the key you used when exporting the DataFrame (or list all keys with 'pd.read_hdf(file_path)')
dataframe_key = 'data'

# Read the DataFrame from the HDF5 file
df = pd.read_hdf(file_path, dataframe_key)

# Now you can work with the pandas DataFrame 'df'
print(df)
