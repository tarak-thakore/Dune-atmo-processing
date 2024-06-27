#!/usr/bin/env python

import pandas as pd
import os
import glob
import sys

def merge_hdf5_files(folder_path, output_file, key):
    """Merges HDF5 files with common DataFrame keys in a folder into a single output file.

    Args:
        folder_path (str): Path to the folder containing the HDF5 files.
        output_file (str): Name of the output HDF5 file.
        key (str): Key under which the DataFrame is stored in the HDF5 files.

    Returns:
        None
    """
    
    files = glob.glob(os.path.join(folder_path, '*.h5'))
    
    if not files:
        print("No HDF5 files found in the specified folder.")
        return

    # Initialize an empty DataFrame to hold the merged data
    merged_df = pd.DataFrame()
    
    for file in files:
        try:
            with pd.HDFStore(file, 'r') as store:
                if key in store:
                    df = store[key]
                    merged_df = pd.concat([merged_df, df], ignore_index=True)
                else:
                    print(f"Key '{key}' not found in file '{file}'. Skipping this file.")
        except Exception as e:
            print(f"Error reading file '{file}': {e}")
    
    # Write the merged DataFrame to a new HDF5 file
    if not merged_df.empty:
        with pd.HDFStore(output_file, 'w', complib='blosc:lz4') as store:
            store.put(key, merged_df)
        print(f"HDF5 files merged successfully into {output_file}!")
    else:
        print("No data to merge. Output file not created.")

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("Usage: python merge_hdf5.py <folder_path> <output_file_name> <key>")
        sys.exit(1)

    folder_path = sys.argv[1]
    output_file = sys.argv[2]
    key = sys.argv[3]

    merge_hdf5_files(folder_path, output_file, key)

