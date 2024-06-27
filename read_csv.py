import pandas as pd
import numpy as np

#df = pd.read_csv('dataset_lstm_ee_fd_fhc_nonswap.csv')
df = pd.read_csv('../ml_datasets/dune_atmo_genie_pandora_300k_extended.csv')
#df = pd.read_csv('output.csv',compression='xz')

# Convert string representations back to numpy arrays
# This give errors if the datatype is int instead of float


#for col in df.select_dtypes(include=[object]).columns:
#  df[col] = df[col].apply(lambda x: np.array(x.split(',')).astype(float))


'''
for col in df.select_dtypes(include=[object]).columns:
  try:
    df[col] = df[col].apply(lambda x: np.array(x.split(',')).astype(float))
  except ValueError:
    df[col] = df[col].apply(lambda x: np.array(x.split(',')).astype(int))
'''


for col in df.select_dtypes(include=[object]).columns:
  df[col] = pd.to_numeric(df[col], errors='coerce')

