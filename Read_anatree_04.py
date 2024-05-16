#!/usr/bin/python

import uproot
import pandas as pd
import numpy as np
import sys
from parallel_pandas import ParallelPandas

# Define the ROOT file and TTree name
#root_file_path = 'ana_tree_hd_9993.root'
root_file_path = 'anatree_hd_AV_2dot5_random_sum_300k_new.root'
tree_name = 'analysistree/anatree'

np.set_printoptions(threshold=sys.maxsize)
#initialize parallel-pandas
#ParallelPandas.initialize(n_cpu=16, split_factor=4, disable_pr_bar=False)
ParallelPandas.initialize(n_cpu=1, split_factor=1, disable_pr_bar=False)


# Define the list of TBranches to read
#branches = ['enu_truth','nuPDG_truth','nuvtxx_truth','nuvtxy_truth','nuvtxz_truth']
#branches = ['run','enu_truth','nuPDG_truth','nuvtxx_truth','nuvtxy_truth','nuvtxz_truth','genie_Eng','EndPointx_geant']

branches_genie = ['nuPDG_truth','ccnc_truth', 'nuvtxx_truth','nuvtxy_truth','nuvtxz_truth', 'enu_truth','nu_dcosx_truth','nu_dcosy_truth','nu_dcosz_truth','lep_mom_truth', 'lep_dcosx_truth', 'lep_dcosy_truth', 'lep_dcosz_truth', 'mode_truth', 'nuWeight_truth','Q2_truth', 'W_truth', 'X_truth', 'Y_truth','genie_no_primaries','genie_primaries_pdg','genie_Eng', 'genie_Px','genie_Py','genie_Pz','genie_P','genie_mass','genie_status_code']

genie_branches_to_filter = ['genie_primaries_pdg','genie_Eng', 'genie_Px','genie_Py','genie_Pz','genie_P','genie_mass','genie_status_code']

branches_test = ['enu_truth','genie_no_primaries','genie_primaries_pdg','genie_P']


branches = branches_genie


# list of branches to be removed 
branches_to_remove = ['processname_geant']


# Define chunk size
chunk_size = 10000  # Adjust the chunk size as needed

# Initialize an empty list to store DataFrames
dfs = []
# branch dictionary
branch_dict={}

# function to filter genie arrays by their status codes
def filter_genie(row, genie_columns):
  #return {k: [x for x, status in zip(row[k], row['genie_status_code']) if status in [1]] if k in genie_columns else row[k] for k in row.keys()}

  # Get the status codes
  status_codes = row['genie_status_code']

  # Filter each genie column based on the status codes
  filtered_genies = {col: [x for x, status in zip(row[col], status_codes) if status in [1]] for col in genie_columns}

  # Calculate Pz/|P|
  #genie_Eng_filtered = np.array(filtered_genies['genie_Eng'], dtype=float)
  #genie_Pz_filtered = np.array(filtered_genies['genie_Pz'], dtype=float)
  #genie_cos_th_z = np.divide(genie_Pz_filtered,genie_Eng_filtered)
  #filtered_genies['genie_cos_th_z'] = genie_cos_th_z

    
  return pd.Series(filtered_genies)



def flatten_branch(branch):
  flattened_arr = []
  for element in branch:
    a = element.ravel() #numpy
    flattened_arr.append(a)
  return flattened_arr

# Function to convert numpy arrays to string representations without square brackets
def array_to_string(array):
  return ', '.join(map(str, array))

# Open the ROOT file
with uproot.open(root_file_path) as root_file:
  # Access the TTree
  tree = root_file[tree_name]

  # all branches
  #branches = tree.keys()
  #branches = [b for b in branches if b not in branches_to_remove]

  # Get the total number of entries
  total_entries = tree.num_entries
  #total_entries = 500 # test case
  print('Total entries: ',total_entries)
    
  # Iterate over the file in chunks
  for chunk_start in range(0, total_entries, chunk_size):
    chunk_stop = min(chunk_start + chunk_size, total_entries)
    print('chunk_start/total :',chunk_start,'/',total_entries)
    chunk_id = chunk_start//chunk_size
      
    # Create an empty dictionary to store arrays for each TBranch
    chunk_data = {}
    
    # Loop over TBranches
    for b in branches:
      # Read the chunk of data for the current TBranch
      branch = tree[b]
      arr = branch.array(library='np', entry_start=chunk_start, entry_stop=chunk_stop)
      if not hasattr(arr[0], "__len__") or len(arr[0])==1:
        a=arr
      else:
        #print(b,len(arr[0]),type(b))
        a = flatten_branch(arr)
      chunk_data[b]= a
      #branch_dict[b] = type(a)

    '''
    mylist = zip('genie_primaries_pdg','genie_Eng', 'genie_Px','genie_Py','genie_Pz','genie_P','genie_status_code')

    if b is in mylist:
      for item in status_code:

    '''
         
    #print('chunkdata len: ',len(chunk_data)) 
    # Convert the dictionary of arrays to a DataFrame
    chunk_df = pd.DataFrame(chunk_data)


    # create a filter on genie vars
    filtered_genies = chunk_df.apply(filter_genie, axis=1, genie_columns=genie_branches_to_filter)
    #filtered_df = pd.DataFrame(filtered_genies.tolist())
    #chunk_df[genie_branches_to_filter] = filtered_df[genie_branches_to_filter]
    chunk_df[genie_branches_to_filter] = filtered_genies[genie_branches_to_filter]


    #print("Go to the next chunk")
    # Process the chunk if needed
    # For example, perform filtering, transformation, etc.
    # Apply the function to numpy array columns
    for col in chunk_df.select_dtypes(include=[np.ndarray]).columns:
      chunk_df[col] = chunk_df[col].p_apply(array_to_string)

    csvfile = './output/' + 'output-' + str(chunk_id) + '.csv'
    print('Writing file, ',csvfile)
    chunk_df.to_csv(csvfile, index=False, header=True)
    del chunk_df
    del chunk_data
    
    # Append the chunk DataFrame to the list
    #dfs.append(chunk_df)

# Concatenate all DataFrames in the list into a single DataFrame
#final_df = pd.concat(dfs, ignore_index=True)

# Perform further analysis or save the final DataFrame to a CSV file
#final_df.to_csv("output.csv", index=False, header=True)

print("CSV file saved successfully!")

