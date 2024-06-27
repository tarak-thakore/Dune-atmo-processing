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

branches_reco = ['nshowers_pandoraShower','shwr_length_pandoraShower',
'shwr_startdcosx_pandoraShower', 'shwr_startdcosy_pandoraShower', 'shwr_startdcosz_pandoraShower', 'shwr_startx_pandoraShower', 'shwr_starty_pandoraShower', 'shwr_startz_pandoraShower', 'shwr_totEng_pandoraShower','shwr_dedx_pandoraShower',
'ntracks_pandoraTrack', 'trkendx_pandoraTrack', 'trkendy_pandoraTrack', 'trkendz_pandoraTrack', 'trkstartdcosx_pandoraTrack', 'trkstartdcosy_pandoraTrack', 'trkstartdcosz_pandoraTrack', 'trkstartx_pandoraTrack', 'trkstarty_pandoraTrack', 'trkstartz_pandoraTrack', 'ntrkhits_pandoraTrack','trkmomrange_pandoraTrack']


genie_branches_to_filter = ['genie_primaries_pdg','genie_Eng', 'genie_Px','genie_Py','genie_Pz','genie_P','genie_mass','genie_status_code']

branches_test = ['enu_truth','genie_no_primaries','genie_primaries_pdg','genie_P']


branches = branches_genie + branches_reco


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

  # Get the status codes
  status_codes = row['genie_status_code']

  # Filter each genie column based on the status codes
  filtered_genies = {col: [x for x, status in zip(row[col], status_codes) if status in [1]] for col in genie_columns}

  genie_Px_filtered = np.array(filtered_genies['genie_Px'], dtype=float)
  genie_Py_filtered = np.array(filtered_genies['genie_Py'], dtype=float)
  genie_Pz_filtered = np.array(filtered_genies['genie_Pz'], dtype=float)
  genie_Eng_filtered = np.array(filtered_genies['genie_Eng'], dtype=float)
  genie_P_filtered = np.sqrt(np.square(genie_Px_filtered) + np.square(genie_Py_filtered) + np.square(genie_Pz_filtered))
  filtered_genies['genie_P'] = genie_P_filtered
  #filtered_genies['genie_Px_sum'] = np.sum(genie_Px_filtered)
  #filtered_genies['genie_Py_sum'] = np.sum(genie_Py_filtered)
  #filtered_genies['genie_Pz_sum'] = np.sum(genie_Pz_filtered)
  #filtered_genies['genie_Eng_sum'] = np.sum(genie_Eng_filtered)
   
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
  #for chunk_start in range(0, 1, chunk_size):
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

      # this is a proper scaler, doesn't have len attribute
      if not hasattr(arr[0], "__len__"):
        a=arr
      # this is a 1d array
      elif len(arr[0])==1 and 'shwr' not in b and 'trk' not in b:
        #a=arr
        #print(b, arr.shape,arr.ndim,arr.size,type(arr),arr.dtype)
        #a=arr.astype(np.float32)  # (10000,) 1 10000 depricated
        a = np.vectorize(np.float32)(arr)
      # this is a vector
      else:
        #print(b,len(arr[0]),type(b))
        a = flatten_branch(arr)
      chunk_data[b]= a
      #branch_dict[b] = type(a)
        
    #print('chunkdata len: ',len(chunk_data)) 
    # Convert the dictionary of arrays to a DataFrame
    chunk_df = pd.DataFrame(chunk_data)

    
    # create a filter on genie vars
    filtered_genies = chunk_df.apply(filter_genie, axis=1, genie_columns=genie_branches_to_filter)
    chunk_df[genie_branches_to_filter] = filtered_genies[genie_branches_to_filter]
    # add final state momentum to understand effect of fermi motion
    #chunk_df['genie_Px_sum'] = filtered_genies['genie_Px_sum']
    #chunk_df['genie_Py_sum'] = filtered_genies['genie_Py_sum']
    #chunk_df['genie_Pz_sum'] = filtered_genies['genie_Pz_sum']
    #chunk_df['genie_Eng_sum'] = filtered_genies['genie_Eng_sum']


    # add new columns
    #chunk_df = chunk_df.apply(pd.to_numeric, errors='coerce')
    nu_theta = np.arccos(chunk_df['nu_dcosy_truth'])*180./np.pi
    nu_phi = np.arctan2(chunk_df['nu_dcosx_truth'], chunk_df['nu_dcosz_truth'])*180/np.pi
    chunk_df.insert(9,'nu_theta',nu_theta)
    chunk_df.insert(10,'nu_phi',nu_phi)
  

    if chunk_df['genie_P'].isna().any():
      raise ValueError("NaN value(s) found in the Series")

    #if chunk_df['W_truth'].isna().any():
    #  raise ValueError("NaN value(s) found in the Series")


    #chunk_df['W_truth'].replace("nan", np.nan, inplace=True)
    #chunk_df['W_truth'].replace("-nan", np.nan, inplace=True)
    chunk_df.dropna(subset=['W_truth'], inplace=True)

    #drop columns with nshowers==0 or ntracks==0
    #chunk_df = chunk_df[chunk_df[nshowers_pandoraShower]!=0]
    shw_indices_to_drop = chunk_df[chunk_df['nshowers_pandoraShower'] == 0 ].index
    #indices_to_drop = shw_indices_to_drop + trk_indices_to_drop
    chunk_df = chunk_df.drop(shw_indices_to_drop)
    trk_indices_to_drop = chunk_df[chunk_df['ntracks_pandoraTrack'] == 0 ].index
    chunk_df = chunk_df.drop(trk_indices_to_drop)
    

    #print("Go to the next chunk")
    # Process the chunk if needed
    # For example, perform filtering, transformation, etc.
    # Apply the function to numpy array columns
    for col in chunk_df.select_dtypes(include=[np.ndarray]).columns:
      chunk_df[col] = chunk_df[col].p_apply(array_to_string)
     

    # write csv files
    csvfile = './output/' + 'output-' + str(chunk_id) + '.csv'
    #csvfile = 'merged' + '.csv'
    print('Writing file, ',csvfile)
    chunk_df.to_csv(csvfile, index=False, header=True)

    '''
    # Define the file path and key for storing the dataframe
    hdf5file = './output/' + 'output-' + str(chunk_id) + '.h5'
    key = 'data'  # Key within the HDF5 file (can use different paths with '/')

    
    print(f"Writing HDF5 file: {hdf5file} with key: {key}")
    # Export the dataframe to HDF5 using 'to_hdf' method
    with pd.HDFStore(hdf5file, mode='w') as store:
        chunk_df.to_hdf(store, key = key, complib='blosc:lz4')
    '''

    del chunk_df
    del chunk_data
    del filtered_genies

    # Append the chunk DataFrame to the list
    #dfs.append(chunk_df)

# Concatenate all DataFrames in the list into a single DataFrame
#final_df = pd.concat(dfs, ignore_index=True)

# Perform further analysis or save the final DataFrame to a CSV file
#final_df.to_csv("output.csv", index=False, header=True)

print("All files converted!")

