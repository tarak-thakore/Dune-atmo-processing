#!/usr/bin/python

import uproot
import awkward as ak
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import csv
import tqdm

# root - uproot - pandas approach, easier to work with

f1 = uproot.open('ana_tree_hd_9993.root')
#f1 = uproot.open('anatree_hd_AV_2dot5_random_sum_300k_new.root')
'''
f1.keys()

'''

#Read the TTree
t1=f1['analysistree/anatree']

#list of branches to read
#branches = ['enu_truth','nuPDG_truth','nuvtxx_truth','nuvtxy_truth','nuvtxz_truth']

branches_genie = ['nuPDG_truth','ccnc_truth', 'nuvtxx_truth','nuvtxy_truth','nuvtxz_truth', 'enu_truth','nu_dcosx_truth','nu_dcosy_truth','nu_dcosz_truth','lep_mom_truth', 'lep_dcosx_truth', 'lep_dcosy_truth', 'lep_dcosz_truth', 'mode_truth', 'nuWeight_truth','Q2_truth', 'W_truth', 'X_truth', 'Y_truth','genie_no_primaries','genie_primaries_pdg','genie_Eng', 'genie_Px','genie_Py','genie_Pz','genie_P','genie_status_code']

genie_branches_to_filter = ['genie_primaries_pdg','genie_Eng', 'genie_Px','genie_Py','genie_Pz','genie_P','genie_status_code']


branches = ['run','enu_truth','nuPDG_truth','nuvtxx_truth','nuvtxy_truth','nuvtxz_truth','genie_Eng','EndPointx_geant']
#branches = t1.keys()

# list of branches to be removed 
branches_to_remove = ['processname_geant']
#branches.remove('processname_geant')
#branches = [b for b in branches if b not in branches_to_remove]

#Different uproot functions to read TBranch into arrays
#df = t1.arrays(['enu_truth','nuPDG_truth'], library="pd")
#df = t1.arrays(branches,library='ak')

arr={}
branch_dict={}
np.set_printoptions(threshold=sys.maxsize)

# function to filter genie arrays by their status codes
def filter_genie(row, genie_columns):
 return {k: [x for x, status in zip(row[k], row['genie_status_code']) if status in [0,1]] if k in genie_columns else row[k] for k in row.keys()}

# Function to convert numpy arrays to string representations without square brackets
def array_to_string(array):
  return ', '.join(map(str, array))

def flatten_branch(branch):
  #a1 = lambda branch: [ak.ravel(element) for element in branch]
  flattened_arr = []
  for element in branch:
    #a = ak.ravel(element) #ak
    a = element.ravel() #numpy
    flattened_arr.append(a)
    #print(len(a),type(a))
  return flattened_arr

for b in branches_genie:
  branch = t1[b].array(library='np')
  #print(b)
  if not hasattr(branch[0], "__len__"):
    a=branch
  elif len(branch[0])==1:
    a=branch
  else:
    print(b,len(branch[0]),type(b))
    #flatten_branch= lambda branch: [ak.ravel(element) for element in branch] #ak
    #flatten_branch= lambda branch: [element.ravel() for element in branch] #np
    a = flatten_branch(branch)
    #print(len(a),type(a))
    #a = branch
  branch_dict[b] = type(a)
  arr[b] = a

'''
print('######### Printing branch_dict################')
for key,value in branch_dict.items():
  print(f"{key}: {value}")
print('##############################################')
'''
#arr = {branch_name: t1[branch_name].array(library="np").tolist() for branch_name in branches}

df = pd.DataFrame(arr)

# create a filter on genie vars
filtered_genies = df.apply(filter_genie, axis=1, genie_columns=genie_branches_to_filter)
filtered_df = pd.DataFrame(filtered_genies.tolist())
df[genie_branches_to_filter] = filtered_df[genie_branches_to_filter]

# Apply the function to numpy array columns
# convert to string
#for col in df.select_dtypes(include=[np.ndarray]).columns:
#  df[col] = df[col].apply(array_to_string)

'''
# select events according to some criterion
df_en = df[df['enu_truth']<10]
print(df_en.shape)
'''

'''
# Histogram of enu_truth
plt.figure(figsize=(6, 4))
plt.hist(df_en['enu_truth'], bins=5, edgecolor='black')  # Adjust the number of bins as needed
plt.xlabel('enu_truth')
plt.ylabel('Count')
plt.title('Histogram')
#plt.grid(axis='y', linestyle='--', alpha=0.7)
#plt.show()
'''

'''
print(df)
print(type(df))
print(df.loc[1])
'''

df.to_csv('output.csv',header=True,index=False,float_format='%.8f',    quoting=csv.QUOTE_ALL)
#df.to_csv('output.csv.xz',header=True,index=False,float_format='%.8f',    quoting=csv.QUOTE_ALL, compression='xz') # pandas

#df.to_excel('output.xlsx',header=True,index=False) # pandas
#df_en.to_csv('output_en.csv',header=True,index=False) # pandas
#pldf.write_csv("output.csv")
#df = pl.from_pandas(df)

# xz -v output.csv

'''
# Define the file path and key for storing the dataframe
file_path = 'output.h5'
key = 'data'  # Key within the HDF5 file (can use different paths with '/')

# Export the dataframe to HDF5 using 'to_hdf' method
with pd.HDFStore(file_path, mode='w') as store:
    df.to_hdf(store, key, complib='blosc:lz4')

# Print confirmation message
print(f"Dataframe exported to HDF5 file: {file_path} with key: {key}")
'''

#df_new = pd.read_csv('output.csv')

