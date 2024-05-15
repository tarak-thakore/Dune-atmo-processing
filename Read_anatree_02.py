#!/usr/bin/python

import sys
import os

# using getlogin() returning username
user_name = os.getlogin()

# using Pierrer's anatree loader classes - uses similar function as the uproot approach
# needs some pandas/numpy inner join to merge different trees and export to csv
sys.path.append(f'/exp/dune/app/users/tthakore/apc_atmo_repo/Anatree/')
from anatree_class import Anatree
from ana_tools import *

anatree = Anatree('ana_tree_hd_9993.root')

pldf_nu = anatree.nu
pldf_g4 = anatree.geant
pldf_reco_trk = anatree.reco_tracks
pldf_reco_shw = anatree.reco_showers
#pldf_reco_hits = anatree.reco_hits

#pldf.merge_sorted(pldf_reco_tracks,key='event')

df_nu = pldf_nu.to_pandas()
#df_nu = pldf_g4.to_pandas()
df_nu.to_csv('anatree_nu.csv',header=True) # pandas


#df_reco_trk = pldf_reco_trk.to_pandas()
#df_reco_trk.to_csv('anatree_rec_trk.csv',header=True)

#anatree.write_polars_parquet(fpath="./")
#df_full = anatree.get_full_pfp()
