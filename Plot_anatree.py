#!/usr/bin/python

import uproot
import pandas as pd
import numpy as np
import sys
import os
import csv
from parallel_pandas import ParallelPandas
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('agg')
plt.ion()

csvfile = '../ml_datasets/dune_atmo_genie_300k_extended.csv'
df_genie = pd.read_csv(csvfile)

'''
# select events according to some criterion
df_en = df[df['enu_truth']<10]
print(df_en.shape)
'''

# Histogram of enu_truth
plt.figure(figsize=(6, 4))
plt.hist(df_genie['enu_truth'], bins=100)  # Adjust the number of bins as needed
plt.xlabel('enu_truth')
plt.ylabel('Count')
plt.yscale('log')
#plt.title('Histogram')
#plt.grid(axis='y', linestyle='--', alpha=0.7)
#plt.show()
plt.savefig('plot_enu.png')

plt.figure(figsize=(6, 4))
plt.hist(df_genie['nu_theta'], bins=100)  # Adjust the number of bins as needed
plt.xlabel('theta(deg)')
plt.ylabel('Count')
#plt.yscale('log')
#plt.title('Histogram')
#plt.grid(axis='y', linestyle='--', alpha=0.7)
#plt.show()
plt.savefig('plot_theta.png')

plt.figure(figsize=(6, 4))
plt.hist2d(df_genie['enu_truth'],df_genie['nu_theta'], bins=1000, cmap='viridis')  # Adjust the number of bins as needed
plt.xscale('log')
#plt.yscale('log')
plt.ylabel('theta(deg)')
plt.xlabel('enu (GeV)')
#plt.yscale('log')
#plt.title('Histogram')
#plt.grid(axis='y', linestyle='--', alpha=0.7)
#plt.show()
plt.colorbar(label='Counts')
plt.savefig('plot_e_theta.png')

'''
print(df)
print(type(df))
print(df.loc[1])
'''


