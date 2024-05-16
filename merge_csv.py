#!/usr/bin/env python

import csv
import glob
import os
import sys

def merge_csv_files(folder_path, output_file):
   """Merges CSV files with common headers in a folder into a single output file.

   Args:
       folder_path (str): Path to the folder containing the CSV files.
       output_file (str): Name of the output CSV file.

   Returns:
       None
   """

   header_written = False
   with open(output_file, 'w', newline='') as outfile:
       writer = csv.writer(outfile)

       for filename in glob.glob(os.path.join(folder_path, '*.csv')):
           with open(filename, 'r') as infile:
               reader = csv.reader(infile)
               header = next(reader)  # Read the header row
               if not header_written:
                   writer.writerow(header)
                   header_written = True
               writer.writerows(reader)

if __name__ == '__main__':
   if len(sys.argv) != 3:
       print("Usage: python merge_csv.py <folder_path> <output_file_name>")
       sys.exit(1)

   folder_path = sys.argv[1]
   output_file = sys.argv[2]

   merge_csv_files(folder_path, output_file)
   print("CSV files merged successfully!")

