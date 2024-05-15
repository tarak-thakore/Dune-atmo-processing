#!/usr/bin/python

from tqdm import tqdm

# Simulate a long-running loop with a wait time
total_items = 100
for i in tqdm(range(total_items)):
  # Simulate work being done
  import time
  time.sleep(0.1)  # Sleep for 0.1 seconds

print("Finished processing all items!")

#########################################
import pandas as pd

# Create two sample DataFrames
data_left = {'CustomerID': [1, 2, 3], 'Name': ['Alice', 'Bob', 'Charlie']}
df_left = pd.DataFrame(data_left)

data_right = {'CustomerID': [1, 2, 4], 'City': ['New York', 'Los Angeles', 'Chicago']}
df_right = pd.DataFrame(data_right)

# Inner Join (default behavior) - Keeps rows with matching CustomerID in both DataFrames
inner_join = pd.merge(df_left, df_right, on='CustomerID')
print("Inner Join:")
print(inner_join)

# Outer Join (preserves all rows from both DataFrames)
# Left Outer Join - Keeps all rows from df_left and matching rows from df_right (fills missing values with NaN)
left_join = pd.merge(df_left, df_right, on='CustomerID', how='left')
print("\nLeft Outer Join:")
print(left_join)

# Right Outer Join - Keeps all rows from df_right and matching rows from df_left (fills missing values with NaN)
right_join = pd.merge(df_left, df_right, on='CustomerID', how='right')
print("\nRight Outer Join:")
print(right_join)

##########################################


