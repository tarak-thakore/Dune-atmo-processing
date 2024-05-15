import pandas as pd
import numpy as np

# Example dataframe
data = {
    'genie_Eng': [[1, 2, 3], [4, 5], [6, 7, 8, 9]],
    'genie_P': [[10, 20, 30], [40, 50], [60, 70, 80, 90]],
    'genie_status_code': [[1, 0, 1], [1, 1], [0, 1, 1, 0]],
    'other1': [['a', 'b', 'c'], ['d', 'e'], ['f', 'g', 'h', 'i']],
    'other2': [['x', 'y', 'z'], ['u', 'v'], ['m', 'n', 'o', 'p']]
}

df = pd.DataFrame(data)

def filter_genie(row, genie_columns):
    return {k: [x for x, status in zip(row[k], row['genie_status_code']) if status == 1] if k in genie_columns else row[k] for k in row.keys()}

genie_col = [ 'genie_Eng', 'genie_P', 'genie_status_code' ]

# Apply the function to each row
filtered_genies = df.apply(filter_genie, axis=1, genie_columns=genie_col)

# Convert the resulting dictionary to a DataFrame
filtered_df = pd.DataFrame(filtered_genies.tolist())

print(filtered_df)
df[genie_columns] = filtered_df[genie_columns]

