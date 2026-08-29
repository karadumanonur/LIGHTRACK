import pandas as pd

# 1. Define your files and your intended date
input_file = 'infrabel_may_2026.csv'
target_date = '14MAY2026'

# This will automatically name the new file 'infrabel_14MAY2026.csv'
output_file = f'infrabel_{target_date}.csv'

print("Loading the massive dataset... This might take a moment.")

# 2. Load the raw dataset
# (low_memory=False helps Pandas handle massive files without guessing data types)
df = pd.read_csv(input_file, low_memory=False)

# 3. Filter the dataset to only include rows matching your exact target date
# If you prefer to filter by departure, change 'REAL_DATE_ARR' to 'DATDEP'
filtered_df = df[df['REAL_DATE_ARR'] == target_date]

# 4. Save the isolated data into a new, lightweight CSV file
filtered_df.to_csv(output_file, index=False)

print(f"Success! Extracted {len(filtered_df)} records for {target_date}.")
print(f"Clean data saved to: {output_file}")