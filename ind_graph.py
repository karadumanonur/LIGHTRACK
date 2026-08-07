import pandas as pd
import matplotlib.pyplot as plt
import glob
from io import StringIO

# 1. Target your exact file name without changing it
file_name = '20260519T093614,145Z.txt'

# Define the column headers (1 Time column + 12 Wavelength columns)
column_names = ['Time [sec]'] + [f'Wavelength {i} [nm]' for i in range(1, 13)]

# 2. Safely handle mixed number formatting (commas vs. periods)
with open(file_name, 'r', encoding='utf-8') as f:
    file_content = f.read().replace(',', '.')

# Load the cleaned text into Pandas
df = pd.read_csv(StringIO(file_content), sep=r'\s+', names=column_names, skiprows=1)

# ---------------------------------------------------------
# STITCHING MULTIPLE FILES (Uncomment to use later)
# If you have multiple files in the same folder and want to 
# combine them all into one continuous dataset automatically:
#
# all_files = glob.glob("*.txt")
# df_list = []
# for f in all_files:
#     with open(f, 'r', encoding='utf-8') as file:
#         content = file.read().replace(',', '.')
#     df_list.append(pd.read_csv(StringIO(content), sep=r'\s+', names=column_names, skiprows=1))
# df = pd.concat(df_list, ignore_index=True)
# ---------------------------------------------------------

# 3. Initialize the plot
plt.figure(figsize=(14, 7))

# 4. Loop through and plot each wavelength column against Time
for i in range(1, 13):
    plt.plot(df['Time [sec]'], df[f'Wavelength {i} [nm]'], marker='.', linestyle='-', linewidth=1.5, label=f'Channel {i}')

# 5. Format the graph
plt.title(f'FBG Sensor Wavelength Variations ({file_name})', fontsize=16, fontweight='bold')
plt.xlabel('Time [sec]', fontsize=12)
plt.ylabel('Wavelength [nm]', fontsize=12)

# Place the legend outside the plot
plt.legend(title='Sensor Channels', loc='upper left', bbox_to_anchor=(1.02, 1))

plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()

# 6. Display the graph
plt.show()