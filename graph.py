import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
from io import StringIO

all_files = sorted(glob.glob("*.txt")) # Adjust path as needed
column_names = ['Time [sec]'] + [f'Wavelength {i} [nm]' for i in range(1, 13)]

for file_path in all_files:
    display_name = os.path.basename(file_path)
    print(f"Loading and plotting: {display_name}...")
    
    # Read the raw text and replace European comma decimals with standard periods
    with open(file_path, 'r', encoding='utf-8') as f:
        file_content = f.read().replace(',', '.')
        
    # Load the cleaned text into Pandas using StringIO
    df = pd.read_csv(StringIO(file_content), sep=r'\s+', skiprows=1, names=column_names)

    # Initialize and format the plot
    plt.figure(figsize=(14, 7))
    for i in range(1, 13):
        plt.plot(df['Time [sec]'], df[f'Wavelength {i} [nm]'], marker='.', linestyle='-', linewidth=1.5, label=f'Channel {i}')

    plt.title(f'FBG Sensor Wavelength Variations ({display_name})', fontsize=16, fontweight='bold')
    plt.xlabel('Time [sec]', fontsize=12)
    plt.ylabel('Wavelength [nm]', fontsize=12)
    plt.legend(title='Sensor Channels', loc='upper left', bbox_to_anchor=(1.02, 1))
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()