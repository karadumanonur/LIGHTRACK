import glob
import os
import pandas as pd

# 1. Grab all text files and sort them chronologically by filename
all_files = sorted(glob.glob("data/data/*.txt"))

column_names = ['Time [sec]'] + [f'Wavelength {i} [nm]' for i in range(1, 13)]

# Set a threshold for an "active" train (e.g., 0.05 nm or 50 pm shift from baseline)
# You may need to tweak this depending on your heaviest coaches
STRAIN_THRESHOLD = 0.050 

split_events = []

print("Scanning for split train events...")

# 2. Iterate through adjacent pairs of files
for i in range(len(all_files) - 1):
    file_A = all_files[i]
    file_B = all_files[i+1]
    
    # Load File A
    df_A = pd.read_csv(file_A, sep=r'\s+', names=column_names, skiprows=1)
    
    # Calculate a simple baseline (median) and grab the last 15 points
    baseline_A = df_A['Wavelength 1 [nm]'].median()
    tail_A = df_A['Wavelength 1 [nm]'].tail(15)
    
    # Check if the maximum deviation at the very end is above our threshold
    is_active_at_end = (tail_A - baseline_A).abs().max() > STRAIN_THRESHOLD
    
    # SHORT-CIRCUIT: Only load File B if File A ended actively
    if is_active_at_end:
        df_B = pd.read_csv(file_B, sep=r'\s+', names=column_names, skiprows=1)
        baseline_B = df_B['Wavelength 1 [nm]'].median()
        
        # Grab the first 15 points of File B
        head_B = df_B['Wavelength 1 [nm]'].head(15)
        
        is_active_at_start = (head_B - baseline_B).abs().max() > STRAIN_THRESHOLD
        
        # If both the end of A and start of B are active, we found a split!
        if is_active_at_start:
            name_A = os.path.basename(file_A)
            name_B = os.path.basename(file_B)
            print(f"Split detected between:\n -> {name_A}\n -> {name_B}\n")
            split_events.append((file_A, file_B))

print(f"Scan complete. Total split events found: {len(split_events)}")