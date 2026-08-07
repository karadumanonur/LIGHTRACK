import pandas as pd

FILE_PATH = 'infrabel_14MAY2026.csv'
DISTANCE_RATIO = 0.76  # 76% of the track distance from Namur to Charleroi

# 1. Load the dataset
df = pd.read_csv(FILE_PATH, low_memory=False)

# 2. Extract unique departures from Namur
namur_df = df[df['PTCAR_LG_NM_NL'].str.contains('NAMUR', case=False, na=False)].copy()
namur_df = namur_df.dropna(subset=['REAL_TIME_DEP']).drop_duplicates(subset=['TRAIN_NO'])
namur_df = namur_df[['TRAIN_NO', 'RELATION_DIRECTION', 'REAL_DATE_DEP', 'REAL_TIME_DEP']]

# 3. Extract unique arrivals at Charleroi
charleroi_df = df[df['PTCAR_LG_NM_NL'].str.contains('CHARLEROI', case=False, na=False)].copy()
charleroi_df = charleroi_df.dropna(subset=['REAL_TIME_ARR']).drop_duplicates(subset=['TRAIN_NO'])
charleroi_df = charleroi_df[['TRAIN_NO', 'REAL_DATE_ARR', 'REAL_TIME_ARR']]

# 4. Merge to find trains that physically visited both stations
merged_df = pd.merge(namur_df, charleroi_df, on='TRAIN_NO', how='inner')

# 5. Parse Datetimes
merged_df['Departure_Namur'] = pd.to_datetime(
    merged_df['REAL_DATE_DEP'].astype(str) + ' ' + merged_df['REAL_TIME_DEP'].astype(str),
    format='%d%b%Y %H:%M:%S',
    errors='coerce'
)
merged_df['Arrival_Charleroi'] = pd.to_datetime(
    merged_df['REAL_DATE_ARR'].astype(str) + ' ' + merged_df['REAL_TIME_ARR'].astype(str),
    format='%d%b%Y %H:%M:%S',
    errors='coerce'
)

# 6. Enforce Direction (Namur -> Charleroi)
merged_df = merged_df[merged_df['Departure_Namur'] < merged_df['Arrival_Charleroi']].copy()

# 7. Interpolate the passing time at the FBG sensors
merged_df['Total_Travel_Time'] = merged_df['Arrival_Charleroi'] - merged_df['Departure_Namur']
merged_df['Time_to_Sensor'] = merged_df['Total_Travel_Time'] * DISTANCE_RATIO
merged_df['Estimated_Sensor_Time_Local'] = merged_df['Departure_Namur'] + merged_df['Time_to_Sensor']

# 8. Convert to UTC to match FBG filenames
merged_df['Estimated_UTC'] = merged_df['Estimated_Sensor_Time_Local'] - pd.Timedelta(hours=2)
merged_df['Expected_File_Prefix'] = merged_df['Estimated_UTC'].dt.strftime('%Y%m%dT%H%M')

# 9. Isolate the afternoon window and print
merged_df = merged_df.sort_values(by='Estimated_Sensor_Time_Local')
afternoon_trains = merged_df[(merged_df['Estimated_UTC'].dt.hour >= 14) & (merged_df['Estimated_UTC'].dt.hour < 16)]

output_cols = ['TRAIN_NO', 'RELATION_DIRECTION', 'Estimated_UTC', 'Expected_File_Prefix']
print("\n--- EXACT TRAINS PASSING FBG SENSORS ---")
print(afternoon_trains[output_cols].to_string(index=False))