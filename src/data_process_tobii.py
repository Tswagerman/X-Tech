import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
import seaborn as sns

pd.options.mode.chained_assignment = None  # Suppress warnings for false positives in this case

def merge_intervals(df):
    combined_intervals = []
    current_interval = None
    end = 0

    for index, row in df.iterrows():
        if current_interval is None:
            current_interval = row
            current_interval['angular_velocities'] = [row['angular_velocity']]  # Initialize the list
            current_interval['start_time'] = row['device_time_stamp']
        else:
            if index == 0:
                time_diff = row['device_time_stamp']
                end = row['device_time_stamp']
            else:
                time_diff = row['device_time_stamp'] - end
                end = row['device_time_stamp']
            if time_diff <= 20 and row['type'] == current_interval['type']:
                current_interval['angular_velocities'].append(row['angular_velocity'])
                current_interval['end_time'] = row['device_time_stamp']
            else:
                current_interval['end_time'] = row['device_time_stamp']
                current_interval['duration'] = current_interval['end_time'] - current_interval['start_time']
                current_interval['data_point_size'] = len(current_interval['angular_velocities'])
                combined_intervals.append(current_interval)
                current_interval = row
                current_interval['angular_velocities'] = [row['angular_velocity']]  # Initialize for the next interval
                current_interval['start_time'] = row['device_time_stamp']

    combined_intervals_df = pd.DataFrame(combined_intervals)
    combined_intervals_df = combined_intervals_df.drop(columns=['device_time_stamp'])

    return combined_intervals_df

def add_extra_columns(df):
    df['duration'] = df['device_time_stamp'].apply(lambda x: x - df['device_time_stamp'].min())
    df['size'] = df.groupby('type').cumcount() + 1
    return df

def calculate_angular_velocity(gaze_origin1, gaze_origin2, time1, time2):
    delta_time = (time2 - time1) / 1000 # Seconds
    rotation1 = Rotation.from_euler('xyz', gaze_origin1, degrees=True)
    rotation2 = Rotation.from_euler('xyz', gaze_origin2, degrees=True)
    delta_rotation = rotation1.inv() * rotation2
    angular_velocity = delta_rotation.magnitude() / delta_time
    return angular_velocity * (180 / np.pi)  # Convert radians to degrees

def classify_saccade_or_fixation(df, velocity_threshold):
    df = df.copy()  # Create a copy of the DataFrame
    df['type'] = 'none'
    df['angular_velocity'] = 0
    start = df['device_time_stamp'].iloc[0]
    df['device_time_stamp'] = (df['device_time_stamp'] - start) / 1000  # Convert microseconds to milliseconds

    for i in range(1, len(df)):
        gaze_origin1 = np.array([df['left_gaze_origin_in_user_x'].iloc[i - 1],
                                 df['left_gaze_origin_in_user_y'].iloc[i - 1],
                                 df['left_gaze_origin_in_user_z'].iloc[i - 1]])

        gaze_origin2 = np.array([df['left_gaze_origin_in_user_x'].iloc[i],
                                 df['left_gaze_origin_in_user_y'].iloc[i],
                                 df['left_gaze_origin_in_user_z'].iloc[i]])

        time1 = df['device_time_stamp'].iloc[i - 1]
        time2 = df['device_time_stamp'].iloc[i]
        if gaze_origin1[0] == gaze_origin2[0] and gaze_origin1[1] == gaze_origin2[1] and gaze_origin1[2] == gaze_origin2[2]:
            angular_velocity = 0
        else:
            angular_velocity = calculate_angular_velocity(gaze_origin1, gaze_origin2, time1, time2)
        df['angular_velocity'].iloc[i] = float(angular_velocity)

        # Differentiate between saccades and fixations based on angular velocity
        df['type'] = np.where(df['angular_velocity'] >= velocity_threshold, 'saccade', 'fixation')

    return df

# Read your gaze data from a CSV file
df = pd.read_csv(r'C:\Users\Thoma\xtech\Data\preprocessed\gaze_databox.csv')
# Remove rows with 'nan' values
df = df.dropna()

# Define a velocity threshold to classify saccades
velocity_threshold = 20  # Adjust this threshold as needed

# Call the function to classify saccades and fixations based on gaze origin
df = classify_saccade_or_fixation(df, velocity_threshold)

# Sort the DataFrame by 'device_time_stamp' to ensure it's in time order
df = df.sort_values(by='device_time_stamp')

# Merge intervals
merged_intervals = merge_intervals(df)

# Define the columns to keep
columns_to_keep = [
    'type', 'start_time', 'end_time', 'duration', 'data_point_size', 'angular_velocities', 'left_gaze_point_on_display_area_x', 'left_gaze_point_on_display_area_y', 'right_gaze_point_on_display_area_x', 'right_gaze_point_on_display_area_y'
]

merged_intervals = merged_intervals[columns_to_keep]

# Save the sorted, combined DataFrame to a CSV file
merged_intervals.to_csv(r'C:\Users\Thoma\xtech\Data\processed\data.csv', index=False)
print("len = ", len(merged_intervals))

# Create a heatmap of saccade positions
plt.figure(figsize=(10, 6))

# Invert the y-axis, coordinate system starts at 0,0 at top 
merged_intervals['left_gaze_point_on_display_area_y'] = 1 - merged_intervals['left_gaze_point_on_display_area_y']
merged_intervals['right_gaze_point_on_display_area_y'] = 1 - merged_intervals['right_gaze_point_on_display_area_y']

# Create the first heatmap
ax1 = sns.kdeplot(data=merged_intervals, x='left_gaze_point_on_display_area_x', y='right_gaze_point_on_display_area_x', fill=True, cmap="YlGnBu", levels=10)

# Create the second heatmap
ax2 = sns.kdeplot(data=merged_intervals, x='right_gaze_point_on_display_area_x', y='right_gaze_point_on_display_area_y', fill=True, cmap="YlGnBu", levels=10)

plt.title("Saccade Heatmap")
plt.xlabel("X coordinate on display")
plt.ylabel("Y coordinate on display")

# Set the axis limits to a maximum of 1. The display area is 1x1. Everything outside that is irrelevant for now
ax1.set_xlim(0, 1)
ax1.set_ylim(0, 1)
ax2.set_xlim(0, 1)
ax2.set_ylim(0, 1)

# Add a colorbar and label it
cbar1 = plt.colorbar(ax1.collections[0], label='Density')
cbar2 = plt.colorbar(ax2.collections[0], label='Density')

plt.show()

# Create a bar chart of saccade durations
'''plt.figure(figsize=(10, 6))
sns.barplot(data=merged_intervals, x='duration', y=merged_intervals.index, orient='h', palette='Blues')
plt.xlabel('Duration (ms)')
plt.ylabel('Saccade Index')
plt.title("Saccade Durations")
plt.show()'''
