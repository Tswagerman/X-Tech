import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
import seaborn as sns

pd.options.mode.chained_assignment = None  # Suppress warnings for false positives in this case

preprocessed_data_path = r'C:\Users\Thoma\xtech\Data\preprocessed\gaze_data3_Thomas.csv'
processed_data_path = r'C:\Users\Thoma\xtech\Data\processed\data.csv'

def merge_intervals(df):
    combined_intervals = []
    current_interval = None
    end = 0
    for index, row in df.iterrows():
        if current_interval is None:
            current_interval = row
            current_interval['angular_velocities'] = [row['angular_velocity']]  # Initialize the list
            current_interval['positionLeft'] = [[row['left_gaze_point_on_display_area_x'], row['left_gaze_point_on_display_area_y']]]
            current_interval['positionRight'] = [[row['right_gaze_point_on_display_area_x'], row['right_gaze_point_on_display_area_y']]]
            current_interval['start_time'] = row['device_time_stamp']
        end = row['device_time_stamp']
        if index == 0:
            time_diff = row['device_time_stamp']
        else:
            time_diff = row['device_time_stamp'] - end
        if time_diff <= 20 and row['type'] == current_interval['type']:
            current_interval['angular_velocities'].append(row['angular_velocity'])
            current_interval['positionLeft'].append([row['left_gaze_point_on_display_area_x'], row['left_gaze_point_on_display_area_y']])
            current_interval['positionRight'].append([row['right_gaze_point_on_display_area_x'], row['right_gaze_point_on_display_area_y']])
            current_interval['end_time'] = end
        else:
            current_interval['end_time'] = row['device_time_stamp']
            current_interval['duration'] = current_interval['end_time'] - current_interval['start_time']
            current_interval['size'] = len(current_interval['angular_velocities'])
            combined_intervals.append(current_interval)
            # Current interval is done, start a new one
            current_interval = None

    combined_intervals_df = pd.DataFrame(combined_intervals)
    return combined_intervals_df

# Function to calculate angular velocity between two gaze origin points
def calculate_angular_velocity(gaze_origin1, gaze_origin2, time1, time2):
    delta_time = (time2 - time1) / 1000 # convert microseconds to seconds
    rotation1 = Rotation.from_euler('xyz', gaze_origin1, degrees=True)
    rotation2 = Rotation.from_euler('xyz', gaze_origin2, degrees=True)
    delta_rotation = rotation1.inv() * rotation2
    angular_velocity = delta_rotation.magnitude() / delta_time
    return angular_velocity * (180 / np.pi)  # Convert radians to degrees

# Function to classify saccades or fixations based on angular velocity
def classify_saccade_or_fixation(df, velocity_threshold):
    df = df.copy()  # Create a copy of the DataFrame
    df['type'] = 'none'
    df['angular_velocity'] = 0.0
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
df = pd.read_csv(preprocessed_data_path)
# Remove rows with 'nan' values
df = df.dropna()
print("length df after dropping rows containing NAN = ", len(df))

# Define a velocity threshold to classify saccades
velocity_threshold = 20  # Adjust this threshold as needed

# Call the function to classify saccades and fixations based on gaze origin
df = classify_saccade_or_fixation(df, velocity_threshold)

# Sort the DataFrame by 'device_time_stamp' to ensure it's in time order
df = df.sort_values(by='device_time_stamp')

# Merge intervals
merged_intervals = merge_intervals(df)

# Define the columns to keep
columns_to_keep = ['type', 'start_time', 'end_time', 'duration', 'size', 'angular_velocities', 'positionLeft' ,'positionRight']
merged_intervals = merged_intervals[columns_to_keep]

# Save the sorted, combined DataFrame to a CSV file
merged_intervals.to_csv(processed_data_path, index=False)
print("length df after merging = ", len(merged_intervals))

# Extract the 'position' column as a list of lists
positionsL = merged_intervals[merged_intervals['type'] == 'saccade']['positionLeft'].tolist()
positionsR = merged_intervals[merged_intervals['type'] == 'saccade']['positionRight'].tolist()
print("positionsL = ", positionsL, "positionsR = ", positionsR)
# For the left eye positions
x_coordinatesL = [pos[0][0] for pos in positionsL]
y_coordinatesL = [pos[0][1] for pos in positionsL]
print("x_coordinatesL = ", x_coordinatesL, "y_coordinatesL = ", y_coordinatesL)
# For the right eye positions
x_coordinatesR = [pos[0][0] for pos in positionsR]
y_coordinatesR = [pos[0][1] for pos in positionsR]

# Create a list of saccade durations
durations = merged_intervals[merged_intervals['type'] == 'saccade']['duration'].tolist()

fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlim(0, 1)  # Set X-axis limits
ax.set_ylim(0, 1)  # Set Y-axis limits

for i in range(len(positionsL)):
    xL, yL = zip(*positionsL[i])  # Extract x and y coordinates from the left eye data
    xR, yR = zip(*positionsR[i])  # Extract x and y coordinates from the right eye data

    # Color the lines based on duration
    cmap = plt.get_cmap('coolwarm')
    normalized_duration = (durations[i] - min(durations)) / (max(durations) - min(durations))
    color = cmap(normalized_duration)

    # Plot the lines for the left and right eyes
    ax.plot(xL, yL, color=color, label=f'Saccade {i + 1}')
    ax.plot(xR, yR, color=color)

# Set labels and title
ax.set_xlabel('X coordinate on display')
ax.set_ylabel('Y coordinate on display')
ax.set_title('Saccade Duration Heatmap')

# Show a colorbar indicating duration
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=min(durations), vmax=max(durations)))
sm.set_array([])  # Required for the ScalarMappable
cbar = plt.colorbar(sm, ax=ax, label='Duration (ms)')

# Display the plot
plt.gca().invert_yaxis()

#Heatmap
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlim(0, 1)  # Set X-axis limits
ax.set_ylim(0, 1)  # Set Y-axis limits

# Create a 2D histogram for saccade locations
hist, xedges, yedges = np.histogram2d(x_coordinatesL + x_coordinatesR, y_coordinatesL + y_coordinatesR, bins=50, range=[[0, 1], [0, 1]])
extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

# Plot the heatmap
cax = ax.imshow(hist, extent=extent, origin='lower', cmap='coolwarm')

# Set labels and title
ax.set_xlabel('X coordinate on display')
ax.set_ylabel('Y coordinate on display')
ax.set_title('Saccade Locations Heatmap')

# Add a colorbar
cbar = plt.colorbar(cax, label='Count')

# Display the plot
ax.invert_yaxis()
plt.show()

