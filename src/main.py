import matplotlib.pyplot as plt
import seaborn as sns
from process_data import process_data, np

def plot_saccades_and_heatmap(merged_intervals):
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

    # Set labels and title for the duration heatmap
    ax.set_xlabel('X coordinate on display')
    ax.set_ylabel('Y coordinate on display')
    ax.set_title('Saccade Duration Heatmap')

    # Show a colorbar indicating duration
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=min(durations), vmax=max(durations)))
    sm.set_array([])  # Required for the ScalarMappable
    cbar = plt.colorbar(sm, ax=ax, label='Duration (ms)')

    ax.invert_yaxis()

    # Create a new figure and axis for the location heatmap
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(0, 1)  # Set X-axis limits
    ax.set_ylim(0, 1)  # Set Y-axis limits

    # Create a 2D histogram for saccade locations
    hist, xedges, yedges = np.histogram2d(x_coordinatesL + x_coordinatesR, y_coordinatesL + y_coordinatesR, bins=50, range=[[0, 1], [0, 1]])
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

    # Plot the heatmap for locations
    cax = ax.imshow(hist, extent=extent, origin='lower', cmap='coolwarm')

    # Set labels and title for the location heatmap
    ax.set_xlabel('X coordinate on display')
    ax.set_ylabel('Y coordinate on display')
    ax.set_title('Saccade Locations Heatmap')

    # Add a colorbar
    cbar = plt.colorbar(cax, label='Count')

    ax.invert_yaxis()

    # Display the plot
    plt.show()


def main():
    preprocessed_data_path = r'C:\Users\Thoma\xtech\Data\preprocessed\gaze_data_LEFT&RIGHT.csv'
    processed_data_path = r'C:\Users\Thoma\xtech\Data\processed\data.csv'
    velocity_threshold = 20  # Adjust this threshold as needed

    df = process_data(preprocessed_data_path, processed_data_path, velocity_threshold)
    plot_saccades_and_heatmap(df)

if __name__ == "__main__":
    main()