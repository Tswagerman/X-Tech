import numpy as np
import matplotlib.pyplot as plt


def create_dummy_gaze_data(num_points=1000):
    """
    Create dummy gaze data for demonstration purposes.

    Args:
        num_points (int): Number of data points to generate.

    Returns:
        List of (x, y) gaze data points.
    """
    time = np.arange(num_points)
    gaze_x = 400 + 100 * np.sin(0.1 * time) + np.random.normal(0, 10, num_points)
    gaze_y = 300 + 50 * np.sin(0.2 * time) + np.random.normal(0, 5, num_points)
    gaze_data = list(zip(gaze_x, gaze_y))
    return gaze_data

def detect_saccades_fixations(gaze_data, velocity_threshold=30.0, min_duration=20):
    """
    Detect saccades and fixations in gaze data.

    Args:
        gaze_data (list): List of gaze data points over time, where each point is a tuple of (x, y) coordinates.
        velocity_threshold (float): Threshold for saccade detection (degrees/second).
        min_duration (int): Minimum duration (in data points) for a fixation.

    Returns:
        List of tuples representing saccades and fixations.
        Each tuple has the format (start_index, end_index, type), where:
        - start_index: Index of the data point where the event starts.
        - end_index: Index of the data point where the event ends.
        - type: 'saccade' or 'fixation'.
    """
    events = []
    velocity_data = []

    # Calculate velocity for each gaze point
    for i in range(1, len(gaze_data)):
        delta_x = gaze_data[i][0] - gaze_data[i - 1][0]
        delta_y = gaze_data[i][1] - gaze_data[i - 1][1]
        time_diff = 1.0  # Assuming a fixed time interval (you may adjust this)

        # Calculate angular velocity (degrees per second)
        velocity = np.sqrt((delta_x ** 2 + delta_y ** 2) / time_diff)

        velocity_data.append(velocity)

    in_fixation = False
    fixation_start = 0

    for i, velocity in enumerate(velocity_data):
        if not in_fixation and velocity <= velocity_threshold:
            fixation_start = i
            in_fixation = True
        elif in_fixation and velocity > velocity_threshold:
            fixation_end = i - 1
            fixation_duration = fixation_end - fixation_start + 1

            if fixation_duration >= min_duration:
                events.append((fixation_start, fixation_end, 'fixation'))
            else:
                events.append((fixation_start, fixation_end, 'saccade'))

            in_fixation = False

    # Check if the last event is a fixation
    if in_fixation:
        fixation_end = len(velocity_data) - 1
        fixation_duration = fixation_end - fixation_start + 1

        if fixation_duration >= min_duration:
            events.append((fixation_start, fixation_end, 'fixation'))
        else:
            events.append((fixation_start, fixation_end, 'saccade'))

    return events

def main():
    # Generate dummy gaze data
    num_points = 1000
    gaze_data = create_dummy_gaze_data(num_points)

    # Plot the simulated gaze data
    plt.figure(figsize=(8, 6))
    plt.plot(*zip(*gaze_data), label='Gaze Data', marker='o', linestyle='-', markersize=5)
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('Simulated Gaze Data')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Detect saccades and fixations
    events = detect_saccades_fixations(gaze_data)

    # Print detected events
    for event in events:
        start_idx, end_idx, event_type = event
        print(f"Event Type: {event_type}, Start Index: {start_idx}, End Index: {end_idx}")

if __name__ == "__main__":
    main()