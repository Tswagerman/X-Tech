# Eye Tracking Data Analysis For early Glaucoma Detection

This Python script is designed for analyzing eye tracking data obtained from gaze tracking devices. The script performs various data processing and visualization tasks to gain insights into eye movement patterns, classify saccades and fixations, and generate visualizations. The goal is to detect early stage glaucoma.

## Getting Started

Before running the code, make sure you have the required libraries installed. You can install them using pip: pip install -r requirements.txt


## Usage

### 1. Data Preprocessing

- Ensure you have your gaze data in a CSV file. Update the `preprocessed_data_path` variable in `main.py` with the path to your CSV file.

- Run the script to perform the initial data preprocessing. The script reads the CSV file, removes rows with 'nan' values, and calculates angular velocities to classify saccades and fixations.

```console
python main.py
```

### 2. Saccade Classification

- The saccade classification function is defined in process_data.py. You can use this function independently as follows:

from data_process import classify_saccade_or_fixation
Load the preprocessed DataFrame (ensure data is already preprocessed)

df = load_preprocessed_data()
Define a velocity threshold to classify saccades

velocity_threshold = 20
Call the classification function

df = classify_saccade_or_fixation(df, velocity_threshold)

### 3. Data Visualization

The script will generate and display visualizations showing saccade duration heatmaps, scatter plots of saccade locations, and other relevant insights.

## File Structure

    main.py: Main script for executing the analysis and generating visualizations.
    process_data.py: Contains functions for data preprocessing.

## Requirements

You can install the required libraries by running:
```bash
pip install -r requirements.txt
```
