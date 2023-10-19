# Eye Tracking Data Analysis For early Glaucoma Detection

This Python script is designed for analyzing eye tracking data obtained from gaze tracking devices. The script performs various data processing and visualization tasks to gain insights into eye movement patterns, classify saccades and fixations, and generate visualizations. The goal is to detect early stage glaucoma.

Early glaucoma detection using saccades is based on the premise that individuals with glaucoma may exhibit specific patterns of eye movement, particularly during saccades (rapid eye movements). Glaucoma is a progressive eye condition characterized by damage to the optic nerve, often resulting in vision loss. Research has shown that individuals with glaucoma may have alterations in their saccade characteristics, such as saccade velocity and accuracy.

By analyzing saccades, we can potentially detect subtle changes in eye movement patterns that are indicative of early-stage glaucoma, even before more obvious vision impairments occur. The rationale is that early detection allows for early intervention and treatment, which can help preserve a patient's vision and quality of life. Therefore, this code focuses on saccade analysis as a potential tool for early glaucoma detection, as it may reveal important insights into the disease's progression.

## Getting Started

Before running the code, make sure you have the required libraries installed. You can install them using pip: 

```bash
pip install -r requirements.txt
```

## Usage

### 1. Data Preprocessing

- Ensure you have your gaze data in a CSV file. Update the `preprocessed_data_path` variable in `main.py` with the path to your CSV file.

- Run the script to perform the initial data preprocessing. The script reads the CSV file, removes rows with 'nan' values, and calculates angular velocities to classify saccades and fixations. As the final step in data preprocessing, the script merges subsequent data points that correspond to the same saccades or fixations. This merging process groups together related data points, creating a more comprehensive and organized dataset for subsequent analysis and visualization

```console
python main.py
```

### 2. Saccade Classification

- Saccade classification is the process of identifying rapid eye movements (saccades) and stable eye positions (fixations). This is typically done by setting thresholds for criteria like angular velocity, duration, acceleration, and spatial displacement. Accurate classification helps understand eye movement patterns for applications like eye condition diagnosis, visual attention studies, and human-computer interaction.

-The classification in thius code is based on angular velocity, with a user-defined threshold. The script calculates the angular velocities between consecutive gaze origin points and uses this information to differentiate between saccades (rapid eye movements) and fixations (stable eye positions). The result is a more structured dataset that groups similar eye movements together.

### 3. Data Visualization

The script will generate and display visualizations

## File Structure

    main.py: Main script for executing the analysis and generating visualizations.
    process_data.py: Contains functions for data preprocessing.


