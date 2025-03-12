import numpy as np
import os
import matplotlib.pyplot as plt
import re
import datetime

# Directory containing the .npy files
directory = "./npy"

# Function to generate the grid report image
def generate_grid_report(directory):
    # List all .npy files
    file_names = [f for f in os.listdir(directory) if f.endswith('.npy')]
    
    # Pattern to match x and y files, e.g., train_x_4x9.npy and train_y_4x9.npy
    pattern = re.compile(r"train_(x|y)_(\d+)x(\d+)_(\d+)\.npy")
    
    # Create a dictionary to store file pairs by their shape
    file_pairs = {}

    for file in file_names:
        match = pattern.match(file)
        if match:
            file_type = match.group(1)  # x or y
            x_dim = int(match.group(2))
            y_dim = int(match.group(3))
            shape = (x_dim, y_dim)

            # Add file to the dictionary by its shape
            if shape not in file_pairs:
                file_pairs[shape] = {"x": None, "y": None}

            # Store the file paths in the dictionary
            if file_type == "x":
                file_pairs[shape]["x"] = file
            elif file_type == "y":
                file_pairs[shape]["y"] = file

    # Determine grid size based on the number of file pairs
    num_files = len(file_pairs)
    grid_cols = 4  # Number of columns in the grid
    grid_rows = (num_files // grid_cols) + (num_files % grid_cols > 0)  # Calculate rows needed

    # Create a figure with subplots in grid format
    fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(16, grid_rows * 3))
    axes = axes.flatten()  # Flatten the axes array to easily iterate through

    # Loop over each file pair (x, y) and fill the grid cells
    for idx, (shape, files) in enumerate(file_pairs.items()):
        x_file = files["x"]
        y_file = files["y"]

        # Load x and y data if files exist
        if x_file and y_file:
            x_file_path = os.path.join(directory, x_file)
            y_file_path = os.path.join(directory, y_file)

            print(f"Loading files: x = {x_file_path}, y = {y_file_path}")  # Debugging line

            x_data = np.load(x_file_path)
            y_data = np.load(y_file_path)

            # Print data shapes to check if they are loaded correctly
            print(f"x_data shape: {x_data.shape}, y_data shape: {y_data.shape}")  # Debugging line

            x_max_value = np.max(x_data)
            x_min_value = np.min(x_data)
            y_max_value = np.max(y_data)
            y_min_value = np.min(y_data)

            # Display the file information in the grid
            ax = axes[idx]
            ax.axis('off')  # Turn off the axis

            # Display x and y file information
            ax.text(0.5, 0.9, f"X: {x_file}", ha='center', va='center', fontsize=12, fontweight='bold')
            ax.text(0.5, 0.75, f"Y: {y_file}", ha='center', va='center', fontsize=12, fontweight='bold')
            ax.text(0.5, 0.6, f"Shape(X): {x_data.shape} | Shape(Y): {y_data.shape}", ha='center', va='center', fontsize=10)
            ax.text(0.5, 0.45, f"Max(X): {x_max_value} | Max(Y): {y_max_value}", ha='center', va='center', fontsize=10)
            ax.text(0.5, 0.3, f"Min(X): {x_min_value} | Min(Y): {y_min_value}", ha='center', va='center', fontsize=10)
            ax.text(0.5, 0.15, f"Count(X): {x_data.size} | Count(Y): {y_data.size}", ha='center', va='center', fontsize=10)
        else:
            ax = axes[idx]
            ax.axis('off')
            ax.text(0.5, 0.5, "X or Y file missing", ha='center', va='center', fontsize=12)


    # Remove unused axes (when there are more axes than files)
    for idx in range(num_files, len(axes)):
        axes[idx].axis('off')

    # Adjust layout for better spacing
    plt.tight_layout()
    

    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    # Create the npy_report directory if it doesn't exist
    if not os.path.exists('./npy_report'):
        os.makedirs('./npy_report')

    plt.savefig(f'./npy_report/npy_report_{timestamp}.png', dpi=300)
    
    plt.close()

# Generate and save the grid report image
generate_grid_report(directory)
