import csv 
import math


filename = "Timing_info_camshift.csv"

# Open the file in read mode
with open(filename, mode='r') as file:
    reader = csv.reader(file)
    
    # Skip the header row
    next(reader)
    
    # Initialize the total runtime
    total_runtime = 0
    video_count = 0
    total_iterations = 0
    
    # Iterate over the rows in the file
    for row in reader:
        # Extract the runtime from the row
        runtime = float(row[1])
        iterations = int(row[2])
        # Add the runtime to the total runtime
        total_runtime += runtime
        total_iterations += iterations

# Calculate the average runtime
average_runtime = total_runtime / total_iterations
print(f"File: {filename}, Average Runtime: {average_runtime:.3f} seconds, FPS: {1/average_runtime:.2f}")

