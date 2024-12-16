import json
import os

def count_dataset_stats(error_file_path, dataset_name):
    # Load the error file
    with open(error_file_path, 'r') as f:
        errors = json.load(f)
    
    # Count invalid images
    num_invalid = len(errors)
    
    # Extract base directory path from first error entry
    if num_invalid > 0:
        first_key = list(errors.keys())[0]
        # Get the path up to /sketchy2pix/dataset_name/
        base_dir = first_key.split(dataset_name)[0] + dataset_name
        
        # Count total directories (each is a datapoint)
        total_datapoints = len([d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))])
    else:
        total_datapoints = 0
    
    return num_invalid, total_datapoints

# List of validation files to process
validation_files = [
    "final_bincheck_validation_errors.json"
]

# Process each file and write results
with open('dataset-final2-stats.txt', 'w') as out_file:
    out_file.write("Dataset Statistics Summary\n")
    out_file.write("=======================\n\n")
    
    for val_file in validation_files:
        if os.path.exists(val_file):
            dataset_name = val_file.split('_')[0]
            num_invalid, total = count_dataset_stats(val_file, dataset_name)
            
            # Write statistics
            out_file.write(f"{dataset_name} Dataset:\n")
            out_file.write(f"  Invalid datapoints: {num_invalid}\n")
            out_file.write(f"  Total datapoints: {total}\n")
            out_file.write(f"  Valid datapoints: {total - num_invalid}\n")
            out_file.write(f"  Validity percentage: {((total - num_invalid) / total * 100):.2f}%\n\n")