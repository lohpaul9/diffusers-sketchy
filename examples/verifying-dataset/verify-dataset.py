import os
import json
from PIL import Image
import numpy as np
from pathlib import Path
from tqdm import tqdm
import concurrent.futures
from functools import partial
import shutil

def load_image_as_array(path):
    with Image.open(path) as img:
        return np.array(img)

def check_image_set(base_path, suffixes=['_0', '_1', '_2', '_3']):
    errors = []
    base_name = os.path.splitext(base_path)[0]
    
    # Check if all images exist and have same size
    images = []
    sizes = []
    for suffix in suffixes:
        img_path = f"{base_name}{suffix}.png"
        if not os.path.exists(img_path):
            errors.append(f"Missing image: {img_path}")
            continue
        try:
            img = Image.open(img_path)
            sizes.append(img.size)
            images.append(img_path)
        except Exception as e:
            errors.append(f"Error loading {img_path}: {str(e)}")
    
    if len(set(sizes)) > 1:
        errors.append(f"Inconsistent image sizes in {base_name}: {sizes}")

    # Check prompt.json in the same directory
    dir_path = os.path.dirname(base_path)
    json_path = os.path.join(dir_path, "prompt.json")
    if not os.path.exists(json_path):
        errors.append(f"Missing prompt.json file in directory: {dir_path}")
    else:
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
                if "edit" not in data:
                    errors.append(f"Missing 'edit' field in {json_path}")
        except Exception as e:
            errors.append(f"Error loading JSON {json_path}: {str(e)}")

    # Check mask validity
    if len(errors) == 0:  # Only check if no previous errors
        source = load_image_as_array(f"{base_name}_0.png")
        target = load_image_as_array(f"{base_name}_1.png")
        mask = load_image_as_array(f"{base_name}_2.png")

        # Convert mask to binary (0 or 1)
        if mask.dtype == np.bool:
            mask_binary = mask
        elif mask.dtype == np.uint8:
            # the mask is not actually binary, and has weird tapering at the side
            mask_binary = (mask > 10).astype(np.float32)
        else:
            raise ValueError("mask is not a boolean or uint8")
        
        # Calculate actual differences
        diff = np.abs(source - target).mean(axis=2)
        diff_binary = (diff > 30).astype(np.float32)  # threshold of 30 for difference
        
        # Compare mask area with actual differences
        mask_area = mask_binary.sum()
        diff_area = diff_binary.sum()
        overlap = (mask_binary * diff_binary).sum()
        
        if mask_area > 0:
            overlap_ratio = overlap / mask_area
            if overlap_ratio < 0.7:  # Less than 40% overlap
                errors.append(f"Low overlap between mask and actual differences: {overlap_ratio:.2f}")
        else:
            raise ValueError(f"Mask area is 0 for {base_name}")

    # Add sketch binary check
    sketch_path = f"{base_name}_3.png"
    if os.path.exists(sketch_path):
        try:
            sketch = load_image_as_array(f"{base_name}_3.png")
            # Check if values are only 0 or 255
            unique_values = np.unique(sketch)
            if len(unique_values) > 2 or not all(v in [0, 255] for v in unique_values):
                errors.append(f"Sketch image {sketch_path} is not properly binarized. Found values: {unique_values}")
        except Exception as e:
            errors.append(f"Error checking sketch binarization {sketch_path}: {str(e)}")
            raise ValueError(f"Error checking sketch binarization {sketch_path}: {str(e)}")

    return errors

def check_directory_image_sizes(dir_path):
    """Check if all PNG images in the directory have the same dimensions."""
    errors = []
    sizes = {}
    
    for file in os.listdir(dir_path):
        if file.endswith('.png'):
            try:
                img_path = os.path.join(dir_path, file)
                with Image.open(img_path) as img:
                    size = img.size
                    sizes[file] = size
            except Exception as e:
                errors.append(f"Error loading {file}: {str(e)}")
    
    if len(set(sizes.values())) > 1:
        errors.append(f"Inconsistent image sizes in directory: {sizes}")
    
    return errors

def validate_directory(dir_name, root_dir):
    """Process a single directory"""
    all_errors = {}
    total_checked = 0
    
    dir_path = os.path.join(root_dir, dir_name)
    if not os.path.isdir(dir_path):
        print(f"Directory {dir_name} does not exist")
        return {}, 0
        
    try:
        # Check if all images in the directory have the same dimensions
        dir_errors = check_directory_image_sizes(dir_path)
        if dir_errors:
            all_errors[dir_path] = dir_errors
            print(f"Found directory size errors in {dir_name}: {dir_errors}")  # Debug output
            raise Exception(f"Directory size check failed: {dir_errors}")
            
        # Look for image sets within each directory
        for file in os.listdir(dir_path):
            if file.endswith('_0.png'):
                base_name = os.path.join(dir_path, file[:-6])  # Remove '_0.png'
                errors = check_image_set(base_name)
                if errors:
                    all_errors[base_name] = errors
                    # print(f"Found errors in image set {base_name}: {errors}")  # Debug output
                    raise Exception(f"Image set check failed: {errors}")
                total_checked += 1
                
        return all_errors, total_checked
        
    except Exception as e:
        # Create error directory if it doesn't exist
        error_dir = os.path.join(os.path.dirname(root_dir), "error_directories")
        os.makedirs(error_dir, exist_ok=True)
        
        # Move problematic directory to error_directories
        error_path = os.path.join(error_dir, dir_name)
        try:
            # print(f"Attempting to move {dir_path} to {error_path}")  # Debug output
            # Remove destination directory if it already exists
            if os.path.exists(error_path):
                print(f"Removing existing directory at {error_path}")  # Debug output
                shutil.rmtree(error_path)
            # Move the problematic directory
            shutil.move(dir_path, error_path)
            # print(f"Successfully moved problematic directory {dir_name} to error_directories due to: {str(e)}")
        except Exception as move_error:
            print(f"Failed to move directory {dir_name}: {str(move_error)}")
            
        return {dir_path: [f"Directory processing error: {str(e)}"]}, 0

def validate_dataset(root_dir):
    print("Starting validation...")
    all_errors = {}
    total_checked = 0
    
    # Get list of directories
    directories = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    directory_len = len(directories)
    print(f"Number of directories: {directory_len}")

    # Create a partial function with root_dir
    validate_dir_partial = partial(validate_directory, root_dir=root_dir)

    # Process directories in parallel
    with concurrent.futures.ProcessPoolExecutor(max_workers=48) as executor:
        futures = []
        for dir_name in directories:
            futures.append(executor.submit(validate_dir_partial, dir_name))
        print(f"Submitted {len(futures)} directories for validation")

        # Process results as they complete
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            try:
                dir_errors, dir_total = future.result()
                # if dir_errors:
                #     print(f"Found errors in directory: {list(dir_errors.keys())[0]}")  # Debug output
                if (i + 1) % 500 == 0:
                    print(f"Processed {i + 1} of {directory_len} directories")
                
                all_errors.update(dir_errors)
                total_checked += dir_total
            except Exception as e:
                print(f"Error processing future {i}: {str(e)}")

    print(f"\nTotal image sets checked: {total_checked}")
    return all_errors

# Main execution
if __name__ == '__main__':
    subdir = "final"
    root_dir = "/bigdrive/datasets/sketchy2pix/" + subdir
    print("Path exists:", os.path.exists(root_dir))
    
    # Create error_directories path and verify permissions
    error_dir = os.path.join(os.path.dirname(root_dir), "error_directories")
    os.makedirs(error_dir, exist_ok=True)
    
    # Verify permissions
    print(f"Root dir permissions: {oct(os.stat(root_dir).st_mode)[-3:]}")
    print(f"Error dir permissions: {oct(os.stat(error_dir).st_mode)[-3:]}")

    errors = validate_dataset(root_dir)

    # Save errors to JSON
    error_output_path = "./" + subdir + "_bincheck_validation_errors.json"
    with open(error_output_path, 'w') as f:
        json.dump(errors, f, indent=2)

    print("Errors saved to", error_output_path)
    # Print summary
    print("\nValidation Summary:")
    print(f"Total image sets with errors: {len(errors)}")

    # check the number of errors due to image "low overlap"
    low_overlap_errors = 0
    for path, error_list in errors.items():
        for error in error_list:
            if "Low overlap" in error:
                low_overlap_errors += 1
    print(f"Number of errors due to low overlap: {low_overlap_errors}")

    # for path, error_list in errors.items():
    #     print(f"\n{path}:")
    #     for error in error_list:
    #         print(f"  - {error}")


    # check_image_set("/bigdrive/datasets/sketchy2pix/final/5668a28d65d0422d/5668a28d65d0422d")
    # check_image_set("/bigdrive/datasets/sketchy2pix/final/468193/468193")
    # print("done")