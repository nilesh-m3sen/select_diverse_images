import os

def count_files_in_directory(directory_path):
    total_files = 0
    # Walk through the directory and count files
    for root, dirs, files in os.walk(directory_path):
        total_files += len(files)
    return total_files

# List of directories to count files in
labeled_no_array = [0, 1, 2, 3, 4, 5]
base_path = f"E:/jan_13_data/HI/20250113/RGB_selected_label/"
for label_number in labeled_no_array: 
    labeled_based_path = f"{base_path}/{label_number}"
    directories = [ f'{labeled_based_path}/{label_number}_label' , f'{labeled_based_path}/{label_number}_bin_used' , f'{labeled_based_path}/{label_number}_image' ]

    # Count and print the total files for each directory
    for directory in directories:
        try:
            total_files = count_files_in_directory(directory)
            print(f"Total files in '{directory}': {total_files}")
        except FileNotFoundError:
            print(f"Directory '{directory}' not found.")
        except PermissionError:
            print(f"Permission denied to access '{directory}'.")
