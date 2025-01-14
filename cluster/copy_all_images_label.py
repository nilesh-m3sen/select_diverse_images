import shutil
import os

def copy_files_from_dirs(source_dir, destination_dir):

    if os.path.exists(source_dir):
        # Loop through files in the source directory
        for filename in os.listdir(source_dir):
            source_file = os.path.join(source_dir, filename)
            # Check if it's a file
            if os.path.isfile(source_file):
                # Copy the file to the destination directory
                shutil.copy(source_file, destination_dir)
                print(f"Copied: {source_file} to {destination_dir}")
    else:
        print(f"Source directory does not exist: {source_dir}")

# List of directories to copy files from

label_array = [0, 1, 2 , 3 , 4 , 5 ]

base_path = f"E:/jan_13_data/ID/20250112/RGB_selected_label"
destination_dir = f'E:/jan_13_data/ID_mixed'

# Check if the destination directory exists, create it if it doesn't
if not os.path.exists(destination_dir):
    os.makedirs(destination_dir)

# copy images
for label_number in label_array:
    image_dir = f"{base_path}/{label_number}/{label_number}_image/"
    copy_files_from_dirs(image_dir, destination_dir)
    
# copy labels
for label_number in label_array:
    label_dir = f"{base_path}/{label_number}/{label_number}_label/"
    copy_files_from_dirs(label_dir, destination_dir)