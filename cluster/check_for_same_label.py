import os
import shutil

label_array = [0, 1, 2, 3, 4, 5]

for label_number in label_array: 
    base_path = f"E:/jan_13_data/HI/20250113/RGB_selected_label"

    source_directory = f"{base_path}/{label_number}/{label_number}_label"
    destination_directory = f"{base_path}/{label_number}/{label_number}_label_incorrect"

    # Ensure destination folder exists
    if not os.path.exists(destination_directory):
        os.makedirs(destination_directory)

    # Iterate over files in the source directory
    for filename in os.listdir(source_directory):
        # Check if the file is a .txt file
        if filename.endswith('.txt'):
            source_file = os.path.join(source_directory, filename)
            
            # Open the file and read the first character of the content
            with open(source_file, 'r') as file:
                content = file.read()
                
                if not content.startswith(str(label_number)):
                    print('Found something wrong', label_number)
                    destination_file = os.path.join(destination_directory, filename)
                    
                    #Copy the file to the destination folder
                    shutil.copy(source_file, destination_file)
                    print(f"Copied: {filename}")
