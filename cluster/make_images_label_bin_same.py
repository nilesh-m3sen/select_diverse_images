
import os
import shutil


labeled_no_array = [0, 1, 2, 3, 4, 5]

for label_number in labeled_no_array:
    base_path = f"E:/jan_13_data/HI/20250113/RGB_selected_label/{label_number}"

    text_dir = f'{base_path}/{label_number}_label' 
    bin_dir = f'E:/jan_13_data/HI/20250113/F' 
    target_dir = f'{base_path}/{label_number}_bin_used' 

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        print(f"Created target directory: {target_dir}")

    text_files = {os.path.splitext(file)[0] for file in os.listdir(text_dir) if os.path.isfile(os.path.join(text_dir, file))}

    image_files = {os.path.splitext(file)[0] for file in os.listdir(bin_dir) if os.path.isfile(os.path.join(bin_dir, file))}

    matching_files = image_files & text_files

    for file in matching_files:
        bin_path = os.path.join(bin_dir, file + '.bin') 
        target_path = os.path.join(target_dir, file + '.bin')  
        if os.path.exists(bin_path):
            shutil.copy(bin_path, target_path)
            print(f"Copied {bin_path} to {target_path}")
        else:
            print(f"Image {bin_path} does not exist.")
