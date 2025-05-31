import os
import shutil
import re

# Source directory containing all pesticide images
source_folder = "e:/MY PROJECT 2/plant_disease_dataset/Pesticide_Affected 2"

# Define folder ranges based on image filenames
folders = {
    "insecticide": ("pesticide_1.jpg", "pesticide_300.jpg"),
    "fungicide": ("pesticide_301.jpg", "pesticide_600.jpg"),
    "herbicide": ("pesticide_601.jpg", "pesticide_900.jpg"),
    "bactericide": ("pesticide_901.jpg", "pesticide_1200.jpg"),
    "Rodenticide": ("pesticide_1201.jpg", "pesticide_1500.jpg"),
    "Nematicide": ("pesticide_1501.jpg", "pesticide_1800.jpg"),
    "Miticide": ("pesticide_1801.jpg", "pesticide_2000.jpg")
}

# Convert range filenames to integer start and end points
numeric_folders = {
    name: (int(start.split("_")[1].split(".")[0]), int(end.split("_")[1].split(".")[0]))
    for name, (start, end) in folders.items()
}

# Create destination folders if not already present
for folder_name in numeric_folders:
    os.makedirs(os.path.join(source_folder, folder_name), exist_ok=True)

# Process and move files
for file_name in os.listdir(source_folder):
    if file_name.startswith("pesticide_") and file_name.endswith(".jpg"):
        match = re.match(r"pesticide_(\d+)\.jpg", file_name)
        if match:
            img_num = int(match.group(1))
            # Determine the target folder
            for folder_name, (start, end) in numeric_folders.items():
                if start <= img_num <= end:
                    src_path = os.path.join(source_folder, file_name)
                    dest_path = os.path.join(source_folder, folder_name, file_name)
                    shutil.move(src_path, dest_path)
                    print(f"Moved {file_name} to {folder_name}")
                    break
