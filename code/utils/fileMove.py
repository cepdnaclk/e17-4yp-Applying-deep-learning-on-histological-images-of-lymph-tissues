import os
import shutil

def move_images(source_path, destination_path):
    # Ensure the destination directory exists
    os.makedirs(destination_path, exist_ok=True)

    # Walk through the source directory and its subdirectories
    for root, _, files in os.walk(source_path):
        for file in files:
            # Check if the file is an image (you can add more image extensions if needed)
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.gif')):
                source_file_path = os.path.join(root, file)
                destination_file_path = os.path.join(destination_path, file)

                # Move the image file
                shutil.move(source_file_path, destination_file_path)
                print(f"Moved: {source_file_path} -> {destination_file_path}")

# Example usage:
source_path = '/home/e17358/4yp/MedNET/Breast Histopathology Images/IDC_regular_ps50_idx5/'
destination_path = '/home/e17358/4yp/MedNET/medicalimages/'
move_images(source_path, destination_path)
