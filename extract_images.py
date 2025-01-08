import os
import shutil
from pathlib import Path

def extract_images():
    # Source directory containing brand folders with images
    source_dir = Path('dateset/original/Images')
    # Destination directory for all images
    dest_dir = Path('dataset/original')  # Changed to dataset instead of dateset
    
    # Create destination directory if it doesn't exist
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    # Walk through all subdirectories
    for brand_dir in source_dir.iterdir():
        if brand_dir.is_dir():  # Process only directories
            print(f"Processing {brand_dir.name}...")
            
            # Process all image files in the brand directory
            for img_file in brand_dir.glob('*.jpg'):
                # Get destination path
                dest_path = dest_dir / img_file.name
                
                # Copy the file
                try:
                    shutil.copy2(img_file, dest_path)
                    print(f"Copied {img_file.name}")
                except Exception as e:
                    print(f"Error copying {img_file.name}: {str(e)}")

if __name__ == '__main__':
    extract_images()
