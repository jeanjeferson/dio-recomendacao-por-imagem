import os
import random
import shutil

# Define paths
source_dir = 'dateset/original'
train_dir = 'dateset/train' 
test_dir = 'dateset/test'

# Create train/test directories if they don't exist
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Get all image files
image_files = [f for f in os.listdir(source_dir) if f.endswith('.jpg')]

# Shuffle images randomly
random.shuffle(image_files)

# Calculate split index (80% train, 20% test)
split_index = int(len(image_files) * 0.8)

# Split into train and test sets
train_files = image_files[:split_index]
test_files = image_files[split_index:]

# Move files to respective directories
for file in train_files:
    shutil.move(os.path.join(source_dir, file), os.path.join(train_dir, file))
    
for file in test_files:
    shutil.move(os.path.join(source_dir, file), os.path.join(test_dir, file))

print(f"Moved {len(train_files)} images to train directory")
print(f"Moved {len(test_files)} images to test directory")
