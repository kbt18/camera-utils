import os
import shutil
import random
from pathlib import Path

def split_dataset(images_dir, labels_dir, output_dir="dataset", train_ratio=0.8, random_seed=42):
    """
    Split images and labels into train/test directories with shuffling.
    
    Args:
        images_dir (str): Path to images directory
        labels_dir (str): Path to labels directory
        output_dir (str): Output directory for train/test splits
        train_ratio (float): Ratio for training data (0.8 = 80%)
        random_seed (int): Random seed for reproducibility
    """
    
    # Set random seed for reproducibility
    random.seed(random_seed)
    
    # Convert to Path objects
    images_path = Path(images_dir)
    labels_path = Path(labels_dir)
    output_path = Path(output_dir)
    
    # Check if directories exist
    if not images_path.exists():
        raise FileNotFoundError(f"Images directory '{images_dir}' not found")
    if not labels_path.exists():
        raise FileNotFoundError(f"Labels directory '{labels_dir}' not found")
    
    # Get list of image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    image_files = [f for f in images_path.iterdir() 
                   if f.is_file() and f.suffix.lower() in image_extensions]
    
    if not image_files:
        raise ValueError(f"No image files found in '{images_dir}'")
    
    # Create output directories
    train_images_dir = output_path / "train" / "images"
    train_labels_dir = output_path / "train" / "labels"
    test_images_dir = output_path / "test" / "images"
    test_labels_dir = output_path / "test" / "labels"
    
    for dir_path in [train_images_dir, train_labels_dir, test_images_dir, test_labels_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Shuffle the image files
    random.shuffle(image_files)
    
    # Calculate split point
    train_count = int(len(image_files) * train_ratio)
    
    # Split files
    train_files = image_files[:train_count]
    test_files = image_files[train_count:]
    
    print(f"Total files: {len(image_files)}")
    print(f"Training files: {len(train_files)}")
    print(f"Testing files: {len(test_files)}")
    
    # Copy files to respective directories
    def copy_files(file_list, img_dest, label_dest):
        copied_images = 0
        copied_labels = 0
        missing_labels = []
        
        for img_file in file_list:
            # Copy image
            shutil.copy2(img_file, img_dest / img_file.name)
            copied_images += 1
            
            # Find corresponding label file
            # Try common label extensions
            label_extensions = ['.txt', '.xml', '.json']
            label_file = None
            
            for ext in label_extensions:
                potential_label = labels_path / (img_file.stem + ext)
                if potential_label.exists():
                    label_file = potential_label
                    break
            
            if label_file:
                shutil.copy2(label_file, label_dest / label_file.name)
                copied_labels += 1
            else:
                missing_labels.append(img_file.name)
        
        return copied_images, copied_labels, missing_labels
    
    # Copy training files
    print("\nCopying training files...")
    train_img_count, train_label_count, train_missing = copy_files(
        train_files, train_images_dir, train_labels_dir
    )
    
    # Copy testing files
    print("Copying testing files...")
    test_img_count, test_label_count, test_missing = copy_files(
        test_files, test_images_dir, test_labels_dir
    )
    
    # Print summary
    print(f"\n{'='*50}")
    print("SPLIT SUMMARY")
    print(f"{'='*50}")
    print(f"Train - Images: {train_img_count}, Labels: {train_label_count}")
    print(f"Test  - Images: {test_img_count}, Labels: {test_label_count}")
    
    if train_missing:
        print(f"\nMissing labels in training set: {len(train_missing)}")
        for missing in train_missing[:5]:  # Show first 5
            print(f"  - {missing}")
        if len(train_missing) > 5:
            print(f"  ... and {len(train_missing) - 5} more")
    
    if test_missing:
        print(f"\nMissing labels in test set: {len(test_missing)}")
        for missing in test_missing[:5]:  # Show first 5
            print(f"  - {missing}")
        if len(test_missing) > 5:
            print(f"  ... and {len(test_missing) - 5} more")
    
    print(f"\nDataset split completed! Files saved to '{output_dir}'")

if __name__ == "__main__":
    # Example usage
    images_directory = r"C:\Users\kbtod\datasets\drone_blob_synthetic\drones_dataset\images"
    labels_directory = r"C:\Users\kbtod\datasets\drone_blob_synthetic\drones_dataset\labels"
    
    try:
        split_dataset(
            images_dir=images_directory,
            labels_dir=labels_directory,
            output_dir="dataset",
            train_ratio=0.8,
            random_seed=42
        )
    except Exception as e:
        print(f"Error: {e}")