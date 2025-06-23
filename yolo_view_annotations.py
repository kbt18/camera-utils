import cv2
import os
import numpy as np
from pathlib import Path
import argparse

class YOLOViewer:
    def __init__(self, image_dir, label_dir, class_names=None):
        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir)
        self.class_names = class_names or {}
        self.current_index = 0
        
        # Get all image files
        self.image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        self.image_files = []
        
        for ext in self.image_extensions:
            self.image_files.extend(list(self.image_dir.glob(f'*{ext}')))
            self.image_files.extend(list(self.image_dir.glob(f'*{ext.upper()}')))
        
        self.image_files.sort()
        
        if not self.image_files:
            raise ValueError(f"No image files found in {image_dir}")
        
        print(f"Found {len(self.image_files)} images")
        
    def load_annotations(self, image_path):
        """Load YOLO annotations for an image"""
        label_file = self.label_dir / (image_path.stem + '.txt')
        annotations = []
        
        if label_file.exists():
            with open(label_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        parts = line.split()
                        if len(parts) >= 5:
                            class_id = int(parts[0])
                            x_center = float(parts[1])
                            y_center = float(parts[2])
                            width = float(parts[3])
                            height = float(parts[4])
                            annotations.append((class_id, x_center, y_center, width, height))
        
        return annotations
    
    def yolo_to_bbox(self, x_center, y_center, width, height, img_width, img_height):
        """Convert YOLO format to bounding box coordinates"""
        x_center *= img_width
        y_center *= img_height
        width *= img_width
        height *= img_height
        
        x1 = int(x_center - width / 2)
        y1 = int(y_center - height / 2)
        x2 = int(x_center + width / 2)
        y2 = int(y_center + height / 2)
        
        return x1, y1, x2, y2
    
    def draw_annotations(self, image, annotations):
        """Draw bounding boxes and labels on image"""
        img_height, img_width = image.shape[:2]
        
        # Colors for different classes (BGR format)
        colors = [
            (0, 255, 0),    # Green
            (255, 0, 0),    # Blue
            (0, 0, 255),    # Red
            (255, 255, 0),  # Cyan
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Yellow
            (128, 0, 128),  # Purple
            (255, 165, 0),  # Orange
            (255, 192, 203), # Pink
            (0, 128, 128),  # Teal
        ]
        
        for class_id, x_center, y_center, width, height in annotations:
            # Convert YOLO to bbox coordinates
            x1, y1, x2, y2 = self.yolo_to_bbox(x_center, y_center, width, height, img_width, img_height)
            
            # Get color for this class
            color = colors[class_id % len(colors)]
            
            # Draw bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            
            # Prepare label text
            class_name = self.class_names.get(class_id, f"Class {class_id}")
            label = f"{class_name} ({class_id})"
            
            # Draw label background
            (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(image, (x1, y1 - text_height - baseline), (x1 + text_width, y1), color, -1)
            
            # Draw label text
            cv2.putText(image, label, (x1, y1 - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Draw center point
            center_x = int(x_center * img_width)
            center_y = int(y_center * img_height)
            cv2.circle(image, (center_x, center_y), 3, color, -1)
        
        return image
    
    def display_image(self):
        """Display current image with annotations"""
        if not self.image_files:
            return False
            
        image_path = self.image_files[self.current_index]
        
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Could not load image: {image_path}")
            return False
        
        # Load annotations
        annotations = self.load_annotations(image_path)
        
        # Draw annotations
        annotated_image = self.draw_annotations(image.copy(), annotations)
        
        # Add info text
        info_text = f"Image {self.current_index + 1}/{len(self.image_files)}: {image_path.name}"
        cv2.putText(annotated_image, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        annotation_text = f"Annotations: {len(annotations)}"
        cv2.putText(annotated_image, annotation_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Resize image if too large
        max_height = 800
        if annotated_image.shape[0] > max_height:
            scale = max_height / annotated_image.shape[0]
            new_width = int(annotated_image.shape[1] * scale)
            annotated_image = cv2.resize(annotated_image, (new_width, max_height))
        
        # Display image
        cv2.imshow('YOLO Annotation Viewer', annotated_image)
        
        print(f"\nDisplaying: {image_path.name}")
        print(f"Annotations: {len(annotations)}")
        for i, (class_id, x_center, y_center, width, height) in enumerate(annotations):
            class_name = self.class_names.get(class_id, f"Class {class_id}")
            print(f"  {i+1}. {class_name}: center=({x_center:.3f}, {y_center:.3f}), size=({width:.3f}, {height:.3f})")
        
        return True
    
    def run(self):
        """Main loop for the viewer"""
        print("\nYOLO Annotation Viewer")
        print("Controls:")
        print("  'n' or Right Arrow - Next image")
        print("  'p' or Left Arrow - Previous image")
        print("  'q' or ESC - Quit")
        print("  'r' - Reload current image")
        print("  Space - Next image")
        print("\nPress any key to start...")
        
        cv2.namedWindow('YOLO Annotation Viewer', cv2.WINDOW_AUTOSIZE)
        
        while True:
            if not self.display_image():
                break
                
            key = cv2.waitKey(0) & 0xFF
            
            if key == ord('q') or key == 27:  # 'q' or ESC
                break
            elif key == ord('n') or key == 83:  # 'n' or Right arrow
                self.current_index = (self.current_index + 1) % len(self.image_files)
            elif key == ord('p') or key == 81:  # 'p' or Left arrow
                self.current_index = (self.current_index - 1) % len(self.image_files)
            elif key == ord(' '):  # Space
                self.current_index = (self.current_index + 1) % len(self.image_files)
            elif key == ord('r'):  # Reload
                continue
        
        cv2.destroyAllWindows()

def load_class_names(class_file):
    """Load class names from a file"""
    class_names = {}
    if class_file and os.path.exists(class_file):
        with open(class_file, 'r') as f:
            for i, line in enumerate(f):
                class_names[i] = line.strip()
    return class_names

def main():
    parser = argparse.ArgumentParser(description='YOLO Annotation Viewer')
    parser.add_argument('image_dir', help='Directory containing images')
    parser.add_argument('label_dir', help='Directory containing YOLO label files')
    parser.add_argument('--classes', help='File containing class names (one per line)', default=None)
    
    args = parser.parse_args()
    
    # Validate directories
    if not os.path.exists(args.image_dir):
        print(f"Error: Image directory '{args.image_dir}' does not exist")
        return
    
    if not os.path.exists(args.label_dir):
        print(f"Error: Label directory '{args.label_dir}' does not exist")
        return
    
    # Load class names if provided
    class_names = load_class_names(args.classes)
    
    try:
        viewer = YOLOViewer(args.image_dir, args.label_dir, class_names)
        viewer.run()
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()