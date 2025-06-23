import cv2
import numpy as np
import os
import glob
from pathlib import Path

class DroneDatasetCreator:
    def __init__(self, drone_image_path, background_dir, output_dir="drones_dataset"):
        self.drone_image_path = drone_image_path
        self.background_dir = background_dir
        self.output_dir = output_dir
        
        # Create output directories
        self.images_dir = os.path.join(output_dir, "images")
        self.labels_dir = os.path.join(output_dir, "labels")
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.labels_dir, exist_ok=True)
        
        # Load drone image with alpha channel if available
        self.drone_img = cv2.imread(drone_image_path, cv2.IMREAD_UNCHANGED)
        if self.drone_img is None:
            raise ValueError(f"Could not load drone image from {drone_image_path}")
        
        # Print drone image info for debugging
        print(f"Drone image loaded: {drone_image_path}")
        print(f"Drone image shape: {self.drone_img.shape}")
        print(f"Drone image dtype: {self.drone_img.dtype}")
        
        # Get background image paths
        self.bg_paths = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            self.bg_paths.extend(glob.glob(os.path.join(background_dir, ext)))
            self.bg_paths.extend(glob.glob(os.path.join(background_dir, ext.upper())))
        
        if not self.bg_paths:
            raise ValueError(f"No background images found in {background_dir}")
        
        self.current_bg_idx = 0
        self.current_bg = None
        self.current_bg_original = None
        self.image_counter = 0
        self.drone_positions = []
        
        print(f"Loaded {len(self.bg_paths)} background images")
        print("\nControls:")
        print("- Click to place drone")
        print("- Press 'n' for next image")
        print("- Press 'r' to reset current image")
        print("- Press 'q' to quit")
        print("- Press 's' to save current placements and continue")
    
    def resize_drone(self, target_size=None):
        """Return drone image without resizing"""
        return self.drone_img
    
    def blend_drone_on_background(self, bg_img, drone_img, x, y):
        """Blend drone image onto background at specified position"""
        drone_h, drone_w = drone_img.shape[:2]
        bg_h, bg_w = bg_img.shape[:2]
        
        # Ensure drone fits within background
        if x + drone_w > bg_w or y + drone_h > bg_h or x < 0 or y < 0:
            return bg_img
        
        # Create a copy of background
        result = bg_img.copy()
        
        if drone_img.shape[2] == 4:  # Drone has alpha channel
            # Extract alpha channel
            alpha = drone_img[:, :, 3] / 255.0
            drone_rgb = drone_img[:, :, :3]
            
            # Blend using alpha
            for c in range(3):
                result[y:y+drone_h, x:x+drone_w, c] = (
                    alpha * drone_rgb[:, :, c] + 
                    (1 - alpha) * bg_img[y:y+drone_h, x:x+drone_w, c]
                )
        else:
            # Simple overlay without alpha
            result[y:y+drone_h, x:x+drone_w] = drone_img
        
        return result
    
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse clicks"""
        if event == cv2.EVENT_LBUTTONDOWN:
            # Use drone at original size
            drone_resized = self.resize_drone()
            drone_h, drone_w = drone_resized.shape[:2]
            
            # Center drone on click position
            drone_x = max(0, min(x - drone_w // 2, self.current_bg.shape[1] - drone_w))
            drone_y = max(0, min(y - drone_h // 2, self.current_bg.shape[0] - drone_h))
            
            # Add drone to image
            self.current_bg = self.blend_drone_on_background(
                self.current_bg, drone_resized, drone_x, drone_y
            )
            
            # Store position for YOLO format (normalized coordinates)
            bg_h, bg_w = self.current_bg.shape[:2]
            center_x = (drone_x + drone_w / 2) / bg_w
            center_y = (drone_y + drone_h / 2) / bg_h
            width = drone_w / bg_w
            height = drone_h / bg_h
            
            self.drone_positions.append((center_x, center_y, width, height))
            
            print(f"Placed drone at ({x}, {y}). Total drones: {len(self.drone_positions)}")
    
    def save_current_image_and_labels(self):
        """Save current image with drones and corresponding labels"""
        if not self.drone_positions:
            print("No drones placed, nothing to save.")
            return
        
        # Save image
        img_filename = f"drone_{self.image_counter:04d}.jpg"
        img_path = os.path.join(self.images_dir, img_filename)
        cv2.imwrite(img_path, self.current_bg)
        
        # Save labels in YOLO format
        label_filename = f"drone_{self.image_counter:04d}.txt"
        label_path = os.path.join(self.labels_dir, label_filename)
        
        with open(label_path, 'w') as f:
            for pos in self.drone_positions:
                # YOLO format: class_id center_x center_y width height
                # Using class_id = 0 for drone
                f.write(f"0 {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f} {pos[3]:.6f}\n")
        
        print(f"Saved: {img_filename} with {len(self.drone_positions)} drone(s)")
        self.image_counter += 1
    
    def load_next_background(self):
        """Load next background image"""
        if self.current_bg_idx >= len(self.bg_paths):
            print("All background images processed!")
            return False
        
        bg_path = self.bg_paths[self.current_bg_idx]
        self.current_bg_original = cv2.imread(bg_path)
        
        if self.current_bg_original is None:
            print(f"Could not load background image: {bg_path}")
            self.current_bg_idx += 1
            return self.load_next_background()
        
        self.current_bg = self.current_bg_original.copy()
        self.drone_positions = []
        
        print(f"\nLoaded background {self.current_bg_idx + 1}/{len(self.bg_paths)}: {os.path.basename(bg_path)}")
        print(f"Image size: {self.current_bg.shape}")
        return True
    
    def reset_current_image(self):
        """Reset current image to original background"""
        if self.current_bg_original is not None:
            self.current_bg = self.current_bg_original.copy()
            self.drone_positions = []
            print("Reset current image")
    
    def run(self):
        """Main loop"""
        cv2.namedWindow('Drone Dataset Creator', cv2.WINDOW_NORMAL)
        cv2.setMouseCallback('Drone Dataset Creator', self.mouse_callback)
        
        # Load first background
        if not self.load_next_background():
            return
        
        while True:
            # Display current image
            display_img = self.current_bg.copy()
            
            # Add instruction text
            cv2.putText(display_img, f"Image {self.current_bg_idx + 1}/{len(self.bg_paths)} | Drones: {len(self.drone_positions)}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(display_img, "Click to place drone | 'n': next | 'r': reset | 's': save | 'q': quit", 
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow('Drone Dataset Creator', display_img)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('n'):
                # Save current image if drones are placed
                if self.drone_positions:
                    self.save_current_image_and_labels()
                
                # Move to next background
                self.current_bg_idx += 1
                if not self.load_next_background():
                    break
            elif key == ord('r'):
                self.reset_current_image()
            elif key == ord('s'):
                if self.drone_positions:
                    self.save_current_image_and_labels()
                    # Reset for more placements on same background
                    self.reset_current_image()
        
        cv2.destroyAllWindows()
        print(f"\nDataset creation completed! Generated {self.image_counter} images.")
        print(f"Images saved in: {self.images_dir}")
        print(f"Labels saved in: {self.labels_dir}")

def main():
    # Configuration
    drone_image_path = "drone.png"  # Change this to your drone image path
    background_dir = "backgrounds"   # Change this to your background images directory
    output_dir = "drones_dataset"
    
    try:
        creator = DroneDatasetCreator(drone_image_path, background_dir, output_dir)
        creator.run()
    except Exception as e:
        print(f"Error: {e}")
        print("\nMake sure you have:")
        print("1. A drone image file (drone.png)")
        print("2. A directory with background images (backgrounds/)")
        print("3. OpenCV installed: pip install opencv-python")

if __name__ == "__main__":
    main()