import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import tkinter as tk
from tkinter import filedialog
import os

class BinaryImageViewer:
    def __init__(self):
        self.image = None
        self.gray_image = None
        self.fig = None
        self.ax_original = None
        self.ax_binary = None
        self.slider = None
        self.threshold = 127
        
    def load_image(self):
        """Load an image using file dialog"""
        root = tk.Tk()
        root.withdraw()  # Hide the main window
        
        file_path = r"C:\Users\kbtod\git\camera-utils\screenshot_20250623_113559_frame27508.jpg"
        
        if file_path:
            self.image = cv2.imread(file_path)
            if self.image is not None:
                # Convert BGR to RGB for matplotlib display
                self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
                # Convert to grayscale for binary processing
                self.gray_image = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)
                print(f"Image loaded successfully: {os.path.basename(file_path)}")
                return True
            else:
                print("Error: Could not load the image")
                return False
        else:
            print("No file selected")
            return False
    
    def update_binary_image(self, val):
        """Update binary image when slider value changes"""
        self.threshold = int(self.slider.val)
        
        # Create binary image
        _, binary_img = cv2.threshold(self.gray_image, self.threshold, 255, cv2.THRESH_BINARY)
        
        # Update the binary image display
        self.ax_binary.clear()
        self.ax_binary.imshow(binary_img, cmap='gray')
        self.ax_binary.set_title(f'Binary Image (Threshold: {self.threshold})')
        self.ax_binary.axis('off')
        
        # Refresh the display
        self.fig.canvas.draw()
    
    def display_images(self):
        """Display original and binary images side by side with slider"""
        if self.image is None or self.gray_image is None:
            print("No image loaded. Please load an image first.")
            return
        
        # Create figure and subplots
        self.fig, (self.ax_original, self.ax_binary) = plt.subplots(1, 2, figsize=(12, 6))
        
        # Display original image
        self.ax_original.imshow(self.image)
        self.ax_original.set_title('Original Image')
        self.ax_original.axis('off')
        
        # Display initial binary image
        _, binary_img = cv2.threshold(self.gray_image, self.threshold, 255, cv2.THRESH_BINARY)
        self.ax_binary.imshow(binary_img, cmap='gray')
        self.ax_binary.set_title(f'Binary Image (Threshold: {self.threshold})')
        self.ax_binary.axis('off')
        
        # Create slider
        plt.subplots_adjust(bottom=0.2)
        ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03])
        self.slider = Slider(ax_slider, 'Threshold', 0, 255, valinit=self.threshold, valfmt='%d')
        self.slider.on_changed(self.update_binary_image)
        
        plt.tight_layout()
        plt.show()
    
    def run(self):
        """Main function to run the application"""
        print("Binary Image Viewer")
        print("==================")
        
        if self.load_image():
            self.display_images()
        else:
            print("Failed to load image. Exiting.")

def main():
    """Main function with example usage"""
    viewer = BinaryImageViewer()
    viewer.run()

# Alternative function to use with a specific image path
def binary_viewer_with_path(image_path):
    """Create binary viewer with a specific image path"""
    viewer = BinaryImageViewer()
    
    # Load image directly from path
    viewer.image = cv2.imread(image_path)
    if viewer.image is not None:
        viewer.image = cv2.cvtColor(viewer.image, cv2.COLOR_BGR2RGB)
        viewer.gray_image = cv2.cvtColor(viewer.image, cv2.COLOR_RGB2GRAY)
        print(f"Image loaded: {os.path.basename(image_path)}")
        viewer.display_images()
    else:
        print(f"Error: Could not load image from {image_path}")

if __name__ == "__main__":
    # Run the interactive version
    main()
    
    # Example of using with a specific path (uncomment to use):
    # binary_viewer_with_path("path/to/your/image.jpg")