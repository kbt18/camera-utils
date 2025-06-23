import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from matplotlib.widgets import Button
import tkinter as tk
from tkinter import filedialog
import os

class ImagePixelViewer:
    def __init__(self):
        self.image = None
        self.fig = None
        self.ax = None
        self.original_xlim = None
        self.original_ylim = None
        self.zoom_factor = 0.5
        
    def load_image(self, image_path):
        """Load an image file"""
            
        try:
            self.image = mpimg.imread(image_path)
            print(f"Loaded image: {os.path.basename(image_path)}")
            print(f"Image shape: {self.image.shape}")
            print(f"Image dtype: {self.image.dtype}")
            return True
        except Exception as e:
            print(f"Error loading image: {e}")
            return False
    
    def on_click(self, event):
        """Handle mouse clicks on the image"""
        if event.inaxes != self.ax or self.image is None:
            return
            
        # Get click coordinates
        x, y = int(event.xdata), int(event.ydata)
        
        # Check if coordinates are within image bounds
        if (0 <= x < self.image.shape[1] and 0 <= y < self.image.shape[0]):
            # Get pixel intensity
            if len(self.image.shape) == 3:  # Color image
                pixel_value = self.image[y, x]
                if self.image.dtype == np.uint8:
                    intensity_text = f"RGB: ({pixel_value[0]}, {pixel_value[1]}, {pixel_value[2]})"
                else:
                    intensity_text = f"RGB: ({pixel_value[0]:.3f}, {pixel_value[1]:.3f}, {pixel_value[2]:.3f})"
            else:  # Grayscale image
                pixel_value = self.image[y, x]
                if self.image.dtype == np.uint8:
                    intensity_text = f"Intensity: {pixel_value}"
                else:
                    intensity_text = f"Intensity: {pixel_value:.3f}"
            
            print(f"Pixel at ({x}, {y}): {intensity_text}")
            
            # Update the title with pixel information
            self.ax.set_title(f"Click pixel info - ({x}, {y}): {intensity_text}")
            self.fig.canvas.draw()
    
    def zoom_in(self, event):
        """Zoom in to the image"""
        if self.ax is None:
            return
            
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        
        # Calculate new limits (zoom in by zoom_factor)
        x_center = (xlim[0] + xlim[1]) / 2
        y_center = (ylim[0] + ylim[1]) / 2
        x_range = (xlim[1] - xlim[0]) * self.zoom_factor
        y_range = (ylim[1] - ylim[0]) * self.zoom_factor
        
        new_xlim = [x_center - x_range/2, x_center + x_range/2]
        new_ylim = [y_center - y_range/2, y_center + y_range/2]
        
        self.ax.set_xlim(new_xlim)
        self.ax.set_ylim(new_ylim)
        self.fig.canvas.draw()
    
    def zoom_out(self, event):
        """Zoom out of the image"""
        if self.ax is None:
            return
            
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        
        # Calculate new limits (zoom out by 1/zoom_factor)
        x_center = (xlim[0] + xlim[1]) / 2
        y_center = (ylim[0] + ylim[1]) / 2
        x_range = (xlim[1] - xlim[0]) / self.zoom_factor
        y_range = (ylim[1] - ylim[0]) / self.zoom_factor
        
        new_xlim = [x_center - x_range/2, x_center + x_range/2]
        new_ylim = [y_center - y_range/2, y_center + y_range/2]
        
        # Don't zoom out beyond original image bounds
        if self.original_xlim and self.original_ylim:
            new_xlim[0] = max(new_xlim[0], self.original_xlim[0])
            new_xlim[1] = min(new_xlim[1], self.original_xlim[1])
            new_ylim[0] = max(new_ylim[0], self.original_ylim[0])
            new_ylim[1] = min(new_ylim[1], self.original_ylim[1])
        
        self.ax.set_xlim(new_xlim)
        self.ax.set_ylim(new_ylim)
        self.fig.canvas.draw()
    
    def reset_view(self, event):
        """Reset to original view"""
        if self.ax is None or self.original_xlim is None:
            return
            
        self.ax.set_xlim(self.original_xlim)
        self.ax.set_ylim(self.original_ylim)
        self.ax.set_title("Click on pixels to see intensity values")
        self.fig.canvas.draw()
    
    def show_image(self):
        """Display the image with interactive features"""
        if self.image is None:
            print("No image loaded. Please load an image first.")
            return
        
        # Create figure and axis
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        
        # Display image
        if len(self.image.shape) == 3:
            self.ax.imshow(self.image)
        else:
            self.ax.imshow(self.image, cmap='gray')
        
        # Store original view limits
        self.original_xlim = self.ax.get_xlim()
        self.original_ylim = self.ax.get_ylim()
        
        # Set up the interface
        self.ax.set_title("Click on pixels to see intensity values")
        
        # Add buttons
        ax_zoom_in = plt.axes([0.02, 0.7, 0.08, 0.04])
        ax_zoom_out = plt.axes([0.02, 0.65, 0.08, 0.04])
        ax_reset = plt.axes([0.02, 0.6, 0.08, 0.04])
        
        btn_zoom_in = Button(ax_zoom_in, 'Zoom In')
        btn_zoom_out = Button(ax_zoom_out, 'Zoom Out')
        btn_reset = Button(ax_reset, 'Reset')
        
        # Connect button events
        btn_zoom_in.on_clicked(self.zoom_in)
        btn_zoom_out.on_clicked(self.zoom_out)
        btn_reset.on_clicked(self.reset_view)
        
        # Connect mouse click event
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        
        # Add instructions
        plt.figtext(0.02, 0.55, "Instructions:", fontsize=10, weight='bold')
        plt.figtext(0.02, 0.52, "• Click on image to see pixel values", fontsize=9)
        plt.figtext(0.02, 0.49, "• Use buttons to zoom in/out", fontsize=9)
        plt.figtext(0.02, 0.46, "• Mouse wheel also zooms", fontsize=9)
        plt.figtext(0.02, 0.43, "• Values printed in console", fontsize=9)
        
        # Enable mouse wheel zoom
        def on_scroll(event):
            if event.inaxes != self.ax:
                return
            if event.button == 'up':
                self.zoom_in(event)
            elif event.button == 'down':
                self.zoom_out(event)
        
        self.fig.canvas.mpl_connect('scroll_event', on_scroll)
        
        plt.tight_layout()
        plt.show()

def main():
    """Main function to run the image pixel viewer"""
    viewer = ImagePixelViewer()
    
    print("Image Pixel Intensity Viewer")
    print("=" * 30)
    
    # Load image
    if viewer.load_image(r"C:\Users\kbtod\git\camera-utils\screenshot_20250623_120452_frame5200.jpg"):
        print("\nImage loaded successfully!")
        print("Opening interactive viewer...")
        viewer.show_image()
    else:
        print("Failed to load image.")

if __name__ == "__main__":
    main()