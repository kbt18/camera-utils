import cv2
import os
import numpy as np
from pathlib import Path

class ImageCropper:
    def __init__(self, input_dir, crop_size=640):
        self.input_dir = Path(input_dir)
        self.crop_dir = Path("crop")
        self.crop_size = crop_size
        self.current_image = None
        self.current_filename = None
        self.image_files = []
        self.current_index = 0
        
        # Create crop directory if it doesn't exist
        self.crop_dir.mkdir(exist_ok=True)
        
        # Get all image files
        self.get_image_files()
        
    def get_image_files(self):
        """Get all image files from the input directory"""
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
        for ext in extensions:
            self.image_files.extend(self.input_dir.glob(f'*{ext}'))
            self.image_files.extend(self.input_dir.glob(f'*{ext.upper()}'))
        
        if not self.image_files:
            print(f"No image files found in {self.input_dir}")
            return False
        
        print(f"Found {len(self.image_files)} image files")
        return True
    
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse clicks"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.crop_image(x, y)
    
    def crop_image(self, click_x, click_y):
        """Crop 640x640 area around the click point"""
        if self.current_image is None:
            return
        
        h, w = self.current_image.shape[:2]
        half_crop = self.crop_size // 2
        
        # Calculate crop boundaries, ensuring we stay within image bounds
        start_x = max(0, min(click_x - half_crop, w - self.crop_size))
        start_y = max(0, min(click_y - half_crop, h - self.crop_size))
        end_x = start_x + self.crop_size
        end_y = start_y + self.crop_size
        
        # If image is smaller than crop size, pad with black
        if w < self.crop_size or h < self.crop_size:
            padded_image = np.zeros((max(h, self.crop_size), max(w, self.crop_size), 3), dtype=np.uint8)
            padded_image[:h, :w] = self.current_image
            cropped = padded_image[start_y:end_y, start_x:end_x]
        else:
            cropped = self.current_image[start_y:end_y, start_x:end_x]
        
        # Generate output filename
        base_name = self.current_filename.stem
        crop_filename = self.crop_dir / f"{base_name}_crop_{click_x}_{click_y}.jpg"
        
        # Save the cropped image
        cv2.imwrite(str(crop_filename), cropped)
        print(f"Saved crop: {crop_filename}")
        
        # Show a preview of the cropped area on the original image
        self.show_crop_preview(start_x, start_y, end_x, end_y)
    
    def show_crop_preview(self, start_x, start_y, end_x, end_y):
        """Show a rectangle preview of the cropped area"""
        preview_image = self.current_image.copy()
        cv2.rectangle(preview_image, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)
        
        # Scale image to fit screen if it's too large
        display_image = self.scale_image_for_display(preview_image)
        cv2.imshow('Image Cropper', display_image)
        cv2.waitKey(500)  # Show preview for 500ms
        
        # Show original image again
        display_image = self.scale_image_for_display(self.current_image)
        cv2.imshow('Image Cropper', display_image)
    
    def scale_image_for_display(self, image):
        """Scale image to fit screen while maintaining aspect ratio"""
        max_height = 800
        max_width = 1200
        
        h, w = image.shape[:2]
        
        if h <= max_height and w <= max_width:
            return image
        
        # Calculate scaling factor
        scale_h = max_height / h
        scale_w = max_width / w
        scale = min(scale_h, scale_w)
        
        new_width = int(w * scale)
        new_height = int(h * scale)
        
        return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    
    def load_current_image(self):
        """Load the current image"""
        if self.current_index >= len(self.image_files):
            return False
        
        self.current_filename = self.image_files[self.current_index]
        self.current_image = cv2.imread(str(self.current_filename))
        
        if self.current_image is None:
            print(f"Error loading image: {self.current_filename}")
            return False
        
        print(f"Loaded: {self.current_filename} ({self.current_index + 1}/{len(self.image_files)})")
        return True
    
    def run(self):
        """Main loop to process images"""
        if not self.image_files:
            return
        
        cv2.namedWindow('Image Cropper', cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback('Image Cropper', self.mouse_callback)
        
        print("\nControls:")
        print("- Click anywhere on the image to crop a 640x640 area")
        print("- Press 'n' or 'd' for next image")
        print("- Press 'p' or 'a' for previous image")
        print("- Press 'q' or ESC to quit")
        print("- Press 'r' to reload current image")
        
        while self.current_index < len(self.image_files):
            if not self.load_current_image():
                self.current_index += 1
                continue
            
            # Scale image for display
            display_image = self.scale_image_for_display(self.current_image)
            cv2.imshow('Image Cropper', display_image)
            
            while True:
                key = cv2.waitKey(0) & 0xFF
                
                if key == ord('q') or key == 27:  # 'q' or ESC
                    cv2.destroyAllWindows()
                    print("Exiting...")
                    return
                
                elif key == ord('n') or key == ord('d'):  # Next image
                    self.current_index += 1
                    break
                
                elif key == ord('p') or key == ord('a'):  # Previous image
                    if self.current_index > 0:
                        self.current_index -= 1
                    break
                
                elif key == ord('r'):  # Reload current image
                    display_image = self.scale_image_for_display(self.current_image)
                    cv2.imshow('Image Cropper', display_image)
        
        cv2.destroyAllWindows()
        print("Finished processing all images!")

def main():
    # Get input directory from user
    input_dir = input("Enter the directory path containing images (or press Enter for current directory): ").strip()
    
    if not input_dir:
        input_dir = "."
    
    if not os.path.exists(input_dir):
        print(f"Directory '{input_dir}' does not exist!")
        return
    
    # Initialize and run the cropper
    cropper = ImageCropper(input_dir)
    cropper.run()

if __name__ == "__main__":
    main()