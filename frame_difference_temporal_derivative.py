import cv2
import numpy as np
from collections import deque
import argparse

class TemporalDerivativeMotionDetector:
    def __init__(self, method='central', buffer_size=5, gaussian_kernel=5, 
                 threshold_low=20, threshold_high=40, morph_kernel_size=3):
        """
        Initialize temporal derivative motion detector
        
        Args:
            method: 'simple', 'central', or 'higher_order'
            buffer_size: Number of frames to keep in buffer
            gaussian_kernel: Size of Gaussian blur kernel (0 to disable)
            threshold_low: Lower threshold for hysteresis
            threshold_high: Upper threshold for hysteresis
            morph_kernel_size: Kernel size for morphological operations
        """
        self.method = method
        self.buffer_size = max(buffer_size, 3)  # Minimum 3 for central difference
        self.frame_buffer = deque(maxlen=self.buffer_size)
        self.gaussian_kernel = gaussian_kernel
        self.threshold_low = threshold_low
        self.threshold_high = threshold_high
        
        # Morphological kernel
        self.morph_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (morph_kernel_size, morph_kernel_size)
        )
        
        # For adaptive background subtraction (optional enhancement)
        self.background = None
        self.learning_rate = 0.1
        
    def preprocess_frame(self, frame):
        """Preprocess frame: convert to grayscale and apply Gaussian blur"""
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame.copy()
        
        if self.gaussian_kernel > 0:
            gray = cv2.GaussianBlur(gray, (self.gaussian_kernel, self.gaussian_kernel), 0)
        
        return gray.astype(np.float32)
    
    def compute_temporal_derivative(self):
        """Compute temporal derivative based on selected method"""
        if len(self.frame_buffer) < 2:
            return None
            
        frames = list(self.frame_buffer)
        
        if self.method == 'simple':
            # Simple frame differencing: I(t) - I(t-1)
            return np.abs(frames[-1] - frames[-2])
            
        elif self.method == 'central' and len(frames) >= 3:
            # Central difference: [I(t+1) - I(t-1)] / 2
            return np.abs((frames[-1] - frames[-3]) / 2.0)
            
        elif self.method == 'higher_order' and len(frames) >= 5:
            # Higher order: [-I(t+2) + 8*I(t+1) - 8*I(t-1) + I(t-2)] / 12
            derivative = (-frames[-1] + 8*frames[-2] - 8*frames[-4] + frames[-5]) / 12.0
            return np.abs(derivative)
        
        else:
            # Fallback to simple differencing
            return np.abs(frames[-1] - frames[-2])
    
    def apply_hysteresis_threshold(self, temporal_deriv):
        """Apply hysteresis thresholding to reduce noise"""
        # High threshold - definitely motion
        high_thresh = temporal_deriv > self.threshold_high
        
        # Low threshold - possible motion
        low_thresh = temporal_deriv > self.threshold_low
        
        # Hysteresis: keep low threshold pixels connected to high threshold pixels
        motion_mask = cv2.morphologyEx(
            high_thresh.astype(np.uint8), cv2.MORPH_DILATE, self.morph_kernel
        )
        
        # Find connected components
        motion_mask = low_thresh & (motion_mask > 0)
        
        return motion_mask.astype(np.uint8) * 255
    
    def apply_morphological_filtering(self, motion_mask):
        """Apply morphological operations to clean up the motion mask"""
        # Opening to remove small noise
        cleaned = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, self.morph_kernel)
        
        # Closing to fill gaps
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, self.morph_kernel)
        
        return cleaned
    
    def update_background(self, frame):
        """Update background model using exponential averaging"""
        if self.background is None:
            self.background = frame.copy()
        else:
            self.background = (self.learning_rate * frame + 
                             (1 - self.learning_rate) * self.background)
    
    def detect_motion(self, frame):
        """Main function to detect motion in a frame"""
        # Preprocess frame
        processed_frame = self.preprocess_frame(frame)
        
        # Add to buffer
        self.frame_buffer.append(processed_frame)
        
        # Update background model (optional)
        self.update_background(processed_frame)
        
        # Compute temporal derivative
        temporal_deriv = self.compute_temporal_derivative()
        
        if temporal_deriv is None:
            return None, None
        
        # Apply thresholding
        motion_mask = self.apply_hysteresis_threshold(temporal_deriv)
        
        # Clean up with morphological operations
        motion_mask = self.apply_morphological_filtering(motion_mask)
        
        return temporal_deriv, motion_mask

def main():
    parser = argparse.ArgumentParser(description='Temporal Derivative Motion Detection')
    parser.add_argument('--input', type=str, default=0, 
                       help='Input video file or camera index (default: 0)')
    parser.add_argument('--method', type=str, default='central',
                       choices=['simple', 'central', 'higher_order'],
                       help='Temporal derivative method')
    parser.add_argument('--gaussian', type=int, default=5,
                       help='Gaussian blur kernel size (0 to disable)')
    parser.add_argument('--threshold_low', type=int, default=20,
                       help='Lower threshold for hysteresis')
    parser.add_argument('--threshold_high', type=int, default=40,
                       help='Upper threshold for hysteresis')
    
    args = parser.parse_args()
    
    # Initialize detector
    detector = TemporalDerivativeMotionDetector(
        method=args.method,
        gaussian_kernel=args.gaussian,
        threshold_low=args.threshold_low,
        threshold_high=args.threshold_high
    )
    
    # Open video source
    if isinstance(args.input, str) and args.input.isdigit():
        cap = cv2.VideoCapture(int(args.input))
    else:
        cap = cv2.VideoCapture(args.input)
    
    if not cap.isOpened():
        print("Error: Could not open video source")
        return
    
    print(f"Using method: {args.method}")
    print("Press 'q' to quit, 's' to save current frame, 'r' to reset background")
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Detect motion
        temporal_deriv, motion_mask = detector.detect_motion(frame)
        
        if temporal_deriv is not None and motion_mask is not None:
            # Normalize temporal derivative for display
            temp_display = cv2.normalize(temporal_deriv, None, 0, 255, cv2.NORM_MINMAX)
            temp_display = temp_display.astype(np.uint8)
            
            # Create colored motion overlay
            motion_overlay = frame.copy()
            motion_overlay[motion_mask > 0] = [0, 0, 255]  # Red for motion
            result = cv2.addWeighted(frame, 0.7, motion_overlay, 0.3, 0)
            
            # Display results
            cv2.imshow('Original', frame)
            cv2.imshow('Temporal Derivative', temp_display)
            cv2.imshow('Motion Mask', motion_mask)
            cv2.imshow('Motion Overlay', result)
        else:
            cv2.imshow('Original', frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s') and motion_mask is not None:
            cv2.imwrite(f'motion_frame_{frame_count}.jpg', result)
            print(f"Saved motion_frame_{frame_count}.jpg")
        elif key == ord('r'):
            detector.background = None
            detector.frame_buffer.clear()
            print("Reset background and frame buffer")
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()