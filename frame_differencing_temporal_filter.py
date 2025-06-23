import cv2
import numpy as np
import argparse

class OpticalFlowMotionDetector:
    def __init__(self, learning_rate=0.05, max_features=200, quality_level=0.01,
                 min_distance=10, threshold=25, gaussian_kernel=5, morph_kernel_size=3):
        """
        Initialize optical flow motion detector with background modeling
        
        Args:
            learning_rate: Background update rate (0.01-0.1, lower = more stable)
            max_features: Maximum number of features to track
            quality_level: Quality level for corner detection
            min_distance: Minimum distance between features
            threshold: Threshold for motion detection after background subtraction
            gaussian_kernel: Size of Gaussian blur kernel (0 to disable)
            morph_kernel_size: Kernel size for morphological operations
        """
        self.learning_rate = learning_rate
        self.background = None
        self.prev_frame = None
        self.prev_points = None
        
        # Feature detection parameters
        self.feature_params = dict(
            maxCorners=max_features,
            qualityLevel=quality_level,
            minDistance=min_distance,
            blockSize=7
        )
        
        # Lucas-Kanade optical flow parameters
        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        
        self.threshold = threshold
        self.gaussian_kernel = gaussian_kernel
        
        # Morphological kernel
        self.morph_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (morph_kernel_size, morph_kernel_size)
        )
        
        # For visualization
        self.flow_vectors = []
        
    def preprocess_frame(self, frame):
        """Preprocess frame: convert to grayscale and apply Gaussian blur"""
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame.copy()
        
        if self.gaussian_kernel > 0:
            gray = cv2.GaussianBlur(gray, (self.gaussian_kernel, self.gaussian_kernel), 0)
        
        return gray
    
    def detect_and_track_features(self, current_frame):
        """Detect new features and track existing ones"""
        if self.prev_frame is None or self.prev_points is None:
            # First frame - detect initial features
            self.prev_points = cv2.goodFeaturesToTrack(current_frame, **self.feature_params)
            return None, None
        
        # Track features using Lucas-Kanade optical flow
        if self.prev_points is not None and len(self.prev_points) > 0:
            next_points, status, error = cv2.calcOpticalFlowPyrLK(
                self.prev_frame, current_frame, self.prev_points, None, **self.lk_params
            )
            
            # Select good points
            if next_points is not None:
                good_new = next_points[status == 1]
                good_old = self.prev_points[status == 1]
                
                # Store flow vectors for visualization
                self.flow_vectors = [(good_old[i], good_new[i]) for i in range(len(good_new))]
                
                return good_old, good_new
        
        return None, None
    
    def warp_frame_with_optical_flow(self, frame, good_old, good_new):
        """Warp previous frame to current frame using optical flow"""
        if good_old is None or good_new is None or len(good_old) < 4:
            return frame  # Not enough points for warping
        
        try:
            # Estimate homography from optical flow
            H, mask = cv2.findHomography(good_old, good_new, 
                                       cv2.RANSAC, ransacReprojThreshold=3.0)
            
            if H is not None:
                # Warp the frame
                h, w = frame.shape[:2]
                warped = cv2.warpPerspective(frame, H, (w, h))
                return warped
            else:
                return frame
                
        except cv2.error:
            return frame
    
    def update_background_model(self, current_frame, motion_mask=None):
        """Update background model with temporal filtering"""
        if self.background is None:
            # Initialize background with first frame
            self.background = current_frame.astype(np.float32)
            return
        
        # Standard background update
        current_float = current_frame.astype(np.float32)
        
        if motion_mask is not None:
            # Don't update background in motion areas
            update_mask = (motion_mask == 0).astype(np.float32)
            self.background = (self.learning_rate * current_float * update_mask + 
                             self.background * (1 - self.learning_rate * update_mask))
        else:
            # Simple exponential averaging
            self.background = (self.learning_rate * current_float + 
                             (1 - self.learning_rate) * self.background)
    
    def detect_motion_with_background_subtraction(self, current_frame):
        """Detect motion using background subtraction"""
        if self.background is None:
            return None
        
        # Compute absolute difference from background
        diff = cv2.absdiff(current_frame, self.background.astype(np.uint8))
        
        # Threshold the difference
        _, motion_mask = cv2.threshold(diff, self.threshold, 255, cv2.THRESH_BINARY)
        
        # Clean up with morphological operations
        motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, self.morph_kernel)
        motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_CLOSE, self.morph_kernel)
        
        return motion_mask
    
    def refresh_features(self, current_frame, motion_mask=None):
        """Refresh feature points, avoiding motion areas if mask provided"""
        if motion_mask is not None:
            # Create mask to avoid detecting features in motion areas
            feature_mask = cv2.bitwise_not(motion_mask)
            # Erode to create buffer around motion areas
            feature_mask = cv2.erode(feature_mask, self.morph_kernel, iterations=2)
        else:
            feature_mask = None
        
        # Detect new features
        new_features = cv2.goodFeaturesToTrack(current_frame, mask=feature_mask, 
                                             **self.feature_params)
        
        # Combine with existing good features if any
        if self.prev_points is not None and len(self.prev_points) > 0:
            if new_features is not None:
                self.prev_points = np.vstack([self.prev_points, new_features])
            # Remove duplicate points (simple distance-based)
            if len(self.prev_points) > self.feature_params['maxCorners']:
                self.prev_points = self.prev_points[:self.feature_params['maxCorners']]
        else:
            self.prev_points = new_features
    
    def detect_motion(self, frame):
        """Main function to detect motion in a frame"""
        # Preprocess frame
        current_frame = self.preprocess_frame(frame)
        
        # Detect and track features
        good_old, good_new = self.detect_and_track_features(current_frame)
        
        # Warp previous frame if we have optical flow
        warped_prev = None
        if self.prev_frame is not None:
            if good_old is not None and good_new is not None:
                warped_prev = self.warp_frame_with_optical_flow(self.prev_frame, good_old, good_new)
            else:
                warped_prev = self.prev_frame
        
        # Compute frame difference (warped previous vs current)
        frame_diff = None
        if warped_prev is not None:
            frame_diff = cv2.absdiff(current_frame, warped_prev)
        
        # Detect motion using background subtraction
        motion_mask = self.detect_motion_with_background_subtraction(current_frame)
        
        # Update background model (avoiding motion areas)
        self.update_background_model(current_frame, motion_mask)
        
        # Refresh features periodically or when we have too few
        if (self.prev_points is None or len(self.prev_points) < 50 or 
            np.random.random() < 0.1):  # 10% chance to refresh
            self.refresh_features(current_frame, motion_mask)
        else:
            # Update feature points for next iteration
            if good_new is not None:
                self.prev_points = good_new.reshape(-1, 1, 2)
        
        # Update previous frame
        self.prev_frame = current_frame.copy()
        
        return frame_diff, motion_mask, self.background
    
    def draw_flow_vectors(self, frame):
        """Draw optical flow vectors on frame"""
        if not self.flow_vectors:
            return frame
        
        result = frame.copy()
        for old_pt, new_pt in self.flow_vectors:
            a, b = new_pt.ravel().astype(int)
            c, d = old_pt.ravel().astype(int)
            
            # Draw flow vector
            cv2.arrowedLine(result, (c, d), (a, b), (0, 255, 0), 2)
            cv2.circle(result, (a, b), 3, (0, 255, 0), -1)
        
        return result

def main():
    parser = argparse.ArgumentParser(description='Optical Flow Motion Detection with Background Model')
    parser.add_argument('--input', type=str, default=0, 
                       help='Input video file or camera index (default: 0)')
    parser.add_argument('--learning_rate', type=float, default=0.05,
                       help='Background learning rate (0.01-0.1)')
    parser.add_argument('--threshold', type=int, default=25,
                       help='Motion detection threshold')
    parser.add_argument('--max_features', type=int, default=200,
                       help='Maximum number of features to track')
    parser.add_argument('--gaussian', type=int, default=5,
                       help='Gaussian blur kernel size (0 to disable)')
    
    args = parser.parse_args()
    
    # Initialize detector
    detector = OpticalFlowMotionDetector(
        learning_rate=args.learning_rate,
        max_features=args.max_features,
        threshold=args.threshold,
        gaussian_kernel=args.gaussian
    )
    
    # Open video source
    if isinstance(args.input, str) and args.input.isdigit():
        cap = cv2.VideoCapture(int(args.input))
    else:
        cap = cv2.VideoCapture(args.input)
    
    if not cap.isOpened():
        print("Error: Could not open video source")
        return
    
    print(f"Background learning rate: {args.learning_rate}")
    print(f"Motion threshold: {args.threshold}")
    print("Press 'q' to quit, 'r' to reset background, 'f' to toggle flow visualization")
    
    show_flow = False
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Detect motion
        frame_diff, motion_mask, background = detector.detect_motion(frame)
        
        # Prepare displays
        displays = [frame]
        
        if frame_diff is not None:
            displays.append(cv2.applyColorMap(frame_diff, cv2.COLORMAP_HOT))
        
        if motion_mask is not None:
            # Create colored motion overlay
            motion_overlay = frame.copy()
            motion_overlay[motion_mask > 0] = [0, 0, 255]  # Red for motion
            result = cv2.addWeighted(frame, 0.7, motion_overlay, 0.3, 0)
            displays.extend([motion_mask, result])
        
        if background is not None:
            displays.append(background.astype(np.uint8))
        
        # Show flow vectors if enabled
        if show_flow:
            flow_frame = detector.draw_flow_vectors(frame)
            displays.append(flow_frame)
        
        # Display windows
        window_names = ['Original', 'Frame Diff', 'Motion Mask', 'Motion Overlay', 'Background']
        if show_flow:
            window_names.append('Optical Flow')
        
        for i, (display, name) in enumerate(zip(displays, window_names)):
            if display is not None:
                cv2.imshow(name, display)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            detector.background = None
            detector.prev_frame = None
            detector.prev_points = None
            print("Reset background model and tracking")
        elif key == ord('f'):
            show_flow = not show_flow
            if not show_flow:
                cv2.destroyWindow('Optical Flow')
            print(f"Flow visualization: {'ON' if show_flow else 'OFF'}")
        elif key == ord('s') and motion_mask is not None:
            cv2.imwrite(f'motion_frame_{frame_count}.jpg', result)
            print(f"Saved motion_frame_{frame_count}.jpg")
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()