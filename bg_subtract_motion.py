import cv2
import numpy as np
import argparse

class MotionRobustBackgroundSubtractor:
    def __init__(self, method='knn', stabilize=True):
        """
        Initialize background subtractor with motion compensation
        
        Args:
            method: 'mog2', 'knn', or 'gmg'
            stabilize: Whether to apply digital stabilization
        """
        self.method = method
        self.stabilize = stabilize
        
        # Initialize background subtractor
        if method == 'mog2':
            self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
                detectShadows=True, varThreshold=50, history=200)
        elif method == 'knn':
            self.bg_subtractor = cv2.createBackgroundSubtractorKNN(
                detectShadows=True, dist2Threshold=800, history=200)
        elif method == 'gmg':
            self.bg_subtractor = cv2.bgsegm.createBackgroundSubtractorGMG()
        
        # For stabilization
        self.prev_frame = None
        self.transform_matrix = np.eye(2, 3, dtype=np.float32)
        
        # Feature detector for stabilization
        self.detector = cv2.goodFeaturesToTrack
        self.lk_params = dict(winSize=(15, 15),
                             maxLevel=2,
                             criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    
    def stabilize_frame(self, frame):
        """Apply digital stabilization to compensate for camera movement"""
        if not self.stabilize:
            return frame
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if self.prev_frame is None:
            self.prev_frame = gray
            return frame
        
        # Detect features in previous frame
        prev_pts = self.detector(self.prev_frame, maxCorners=200, 
                                qualityLevel=0.01, minDistance=30, blockSize=3)
        
        if prev_pts is not None and len(prev_pts) > 10:
            # Track features in current frame
            curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                self.prev_frame, gray, prev_pts, None, **self.lk_params)
            
            # Filter good points
            good_prev = prev_pts[status == 1]
            good_curr = curr_pts[status == 1]
            
            if len(good_prev) > 10:
                # Calculate transformation matrix
                transform = cv2.estimateAffinePartial2D(good_curr, good_prev)[0]
                
                if transform is not None:
                    # Apply transformation to stabilize
                    h, w = frame.shape[:2]
                    stabilized = cv2.warpAffine(frame, transform, (w, h))
                    self.prev_frame = cv2.cvtColor(stabilized, cv2.COLOR_BGR2GRAY)
                    return stabilized
        
        self.prev_frame = gray
        return frame
    
    def process_frame(self, frame):
        """Process a single frame and return foreground mask"""
        # Apply stabilization if enabled
        stabilized_frame = self.stabilize_frame(frame)
        
        # Apply background subtraction
        fg_mask = self.bg_subtractor.apply(stabilized_frame)
        
        # Post-processing to reduce noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        
        # Remove noise
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        
        # Fill holes
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        
        # Additional filtering for moving camera scenarios
        fg_mask = cv2.medianBlur(fg_mask, 5)
        
        return fg_mask, stabilized_frame

def main():
    parser = argparse.ArgumentParser(description='Background subtraction with camera motion compensation')
    parser.add_argument('--input', type=str, default=0, help='Input video file or camera index')
    parser.add_argument('--method', type=str, default='mog2', choices=['mog2', 'knn', 'gmg'],
                       help='Background subtraction method')
    parser.add_argument('--no-stabilize', action='store_true', help='Disable digital stabilization')
    parser.add_argument('--output', type=str, help='Output video file (optional)')
    
    args = parser.parse_args()
    
    # Initialize video capture
    if args.input.isdigit():
        cap = cv2.VideoCapture(int(args.input))
    else:
        cap = cv2.VideoCapture(args.input)
    
    if not cap.isOpened():
        print("Error: Could not open video source")
        return
    
    # Initialize background subtractor
    bg_sub = MotionRobustBackgroundSubtractor(
        method=args.method, 
        stabilize=not args.no_stabilize
    )
    
    # Setup video writer if output specified
    writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(args.output, fourcc, fps, (width * 2, height))
    
    print("Processing video... Press 'q' to quit")
    print(f"Method: {args.method}, Stabilization: {not args.no_stabilize}")
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Process frame
        fg_mask, processed_frame = bg_sub.process_frame(frame)
        
        # Create colored mask for visualization
        fg_colored = cv2.applyColorMap(fg_mask, cv2.COLORMAP_HOT)
        
        # Combine original and mask for display
        combined = np.hstack((processed_frame, fg_colored))
        
        # Resize for display if too large
        h, w = combined.shape[:2]
        if w > 1280:
            scale = 1280 / w
            new_w, new_h = int(w * scale), int(h * scale)
            combined = cv2.resize(combined, (new_w, new_h))
        
        cv2.imshow('Background Subtraction | Original - Foreground Mask', combined)
        
        # Save frame if output specified
        if writer:
            writer.write(np.hstack((processed_frame, 
                                  cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2BGR))))
        
        # Check for quit
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):  # Reset background model
            if args.method == 'mog2':
                bg_sub.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
                    detectShadows=True, varThreshold=50, history=200)
            elif args.method == 'knn':
                bg_sub.bg_subtractor = cv2.createBackgroundSubtractorKNN(
                    detectShadows=True, dist2Threshold=800, history=200)
            print("Background model reset")
    
    print(f"Processed {frame_count} frames")
    
    # Cleanup
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
