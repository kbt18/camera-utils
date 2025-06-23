import cv2
import sys
import time
import numpy as np

class CSRTLatencyTest:
    def __init__(self, video_path):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        self.tracker = cv2.TrackerCSRT_create()
        self.original_bbox = None
        self.roi_bbox = None  # Bbox relative to ROI crop
        self.roi_center = None
        self.roi_top_left = None
        self.latencies = []
        self.roi_size = 640
        
    def select_roi(self, frame):
        """Let user select ROI by drawing bounding box"""
        print("Select the object to track by drawing a bounding box")
        print("Press SPACE or ENTER to confirm, ESC to cancel")
        
        # Select ROI
        bbox = cv2.selectROI("Select Object - Press SPACE/ENTER to confirm", frame, False)
        cv2.destroyWindow("Select Object - Press SPACE/ENTER to confirm")
        
        # Ensure bbox is in the correct format (x, y, width, height)
        if len(bbox) == 4 and bbox[2] > 0 and bbox[3] > 0:  # Valid selection
            # Convert to tuple of integers if needed
            bbox = tuple(int(x) for x in bbox)
            return bbox
        return None
    
    def calculate_roi_crop(self, frame, bbox):
        """Calculate 640x640 ROI crop centered on bbox center"""
        frame_h, frame_w = frame.shape[:2]
        
        # Calculate bbox center
        bbox_center_x = bbox[0] + bbox[2] // 2
        bbox_center_y = bbox[1] + bbox[3] // 2
        
        # Calculate ROI boundaries (640x640 centered on bbox)
        half_roi = self.roi_size // 2
        roi_x1 = max(0, bbox_center_x - half_roi)
        roi_y1 = max(0, bbox_center_y - half_roi)
        roi_x2 = min(frame_w, bbox_center_x + half_roi)
        roi_y2 = min(frame_h, bbox_center_y + half_roi)
        
        # Adjust if ROI goes outside frame boundaries
        roi_width = roi_x2 - roi_x1
        roi_height = roi_y2 - roi_y1
        
        # If ROI is smaller than desired size due to frame boundaries, pad it
        if roi_width < self.roi_size:
            if roi_x1 == 0:  # Hit left boundary
                roi_x2 = min(frame_w, roi_x1 + self.roi_size)
            else:  # Hit right boundary
                roi_x1 = max(0, roi_x2 - self.roi_size)
        
        if roi_height < self.roi_size:
            if roi_y1 == 0:  # Hit top boundary
                roi_y2 = min(frame_h, roi_y1 + self.roi_size)
            else:  # Hit bottom boundary
                roi_y1 = max(0, roi_y2 - self.roi_size)
        
        # Store ROI parameters
        self.roi_center = (bbox_center_x, bbox_center_y)
        self.roi_top_left = (roi_x1, roi_y1)
        
        # Calculate bbox coordinates relative to ROI crop
        roi_bbox_x = bbox[0] - roi_x1
        roi_bbox_y = bbox[1] - roi_y1
        roi_bbox = (roi_bbox_x, roi_bbox_y, bbox[2], bbox[3])
        
        # Extract ROI crop
        roi_crop = frame[roi_y1:roi_y2, roi_x1:roi_x2]
        
        # Pad ROI if it's smaller than 640x640 due to frame boundaries
        if roi_crop.shape[0] < self.roi_size or roi_crop.shape[1] < self.roi_size:
            padded_roi = np.zeros((self.roi_size, self.roi_size, 3), dtype=roi_crop.dtype)
            h, w = roi_crop.shape[:2]
            padded_roi[:h, :w] = roi_crop
            roi_crop = padded_roi
            
        return roi_crop, roi_bbox
    
    def run_latency_test(self):
        if not self.cap.isOpened():
            print("Error: Could not open video")
            return
        
        # Get total frame count
        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        print(f"Video info: {total_frames} frames at {fps:.2f} FPS")
        
        # Read first frame
        ret, frame = self.cap.read()
        if not ret:
            print("Error: Could not read first frame")
            return
        
        # Initialize tracker with first frame
        self.original_bbox = self.select_roi(frame)
        if self.original_bbox is None:
            print("No valid selection made")
            return
        
        # Debug: Print frame and bbox info
        print(f"Frame shape: {frame.shape}")
        print(f"Selected bbox: {self.original_bbox}")
        
        # Calculate ROI crop
        roi_crop, self.roi_bbox = self.calculate_roi_crop(frame, self.original_bbox)
        
        print(f"ROI crop shape: {roi_crop.shape}")
        print(f"ROI center: {self.roi_center}")
        print(f"ROI top-left: {self.roi_top_left}")
        print(f"Bbox in ROI coordinates: {self.roi_bbox}")
        
        # Ensure ROI crop is in correct format (BGR, uint8)
        if len(roi_crop.shape) == 3 and roi_crop.shape[2] == 3:
            # ROI is already BGR
            init_roi = roi_crop.copy()
        else:
            print("Converting ROI format...")
            init_roi = cv2.cvtColor(roi_crop, cv2.COLOR_GRAY2BGR) if len(roi_crop.shape) == 2 else roi_crop
        
        # Initialize tracker with ROI crop
        init_start = time.perf_counter()
        try:
            print("Attempting to initialize tracker with ROI crop...")
            self.tracker.init(init_roi, self.roi_bbox)
            success = True
            init_time = (time.perf_counter() - init_start) * 1000
            
            if not success:
                print("Failed to initialize tracker - tracker.init() returned False")
                print("Trying with different tracker creation method...")
                
                # Try recreating the tracker
                try:
                    self.tracker = cv2.TrackerCSRT_create()
                    success = self.tracker.init(init_roi, self.roi_bbox)
                    if success:
                        print("Success with recreated tracker!")
                    else:
                        print("Still failed with recreated tracker")
                        return
                except Exception as e2:
                    print(f"Error recreating tracker: {e2}")
                    return
            
        except Exception as e:
            print(f"Exception during tracker initialization: {e}")
            return
            
        print(f"Tracker initialization time: {init_time:.2f} ms")
        print("\nRunning latency test with ROI crops (no preview)...")
        print("Processing frames...")
        
        frame_count = 1  # Already processed first frame
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Extract ROI crop from current frame
            roi_x1, roi_y1 = self.roi_top_left
            roi_x2 = roi_x1 + self.roi_size
            roi_y2 = roi_y1 + self.roi_size
            
            # Ensure ROI boundaries are within frame
            frame_h, frame_w = frame.shape[:2]
            roi_x2 = min(frame_w, roi_x2)
            roi_y2 = min(frame_h, roi_y2)
            
            # Extract ROI crop
            current_roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]
            
            # Pad ROI if needed
            if current_roi.shape[0] < self.roi_size or current_roi.shape[1] < self.roi_size:
                padded_roi = np.zeros((self.roi_size, self.roi_size, 3), dtype=current_roi.dtype)
                h, w = current_roi.shape[:2]
                padded_roi[:h, :w] = current_roi
                current_roi = padded_roi
            
            # Measure tracking update time on ROI crop
            start_time = time.perf_counter()
            success, bbox = self.tracker.update(current_roi)
            end_time = time.perf_counter()
            
            latency_ms = (end_time - start_time) * 1000
            self.latencies.append(latency_ms)
            
            frame_count += 1
            
            # Progress indicator
            if frame_count % 100 == 0:
                progress = (frame_count / total_frames) * 100
                print(f"Processed {frame_count}/{total_frames} frames ({progress:.1f}%)")
        
        # Calculate and display statistics
        self.display_statistics(frame_count - 1)  # Subtract 1 for init frame
        
        # Cleanup
        self.cap.release()
        cv2.destroyAllWindows()
    
    def display_statistics(self, processed_frames):
        """Display latency statistics"""
        if not self.latencies:
            print("No latency data collected")
            return
        
        latencies = np.array(self.latencies)
        
        print("\n" + "="*50)
        print("CSRT TRACKER LATENCY STATISTICS (with 640x640 ROI)")
        print("="*50)
        print(f"Total frames processed: {processed_frames}")
        print(f"ROI size: {self.roi_size}x{self.roi_size}")
        print(f"Average latency: {np.mean(latencies):.2f} ms")
        print(f"Median latency: {np.median(latencies):.2f} ms")
        print(f"Min latency: {np.min(latencies):.2f} ms")
        print(f"Max latency: {np.max(latencies):.2f} ms")
        print(f"Standard deviation: {np.std(latencies):.2f} ms")
        print(f"95th percentile: {np.percentile(latencies, 95):.2f} ms")
        print(f"99th percentile: {np.percentile(latencies, 99):.2f} ms")
        
        # Performance analysis
        avg_latency = np.mean(latencies)
        max_fps = 1000 / avg_latency if avg_latency > 0 else 0
        
        print(f"\nPerformance Analysis:")
        print(f"Theoretical max FPS based on avg latency: {max_fps:.1f} FPS")
        
        # Latency distribution
        fast_frames = np.sum(latencies < 10)
        medium_frames = np.sum((latencies >= 10) & (latencies < 30))
        slow_frames = np.sum(latencies >= 30)
        
        print(f"\nLatency Distribution:")
        print(f"< 10ms (very fast): {fast_frames} frames ({fast_frames/len(latencies)*100:.1f}%)")
        print(f"10-30ms (medium): {medium_frames} frames ({medium_frames/len(latencies)*100:.1f}%)")
        print(f"> 30ms (slow): {slow_frames} frames ({slow_frames/len(latencies)*100:.1f}%)")
        
        # Save detailed results to file
        self.save_results_to_file(latencies)
    
    def save_results_to_file(self, latencies):
        """Save detailed latency results to a CSV file"""
        filename = "csrt_latency_results_roi.csv"
        try:
            with open(filename, 'w') as f:
                f.write("Frame,Latency_ms\n")
                for i, latency in enumerate(latencies, 1):
                    f.write(f"{i},{latency:.4f}\n")
            print(f"\nDetailed results saved to: {filename}")
        except Exception as e:
            print(f"Could not save results to file: {e}")

def main():
    if len(sys.argv) != 2:
        print("Usage: python csrt_latency_test.py <video_path>")
        print("Example: python csrt_latency_test.py video.mp4")
        return
    
    video_path = sys.argv[1]
    tester = CSRTLatencyTest(video_path)
    tester.run_latency_test()

if __name__ == "__main__":
    main()