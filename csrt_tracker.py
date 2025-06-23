import cv2
import sys

class CSRTTracker:
    def __init__(self, video_path):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        self.tracker = cv2.TrackerCSRT_create()
        self.bbox = None
        self.tracking = False
        
    def select_roi(self, frame):
        """Let user select ROI by drawing bounding box"""
        print("Select the object to track by drawing a bounding box")
        print("Press SPACE or ENTER to confirm, ESC to cancel")
        
        # Select ROI
        bbox = cv2.selectROI("Select Object", frame, False)
        cv2.destroyWindow("Select Object")
        
        if bbox[2] > 0 and bbox[3] > 0:  # Valid selection
            return bbox
        return None
    
    def run(self):
        if not self.cap.isOpened():
            print("Error: Could not open video")
            return
        
        # Read first frame
        ret, frame = self.cap.read()
        if not ret:
            print("Error: Could not read first frame")
            return
        
        # Initialize tracker with first frame
        self.bbox = self.select_roi(frame)
        if self.bbox is None:
            print("No valid selection made")
            return
        
        # Initialize tracker
        self.tracker.init(frame, self.bbox)
        self.tracking = True
        
        print("Tracking initialized. Press 'q' to quit, 'r' to reinitialize")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("End of video or error reading frame")
                break
            
            if self.tracking:
                # Update tracker
                success, bbox = self.tracker.update(frame)
                
                if success:
                    # Draw bounding box
                    (x, y, w, h) = [int(v) for v in bbox]
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, "CSRT Tracker", (x, y - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    # Tracking failed
                    cv2.putText(frame, "Tracking failed", (50, 50), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Display frame
            cv2.imshow('CSRT Object Tracking', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                # Reinitialize tracker
                self.bbox = self.select_roi(frame)
                if self.bbox is not None:
                    self.tracker = cv2.TrackerCSRT_create()
                    self.tracker.init(frame, self.bbox)
                    self.tracking = True
                    print("Tracker reinitialized")
        
        # Cleanup
        self.cap.release()
        cv2.destroyAllWindows()

def main():
    if len(sys.argv) != 2:
        print("Usage: python csrt_tracker.py <video_path>")
        print("Example: python csrt_tracker.py video.mp4")
        return
    
    video_path = sys.argv[1]
    tracker = CSRTTracker(video_path)
    tracker.run()

if __name__ == "__main__":
    main()