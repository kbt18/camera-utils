import cv2
import pandas as pd
import os
import argparse
from pathlib import Path

class VideoAnnotator:
    def __init__(self, video_path, output_csv="annotations.csv"):
        self.video_path = video_path
        self.output_csv = output_csv
        self.cap = cv2.VideoCapture(video_path)
        self.annotations = []
        self.current_frame = 0
        
        # Mouse callback variables
        self.clicked_point = None
        
        # Display window name
        self.window_name = "Video Annotation Tool"
        
        print(f"Loaded video: {video_path}")
        print("\nControls:")
        print("- Click on object centroid to annotate (auto-advances to next frame)")
        print("- Press SPACE or 'n' to skip frame (object not visible)")
        print("- Press 'p' to go to previous frame")
        print("- Press 's' to save annotations")
        print("- Press 'r' to remove annotation from current frame")
        print("- Press 'q' to quit")
        
    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.clicked_point = (x, y)
            self.save_annotation()
            print(f"Frame {self.current_frame}: Annotated at ({x}, {y})")
            # Auto-advance to next frame
            # self.current_frame += 1
            
    def get_current_frame(self):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
        ret, frame = self.cap.read()
        return frame if ret else None
        
    def draw_annotations(self, frame):
        # Draw existing annotation for current frame if it exists
        existing_annotation = self.get_frame_annotation(self.current_frame)
        if existing_annotation:
            x, y = int(existing_annotation['x']), int(existing_annotation['y'])
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)  # Green filled circle
            cv2.circle(frame, (x, y), 10, (0, 255, 0), 2)   # Green outline
            
        # Draw current click if any
        if self.clicked_point:
            x, y = self.clicked_point
            cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)   # Red filled circle
            cv2.circle(frame, (x, y), 10, (0, 0, 255), 2)    # Red outline
            
        return frame
        
    def get_frame_annotation(self, frame_num):
        for ann in self.annotations:
            if ann['frame'] == frame_num:
                return ann
        return None
        
    def save_annotation(self):
        if self.clicked_point:
            # Remove existing annotation for this frame
            self.annotations = [ann for ann in self.annotations if ann['frame'] != self.current_frame]
            
            # Add new annotation (no timestamp since we're not relying on FPS)
            annotation = {
                'frame': self.current_frame,
                'x': self.clicked_point[0],
                'y': self.clicked_point[1]
            }
            self.annotations.append(annotation)
            
            self.clicked_point = None
            
    def remove_annotation(self):
        self.annotations = [ann for ann in self.annotations if ann['frame'] != self.current_frame]
        self.clicked_point = None
        print(f"Removed annotation for frame {self.current_frame}")
        
    def save_to_csv(self):
        if not self.annotations:
            print("No annotations to save!")
            return
            
        df = pd.DataFrame(self.annotations)
        df = df.sort_values('frame')  # Sort by frame number
        df.to_csv(self.output_csv, index=False)
        print(f"Saved {len(self.annotations)} annotations to {self.output_csv}")
        
    def load_existing_annotations(self):
        if os.path.exists(self.output_csv):
            try:
                df = pd.read_csv(self.output_csv)
                self.annotations = df.to_dict('records')
                print(f"Loaded {len(self.annotations)} existing annotations")
            except Exception as e:
                print(f"Could not load existing annotations: {e}")
                
    def run(self):
        # Load existing annotations if available
        self.load_existing_annotations()
        
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        
        while True:
            frame = self.get_current_frame()
            if frame is None:
                print("End of video or cannot read frame")
                break
                
            # Draw annotations on frame
            display_frame = self.draw_annotations(frame.copy())
            
            # Add frame info text
            info_text = f"Frame: {self.current_frame} | Annotations: {len(self.annotations)}"
            if self.get_frame_annotation(self.current_frame):
                info_text += " | [ANNOTATED]"
                
            cv2.putText(display_frame, info_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow(self.window_name, display_frame)
            
            key = cv2.waitKey(0) & 0xFF  # Wait indefinitely for key press
            
            if key == ord('q'):
                break
            elif key == ord('n') or key == ord(' '):  # Skip frame (object not visible)
                self.current_frame += 1
            elif key == ord('p'):  # Previous frame
                if self.current_frame > 0:
                    self.current_frame -= 1
                else:
                    print("Already at first frame")
            elif key == ord('s'):  # Save
                self.save_to_csv()
            elif key == ord('r'):  # Remove annotation
                self.remove_annotation()
                
        # Save annotations before closing
        self.save_to_csv()
        
        self.cap.release()
        cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description='Video Object Centroid Annotation Tool')
    parser.add_argument('video_path', help='Path to the video file')
    parser.add_argument('--output', '-o', default='annotations.csv', 
                       help='Output CSV file path (default: annotations.csv)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.video_path):
        print(f"Error: Video file '{args.video_path}' not found!")
        return
        
    annotator = VideoAnnotator(args.video_path, args.output)
    annotator.run()

if __name__ == "__main__":
    main()