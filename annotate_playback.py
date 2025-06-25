import cv2
import pandas as pd
import os
import argparse
from pathlib import Path

class AnnotationVisualizer:
    def __init__(self, video_path, annotations_csv):
        self.video_path = video_path
        self.annotations_csv = annotations_csv
        self.cap = cv2.VideoCapture(video_path)
        self.annotations = {}
        self.current_frame = 0
        
        # Display window name
        self.window_name = "Annotation Visualizer"
        
        # Load annotations
        self.load_annotations()
        
        print(f"Loaded video: {video_path}")
        print(f"Loaded {len(self.annotations)} annotations from: {annotations_csv}")
        print("\nControls:")
        print("- Press SPACE or 'n' to go to next frame")
        print("- Press 'p' to go to previous frame")
        print("- Press 'r' to play/resume video")
        print("- Press 's' to stop/pause video")
        print("- Press 'q' to quit")
        
    def load_annotations(self):
        """Load annotations from CSV file"""
        if not os.path.exists(self.annotations_csv):
            print(f"Error: Annotations file '{self.annotations_csv}' not found!")
            return
            
        try:
            df = pd.read_csv(self.annotations_csv)
            # Convert to dictionary for fast lookup by frame number
            for _, row in df.iterrows():
                frame_num = int(row['frame'])
                self.annotations[frame_num] = {
                    'x': int(row['x']),
                    'y': int(row['y'])
                }
            print(f"Successfully loaded annotations for frames: {sorted(self.annotations.keys())}")
        except Exception as e:
            print(f"Error loading annotations: {e}")
            self.annotations = {}
    
    def get_current_frame(self):
        """Get the current frame from video"""
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
        ret, frame = self.cap.read()
        return frame if ret else None
    
    def draw_annotation(self, frame, frame_num):
        """Draw annotation circle if annotation exists for this frame"""
        if frame_num in self.annotations:
            ann = self.annotations[frame_num]
            x, y = ann['x'], ann['y']
            
            # Draw filled circle at centroid
            # cv2.circle(frame, (x, y), 8, (0, 255, 0), -1)  # Green filled circle
            # Draw outline circle
            cv2.circle(frame, (x, y), 15, (0, 255, 0), 2)   # Green outline
            
            # Add small text label
            cv2.putText(frame, f"({x},{y})", (x+20, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        return frame
    
    def add_frame_info(self, frame, frame_num):
        """Add frame information text overlay"""
        # Frame info
        info_text = f"Frame: {frame_num}"
        if frame_num in self.annotations:
            info_text += " | ANNOTATED"
        else:
            info_text += " | No annotation"
            
        cv2.putText(frame, info_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Instructions
        cv2.putText(frame, "SPACE/n:Next | p:Prev | r:Play | s:Stop | q:Quit", 
                   (10, frame.shape[0] - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        return frame
    
    def play_video(self):
        """Play video continuously with annotations"""
        print("Playing video... Press 's' to stop")
        
        while True:
            frame = self.get_current_frame()
            if frame is None:
                print("End of video reached")
                break
                
            # Draw annotations and info
            frame = self.draw_annotation(frame, self.current_frame)
            frame = self.add_frame_info(frame, self.current_frame)
            
            cv2.imshow(self.window_name, frame)
            
            # Short wait for smooth playback, check for stop key
            key = cv2.waitKey(30) & 0xFF  # ~30 FPS playback
            if key == ord('s') or key == ord('q'):
                if key == ord('q'):
                    return True  # Signal to quit
                print("Video stopped")
                break
                
            self.current_frame += 1
            
        return False  # Don't quit
    
    def run(self):
        """Main run loop for manual frame stepping"""
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        
        while True:
            frame = self.get_current_frame()
            if frame is None:
                print("End of video or cannot read frame")
                break
                
            # Draw annotations and frame info
            display_frame = self.draw_annotation(frame.copy(), self.current_frame)
            display_frame = self.add_frame_info(display_frame, self.current_frame)
            
            cv2.imshow(self.window_name, display_frame)
            
            key = cv2.waitKey(0) & 0xFF  # Wait for key press
            
            if key == ord('q'):
                break
            elif key == ord('n') or key == ord(' '):  # Next frame
                self.current_frame += 1
            elif key == ord('p'):  # Previous frame
                if self.current_frame > 0:
                    self.current_frame -= 1
                else:
                    print("Already at first frame")
            elif key == ord('r'):  # Play video
                should_quit = self.play_video()
                if should_quit:
                    break
            elif key == ord('s'):  # Stop (already stopped in manual mode)
                print("Already in manual stepping mode")
                
        self.cap.release()
        cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description='Video Annotation Visualizer')
    parser.add_argument('video_path', help='Path to the video file')
    parser.add_argument('annotations_csv', help='Path to the annotations CSV file')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.video_path):
        print(f"Error: Video file '{args.video_path}' not found!")
        return
        
    if not os.path.exists(args.annotations_csv):
        print(f"Error: Annotations file '{args.annotations_csv}' not found!")
        return
        
    visualizer = AnnotationVisualizer(args.video_path, args.annotations_csv)
    visualizer.run()

if __name__ == "__main__":
    main()