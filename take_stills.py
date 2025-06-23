import cv2
import os
from datetime import datetime

def play_video_with_screenshot(video_path, output_folder="screenshots"):
    """
    Play a video and capture screenshots when spacebar is pressed.
    
    Args:
        video_path (str): Path to the video file
        output_folder (str): Folder to save screenshots
    """
    
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Open video file
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video file '{video_path}'")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    
    print(f"Video loaded: {video_path}")
    print(f"Duration: {duration:.2f} seconds")
    print(f"FPS: {fps}")
    print("\nControls:")
    print("- SPACEBAR: Capture screenshot")
    print("- 'q' or ESC: Quit")
    print("- 'p': Pause/Resume")
    print("- LEFT/RIGHT arrows: Skip backward/forward 10 seconds")
    
    paused = False
    screenshot_count = 0
    
    while True:
        if not paused:
            ret, frame = cap.read()
            
            if not ret:
                print("End of video reached")
                break
        
        # Display the frame
        cv2.imshow('Video Player - Press SPACEBAR to capture', frame)
        
        # Wait for key press (1ms timeout)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q') or key == 27:  # 'q' or ESC key
            break
        elif key == ord(' '):  # Spacebar
            # Generate timestamp for filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            filename = f"screenshot_{timestamp}_frame{current_frame}.jpg"
            filepath = os.path.join(output_folder, filename)
            
            # Save the current frame
            cv2.imwrite(filepath, frame)
            screenshot_count += 1
            print(f"Screenshot saved: {filepath}")
            
        elif key == ord('p'):  # Pause/Resume
            paused = not paused
            status = "Paused" if paused else "Resumed"
            print(f"Video {status}")
            
        elif key == 81:  # Left arrow key (skip backward 10 seconds)
            current_pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
            new_pos = max(0, current_pos - (fps * 10))
            cap.set(cv2.CAP_PROP_POS_FRAMES, new_pos)
            print("Skipped backward 10 seconds")
            
        elif key == 83:  # Right arrow key (skip forward 10 seconds)
            current_pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
            new_pos = min(total_frames - 1, current_pos + (fps * 10))
            cap.set(cv2.CAP_PROP_POS_FRAMES, new_pos)
            print("Skipped forward 10 seconds")
    
    # Clean up
    cap.release()
    cv2.destroyAllWindows()
    print(f"\nTotal screenshots captured: {screenshot_count}")
    print(f"Screenshots saved in: {os.path.abspath(output_folder)}")

def main():
    video_path = r"C:\Users\kbtod\Videos\recordings\recordings\recording-ir-50.h264"

    if not os.path.exists(video_path):
        print(f"Error: Video file not found at '{video_path}'")
        return
    
    # Optional: specify custom output folder
    output_folder = "."
    
    play_video_with_screenshot(video_path, output_folder)

if __name__ == "__main__":
    main()