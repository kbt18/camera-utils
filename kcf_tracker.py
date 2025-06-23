import cv2
import sys 
import numpy as np
import readchar
import msvcrt

pixelIntensityThreshold = 10

def findDroneInPatch(frame, last_good_bbox):
    x, y, w, h = [int(i) for i in last_good_bbox]
    
    # Calculate center of the last good bbox
    bbox_center_x = x + w // 2
    bbox_center_y = y + h // 2
    
    # Define 64x64 patch centered at bbox center
    patch_size = 64
    half_patch = patch_size // 2
    
    # Calculate patch boundaries
    patch_x1 = max(0, bbox_center_x - half_patch)
    patch_y1 = max(0, bbox_center_y - half_patch)
    patch_x2 = min(frame.shape[1], bbox_center_x + half_patch)
    patch_y2 = min(frame.shape[0], bbox_center_y + half_patch)
    
    # Extract the 64x64 patch
    crop = frame[patch_y1:patch_y2, patch_x1:patch_x2]
    
    # DEBUG: Display patch info
    print(f"Patch dimensions: {crop.shape}")
    print(f"Patch coordinates: ({patch_x1}, {patch_y1}) to ({patch_x2}, {patch_y2})")
    
    # Convert to grayscale and threshold to find bright dot
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    # High-pass filter to enhance local contrast
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    highpass = cv2.subtract(blurred, gray)  # highlights dark-on-bright features
    # Normalize to 0–255 range
    norm = cv2.normalize(highpass, None, 0, 255, cv2.NORM_MINMAX)
    # Use a fixed low threshold

    _, thresh = cv2.threshold(norm, pixelIntensityThreshold, 255, cv2.THRESH_BINARY)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cv2.imwrite('debug_patch_original.jpg', crop)
    cv2.imwrite('debug_patch_gray.jpg', gray)
    cv2.imwrite('debug_patch_thresh.jpg', thresh)
    
    # DEBUG: Print contour info
    print(f"Number of contours found: {len(contours)}")
    if contours:
        areas = [cv2.contourArea(c) for c in contours]
        print(f"Contour areas: {areas}")

    if not contours:
        print("DEBUG: No contours found - returning None")
        return None  # No dot found
    

    # Get combined moment of the white pixels
    M = cv2.moments(thresh, binaryImage=True)

    if M["m00"] != 0:
        cx_patch = int(M["m10"] / M["m00"])
        cy_patch = int(M["m01"] / M["m00"])
        print(f"Overall center of mass: ({cx_patch}, {cy_patch})")
    else:
        print("No white pixels detected.")
    
    # Convert patch coordinates to global frame coordinates
    cx_global = patch_x1 + cx_patch
    cy_global = patch_y1 + cy_patch
    
    # Create a 64x64 bbox centered at the dot location
    bbox_x1 = cx_global - half_patch//2
    bbox_y1 = cy_global - half_patch//2

    # Return bbox in format [x1, y1, width, height]
    return [bbox_x1, bbox_y1, patch_size//2, patch_size//2]

def main():
    # Initialize video capture
    # You can use 0 for webcam or provide a video file path
    video_path = r"C:\Users\kbtod\datasets\kopi_av_dataset\data\IMG_2272.MOV"  # Change to video file path if needed
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error: Could not open video source")
        return
    
    # Read the first frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read first frame")
        return
    
    # Create KCF tracker
    tracker = cv2.TrackerKCF_create()
    
    # Select ROI (Region of Interest) on the first frame
    print("Select the object to track and press ENTER or SPACE")
    print("Press 'c' to cancel selection")
    
    # Select bounding box
    bbox = cv2.selectROI("Select Object", frame, False)
    cv2.destroyWindow("Select Object")
    
    # Check if a valid bounding box was selected
    if bbox == (0, 0, 0, 0):
        print("No object selected. Exiting...")
        return
    
    # Initialize tracker with first frame and bounding box
    success = True
    tracker.init(frame, bbox)
    if not success:
        print("Failed to initialize tracker")
        return
    
    print("Tracker initialized successfully!")
    print("Press 'q' to quit, 'r' to reinitialize tracker")
    
    last_good_bbox = bbox

    # Main tracking loop
    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video or failed to read frame")
            break
        
        # Update tracker
        success, bbox = tracker.update(frame)
        
        # Draw bounding box
        if success:
            last_good_bbox=bbox
            # Extract coordinates
            x, y, w, h = [int(i) for i in bbox]
            
            # Draw rectangle
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Add tracking status text
            cv2.putText(frame, "Tracking", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            # Tracking failed
            cv2.putText(frame, "Tracking Lost", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            bbox = findDroneInPatch(frame, last_good_bbox)
            print(bbox)

            tracker = cv2.TrackerKCF_create()
            tracker.init(frame, bbox)
        
        # Display frame
        cv2.imshow("KCF Tracker", frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            # Reinitialize tracker
            print("Select new object to track")
            bbox = cv2.selectROI("Select Object", frame, False)
            cv2.destroyWindow("Select Object")
            
            if bbox != (0, 0, 0, 0):
                tracker = cv2.TrackerKCF_create()
                success = tracker.init(frame, bbox)
                if success:
                    print("Tracker reinitialized successfully!")
                else:
                    print("Failed to reinitialize tracker")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()