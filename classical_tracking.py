

import cv2
import numpy as np
import os

print("Starting drone tracking with sky segmentation... Press ESC to exit")

def detect_horizon(frame):
    """
    Detect horizon line using edge detection and line detection
    Returns the y-coordinate of the horizon, or None if not found
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Edge detection
    edges = cv2.Canny(blurred, 50, 150, apertureSize=3)
    
    # Use HoughLines to detect straight lines
    lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
    
    if lines is not None:
        # Filter for horizontal-ish lines (close to 0 or 180 degrees)
        horizontal_lines = []
        for rho, theta in lines[:, 0]:
            angle_deg = np.degrees(theta)
            # Look for lines that are roughly horizontal (within 30 degrees)
            if abs(angle_deg) < 30 or abs(angle_deg - 180) < 30:
                # Convert to cartesian coordinates
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                
                # Get y-coordinate at the center of the frame
                center_x = frame.shape[1] // 2
                if abs(a) > 0.01:  # Avoid division by zero
                    y_at_center = int(y0 - (x0 - center_x) * b / a)
                    if 0 < y_at_center < frame.shape[0]:
                        horizontal_lines.append(y_at_center)
        
        if horizontal_lines:
            # Return the median y-coordinate of detected horizontal lines
            return int(np.median(horizontal_lines))
    
    return None

def create_sky_mask(frame, horizon_y=None):
    """
    Create a mask that isolates the sky region
    If horizon_y is None, use simple sky segmentation
    """
    height, width = frame.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    
    if horizon_y is not None:
        # Use detected horizon
        mask[0:horizon_y, :] = 255
    else:
        # Fallback: use color-based sky segmentation
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Define range for blue/white sky colors
        # Blue sky
        lower_blue = np.array([100, 50, 50])
        upper_blue = np.array([130, 255, 255])
        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
        
        # White/gray sky (clouds)
        lower_white = np.array([0, 0, 180])
        upper_white = np.array([180, 30, 255])
        white_mask = cv2.inRange(hsv, lower_white, upper_white)
        
        # Combine masks
        sky_color_mask = cv2.bitwise_or(blue_mask, white_mask)
        
        # Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        sky_color_mask = cv2.morphologyEx(sky_color_mask, cv2.MORPH_OPEN, kernel)
        sky_color_mask = cv2.morphologyEx(sky_color_mask, cv2.MORPH_CLOSE, kernel)
        
        # Assume upper portion is sky if color detection fails
        if np.sum(sky_color_mask) < width * height * 0.1:  # Less than 10% sky detected
            mask[0:height//2, :] = 255  # Use upper half as sky
        else:
            mask = sky_color_mask
    
    return mask

# Video file path - modify as needed
video_path = r"C:\Users\kbtod\Downloads\IMG_2272.MOV"

# Check if video file exists
if not os.path.exists(video_path):
    print(f"Video file not found: {video_path}")
    print("Using webcam instead...")
    cap = cv2.VideoCapture(0)  # Use webcam
else:
    cap = cv2.VideoCapture(video_path)

# Check if video capture is successful
if not cap.isOpened():
    print("Error: Could not open video source")
    exit()

# Get and display video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"Video properties:")
print(f"  Resolution: {frame_width} x {frame_height}")
print(f"  FPS: {fps}")
print(f"  Total frames: {total_frames}")
print(f"  Duration: {total_frames/fps:.1f} seconds" if fps > 0 else "  Duration: Unknown")

# Create background subtractor with more sensitive settings for sky detection
bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=50, varThreshold=16, detectShadows=True)

# Kalman Filter setup
kalman = cv2.KalmanFilter(4, 2)
kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                     [0, 1, 0, 0]], np.float32)
kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                    [0, 1, 0, 1],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1]], np.float32)
kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.1

# Tracking state
has_measurement = False
last_predicted_pos = None
frames_without_detection = 0
max_frames_without_detection = 10
horizon_y = None
horizon_detection_interval = 30  # Detect horizon every N frames
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    
    # Display current frame info on first few frames
    if frame_count <= 3:
        print(f"Frame {frame_count}: {frame.shape[1]} x {frame.shape[0]} pixels")
    
    # Create windows with specific sizes to prevent auto-scaling
    if frame_count == 1:
        cv2.namedWindow("Drone Tracker", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Foreground Mask", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Sky Mask", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Original Grayscale", cv2.WINDOW_NORMAL)
        
        # Resize windows to show full resolution (or scale down if too large)
        display_width = min(frame_width, 1200)  # Max width of 1200px
        display_height = int(frame_height * (display_width / frame_width))
        
        cv2.resizeWindow("Drone Tracker", display_width, display_height)
        cv2.resizeWindow("Foreground Mask", display_width, display_height)
        cv2.resizeWindow("Sky Mask", display_width, display_height)
        cv2.resizeWindow("Original Grayscale", display_width, display_height)
        
        print(f"Display windows sized to: {display_width} x {display_height}")
        print(f"Original resolution preserved: {frame_width} x {frame_height}")
    
    # Detect horizon periodically
    if frame_count % horizon_detection_interval == 1:
        detected_horizon = detect_horizon(frame)
        if detected_horizon is not None:
            horizon_y = detected_horizon
            print(f"Horizon detected at y={horizon_y}")
        elif horizon_y is None:
            print("No horizon detected, using fallback sky segmentation")
    
    # Create sky mask
    sky_mask = create_sky_mask(frame, horizon_y)

    # Apply background subtraction
    fg_mask = bg_subtractor.apply(frame)
    
    # Apply additional filtering for better drone detection
    # Convert to grayscale for better contrast detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply adaptive threshold to help detect dark objects against bright sky
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    
    # Combine background subtraction with adaptive threshold
    combined_mask = cv2.bitwise_or(fg_mask, thresh)
    
    # Apply sky mask to filter out non-sky regions
    sky_filtered_mask = cv2.bitwise_and(combined_mask, sky_mask)
    
    # Clean up the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    sky_filtered_mask = cv2.morphologyEx(sky_filtered_mask, cv2.MORPH_OPEN, kernel)
    sky_filtered_mask = cv2.morphologyEx(sky_filtered_mask, cv2.MORPH_CLOSE, kernel)
    
    # Use the sky-filtered mask for detection
    fg_mask = sky_filtered_mask

    # Find contours
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    measured = None
    best_contour = None
    min_distance = float('inf')

    # Find the best contour (closest to last prediction if available)
    contour_count = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        contour_count += 1
        if 10 < area < 5000:  # Very wide area range for debugging
            x, y, w, h = cv2.boundingRect(cnt)
            cx, cy = x + w // 2, y + h // 2
            
            # Draw all potential candidates in yellow for debugging
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 1)
            cv2.putText(frame, f"A:{int(area)}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255), 1)
            
            # If we have a previous prediction, choose the closest contour
            if last_predicted_pos is not None:
                distance = np.sqrt((cx - last_predicted_pos[0])**2 + (cy - last_predicted_pos[1])**2)
                if distance < min_distance and distance < 150:  # Increased distance threshold
                    min_distance = distance
                    best_contour = cnt
                    measured = np.array([[np.float32(cx)], [np.float32(cy)]])
            else:
                # If no previous prediction, take the largest valid contour
                if area > 20:  # Minimum area for consideration
                    best_contour = cnt
                    measured = np.array([[np.float32(cx)], [np.float32(cy)]])
                    break
    
    # Display debug info
    cv2.putText(frame, f"Frame: {frame_count}/{total_frames}", (10, frame_height - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, f"Resolution: {frame_width}x{frame_height}", (10, frame_height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, f"Contours: {contour_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Draw the selected contour
    if best_contour is not None:
        x, y, w, h = cv2.boundingRect(best_contour)
        cx, cy = x + w // 2, y + h // 2
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.circle(frame, (cx, cy), 3, (0, 255, 0), -1)
        cv2.putText(frame, 'Detected', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Kalman predict
    prediction = kalman.predict()
    pred_x, pred_y = int(prediction[0]), int(prediction[1])
    last_predicted_pos = (pred_x, pred_y)

    # Kalman update
    if measured is not None:
        if not has_measurement:
            # Initialize Kalman filter with first measurement
            kalman.statePost = np.array([[measured[0, 0]], 
                                       [measured[1, 0]], 
                                       [0.0], 
                                       [0.0]], dtype=np.float32)
            has_measurement = True
            print("Kalman filter initialized")
        
        kalman.correct(measured)
        frames_without_detection = 0
    else:
        frames_without_detection += 1

    # Draw prediction
    if has_measurement:
        cv2.circle(frame, (pred_x, pred_y), 6, (0, 0, 255), 2)
        cv2.putText(frame, 'Predicted', (pred_x + 10, pred_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        # Show tracking status
        if frames_without_detection > 0:
            status = f"Lost for {frames_without_detection} frames"
            cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        else:
            cv2.putText(frame, "Tracking", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # Draw horizon line if detected
    if horizon_y is not None:
        cv2.line(frame, (0, horizon_y), (frame_width, horizon_y), (255, 0, 255), 2)
        cv2.putText(frame, f"Horizon: {horizon_y}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

    # Reset tracking if lost for too long
    if frames_without_detection > max_frames_without_detection:
        has_measurement = False
        frames_without_detection = 0
        last_predicted_pos = None
        print("Tracking lost - waiting for new detection")

    # Show windows with additional debug info
    cv2.imshow("Drone Tracker", frame)
    cv2.imshow("Foreground Mask", fg_mask)
    cv2.imshow("Original Grayscale", gray)  # Add grayscale view

    # Exit on ESC key
    key = cv2.waitKey(30) & 0xFF
    if key == 27:  # ESC key
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
print("Drone tracking stopped")