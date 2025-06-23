import cv2
import numpy as np

# Path to your grayscale video file
video_path = r"C:\Users\kbtod\Videos\recordings\recordings\recording-ir-50.h264"

# Open the video file
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Threshold value (you can adjust this)
threshold_value = 130

while True:
    ret, frame = cap.read()
    if not ret:
        break  # End of video

    # Since the video is already grayscale, frame is single channel
    # Apply threshold
    _, thresh_frame = cv2.threshold(frame, threshold_value, 255, cv2.THRESH_BINARY)

    # Stack original and thresholded frames horizontally for display
    combined = np.hstack((frame, thresh_frame))

    # Show combined frame
    cv2.imshow('Original (Left) | Thresholded (Right)', combined)

    # Wait for 25 ms and break on 'q' key press
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
