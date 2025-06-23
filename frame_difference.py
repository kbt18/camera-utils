import cv2
import numpy as np

# Open video source (0 for webcam or path to file)
cap = cv2.VideoCapture(r"C:\Users\kbtod\datasets\kopi_av_dataset\data\IMG_3806.MOV")

# Read the first frame
ret, prev_frame = cap.read()
if not ret:
    print("Failed to grab first frame.")
    cap.release()
    exit()

# Convert to grayscale
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

while True:
    ret, curr_frame = cap.read()
    if not ret:
        break

    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

    # === Step 1: Estimate Transform (motion between frames) ===
    # Use good features to track
    prev_pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=200, qualityLevel=0.01, minDistance=30)
    curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None)

    # Filter valid matches
    idx = np.where(status == 1)[0]
    prev_pts = prev_pts[idx]
    curr_pts = curr_pts[idx]

    # Estimate affine transform (rigid: translation + rotation)
    transform, _ = cv2.estimateAffinePartial2D(curr_pts, prev_pts)

    # === Step 2: Warp current frame to align with previous ===
    stabilized_curr = cv2.warpAffine(curr_gray, transform, (curr_gray.shape[1], curr_gray.shape[0]))

    # === Step 3: Frame Differencing ===
    diff = cv2.absdiff(prev_gray, stabilized_curr)

    # Optional: Threshold to highlight moving regions
    _, diff_thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

    # Display results
    cv2.imshow("Original", curr_frame)
    cv2.imshow("Stabilized Difference", diff_thresh)

    # Press ESC to exit
    if cv2.waitKey(1) == 27:
        break

    # Update previous frame
    prev_gray = stabilized_curr.copy()

cap.release()
cv2.destroyAllWindows()