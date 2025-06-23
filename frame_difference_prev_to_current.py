import cv2
import numpy as np

cap = cv2.VideoCapture(r"C:\Users\kbtod\datasets\kopi_av_dataset\data\IMG_2272.MOV")

ret, prev_frame = cap.read()
if not ret:
    print("Failed to grab first frame.")
    cap.release()
    exit()

prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

while True:
    ret, curr_frame = cap.read()
    if not ret:
        break

    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

    # Detect features in prev frame
    prev_pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=200, qualityLevel=0.01, minDistance=30)
    curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None)

    # Keep only valid matches
    idx = np.where(status == 1)[0]
    prev_pts = prev_pts[idx]
    curr_pts = curr_pts[idx]

    # Estimate transform from prev → curr
    transform, _ = cv2.estimateAffinePartial2D(prev_pts, curr_pts)

    if transform is not None:
        # Warp previous frame to align with current frame
        prev_warped = cv2.warpAffine(prev_gray, transform, (curr_gray.shape[1], curr_gray.shape[0]))

        # Frame differencing
        diff = cv2.absdiff(prev_warped, curr_gray)
        _, diff_thresh = cv2.threshold(diff, 15, 255, cv2.THRESH_BINARY)

        cv2.imshow("Stabilized Difference", diff_thresh)
    else:
        print("Transform estimation failed.")

    cv2.imshow("Original", curr_frame)

    if cv2.waitKey(1) == 27:
        break

    prev_gray = curr_gray.copy()

cap.release()
cv2.destroyAllWindows()
