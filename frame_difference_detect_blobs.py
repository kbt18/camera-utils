import cv2
import numpy as np

from collections import deque

blob_history = deque(maxlen=5)  # store the last 5 frames of blobs

# buffer of N previous binary diff masks
frame_buffer = deque(maxlen=7)  # adjust length depending on desired sensitivity

import cv2
import numpy as np

def find_circular_blobs(binary_img, min_area=2, max_area=512, min_circularity=0.3):
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blobs = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue

        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue

        circularity = 4 * np.pi * (area / (perimeter ** 2))
        if circularity < min_circularity:
            continue

        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue

        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        blobs.append(((cx, cy), cnt))

    return blobs

cap = cv2.VideoCapture(r"C:\Users\kbtod\Videos\recordings\recordings\recording-ir-50.h264")

ret, prev_frame = cap.read()
if not ret:
    print("Failed to grab first frame.")
    cap.release()
    exit()

prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

# Make display windows resizable
cv2.namedWindow("Stabilized Difference", cv2.WINDOW_NORMAL)
cv2.namedWindow("Blobs", cv2.WINDOW_NORMAL)

# Optionally set default window size
cv2.resizeWindow("Stabilized Difference", 960, 540)
cv2.resizeWindow("Blobs", 960, 540)

while True:
    ret, curr_frame = cap.read()
    if not ret:
        break

    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

    prev_pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=200, qualityLevel=0.01, minDistance=30)
    if prev_pts is None:
        prev_gray = curr_gray.copy()
        continue

    curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None)

    # Keep only valid matches
    idx = np.where(status == 1)[0]
    if len(idx) < 4:
        prev_gray = curr_gray.copy()
        continue

    prev_pts = prev_pts[idx]
    curr_pts = curr_pts[idx]

    transform, _ = cv2.estimateAffinePartial2D(prev_pts, curr_pts)

    if transform is not None:
        prev_warped = cv2.warpAffine(prev_gray, transform, (curr_gray.shape[1], curr_gray.shape[0]))
        
        # # Before differencing, detect background edges (clouds, trees):
        # background_edges = cv2.Canny(prev_gray, 100, 200)
        # edge_mask = cv2.dilate(background_edges, np.ones((5,5), np.uint8), iterations=1)
        # edge_mask_inv = cv2.bitwise_not(edge_mask)

        # # Create a sky mask to remove trees
        # # --- Compute Texture Map using Laplacian ---
        # laplacian = cv2.Laplacian(curr_gray, cv2.CV_64F, ksize=3)
        # texture_map = cv2.convertScaleAbs(laplacian)  # Convert to 8-bit image

        # # --- Threshold: Sky is Low-Texture (Low Laplacian Magnitude) ---
        # # You can tune 10 -> higher = less sky, lower = more sky
        # _, texture_mask = cv2.threshold(texture_map, 2, 255, cv2.THRESH_BINARY_INV)

        # # --- Heuristic: Keep Only Upper Part (Sky is usually on top) ---
        # sky_mask = np.zeros_like(texture_mask)
        # height = texture_mask.shape[0]
        # sky_mask[:height // 2] = texture_mask[:height // 2]

        # # --- Clean Up with Morphology (Optional) ---
        # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        # sky_mask = cv2.morphologyEx(sky_mask, cv2.MORPH_OPEN, kernel)
        # sky_mask = cv2.morphologyEx(sky_mask, cv2.MORPH_CLOSE, kernel)

        diff = cv2.absdiff(prev_warped, curr_gray)
        _, diff_thresh = cv2.threshold(diff, 8, 255, cv2.THRESH_BINARY)

        # Take the median of the last 5 frames. Suppress anything that doesn't change.
        # on each frame:
        frame_buffer.append(diff_thresh)

        if len(frame_buffer) == frame_buffer.maxlen:
            stacked = np.stack(frame_buffer, axis=0)  # shape: [N, H, W]
            temporal_median = np.median(stacked, axis=0).astype(np.uint8)
            temporal_mask = (temporal_median > 128).astype(np.uint8) * 255
            # Invert temporal mask: 255 becomes 0, 0 becomes 255
            inverted_mask = cv2.bitwise_not(temporal_mask)
            diff_thresh = cv2.bitwise_and(diff_thresh, diff_thresh, mask=temporal_mask)

        # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        # dif_thresh = cv2.morphologyEx(diff_thresh, cv2.MORPH_OPEN, kernel)

        # # try removing trees
        # edges = cv2.Canny(prev_gray, 100, 200)
        # tree_mask = cv2.dilate(edges, np.ones((5,5), np.uint8), iterations=1)
        # tree_mask_inv = cv2.bitwise_not(tree_mask)        
        # diff_thresh = cv2.bitwise_and(diff_thresh, diff_thresh, mask=tree_mask_inv)

        # # fill small holes
        # kernel = np.ones((3, 3), np.uint8)
        # diff_thresh = cv2.morphologyEx(diff_thresh, cv2.MORPH_CLOSE, kernel) 

        # # After thresholding, remove edges:
        # diff_thresh = cv2.bitwise_and(diff_thresh, diff_thresh, mask=edge_mask_inv)

        # apply the sky mask
        # diff_thresh = cv2.bitwise_and(diff_thresh, diff_thresh, mask=sky_mask)

        # Find circular blobs
        blobs = find_circular_blobs(diff_thresh)

        # # Step 1: Track blob history
        # raw_blobs = find_circular_blobs(diff_thresh)
        # blob_history.append(raw_blobs)

        # # Step 2: Filter blobs that appear in at least 3 of the last 5 frames
        # filtered_blobs = []
        # for (cx, cy), cnt in raw_blobs:
        #     count = sum(
        #         # calc dist
        #         any(np.linalg.norm(np.array([cx, cy]) - np.array(prev_cxcy)) < 50 for prev_cxcy, _ in frame_blobs)
        #         for frame_blobs in blob_history
        #     )
        #     if count >= 3:
        #         filtered_blobs.append(((cx, cy), cnt))

        # blobs = filtered_blobs  # use this for drawing

        # Draw blobs on frame
        for (cx, cy), cnt in blobs:
            cv2.circle(curr_frame, (cx, cy), 8, (0, 255, 0), 1)
            cv2.drawContours(curr_frame, [cnt], -1, (0, 255, 0), 1)

        # Resize for display if necessary (optional; remove if you prefer original resolution)
        display_blobs = cv2.resize(curr_frame, (960, 540))
        display_diff = cv2.resize(diff_thresh, (960, 540))

        cv2.imshow("Stabilized Difference", display_diff)
        cv2.imshow("Blobs", display_blobs)
    else:
        print("Transform estimation failed.")

    if cv2.waitKey(1) == 27:  # ESC key to break
        break

    prev_gray = curr_gray.copy()

cap.release()
cv2.destroyAllWindows()
