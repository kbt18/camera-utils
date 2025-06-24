import cv2
import numpy as np

from collections import deque

from k_means import prepare_features, apply_kmeans_clustering
from k_means_with_boundary import find_boundary_line, identify_ground_region

class KMeansFrameDiffPipeline:

    def getFrameDiff(self, prev_frame: np.ndarray, curr_frame: np.ndarray): 
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

        prev_pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=200, qualityLevel=0.01, minDistance=30)
        if prev_pts is None:
            prev_gray = curr_gray.copy()
            return np.zeros_like(curr_frame[:, :, 0])

        curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None)

        # Keep only valid matches
        idx = np.where(status == 1)[0]
        if len(idx) < 4:
            prev_gray = curr_gray.copy()
            return np.zeros_like(curr_frame[:, :, 0])

        prev_pts = prev_pts[idx]
        curr_pts = curr_pts[idx]

        transform, _ = cv2.estimateAffinePartial2D(prev_pts, curr_pts)

        if transform is not None:
            prev_warped = cv2.warpAffine(prev_gray, transform, (curr_gray.shape[1], curr_gray.shape[0]))

        diff = cv2.absdiff(prev_warped, curr_gray)
        _, diff_thresh = cv2.threshold(diff, 8, 255, cv2.THRESH_BINARY)

        return diff_thresh
    
    def getSkySegmentationMask(self, frame, offset=10):
        original_height, original_width = frame.shape[:2]
        
        # Resize while maintaining aspect ratio
        target_width = 200
        aspect_ratio = original_height / original_width
        resized_height = int(target_width * aspect_ratio)
        resized_frame = cv2.resize(frame, (target_width, resized_height), interpolation=cv2.INTER_AREA)
        

        gray = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)

        features = prepare_features(gray)
    
        # Apply K-means clustering with k=2
        print("Applying K-means clustering (k=2)...")
        cluster_labels, kmeans_model, scaler = apply_kmeans_clustering(features, n_clusters=2)
        
        # Reshape to image dimensions
        clustered_image = cluster_labels.reshape(gray.shape)

        disp_clustered = (clustered_image * 255).astype(np.uint8)
        cv2.imshow("Clustered Image", disp_clustered)
        
        # Identify ground region
        ground_cluster = identify_ground_region(gray, cluster_labels)
        
        # Create initial sky mask (sky is not ground cluster)
        initial_sky_mask = (clustered_image != ground_cluster).astype(np.uint8) * 255

        # Resize mask back to original frame size
        full_res_mask = cv2.resize(initial_sky_mask, (original_width, original_height), interpolation=cv2.INTER_NEAREST)


        # Convert to binary OpenCV-style mask
        return full_res_mask

    def run(self, video_path):
        cap = cv2.VideoCapture(video_path)
        prev_n_frames = deque(maxlen=3)
        while True:
            ret, curr_frame = cap.read()
            if not ret:
                break

            if len(prev_n_frames) != 3:
                prev_n_frames.append(curr_frame)
                continue  # skip processing on first frame because no previous frame to diff
            
            diff_mask = self.getFrameDiff(prev_n_frames.popleft(), curr_frame)
            
            sky_mask = self.getSkySegmentationMask(curr_frame)

            # Apply sky mask to diff mask
            sky_motion_mask = cv2.bitwise_and(diff_mask, diff_mask, mask=sky_mask)

            # Show results
            cv2.imshow("Sky Motion Mask", sky_motion_mask)
            cv2.imshow("Sky Mask", sky_mask)
            cv2.imshow("Frame Diff", diff_mask)
            cv2.imshow("Current Frame", curr_frame)

            key = cv2.waitKey(1)
            if key == 27:  # ESC key
                break

            prev_n_frames.append(curr_frame)

        cap.release()
        cv2.destroyAllWindows()

def main():
    pipeline = KMeansFrameDiffPipeline()
    pipeline.run(r"C:\Users\kbtod\Videos\recordings\recordings\recording-visible-50.h264")

if __name__ == "__main__":
    main()
