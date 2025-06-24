import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from PIL import Image
import cv2
from scipy import ndimage
from skimage import measure, morphology

# Import your existing functions (assuming they're in a file called kmeans_image.py)
# from kmeans_image import load_monochrome_image, prepare_features, apply_kmeans_clustering
from k_means import load_monochrome_image, prepare_features, apply_kmeans_clustering


def identify_ground_region(image, cluster_labels):
    """
    Identify which cluster represents the ground region based on multiple criteria
    """
    height, width = image.shape
    clustered_image = cluster_labels.reshape(height, width)
    
    cluster_scores = {}
    
    for cluster_id in np.unique(cluster_labels):
        mask = clustered_image == cluster_id
        
        # Criterion 1: Position bias - ground is typically in lower part of image
        y_coords, x_coords = np.where(mask)
        mean_y_position = np.mean(y_coords) / height  # Normalized to [0,1]
        position_score = mean_y_position  # Higher score for lower regions
        
        # Criterion 2: Texture/variance - ground often has more texture
        masked_region = image[mask]
        variance_score = np.var(masked_region) / 10000  # Normalized roughly
        
        # Criterion 3: Brightness - sky is often brighter
        brightness_score = 1 - (np.mean(masked_region) / 255)  # Inverted for ground preference
        
        # Criterion 4: Bottom edge presence - ground touches bottom of image
        bottom_row_pixels = np.sum(mask[-10:, :])  # Count pixels in bottom 10 rows
        bottom_edge_score = bottom_row_pixels / (10 * width)
        
        # Combined score (weighted)
        total_score = (
            0.1 * position_score +
            0.2 * variance_score +
            0.2 * brightness_score +
            0.1 * bottom_edge_score
        )
        
        cluster_scores[cluster_id] = {
            'total_score': total_score,
            'position_score': position_score,
            'variance_score': variance_score,
            'brightness_score': brightness_score,
            'bottom_edge_score': bottom_edge_score,
            'pixel_count': np.sum(mask)
        }
    
    # Print analysis for debugging
    print("\nCluster Analysis for Ground Detection:")
    print("-" * 60)
    for cluster_id, scores in cluster_scores.items():
        print(f"Cluster {cluster_id}:")
        print(f"  Total Score: {scores['total_score']:.3f}")
        print(f"  Position (lower=higher): {scores['position_score']:.3f}")
        print(f"  Variance: {scores['variance_score']:.3f}")
        print(f"  Darkness preference: {scores['brightness_score']:.3f}")
        print(f"  Bottom edge presence: {scores['bottom_edge_score']:.3f}")
        print(f"  Pixel count: {scores['pixel_count']}")
        print()
    
    # Select cluster with highest score as ground
    ground_cluster = max(cluster_scores.keys(), key=lambda k: cluster_scores[k]['total_score'])
    print(f"Identified ground cluster: {ground_cluster}")
    
    return ground_cluster

def find_boundary_line(clustered_image, ground_cluster, offset=30):
    """
    Find boundary line between sky and ground regions with offset from ground
    """
    height, width = clustered_image.shape
    
    # Create binary mask: 1 for ground, 0 for sky
    ground_mask = (clustered_image == ground_cluster).astype(np.uint8)
    
    # Clean up the mask with morphological operations
    kernel = np.ones((5, 5), np.uint8)
    ground_mask_clean = cv2.morphologyEx(ground_mask, cv2.MORPH_CLOSE, kernel)
    ground_mask_clean = cv2.morphologyEx(ground_mask_clean, cv2.MORPH_OPEN, kernel)
    
    # Find contours of the ground region
    contours, _ = cv2.findContours(ground_mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        print("No contours found!")
        return []
    
    # Get the largest contour (main ground region)
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Create boundary line by finding topmost points of ground region for each x-coordinate
    boundary_points = []
    
    for x in range(width):
        # Find all ground pixels in this column
        ground_pixels_y = np.where(ground_mask_clean[:, x] == 1)[0]
        
        if len(ground_pixels_y) > 0:
            # Get the topmost ground pixel
            top_ground_y = np.min(ground_pixels_y)
            
            # Apply offset (move boundary away from ground)
            boundary_y = max(0, top_ground_y - offset)
            boundary_points.append((x, boundary_y))
    
    # Smooth the boundary line
    if boundary_points:
        boundary_points = smooth_boundary_line(boundary_points)
    
    return boundary_points

def smooth_boundary_line(boundary_points, window_size=15):
    """
    Smooth the boundary line using a moving average
    """
    if len(boundary_points) < window_size:
        return boundary_points
    
    x_coords = [p[0] for p in boundary_points]
    y_coords = [p[1] for p in boundary_points]
    
    # Apply moving average to y coordinates
    smoothed_y = []
    half_window = window_size // 2
    
    for i in range(len(y_coords)):
        start_idx = max(0, i - half_window)
        end_idx = min(len(y_coords), i + half_window + 1)
        smoothed_y.append(np.mean(y_coords[start_idx:end_idx]))
    
    return list(zip(x_coords, smoothed_y))

def visualize_segmentation_with_boundary(original_image, clustered_image, boundary_points, ground_cluster):
    """
    Visualize the segmentation results with boundary line
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Original image
    axes[0, 0].imshow(original_image, cmap='gray')
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # Clustered image
    axes[0, 1].imshow(clustered_image, cmap='tab10')
    axes[0, 1].set_title('K-means Segmentation (k=2)')
    axes[0, 1].axis('off')
    
    # Ground mask
    ground_mask = (clustered_image == ground_cluster)
    axes[1, 0].imshow(ground_mask, cmap='gray')
    axes[1, 0].set_title(f'Ground Region (Cluster {ground_cluster})')
    axes[1, 0].axis('off')
    
    # Original with boundary line
    axes[1, 1].imshow(original_image, cmap='gray')
    if boundary_points:
        x_coords = [p[0] for p in boundary_points]
        y_coords = [p[1] for p in boundary_points]
        axes[1, 1].plot(x_coords, y_coords, 'r-', linewidth=3, label='Sky-Ground Boundary')
    axes[1, 1].set_title('Boundary Line (10px offset from ground)')
    axes[1, 1].legend()
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.show()

def main():
    """
    Main function for sky-ground segmentation with boundary line detection
    """
    # Load image (update path as needed)
    image_path = r"C:\Users\kbtod\git\camera-utils\screenshot_20250624_111235_frame12829.jpg"
    image = load_monochrome_image(image_path)
    
    if image is None:
        print("Failed to load image. Exiting.")
        return
    
    print(f"Image shape: {image.shape}")
    
    # Prepare features for K-means
    print("Preparing features...")
    features = prepare_features(image)
    
    # Apply K-means clustering with k=2
    print("Applying K-means clustering (k=2)...")
    cluster_labels, kmeans_model, scaler = apply_kmeans_clustering(features, n_clusters=2)
    
    # Reshape to image dimensions
    clustered_image = cluster_labels.reshape(image.shape)
    
    # Identify ground region
    ground_cluster = identify_ground_region(image, cluster_labels)
    
    # Find boundary line with 10-pixel offset
    print("Finding boundary line...")
    boundary_points = find_boundary_line(clustered_image, ground_cluster, offset=10)
    
    print(f"Boundary line contains {len(boundary_points)} points")
    
    # Visualize results
    visualize_segmentation_with_boundary(image, clustered_image, boundary_points, ground_cluster)
    
    # Return results for further use
    return {
        'original_image': image,
        'clustered_image': clustered_image,
        'boundary_points': boundary_points,
        'ground_cluster': ground_cluster,
        'cluster_labels': cluster_labels
    }

if __name__ == "__main__":
    results = main()