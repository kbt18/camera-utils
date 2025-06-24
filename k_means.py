import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from PIL import Image
import cv2

def load_monochrome_image(image_path):
    """
    Load and convert image to monochrome (grayscale)
    """
    try:
        # Load image using PIL
        img = Image.open(image_path)
        # Convert to grayscale if not already
        img_gray = img.convert('L')
        # Convert to numpy array
        img_array = np.array(img_gray)
        return img_array
    except Exception as e:
        print(f"Error loading image: {e}")
        return None

def prepare_features(image_array):
    """
    Prepare features for K-means clustering: x, y coordinates and intensity
    """
    height, width = image_array.shape
    
    # Create coordinate matrices
    x_coords, y_coords = np.meshgrid(np.arange(width), np.arange(height))
    
    # Flatten arrays to create feature vectors
    x_flat = x_coords.flatten()
    y_flat = y_coords.flatten()
    intensity_flat = image_array.flatten()
    
    # Stack features: [x, y, intensity]
    features = np.stack([x_flat, y_flat, intensity_flat], axis=1)
    
    return features

def apply_kmeans_clustering(features, n_clusters=5, random_state=42):
    """
    Apply K-means clustering to the features
    """
    # Standardize features to give equal weight to spatial and intensity information
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Apply K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    cluster_labels = kmeans.fit_predict(features_scaled)
    
    return cluster_labels, kmeans, scaler

def visualize_results(original_image, cluster_labels, n_clusters):
    """
    Visualize the original image and clustering results
    """
    height, width = original_image.shape
    
    # Reshape cluster labels back to image dimensions
    clustered_image = cluster_labels.reshape(height, width)
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(original_image, cmap='gray')
    axes[0].set_title('Original Monochrome Image')
    axes[0].axis('off')
    
    # Clustered image with different colors for each cluster
    axes[1].imshow(clustered_image, cmap='tab10')
    axes[1].set_title(f'K-means Clustering (k={n_clusters})')
    axes[1].axis('off')
    
    # Cluster overlay on original
    axes[2].imshow(original_image, cmap='gray', alpha=0.7)
    axes[2].imshow(clustered_image, cmap='tab10', alpha=0.3)
    axes[2].set_title('Overlay: Clusters on Original')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()

def analyze_clusters(features, cluster_labels, n_clusters):
    """
    Analyze and print cluster statistics
    """
    print(f"\nCluster Analysis for {n_clusters} clusters:")
    print("-" * 50)
    
    for i in range(n_clusters):
        cluster_mask = cluster_labels == i
        cluster_features = features[cluster_mask]
        
        # Calculate statistics for each cluster
        mean_x = np.mean(cluster_features[:, 0])
        mean_y = np.mean(cluster_features[:, 1])
        mean_intensity = np.mean(cluster_features[:, 2])
        cluster_size = len(cluster_features)
        
        print(f"Cluster {i}:")
        print(f"  Size: {cluster_size} pixels ({cluster_size/len(features)*100:.1f}%)")
        print(f"  Center: ({mean_x:.1f}, {mean_y:.1f})")
        print(f"  Mean Intensity: {mean_intensity:.1f}")
        print()

def main():
    """
    Main function to run the K-means clustering on a monochrome image
    """
    image = load_monochrome_image(r"C:\Users\kbtod\git\camera-utils\screenshot_20250623_120452_frame5200.jpg")
    
    if image is None:
        print("Failed to load image. Exiting.")
        return
    
    print(f"Image shape: {image.shape}")
    
    # Prepare features
    print("Preparing features (x, y, intensity)...")
    features = prepare_features(image)
    print(f"Feature matrix shape: {features.shape}")
    
    # Apply K-means clustering
    n_clusters = 2
    print(f"Applying K-means clustering with {n_clusters} clusters...")
    cluster_labels, kmeans_model, scaler = apply_kmeans_clustering(features, n_clusters)
    
    # Visualize results
    print("Visualizing results...")
    visualize_results(image, cluster_labels, n_clusters)
    
    # Analyze clusters
    analyze_clusters(features, cluster_labels, n_clusters)
    
    # Optional: Try different numbers of clusters
    print("\nTrying different numbers of clusters...")
    for k in [2, 3, 4]:
        labels_k, _, _ = apply_kmeans_clustering(features, k)
        clustered_img_k = labels_k.reshape(image.shape)
        
        plt.figure(figsize=(8, 6))
        plt.imshow(clustered_img_k, cmap='tab10')
        plt.title(f'K-means Clustering (k={k})')
        plt.axis('off')
        plt.show()

if __name__ == "__main__":
    main()