import numpy as np

def roi_pooling(feature_map, rois, output_size):
    num_rois = rois.shape[0]
    output_height, output_width = output_size if isinstance(output_size, tuple) else (output_size, output_size)
    
    # Calculate the height and width of each ROI
    roi_heights = rois[:, 3] - rois[:, 1]
    roi_widths = rois[:, 2] - rois[:, 0]
    
    # Calculate the stride for pooling
    stride_h = roi_heights / output_height
    stride_w = roi_widths / output_width
    
    # Create empty output tensor
    pooled_features = np.zeros((num_rois, feature_map.shape[1], output_height, output_width))
    
    for i in range(num_rois):
        # Generate grid points for the ROI
        grid_h = np.arange(0, output_height) * stride_h[i] + rois[i, 1]
        grid_w = np.arange(0, output_width) * stride_w[i] + rois[i, 0]
        
        # Calculate the integer indices for pooling
        grid_h0 = np.floor(grid_h).astype(int)
        grid_w0 = np.floor(grid_w).astype(int)
        grid_h1 = np.ceil(grid_h).astype(int)
        grid_w1 = np.ceil(grid_w).astype(int)
        
        # Clip indices to the feature map size
        grid_h0 = np.clip(grid_h0, 0, feature_map.shape[2] - 1)
        grid_w0 = np.clip(grid_w0, 0, feature_map.shape[3] - 1)
        grid_h1 = np.clip(grid_h1, 0, feature_map.shape[2] - 1)
        grid_w1 = np.clip(grid_w1, 0, feature_map.shape[3] - 1)
        
        # Get the corresponding features for each index
        features = feature_map[i].reshape(feature_map.shape[1], -1)
        pooled_features[i] = np.maximum(np.maximum(features[:, grid_h0, grid_w0], features[:, grid_h0, grid_w1]),
                                        np.maximum(features[:, grid_h1, grid_w0], features[:, grid_h1, grid_w1]))
    
    return pooled_features
