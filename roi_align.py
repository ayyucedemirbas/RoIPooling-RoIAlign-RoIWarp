import numpy as np

def bilinear_interpolation(x, y, feature_map):
    x0 = int(np.floor(x))
    x1 = int(np.ceil(x))
    y0 = int(np.floor(y))
    y1 = int(np.ceil(y))
    
    Q11 = feature_map[:, y0, x0]
    Q12 = feature_map[:, y1, x0]
    Q21 = feature_map[:, y0, x1]
    Q22 = feature_map[:, y1, x1]
    
    R1 = (x1 - x) * Q11 + (x - x0) * Q21
    R2 = (x1 - x) * Q12 + (x - x0) * Q22
    
    return (y1 - y) * R1 + (y - y0) * R2


def roi_align(feature_map, rois, output_size):
    
    num_rois = rois.shape[0]
    output_height, output_width = output_size if isinstance(output_size, tuple) else (output_size, output_size)
    
    # Calculate the height and width of each ROI
    roi_heights = rois[:, 3] - rois[:, 1]
    roi_widths = rois[:, 2] - rois[:, 0]
    
    # Calculate the stride for aligning
    stride_h = roi_heights / output_height
    stride_w = roi_widths / output_width
    
    # Create empty output tensor
    aligned_features = np.zeros((num_rois, feature_map.shape[1], output_height, output_width))
    
    for i in range(num_rois):
        # Generate grid points for the ROI
        grid_h = np.arange(0, output_height) * stride_h[i] + rois[i, 1]
        grid_w = np.arange(0, output_width) * stride_w[i] + rois[i, 0]
        
        # Calculate the grid cell boundaries
        y0 = np.floor(grid_h).astype(int)
        x0 = np.floor(grid_w).astype(int)
        y1 = np.ceil(grid_h + 1).astype(int)
        x1 = np.ceil(grid_w + 1).astype(int)
        
        # Clip the boundaries to the feature map size
        y0 = np.clip(y0, 0, feature_map.shape[2] - 1)
        x0 = np.clip(x0, 0, feature_map.shape[3] - 1)
        y1 = np.clip(y1, 0, feature_map.shape[2] - 1)
        x1 = np.clip(x1, 0, feature_map.shape[3] - 1)
        
        for j in range(output_height):
            for k in range(output_width):
                # Calculate the sub-pixel offsets
                dy = grid_h[j] - y0[j]
                dx = grid_w[k] - x0[k]
                
                # Perform bilinear interpolation
                interpolated_values = bilinear_interpolation(dx, dy, feature_map[i])
                
                # Assign the interpolated values to the aligned feature map
                aligned_features[i, :, j, k] = interpolated_values
    
    return aligned_features
