import numpy as np

def roi_warp(feature_map, rois, output_size):
    num_rois = rois.shape[0]
    output_height, output_width = output_size if isinstance(output_size, tuple) else (output_size, output_size)
    
    # Calculate the height and width of each ROI
    roi_heights = rois[:, 3] - rois[:, 1]
    roi_widths = rois[:, 2] - rois[:, 0]
    
    # Calculate the stride for warping
    stride_h = roi_heights / output_height
    stride_w = roi_widths / output_width
    
    # Create empty output tensor
    warped_features = np.zeros((num_rois, feature_map.shape[1], output_height, output_width))
    
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
                # Calculate the affine transformation matrix
                dx = grid_w[k] - x0[k]
                dy = grid_h[j] - y0[j]
                dx_ratio = dx / (x1[k] - x0[k])
                dy_ratio = dy / (y1[j] - y0[j])
                affine_matrix = np.array([[1 - dx_ratio, dx_ratio], [1 - dy_ratio, dy_ratio]])
                
                # Calculate the source coordinates for the affine transformation
                source_x = np.array([x0[k], x1[k]])
                source_y = np.array([y0[j], y1[j]])
                
                # Apply the affine transformation to the feature map
                warped_features[i, :, j, k] = bilinear_transform(feature_map[i], source_x, source_y, affine_matrix)
    
    return warped_features


def bilinear_transform(feature_map, source_x, source_y, affine_matrix):
    
    C, H, W = feature_map.shape
    target_height = source_y.shape[0]
    target_width = source_x.shape[0]
    
    target_x, target_y = np.meshgrid(np.arange(target_width), np.arange(target_height))
    target_x = target_x.flatten()
    target_y = target_y.flatten()
    
    source_coords = np.vstack((source_x, source_y))
    target_coords = np.vstack((target_x, target_y))
    
    # Apply the affine transformation to the target coordinates
    transformed_coords = np.dot(affine_matrix, target_coords)
    transformed_x = transformed_coords[0]
    transformed_y = transformed_coords[1]
    
    # Clip the transformed coordinates to the feature map size
    transformed_x = np.clip(transformed_x, 0, W - 1)
    transformed_y = np.clip(transformed_y, 0, H - 1)
    
    # Perform bilinear interpolation
    x0 = np.floor(transformed_x).astype(int)
    x1 = np.ceil(transformed_x).astype(int)
    y0 = np.floor(transformed_y).astype(int)
    y1 = np.ceil(transformed_y).astype(int)
    
    Q11 = feature_map[:, y0, x0]
    Q12 = feature_map[:, y1, x0]
    Q21 = feature_map[:, y0, x1]
    Q22 = feature_map[:, y1, x1]
    
    R1 = (x1 - transformed_x) * Q11 + (transformed_x - x0) * Q21
    R2 = (x1 - transformed_x) * Q12 + (transformed_x - x0) * Q22
    
    transformed_features = (y1 - transformed_y) * R1 + (transformed_y - y0) * R2
    transformed_features = transformed_features.reshape(C, target_height, target_width)
    
    return transformed_features
