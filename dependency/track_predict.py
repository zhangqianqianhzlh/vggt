# ... existing code ...

# In _forward_on_query function
# Potential indexing issue: If scale*coordinates exceeds conf dimensions
query_points_scaled = (query_points.squeeze(0) * scale).round().long()
query_points_np = query_points_scaled.cpu().numpy()

# These lines could potentially cause index out of bounds errors
pred_conf = conf[query_index][query_points_np[:, 1], query_points_np[:, 0]]
pred_point = points_3d[query_index][query_points_np[:, 1], query_points_np[:, 0]]

# ... existing code ...

# In predict_tracks function
# Concatenation might fail if pred_confs or pred_points are empty lists
pred_confs = np.concatenate(pred_confs, axis=0) if pred_confs else None
pred_points = np.concatenate(pred_points, axis=0) if pred_points else None
# ... existing code ... 