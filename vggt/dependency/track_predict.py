# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from vggt.dependency.vggsfm_utils import *


def predict_tracks(images, masks=None, max_query_pts=2048, query_frame_num=5,
                   keypoint_extractor="aliked+sp", 
                   max_points_num=163840, fine_tracking=True):

    """
    Predict tracks for the given images and masks.

    This function predicts the tracks for the given images and masks using the specified query method
    and track predictor. It finds query points, and predicts the tracks, visibility, and scores for the query frames.

    Args:
        images: Tensor of shape [S, 3, H, W] containing the input images.
        masks: Optional tensor of shape [S, 1, H, W] containing masks. Default is None.
        max_query_pts: Maximum number of query points. Default is 2048.
        query_frame_num: Number of query frames to use. Default is 5.
        keypoint_extractor: Method for keypoint extraction. Default is "aliked+sp".
        max_points_num: Maximum number of points to process at once. Default is 163840.
        fine_tracking: Whether to use fine tracking. Default is True.

    Returns:
        pred_tracks: Numpy array containing the predicted tracks.
        pred_vis_scores: Numpy array containing the visibility scores for the tracks.
    """

    frame_num, _, height, width = images.shape
    device = images.device
    dtype = images.dtype
    tracker = build_vggsfm_tracker().to(device, dtype)

    # Find query frames
    query_frame_indexes = generate_rank_by_dino(images, query_frame_num=query_frame_num, device=device)

    # Add the first image to the front if not already present
    if 0 in query_frame_indexes:
        query_frame_indexes.remove(0)
    query_frame_indexes = [0, *query_frame_indexes]

    if masks is None:
        masks = torch.ones_like(images[:, 0:1])

    keypoint_extractors = initialize_feature_extractors(max_query_pts, extractor_method=keypoint_extractor, device=device)

    pred_tracks = []
    pred_vis_scores = []

    fmaps_for_tracker = tracker.process_images_to_fmaps(images)

    if fine_tracking:
        print("For faster inference, consider disabling fine_tracking") 
        
    for query_index in query_frame_indexes:
        print(f"Predicting tracks for query frame {query_index}")
        query_image = images[query_index]
        query_points = extract_keypoints(query_image, keypoint_extractors, round_keypoints=False)
        query_points = query_points[:, torch.randperm(query_points.shape[1], device=device)]

        reorder_index = calculate_index_mappings(query_index, frame_num, device=device)
        reorder_images = switch_tensor_order([images], reorder_index, dim=0)[0]

        images_feed, fmaps_feed = switch_tensor_order([images, fmaps_for_tracker], reorder_index, dim=0)
        images_feed = images_feed[None]  # add batch dimension
        fmaps_feed = fmaps_feed[None]  # add batch dimension
    
        all_points_num = images_feed.shape[1] * query_points.shape[1]

        # Don't need to be scared, this is just chunking to make GPU happy
        if all_points_num > max_points_num:
            num_splits = (all_points_num + max_points_num - 1) // max_points_num
            query_points = torch.chunk(query_points, num_splits, dim=1)
        else:
            query_points = [query_points]

        pred_track, pred_vis, _ = predict_tracks_in_chunks(
            tracker,
            images_feed,
            query_points,
            fmaps_feed,
            fine_tracking=fine_tracking,
        )

        pred_track, pred_vis = switch_tensor_order(
            [pred_track, pred_vis], reorder_index, dim=1
        )

        # Convert from BFloat16 to Float32 before converting to numpy
        pred_tracks.append(pred_track[0].to(torch.float32).cpu().numpy())
        pred_vis_scores.append(pred_vis[0].to(torch.float32).cpu().numpy())
        
    pred_tracks = np.concatenate(pred_tracks, axis=1)
    pred_vis_scores = np.concatenate(pred_vis_scores, axis=1)
    
    # from vggt.utils.visual_track import visualize_tracks_on_images
    # visualize_tracks_on_images(images[None], torch.from_numpy(pred_tracks[None]), torch.from_numpy(pred_vis_scores[None])>0.2, out_dir="track_visuals")

    return pred_tracks, pred_vis_scores




