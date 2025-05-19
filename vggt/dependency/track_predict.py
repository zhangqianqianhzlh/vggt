# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from vggt.dependency.vggsfm_utils import *


def predict_tracks(images, masks=None, max_query_pts=2048, query_frame_num=5, keypoint_extractor="aliked+sp"):
    
    """
    Predict tracks for the given images and masks.

    This function predicts the tracks for the given images and masks using the specified query method
    and track predictor. It finds query points, and predicts the tracks, visibility, and scores for the query frames.

    images: [S, 3, H, W]
    masks: [S, 1, H, W]
    """

    frame_num, _, height, width = images.shape
    device = images.device
    dtype = images.dtype
    tracker = build_vggsfm_tracker().to(device, dtype)
    
    #  Find query frames
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
    pred_conf_scores = []
    
    fmaps_for_tracker = tracker.process_images_to_fmaps(images)


    for query_index in query_frame_indexes:
        query_image = images[query_index]
        query_points = extract_keypoints(query_image, keypoint_extractors)

        reorder_index = calculate_index_mappings(query_index, frame_num, device=device)
        reorder_images = switch_tensor_order([images], reorder_index, dim=0)[0]

        images_feed, fmaps_feed = switch_tensor_order([images, fmaps_for_tracker], reorder_index, dim=0)
        import pdb;pdb.set_trace()

    
    # Find query frames
    # Find query points
    # Predict tracks
    
    
    
    
    
    return None # placeholder
    #
    
    
