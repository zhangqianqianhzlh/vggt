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

    keypoint_extractors = initialize_feature_extractors(max_query_pts, keypoint_extractor, device)
    
    pred_tracks = []
    pred_vis_scores = []
    pred_conf_scores = []

    for query_index in query_frame_indexes:
        query_image = images[query_index]
        import pdb;pdb.set_trace()
        query_points_round = extract_keypoints(query_image, keypoint_extractors, max_query_num)



    
    # Find query frames
    # Find query points
    # Predict tracks
    
    
    
    
    
    return None # placeholder
    #
    
    
