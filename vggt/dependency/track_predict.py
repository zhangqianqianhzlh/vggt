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

    images: [S, 3, H, W]
    masks: [S, 1, H, W]
    
    # 163840/81920
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
        query_points = extract_keypoints(query_image, keypoint_extractors, round_keypoints=False)
        query_points = query_points[:, torch.randperm(query_points.shape[1], device=device)]

        reorder_index = calculate_index_mappings(query_index, frame_num, device=device)
        reorder_images = switch_tensor_order([images], reorder_index, dim=0)[0]

        images_feed, fmaps_feed = switch_tensor_order([images, fmaps_for_tracker], reorder_index, dim=0)
        images_feed = images_feed[None] # add batch dimension
        fmaps_feed = fmaps_feed[None] # add batch dimension
    
        all_points_num = images_feed.shape[1] * query_points.shape[1]

        if all_points_num > max_points_num:
            num_splits = (all_points_num + max_points_num - 1) // max_points_num
            query_points = torch.chunk(query_points, num_splits, dim=1)
        else:
            query_points = [query_points]

        #########################################################
        # First function call - with CUDA timing
        import time
        import torch.cuda
        
        # Make sure previous operations are completed
        torch.cuda.synchronize()
        
        print("Running first predict_tracks_in_chunks with fine_chunk=10240...")
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        pred_track1, pred_vis1, extra1 = predict_tracks_in_chunks(
            tracker,
            images_feed,
            query_points,
            fmaps_feed,
            fine_tracking=fine_tracking,
            fine_chunk=10240,
        )
        end_event.record()
        
        # Wait for GPU to finish
        torch.cuda.synchronize()
        elapsed_time = start_event.elapsed_time(end_event) / 1000  # convert to seconds
        print(f"First function call took {elapsed_time:.4f} seconds (GPU time)")

        # Second function call - with CUDA timing
        torch.cuda.synchronize()
        
        print("Running second predict_tracks_in_chunks with fine_chunk=-1...")
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        pred_track2, pred_vis2, extra2 = predict_tracks_in_chunks(
            tracker,
            images_feed,
            query_points,
            fmaps_feed,
            fine_tracking=fine_tracking,
            fine_chunk=-1, 
        )
        end_event.record()
        
        # Wait for GPU to finish
        torch.cuda.synchronize()
        elapsed_time = start_event.elapsed_time(end_event) / 1000  # convert to seconds
        print(f"Second function call took {elapsed_time:.4f} seconds (GPU time)")

        # Compare results to ensure they're equivalent
        is_track_equal = torch.allclose(pred_track1, pred_track2, atol=1e-5)
        is_vis_equal = torch.allclose(pred_vis1, pred_vis2, atol=1e-5)
        print(f"Results equal: tracks={is_track_equal}, visibility={is_vis_equal}")

        # Use the second result for the rest of the code
        pred_track, pred_vis = pred_track2, pred_vis2
        
        #########################################################
        
        import pdb;pdb.set_trace()
        # Comment out the debugging code for benchmarking
        # import pdb;pdb.set_trace()
        
        # from vggt.utils.visual_track import visualize_tracks_on_images
        # visualize_tracks_on_images(images_feed, pred_track[:,:, :1000], pred_vis[:,:, :1000]>0.2, out_dir="track_visuals")
        # import pdb;pdb.set_trace()




    # Find query frames
    # Find query points
    # Predict tracks





    return None # placeholder
    #
