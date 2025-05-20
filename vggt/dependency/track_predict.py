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
        # Grid search for optimal fine_chunk value
        import time
        import statistics
        
        # Make sure previous operations are completed
        torch.cuda.synchronize()
        
        # Define the grid of fine_chunk values to test
        fine_chunk_values = [-1, 1024, 4096, 10240, 20480, 40960]
        
        # Dictionary to store results for each value
        results = {}
        
        print("Starting grid search for optimal fine_chunk value...")
        
        # For tracking the best performance
        best_time = float('inf')
        best_fine_chunk = None
        all_tracks = []
        all_vis = []
        
        # Run benchmark for each fine_chunk value
        for fine_chunk in fine_chunk_values:
            times = []
            
            print(f"\nTesting fine_chunk={fine_chunk}")
            
            # Run 10 times for each value
            for run in range(10):
                torch.cuda.synchronize()
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                
                start_event.record()
                pred_track_temp, pred_vis_temp, extra_temp = predict_tracks_in_chunks(
                    tracker,
                    images_feed,
                    query_points,
                    fmaps_feed,
                    fine_tracking=fine_tracking,
                    fine_chunk=fine_chunk,
                )
                end_event.record()
                
                torch.cuda.synchronize()
                elapsed_time = start_event.elapsed_time(end_event) / 1000  # convert to seconds
                times.append(elapsed_time)
                print(f"  Run {run+1}/10: {elapsed_time:.4f} seconds")
            
            # Save the last prediction for comparison
            all_tracks.append(pred_track_temp)
            all_vis.append(pred_vis_temp)
            
            # Calculate statistics
            avg_time = statistics.mean(times)
            std_time = statistics.stdev(times) if len(times) > 1 else 0
            
            # Store results
            results[fine_chunk] = {
                'avg_time': avg_time,
                'std_time': std_time,
                'times': times
            }
            
            # Update best performer
            if avg_time < best_time:
                best_time = avg_time
                best_fine_chunk = fine_chunk
        
        # Print summary results
        print("\n===== Grid Search Results =====")
        print(f"{'fine_chunk':<10} | {'avg_time (s)':<12} | {'std_dev (s)':<12}")
        print("-" * 40)
        
        for fine_chunk in fine_chunk_values:
            res = results[fine_chunk]
            print(f"{fine_chunk:<10} | {res['avg_time']:<12.4f} | {res['std_time']:<12.4f}")
        
        print("\nBest fine_chunk value:", best_fine_chunk)
        print(f"Best average time: {best_time:.4f} seconds")
        
        # Verify all results are equivalent
        print("\nVerifying result consistency...")
        all_results_equal = True
        
        for i in range(1, len(fine_chunk_values)):
            tracks_equal = torch.allclose(all_tracks[0], all_tracks[i], atol=1e-5)
            vis_equal = torch.allclose(all_vis[0], all_vis[i], atol=1e-5)
            
            if not (tracks_equal and vis_equal):
                all_results_equal = False
                print(f"Warning: Results for fine_chunk={fine_chunk_values[i]} differ from fine_chunk={fine_chunk_values[0]}")
        
        if all_results_equal:
            print("All configurations produce equivalent results!")
        
        # Use the results from the best configuration
        best_idx = fine_chunk_values.index(best_fine_chunk)
        pred_track, pred_vis = all_tracks[best_idx], all_vis[best_idx]
        
        print(f"\nUsing results from best configuration (fine_chunk={best_fine_chunk})")
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
