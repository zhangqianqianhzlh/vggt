# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import random
import numpy as np
import glob
import os
import torch
import torch.nn.functional as F

# Configure CUDA settings early
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

import argparse
from pathlib import Path

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images_square
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map


def parse_args():
    parser = argparse.ArgumentParser(description='VGGT Demo')
    parser.add_argument('--scene_dir', type=str, required=True,
                      help='Directory containing the scene images')
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed for reproducibility')
    parser.add_argument('--use_ba', action='store_true', default=False,
                      help='Use BA for reconstruction')
    return parser.parse_args()


def run_VGGT(model, images, dtype, resolution=518):
    # images: [B, 3, H, W]
    
    assert len(images.shape) == 4
    assert images.shape[1] == 3
    
    # hard-coded to use 518
    images = F.interpolate(images, size=(resolution, resolution), mode="bicubic", align_corners=False)
    
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            images = images[None] # add batch dimension
            aggregated_tokens_list, ps_idx = model.aggregator(images)
                    
        # Predict Cameras
        pose_enc = model.camera_head(aggregated_tokens_list)[-1]
        # Extrinsic and intrinsic matrices, following OpenCV convention (camera from world)
        extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])
        # Predict Depth Maps
        depth_map, depth_conf = model.depth_head(aggregated_tokens_list, images, ps_idx)

    extrinsic = extrinsic.squeeze(0).cpu().numpy()
    intrinsic = intrinsic.squeeze(0).cpu().numpy()
    depth_map = depth_map.squeeze(0).cpu().numpy()
    depth_conf = depth_conf.squeeze(0).cpu().numpy()
    return extrinsic, intrinsic, depth_map, depth_conf


def demo_fn(args):
    # Print configuration
    print("Arguments:", vars(args))

    # Set seed for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)  # for multi-GPU
    print(f"Setting seed as: {args.seed}")

    # Set device and dtype
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Using dtype: {dtype}")
    
    # Run VGGT for camera and depth estimation
    model = VGGT()
    _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
    model.load_state_dict(torch.hub.load_state_dict_from_url(_URL))
    model.eval()
    model = model.to(device)
    print(f"Model loaded")
    

    # Get image paths and preprocess them 
    image_dir = os.path.join(args.scene_dir, "images")
    image_path_list = glob.glob(os.path.join(image_dir, "*"))
    if len(image_path_list) == 0:
        raise ValueError(f"No images found in {image_dir}")
    
    
    # Load images and original coordinates
    # Load Image in 1024, while running VGGT with 518
    # TODO: also support masks here    
    vggt_fixed_resolution = 518
    img_load_resolution = 1024
    scale = img_load_resolution /vggt_fixed_resolution

    images, original_coords = load_and_preprocess_images_square(image_path_list, img_load_resolution)
    images = images.to(device)
    original_coords = original_coords.to(device)
    print(f"Loaded {len(images)} images from {image_dir}")
    

    # Run VGGT to estimate camera and depth
    # Run with 518x518 images
    extrinsic, intrinsic, depth_map, depth_conf = run_VGGT(model, images, dtype, vggt_fixed_resolution)
    points_3d = unproject_depth_map_to_point_map(depth_map, extrinsic, intrinsic)

    
    if args.use_ba:
        from vggt.dependency.track_predict import predict_tracks
        from vggt.dependency.np_to_pycolmap import batch_np_matrix_to_pycolmap

        with torch.cuda.amp.autocast(dtype=dtype):
            # Predicting Tracks
            # Using VGGSfM tracker instead of VGGT tracker for efficiency
            # VGGT tracker requires multiple backbone runs to query different frames (this is a problem caused by the training process)
            # Will be fixed in VGGT v2

            # You can also change the pred_tracks to any tracks from other trackers
            # e.g., from COLMAP, from CoTracker, or by chaining 2D matches from Lightglue/LoFTR.
            pred_tracks, pred_vis_scores, pred_confs, pred_points_3d = predict_tracks(images, conf=depth_conf, 
                                                                      points_3d=points_3d,
                                                                      masks=None, max_query_pts=2048, 
                                                                      query_frame_num=5, 
                                                                      keypoint_extractor="aliked+sp", 
                                                                      max_points_num=163840, fine_tracking=True)
            torch.cuda.empty_cache()
        
            # rescale the intrinsic matrix from 518 to 1024
            intrinsic[:, :2, :] *= scale
            
            track_mask = (pred_vis_scores > 0.2) 
            image_size = np.array(images.shape[-2:])
            
            max_reproj_error = 8.0
            shared_camera = True
            camera_type = "SIMPLE_PINHOLE"
            
            # TODO: add support for radial distortion, which needs extra_params
            
            reconstruction, valid_track_mask = batch_np_matrix_to_pycolmap(
                pred_points_3d,
                extrinsic,
                intrinsic,
                pred_tracks,
                image_size,
                masks=track_mask,
                max_reproj_error=max_reproj_error,
                shared_camera=shared_camera,
                camera_type=camera_type,
            )

            
            
            # Now we have the tracks, visibility scores, confidences, and 3D points
            
            
            import pdb; pdb.set_trace()




            # Step 2: Filter out tracks based on visibility scores and depth confidence
            # Step 3: 
    
    # from vggt.dependency.track_predict import predict_track, build_vggsfm_tracker
    # from vggt.dependency.vggsfm_tracker import TrackerPredictor
    # tracker = TrackerPredictor()
    # 
    # tracker.load_state_dict(torch.hub.load_state_dict_from_url("https://huggingface.co/facebook/VGGSfM/resolve/main/vggsfm_v2_tracker.pt"))


    # torchvision.utils.save_image(images[:,:, 85:431, 0:518], "images.png")

    sequence_list = test_dataset.sequence_list
    seq_name = sequence_list[0]  # Run on one Scene

    batch, image_paths = test_dataset.get_data(
        sequence_name=seq_name, return_path=True
    )

    output_dir = batch["scene_dir"]

    images = batch["image"]
    masks = batch["masks"] if batch["masks"] is not None else None
    crop_params = batch["crop_params"] if batch["crop_params"] is not None else None
    original_images = batch["original_images"]

    # QQ:
    # Do I want to implement a new runner for vggt?
    # probably not, I hope to keep it as simple as possible

    print("Demo Finished Successfully")
    return True

if __name__ == "__main__":
    args = parse_args()
    with torch.no_grad():
        demo_fn(args)
        
        
        
        
        
        
# DO NOT TOUCH THE STUFFS BELOW

"""
VGGT Runner Script
=================

A script to run the VGGT model for 3D reconstruction from image sequences.

Directory Structure
------------------
Input:
    input_folder/
    └── images/            # Source images for reconstruction

Output:
    output_folder/
    ├── images/
    ├── sparse/           # Reconstruction results
    │   ├── cameras.bin   # Camera parameters (COLMAP format)
    │   ├── images.bin    # Pose for each image (COLMAP format)
    │   ├── points3D.bin  # 3D points (COLMAP format)
    │   └── points.ply    # Point cloud visualization file
    └── visuals/          # Visualization outputs

Key Features
-----------
• Dual-mode Support: Run reconstructions using either VGGT or VGGT+BA
• Resolution Preservation: Maintains original image resolution in camera parameters and tracks
• COLMAP Compatibility: Exports results in standard COLMAP sparse reconstruction format
• Point Cloud Export: Generates additional PLY file for easy visualization
"""

# Plans
# This should be a script that runs the model given an input folder
# The folder should follow the structure of vggsfm, with a images folder inside, e.g., case1/images, case2/images, etc.
# The script can support vggt and vggt+BA 
# The script should save the reconstruction results (pose, point, track) under "/sparse" folder under the case folder, e.g., case1/sparse
# The results should be saved in the format of colmap sparse reconstruction results. Additionaly, saving the point cloud as a separate ply file.
# The returned cameras and tracks should stay in the original resolution of the images.
# Also provide an instruction about how to run a gaussian/nerf over the returned results. (option: nerfstudio?)
# Support the visualization as in VGGSfM? (Optional)
# (Remind myself: for vggt+BA, it is better to use the tracker from VGGSfM, as its tracker head can support query at any frame with only one run)
# (but need to fix this problem in vggt tracker head in the future)
# This should also be able to handle masks






