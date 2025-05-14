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
    return parser.parse_args()


def run_VGGT(model, images, dtype):
    # images: [B, 3, H, W]
    
    assert len(images.shape) == 4
    assert images.shape[1] == 3
    
    # hard-coded to use 518
    images = F.interpolate(images, size=(518, 518), mode="bicubic", align_corners=False)
    
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
    # Default to square images with 1024x1024 resolution
    images, original_coords = load_and_preprocess_images_square(image_path_list)
    images = images.to(device)
    original_coords = original_coords.to(device)
    print(f"Loaded {len(images)} images from {image_dir}")
    

    # Run VGGT to estimate camera and depth
    # Run with 518x518 images
    extrinsic, intrinsic, depth_map, depth_conf = run_VGGT(model, images, dtype)

    import pdb; pdb.set_trace()
    # from vggt.dependency.vggsfm_tracker import TrackerPredictor
    # tracker = TrackerPredictor()


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






