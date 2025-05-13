# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import torch
import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description='VGGT Demo')
    parser.add_argument('--scene_dir', type=str, required=True,
                      help='Directory containing the scene images')
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed for reproducibility')
    return parser.parse_args()

def demo_fn(args):
    # Print configuration
    print("Arguments:", vars(args))

    # Configure CUDA settings
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

    # Set seed for reproducibility
    seed_all_random_engines(args.seed)

    # TODO: rewrite the demo loader
    # Load Data
    test_dataset = DemoLoader(
        SCENE_DIR=args.scene_dir,
        img_size=args.img_size,
        normalize_cameras=False,
        load_gt=args.load_gt,
    )

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







