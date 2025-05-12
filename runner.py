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

