# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


"""Visual Geometry Grounded Transformer (VGGT).

VGGT is a feed-forward neural network that directly infers all key 3D attributes of a scene,
including extrinsic and intrinsic camera parameters, point maps, depth maps, and 3D point tracks,
from one, a few, or hundreds of its views, within seconds.
"""

__version__ = "0.0.1"

from .models.vggt import VGGT

__all__ = ["VGGT"] 