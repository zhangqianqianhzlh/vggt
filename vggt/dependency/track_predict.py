# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from vggt.dependency.vggsfm_tracker import TrackerPredictor


def build_vggsfm_tracker(model_path=None):
    if model_path is None:
        default_url = "https://huggingface.co/facebook/VGGSfM/resolve/main/vggsfm_v2_tracker.pt"
        tracker = TrackerPredictor()
        tracker.load_state_dict(torch.hub.load_state_dict_from_url(default_url))
    else:
        tracker = TrackerPredictor()
        tracker.load_state_dict(torch.load(model_path))
    return tracker


def predict_track(tracker, query_frame_indexes, images, masks=None, max_query_pts=2048, ):
    
    """
    Predict tracks for the given images and masks.

    This function predicts the tracks for the given images and masks using the specified query method
    and track predictor. It finds query points, and predicts the tracks, visibility, and scores for the query frames.

    """


    
    import pdb;pdb.set_trace()
    
    
    
    return None # placeholder
    #
    
    
