import os
import torch
import numpy as np
import gzip
import json

from vggt.models.vggt import VGGT
from vggt.utils.rotation import mat_to_quat

from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import closed_form_inverse_se3, unproject_depth_map_to_point_map



def convert_pt3d_RT_to_opencv(Rot, Trans):
    # Convert pt3d extrinsic to opencv
    rot_pt3d = np.array(Rot)
    trans_pt3d = np.array(Trans)

    trans_pt3d[:2] *= -1
    rot_pt3d[:, :2] *= -1
    rot_pt3d = rot_pt3d.transpose(1, 0)
    extri_opencv = np.hstack((rot_pt3d, trans_pt3d[:, None]))
    return extri_opencv



def build_pair_index(N, B=1):
    # B, N = se3.shape[:2]
    i1_, i2_ = torch.combinations(torch.arange(N), 2, with_replacement=False).unbind(-1)
    i1, i2 = [(i[None] + torch.arange(B)[:, None] * N).reshape(-1) for i in [i1_, i2_]]

    return i1, i2



def rotation_angle(rot_gt, rot_pred, batch_size=None, eps=1e-15):
    #########
    q_pred = mat_to_quat(rot_pred)
    q_gt = mat_to_quat(rot_gt)

    loss_q = (1 - (q_pred * q_gt).sum(dim=1) ** 2).clamp(min=eps)
    err_q = torch.arccos(1 - 2 * loss_q)

    rel_rangle_deg = err_q * 180 / np.pi

    if batch_size is not None:
        rel_rangle_deg = rel_rangle_deg.reshape(batch_size, -1)

    return rel_rangle_deg



def translation_angle(tvec_gt, tvec_pred, batch_size=None, ambiguity=True):
    # tvec_gt, tvec_pred (B, 3,)
    rel_tangle_deg = compare_translation_by_angle(tvec_gt, tvec_pred)
    rel_tangle_deg = rel_tangle_deg * 180.0 / np.pi

    if ambiguity:
        rel_tangle_deg = torch.min(rel_tangle_deg, (180 - rel_tangle_deg).abs())

    if batch_size is not None:
        rel_tangle_deg = rel_tangle_deg.reshape(batch_size, -1)

    return rel_tangle_deg



def compare_translation_by_angle(t_gt, t, eps=1e-15, default_err=1e6):
    """Normalize the translation vectors and compute the angle between them."""
    t_norm = torch.norm(t, dim=1, keepdim=True)
    t = t / (t_norm + eps)

    t_gt_norm = torch.norm(t_gt, dim=1, keepdim=True)
    t_gt = t_gt / (t_gt_norm + eps)

    loss_t = torch.clamp_min(1.0 - torch.sum(t * t_gt, dim=1) ** 2, eps)
    err_t = torch.acos(torch.sqrt(1 - loss_t))

    err_t[torch.isnan(err_t) | torch.isinf(err_t)] = default_err
    return err_t



def calculate_auc(r_error, t_error, max_threshold=30, return_list=False):
    """
    Calculate the Area Under the Curve (AUC) for the given error arrays using PyTorch.

    :param r_error: torch.Tensor representing R error values (Degree).
    :param t_error: torch.Tensor representing T error values (Degree).
    :param max_threshold: maximum threshold value for binning the histogram.
    :return: cumulative sum of normalized histogram of maximum error values.
    """

    # Concatenate the error tensors along a new axis
    error_matrix = torch.stack((r_error, t_error), dim=1)

    # Compute the maximum error value for each pair
    max_errors, _ = torch.max(error_matrix, dim=1)

    # Calculate histogram of maximum error values
    histogram = torch.histc(
        max_errors, bins=max_threshold + 1, min=0, max=max_threshold
    )

    # Normalize the histogram
    num_pairs = float(max_errors.size(0))
    normalized_histogram = histogram / num_pairs

    if return_list:
        return (
            torch.cumsum(normalized_histogram, dim=0).mean(),
            normalized_histogram,
        )
    # Compute and return the cumulative sum of the normalized histogram
    return torch.cumsum(normalized_histogram, dim=0).mean()



def calculate_auc_np(r_error, t_error, max_threshold=30):
    """
    Calculate the Area Under the Curve (AUC) for the given error arrays.

    :param r_error: numpy array representing R error values (Degree).
    :param t_error: numpy array representing T error values (Degree).
    :param max_threshold: maximum threshold value for binning the histogram.
    :return: cumulative sum of normalized histogram of maximum error values.
    """

    # Concatenate the error arrays along a new axis
    error_matrix = np.concatenate((r_error[:, None], t_error[:, None]), axis=1)

    # Compute the maximum error value for each pair
    max_errors = np.max(error_matrix, axis=1)

    # Define histogram bins
    bins = np.arange(max_threshold + 1)

    # Calculate histogram of maximum error values
    histogram, _ = np.histogram(max_errors, bins=bins)

    # Normalize the histogram
    num_pairs = float(len(max_errors))
    normalized_histogram = histogram.astype(float) / num_pairs

    # Compute and return the cumulative sum of the normalized histogram
    return np.mean(np.cumsum(normalized_histogram)), normalized_histogram



def se3_to_relative_pose_error(pred_se3, gt_se3, num_frames):
    # loss_dict = {}

    pair_idx_i1, pair_idx_i2 = build_pair_index(num_frames)

    # Compute relative camera poses between pairs
    # We use closed_form_inverse to avoid potential numerical loss by torch.inverse()
    # This is possible because of SE3
    relative_pose_gt = closed_form_inverse_se3(gt_se3[pair_idx_i1]).bmm(
        gt_se3[pair_idx_i2]
    )
    relative_pose_pred = closed_form_inverse_se3(pred_se3[pair_idx_i1]).bmm(
        pred_se3[pair_idx_i2]
    )
    # Compute the difference in rotation and translation
    # between the ground truth and predicted relative camera poses
    rel_rangle_deg = rotation_angle(
        relative_pose_gt[:, :3, :3], relative_pose_pred[:, :3, :3]
    )
    rel_tangle_deg = translation_angle(
        relative_pose_gt[:, :3, 3], relative_pose_pred[:, :3, 3]
    )


    return rel_rangle_deg, rel_tangle_deg



# TODO: test below

# 1. how much would bf16 and tf32 affect the tracking results?
# 2. use np instead of torch for computing auc



device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

print("Initializing and loading VGGT model...")
# model = VGGT.from_pretrained("facebook/VGGT-1B")  # another way to load the model

model = VGGT()
_URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
model.load_state_dict(torch.hub.load_state_dict_from_url(_URL))

model.eval()
model = model.to(device)

CO3D_DIR = "/YOUR/CO3D/PATH"
CO3D_ANNOTATION_DIR = "/YOUR/CO3D/ANNO/PATH"


# 50 is the minimum number of images for a sequence to be considered
# otherwise, we think it as corrupted.
min_num_images = 50

num_frames = 10 # use 10 frames for testing






SEEN_CATEGORIES = [
    "apple",
    "backpack",
    "banana",
    "baseballbat",
    "baseballglove",
    "bench",
    "bicycle",
    "bottle",
    "bowl",
    "broccoli",
    "cake",
    "car",
    "carrot",
    "cellphone",
    "chair",
    "cup",
    "donut",
    "hairdryer",
    "handbag",
    "hydrant",
    "keyboard",
    "laptop",
    "microwave",
    "motorcycle",
    "mouse",
    "orange",
    "parkingmeter",
    "pizza",
    "plant",
    "stopsign",
    "teddybear",
    "toaster",
    "toilet",
    "toybus",
    "toyplane",
    "toytrain",
    "toytruck",
    "tv",
    "umbrella",
    "vase",
    "wineglass",
]



per_category_results = {}


for category in SEEN_CATEGORIES:
    print(f"Loading annotation for {category} test set")
    # take the test set
    annotation_file = os.path.join(CO3D_ANNOTATION_DIR, f"{category}_test.jgz")

    with gzip.open(annotation_file, "r") as fin:
        annotation = json.loads(fin.read())

    rError = []
    tError = []

    for seq_name, seq_data in annotation.items():

        print(f"Processing {seq_name} of {len(annotation)} for {category} test set")
        if len(seq_data) < min_num_images:
            continue

        metadata = []
        bad_seq = False
        for data in seq_data:
            # Make sure translations are not ridiculous
            if data["T"][0] + data["T"][1] + data["T"][2] > 1e5:
                bad_seq = True
                break

            extri_opencv = convert_pt3d_RT_to_opencv(data["R"], data["T"])
            # Ignore all unnecessary information.
            metadata.append(
                {
                    "filepath": data["filepath"],
                    "extri": extri_opencv,
                }
            )

        ids = np.random.choice(len(metadata), num_frames, replace=False)

        image_names = [os.path.join(CO3D_DIR, metadata[i]["filepath"]) for i in ids]

        gt_extri = [np.array(metadata[i]["extri"]) for i in ids]
        gt_extri = np.stack(gt_extri, axis=0)


        images = load_and_preprocess_images(image_names).to(device)

        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=dtype):
                predictions = model(images)

        with torch.cuda.amp.autocast(dtype=torch.float64):
            # Convert pose encoding to extrinsic and intrinsic matrices
            print("Converting pose encoding to extrinsic and intrinsic matrices...")

            # TODO: write the BA code here



            extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
            pred_extrinsic = extrinsic[0]



            gt_extrinsic = torch.from_numpy(gt_extri).to(device)
            add_row = torch.tensor([0, 0, 0, 1], device=device).expand(pred_extrinsic.size(0), 1, 4)

            pred_se3 = torch.cat((pred_extrinsic, add_row), dim=1)
            gt_se3 = torch.cat((gt_extrinsic, add_row), dim=1)

            ### set the coordinate of the first camera as the coordinate of the world
            # NOTE: this is always necessary. Do not remove it unless you know what you are doing.
            first_cam_extrinsic_inv = closed_form_inverse_se3(gt_se3[0][None])
            gt_se3 = torch.matmul(gt_se3, first_cam_extrinsic_inv)
            ###

            rel_rangle_deg, rel_tangle_deg = se3_to_relative_pose_error(pred_se3, gt_se3, num_frames)
            print(f"{category} sequence {seq_name} Rot   Error: {rel_rangle_deg.mean().item()}")
            print(f"{category} sequence {seq_name} Trans Error: {rel_tangle_deg.mean().item()}")
            rError.extend(rel_rangle_deg.cpu().numpy())
            tError.extend(rel_tangle_deg.cpu().numpy())


    rError = np.array(rError)
    tError = np.array(tError)

    Auc_30, _ = calculate_auc_np(rError, tError, max_threshold=30)

    for _ in range(10): print("*" * 100)
    print(f"AUC of {category} test set: {Auc_30}")
    for _ in range(10): print("*" * 100)

    per_category_results[category] = {
        "rError": rError,
        "tError": tError,
        "Auc_30": Auc_30
    }

for category in SEEN_CATEGORIES:
    print(f"AUC of {category} test set: {per_category_results[category]['Auc_30']}")
mean_AUC = np.mean([per_category_results[category]["Auc_30"] for category in SEEN_CATEGORIES])
print(f"Mean AUC: {mean_AUC}")
