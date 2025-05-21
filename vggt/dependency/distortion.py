# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import random
from typing import Tuple, Union

import numpy as np
import torch

ArrayLike = Union[np.ndarray, torch.Tensor]

# -----------------------------------------------------------------------------
# —— Backend dispatch helpers ————————————————————————————————————————————
# -----------------------------------------------------------------------------

def _is_numpy(x: ArrayLike) -> bool:
    return isinstance(x, np.ndarray)


def _is_torch(x: ArrayLike) -> bool:
    return isinstance(x, torch.Tensor)


# -----------------------------------------------------------------------------
# —— Public façade ————————————————————————————————————————————————————————
# -----------------------------------------------------------------------------

def single_undistortion(
    params: ArrayLike,
    tracks_normalized: ArrayLike,
):
    """Apply one undistortion step using the appropriate backend."""
    if _is_numpy(tracks_normalized):
        return _single_undistortion_np(params, tracks_normalized)
    elif _is_torch(tracks_normalized):
        return _single_undistortion_torch(params, tracks_normalized)
    else:
        raise TypeError("Unsupported array type. Use NumPy ndarray or torch.Tensor.")


def iterative_undistortion(
    params: ArrayLike,
    tracks_normalized: ArrayLike,
    max_iterations: int = 100,
    max_step_norm: float = 1e-10,
    rel_step_size: float = 1e-6,
):
    """Iterative Newton solve, back‑end agnostic."""
    if _is_numpy(tracks_normalized):
        return _iterative_undistortion_np(
            params, tracks_normalized, max_iterations, max_step_norm, rel_step_size
        )
    elif _is_torch(tracks_normalized):
        return _iterative_undistortion_torch(
            params, tracks_normalized, max_iterations, max_step_norm, rel_step_size
        )
    else:
        raise TypeError("Unsupported array type. Use NumPy ndarray or torch.Tensor.")


# -----------------------------------------------------------------------------
# —— NumPy backend ————————————————————————————————————————————————————————
# -----------------------------------------------------------------------------

def _apply_distortion_np(
    extra_params: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorised NumPy implementation (B × T broadcasting)."""
    num_params = extra_params.shape[1]

    if num_params == 1:
        k = extra_params[:, 0][:, None]
        r2 = u**2 + v**2
        radial = k * r2
        du = u * radial
        dv = v * radial

    elif num_params == 2:
        k1 = extra_params[:, 0][:, None]
        k2 = extra_params[:, 1][:, None]
        r2 = u**2 + v**2
        radial = k1 * r2 + k2 * r2**2
        du = u * radial
        dv = v * radial

    elif num_params == 4:
        k1 = extra_params[:, 0][:, None]
        k2 = extra_params[:, 1][:, None]
        p1 = extra_params[:, 2][:, None]
        p2 = extra_params[:, 3][:, None]
        u2 = u**2
        v2 = v**2
        uv = u * v
        r2 = u2 + v2
        radial = k1 * r2 + k2 * r2**2
        du = u * radial + 2 * p1 * uv + p2 * (r2 + 2 * u2)
        dv = v * radial + 2 * p2 * uv + p1 * (r2 + 2 * v2)
    else:
        raise ValueError("Unsupported number of distortion parameters")

    return u + du, v + dv


def _single_undistortion_np(
    params: np.ndarray, tracks_normalized: np.ndarray
) -> np.ndarray:
    u, v = tracks_normalized[..., 0].copy(), tracks_normalized[..., 1].copy()
    uu, vv = _apply_distortion_np(params, u, v)
    return np.stack([uu, vv], axis=-1)


def _iterative_undistortion_np(
    params: np.ndarray,
    tracks_normalized: np.ndarray,
    max_iterations: int,
    max_step_norm: float,
    rel_step_size: float,
) -> np.ndarray:
    B, T, _ = tracks_normalized.shape
    u = tracks_normalized[..., 0].copy()
    v = tracks_normalized[..., 1].copy()
    orig_u, orig_v = u.copy(), v.copy()

    eps = np.finfo(u.dtype).eps

    for _ in range(max_iterations):
        uu, vv = _apply_distortion_np(params, u, v)
        dx = orig_u - uu
        dy = orig_v - vv

        step_u = np.clip(np.abs(u) * rel_step_size, eps, None)
        step_v = np.clip(np.abs(v) * rel_step_size, eps, None)

        # Finite‑difference Jacobian (vectorised)
        J_00 = (
            _apply_distortion_np(params, u + step_u, v)[0]
            - _apply_distortion_np(params, u - step_u, v)[0]
        ) / (2 * step_u)
        J_01 = (
            _apply_distortion_np(params, u, v + step_v)[0]
            - _apply_distortion_np(params, u, v - step_v)[0]
        ) / (2 * step_v)
        J_10 = (
            _apply_distortion_np(params, u + step_u, v)[1]
            - _apply_distortion_np(params, u - step_u, v)[1]
        ) / (2 * step_u)
        J_11 = (
            _apply_distortion_np(params, u, v + step_v)[1]
            - _apply_distortion_np(params, u, v - step_v)[1]
        ) / (2 * step_v)

        # Assemble 2×2 Jacobian and RHS (…,2,2) and (… ,2)
        J = np.stack(
            [
                np.stack([J_00 + 1, J_01], axis=-1),
                np.stack([J_10, J_11 + 1], axis=-1),
            ],
            axis=-2,
        )
        rhs = np.stack([dx, dy], axis=-1)

        # Solve J · delta = rhs (broadcasted)
        delta = np.linalg.solve(J, rhs)

        u += delta[..., 0]
        v += delta[..., 1]

        if np.max(np.sum(delta**2, axis=-1)) < max_step_norm:
            break

    return np.stack([u, v], axis=-1)


# -----------------------------------------------------------------------------
# —— Torch backend ———————————————
# -----------------------------------------------------------------------------

def _apply_distortion_torch(
    extra_params: torch.Tensor,
    u: torch.Tensor,
    v: torch.Tensor,
):
    num_params = extra_params.shape[1]

    if num_params == 1:
        k = extra_params[:, 0]
        r2 = u * u + v * v
        radial = k[:, None] * r2
        du = u * radial
        dv = v * radial

    elif num_params == 2:
        k1, k2 = extra_params[:, 0], extra_params[:, 1]
        r2 = u * u + v * v
        radial = k1[:, None] * r2 + k2[:, None] * r2 * r2
        du = u * radial
        dv = v * radial

    elif num_params == 4:
        k1, k2, p1, p2 = (
            extra_params[:, 0],
            extra_params[:, 1],
            extra_params[:, 2],
            extra_params[:, 3],
        )
        u2 = u * u
        v2 = v * v
        uv = u * v
        r2 = u2 + v2
        radial = k1[:, None] * r2 + k2[:, None] * r2 * r2
        du = u * radial + 2 * p1[:, None] * uv + p2[:, None] * (r2 + 2 * u2)
        dv = v * radial + 2 * p2[:, None] * uv + p1[:, None] * (r2 + 2 * v2)
    else:
        raise ValueError("Unsupported number of distortion parameters")

    return u + du, v + dv


def _single_undistortion_torch(
    params: torch.Tensor, tracks_normalized: torch.Tensor
) -> torch.Tensor:
    u, v = tracks_normalized[..., 0].clone(), tracks_normalized[..., 1].clone()
    uu, vv = _apply_distortion_torch(params, u, v)
    return torch.stack([uu, vv], dim=-1)


def _iterative_undistortion_torch(
    params: torch.Tensor,
    tracks_normalized: torch.Tensor,
    max_iterations: int,
    max_step_norm: float,
    rel_step_size: float,
) -> torch.Tensor:
    B, T, _ = tracks_normalized.shape
    u = tracks_normalized[..., 0].clone()
    v = tracks_normalized[..., 1].clone()
    orig_u, orig_v = u.clone(), v.clone()

    eps = torch.finfo(u.dtype).eps

    for _ in range(max_iterations):
        uu, vv = _apply_distortion_torch(params, u, v)
        dx = orig_u - uu
        dy = orig_v - vv

        step_u = torch.clamp(torch.abs(u) * rel_step_size, min=eps)
        step_v = torch.clamp(torch.abs(v) * rel_step_size, min=eps)

        J_00 = (
            _apply_distortion_torch(params, u + step_u, v)[0]
            - _apply_distortion_torch(params, u - step_u, v)[0]
        ) / (2 * step_u)
        J_01 = (
            _apply_distortion_torch(params, u, v + step_v)[0]
            - _apply_distortion_torch(params, u, v - step_v)[0]
        ) / (2 * step_v)
        J_10 = (
            _apply_distortion_torch(params, u + step_u, v)[1]
            - _apply_distortion_torch(params, u - step_u, v)[1]
        ) / (2 * step_u)
        J_11 = (
            _apply_distortion_torch(params, u, v + step_v)[1]
            - _apply_distortion_torch(params, u, v - step_v)[1]
        ) / (2 * step_v)

        J = torch.stack(
            [
                torch.stack([J_00 + 1, J_01], dim=-1),
                torch.stack([J_10, J_11 + 1], dim=-1),
            ],
            dim=-2,
        )
        rhs = torch.stack([dx, dy], dim=-1)

        delta = torch.linalg.solve(J, rhs)
        u += delta[..., 0]
        v += delta[..., 1]

        if torch.max((delta ** 2).sum(dim=-1)) < max_step_norm:
            break

    return torch.stack([u, v], dim=-1)


# -----------------------------------------------------------------------------
# —— Self‑test when executed directly ——————————————————————————————————————
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    import random
    import pycolmap
    import time

    # Test against pycolmap reference implementation
    print("Testing against pycolmap reference...")
    max_diff = 0
    for i in range(1000):
        # Define distortion parameters (assuming 1 parameter for simplicity)
        B = random.randint(1, 500)
        track_num = random.randint(100, 1000)
        
        # Generate random parameters and tracks
        params_torch = torch.rand((B, 1), dtype=torch.float32)
        tracks_normalized_torch = torch.rand((B, track_num, 2), dtype=torch.float32)
        
        # Convert to numpy for numpy backend testing
        params_np = params_torch.numpy()
        tracks_normalized_np = tracks_normalized_torch.numpy()
        
        # Undistort the tracks using both backends
        undistorted_tracks_torch = iterative_undistortion(params_torch, tracks_normalized_torch)
        undistorted_tracks_np = iterative_undistortion(params_np, tracks_normalized_np)
        
        # Compare with pycolmap
        for b in range(B):
            pycolmap_intri = np.array([1, 0, 0, params_torch[b].item()])
            pycam = pycolmap.Camera(
                model="SIMPLE_RADIAL",
                width=1,
                height=1,
                params=pycolmap_intri,
                camera_id=0,
            )

            undistorted_tracks_pycolmap = pycam.cam_from_img(
                tracks_normalized_torch[b].numpy()
            )
            
            # Compare torch vs pycolmap
            torch_diff = np.median(np.abs(undistorted_tracks_torch[b].numpy() - undistorted_tracks_pycolmap))
            # Compare numpy vs pycolmap
            np_diff = np.median(np.abs(undistorted_tracks_np[b] - undistorted_tracks_pycolmap))
            # Compare torch vs numpy
            impl_diff = np.median(np.abs(undistorted_tracks_torch[b].numpy() - undistorted_tracks_np[b]))
            
            max_diff = max(max_diff, torch_diff, np_diff, impl_diff)
            
        if i % 10 == 0:
            print(f"Iteration {i}, max_diff: {max_diff}")

    print(f"Maximum difference across all implementations: {max_diff}")
    
    # Benchmark performance
    print("\nPerformance benchmark:")
    for backend, params, tracks in [
        ("NumPy", params_np, tracks_normalized_np),
        ("PyTorch", params_torch, tracks_normalized_torch)
    ]:
        start_time = time.time()
        iterative_undistortion(params, tracks)
        elapsed = time.time() - start_time
        print(f"{backend} backend: {elapsed:.4f} seconds")
        
    print("Test finished ✓")


