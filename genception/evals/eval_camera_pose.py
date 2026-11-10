# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Camera pose evaluation on SINTEL for GenCeption."""

from __future__ import annotations

import copy
import os
from typing import Any, Dict, Tuple

from absl import logging
import numpy as np

from representations4d.genception.evals import eval_common
from representations4d.genception.pipeline import GenCeptionPipeline

SINTEL_VIDEOS = [
    'alley_2',
    'ambush_4',
    'ambush_5',
    'ambush_6',
    'cave_2',
    'cave_4',
    'market_2',
    'market_5',
    'market_6',
    'shaman_3',
    'sleeping_1',
    'sleeping_2',
    'temple_2',
    'temple_3',
]


def _spatial_process_frame_zoom(
    frame: np.ndarray,
    target_width: int = 832,
    target_height: int = 480,
) -> np.ndarray:
  """Resize frame maintaining aspect ratio to cover target dimensions, then center-crop."""
  height, width = frame.shape[:2]
  if height == target_height and width == target_width:
    return frame
  if height == 0 or width == 0:
    if len(frame.shape) == 3:
      return np.zeros(
          (target_height, target_width, frame.shape[2]), dtype=frame.dtype
      )
    return np.zeros((target_height, target_width), dtype=frame.dtype)

  scale = max(target_width / width, target_height / height)
  new_width = max(1, int(round(width * scale)))
  new_height = max(1, int(round(height * scale)))

  interpolation = (
      eval_common.cv2.INTER_AREA
      if scale < 1.0
      else eval_common.cv2.INTER_LINEAR
  )
  resized_frame = eval_common.cv2.resize(
      frame, (new_width, new_height), interpolation=interpolation
  )

  delta_w = new_width - target_width
  delta_h = new_height - target_height
  start_x = delta_w // 2
  start_y = delta_h // 2
  return resized_frame[
      start_y : start_y + target_height, start_x : start_x + target_width
  ]


def _unscale_raymap_origin(raymap_video: np.ndarray) -> np.ndarray:
  """Unscales the central log-scaled origin patch of the raymap."""
  pose_unscaling = lambda x: np.sign(x) * (np.exp(np.abs(x) / 1.0) - 1.0)
  patch_size = 16
  height_start, height_end = patch_size * 10, patch_size * 20
  width_start, width_end = patch_size * 18, patch_size * 34

  raymap_with_unscaled_origin = raymap_video.copy()
  patch = raymap_video[:, height_start:height_end, width_start:width_end]
  raymap_with_unscaled_origin[
      :, height_start:height_end, width_start:width_end
  ] = (pose_unscaling((patch * 2.0 - 1.0)) * 0.5 + 0.5)
  return raymap_with_unscaled_origin


def _create_raymap_masks(
    num_frames: int = 81, height: int = 480, width: int = 832
) -> Tuple[np.ndarray, np.ndarray]:
  """Creates masks for raymap origin region and outer ray direction region."""
  patch_size = 16
  height_start, height_end = patch_size * (10 + 2), patch_size * (20 - 2)
  width_start, width_end = patch_size * (18 + 2), patch_size * (34 - 2)
  origin_mask = np.zeros((num_frames, height, width, 3), dtype=np.float32)
  origin_mask[:, height_start:height_end, width_start:width_end] = 1.0

  height_start, height_end = patch_size * (10 - 2), patch_size * (20 + 2)
  width_start, width_end = patch_size * (18 - 2), patch_size * (34 + 2)
  dir_mask = np.ones((num_frames, height, width, 3), dtype=np.float32)
  dir_mask[:, height_start:height_end, width_start:width_end] = 0.0

  return origin_mask, dir_mask


def _robust_lstsq(a_mat: np.ndarray, b_vec: np.ndarray) -> np.ndarray:
  """Two-step robust regression: initial fit, MAD outlier rejection, refined fit."""
  x = a_mat[:, 0]
  n_pts = len(x)
  sum_x = np.sum(x)
  sum_y = np.sum(b_vec)
  sum_xx = np.dot(x, x)
  sum_xy = np.dot(x, b_vec)
  denom = n_pts * sum_xx - sum_x * sum_x
  m_slope = (n_pts * sum_xy - sum_x * sum_y) / (denom + 1e-12)
  c_inter = (sum_y - m_slope * sum_x) / n_pts

  residuals = np.abs(m_slope * x + c_inter - b_vec)
  mad = np.median(residuals[::10])
  threshold = max(5.0 * mad, 1e-3)
  inlier_mask = residuals < threshold

  x_in = x[inlier_mask]
  b_in = b_vec[inlier_mask]
  n_in = len(x_in)
  sum_x_in = np.sum(x_in)
  sum_y_in = np.sum(b_in)
  sum_xx_in = np.dot(x_in, x_in)
  sum_xy_in = np.dot(x_in, b_in)
  denom_in = n_in * sum_xx_in - sum_x_in * sum_x_in
  m_in = (n_in * sum_xy_in - sum_x_in * sum_y_in) / (denom_in + 1e-12)
  c_in = (sum_y_in - m_in * sum_x_in) / n_in
  return np.array([m_in, c_in], dtype=a_mat.dtype)


def _smooth_trajectory(traj: np.ndarray, window_size: int = 5) -> np.ndarray:
  """Dual temporal smoothing (1D median filter + 1D mean filter)."""
  t_len = len(traj)
  if t_len < 3:
    return traj
  w = min(window_size, t_len if t_len % 2 != 0 else t_len - 1)
  pad_w = w // 2
  padded = np.pad(traj, ((pad_w, pad_w), (0, 0)), mode='edge')
  smoothed_median = np.zeros_like(traj)
  for i in range(t_len):
    smoothed_median[i] = np.median(padded[i : i + w], axis=0)
  padded_mean = np.pad(smoothed_median, ((pad_w, pad_w), (0, 0)), mode='edge')
  final_smoothed = np.zeros_like(traj)
  for i in range(t_len):
    final_smoothed[i] = np.mean(padded_mean[i : i + w], axis=0)
  return final_smoothed


def _smooth_rotations_svd(
    r_seq: np.ndarray, window_size: int = 5
) -> np.ndarray:
  """Smooths rotation sequence on SO(3) manifold using SVD projection."""
  t_len = len(r_seq)
  if t_len < 3:
    return r_seq
  w = min(window_size, t_len if t_len % 2 != 0 else t_len - 1)
  pad_w = w // 2
  padded_r = np.pad(r_seq, ((pad_w, pad_w), (0, 0), (0, 0)), mode='edge')
  smoothed_r = np.zeros_like(r_seq)
  for i in range(t_len):
    m_mat = np.mean(padded_r[i : i + w], axis=0)
    u_mat, _, vt_mat = np.linalg.svd(m_mat)
    r_new = u_mat @ vt_mat
    if np.linalg.det(r_new) < 0:
      u_copy = u_mat.copy()
      u_copy[:, -1] *= -1.0
      r_new = u_copy @ vt_mat
    smoothed_r[i] = r_new
  return smoothed_r


def _compute_optimal_rotation_alignment_np(
    a_mat: np.ndarray, b_mat: np.ndarray
) -> np.ndarray:
  """Kabsch algorithm: optimal rotation minimizing || A - B @ R ||_F."""
  h_mat = b_mat.T @ a_mat
  u_mat, _, vh_mat = np.linalg.svd(h_mat, full_matrices=True)
  sign_det = np.linalg.det(u_mat @ vh_mat)
  s_prime = np.diag([1.0, 1.0, np.sign(sign_det)])
  return u_mat @ s_prime @ vh_mat


def _estimate_global_params_with_mask_robust2(
    raymap_3d: np.ndarray, origin_mask: np.ndarray, dir_mask: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Estimates camera intrinsics K and extrinsics R using covariance / Gram-Schmidt."""
  t_frames, height, width, _ = raymap_3d.shape
  rs_list, ts_list = [], []
  all_x_over_z, all_y_over_z, all_u, all_v = [], [], [], []
  uu_full, vv_full = np.meshgrid(np.arange(width), np.arange(height))

  for i in range(t_frames):
    current_map = copy.deepcopy(raymap_3d[i])
    current_map[origin_mask[i, :, :, 0] == 1] = (
        current_map[origin_mask[i, :, :, 0] == 1] * 2.0 - 1.0
    )

    valid_dir_mask = dir_mask[i, :, :, 0] == 1
    current_map[valid_dir_mask] = (current_map[valid_dir_mask] * 2.0 - 1.0) / (
        np.linalg.norm(
            current_map[valid_dir_mask] * 2.0 - 1.0, axis=-1, keepdims=True
        )
        + 1e-6
    )

    t_i = np.median(current_map[origin_mask[i, :, :, 0] == 1], axis=0)
    ts_list.append(t_i)

    valid_rays = current_map[valid_dir_mask]
    uu = uu_full[valid_dir_mask]
    vv = vv_full[valid_dir_mask]

    z_axis = np.median(valid_rays, axis=0)
    z_axis /= np.linalg.norm(z_axis) + 1e-6

    uu_c = uu - np.mean(uu)
    x_raw = uu_c @ valid_rays
    x_axis = x_raw - np.dot(x_raw, z_axis) * z_axis
    x_axis /= np.linalg.norm(x_axis) + 1e-6

    y_axis = np.cross(z_axis, x_axis)
    r_i = np.stack([x_axis, y_axis, z_axis], axis=1)
    rs_list.append(r_i)

    rays_cam = (r_i.T @ valid_rays.T).T
    all_x_over_z.append(rays_cam[:, 0] / (rays_cam[:, 2] + 1e-8))
    all_y_over_z.append(rays_cam[:, 1] / (rays_cam[:, 2] + 1e-8))
    all_u.append(uu)
    all_v.append(vv)

  x_over_z = np.concatenate(all_x_over_z)
  y_over_z = np.concatenate(all_y_over_z)
  u_vec = np.concatenate(all_u)
  v_vec = np.concatenate(all_v)

  a_u = np.stack([x_over_z, np.ones_like(x_over_z)], axis=1)
  fx, cx = _robust_lstsq(a_u, u_vec)

  a_v = np.stack([y_over_z, np.ones_like(y_over_z)], axis=1)
  fy, cy = _robust_lstsq(a_v, v_vec)

  k_mat = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

  ts_array = np.stack(ts_list)
  rs_array = np.stack(rs_list)

  ts_smooth = _smooth_trajectory(ts_array, window_size=5)
  rs_smooth = _smooth_rotations_svd(rs_array, window_size=5)

  k_seq = np.repeat(k_mat[None, ...], t_frames, axis=0)
  r_smooth = np.concatenate((rs_smooth, ts_smooth[..., None]), axis=-1)
  r_non_smooth = np.concatenate((rs_array, ts_array[..., None]), axis=-1)

  return k_seq, r_smooth, r_non_smooth


def _estimate_global_params_with_mask_robust_svd(
    raymap_3d: np.ndarray, origin_mask: np.ndarray, dir_mask: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Estimates camera intrinsics K and extrinsics R using Kabsch SVD alignment."""
  t_frames, height, width, _ = raymap_3d.shape
  rs_list, ts_list = [], []
  all_x_over_z, all_y_over_z, all_u, all_v = [], [], [], []
  uu_full, vv_full = np.meshgrid(np.arange(width), np.arange(height))

  f_prior = max(height, width)
  for i in range(t_frames):
    current_map = copy.deepcopy(raymap_3d[i])
    current_map[origin_mask[i, :, :, 0] == 1] = (
        current_map[origin_mask[i, :, :, 0] == 1] * 2.0 - 1.0
    )

    valid_dir_mask = dir_mask[i, :, :, 0] == 1
    current_map[valid_dir_mask] = (current_map[valid_dir_mask] * 2.0 - 1.0) / (
        np.linalg.norm(
            current_map[valid_dir_mask] * 2.0 - 1.0, axis=-1, keepdims=True
        )
        + 1e-6
    )

    t_i = np.median(current_map[origin_mask[i, :, :, 0] == 1], axis=0)
    ts_list.append(t_i)

    valid_rays = current_map[valid_dir_mask]
    uu = uu_full[valid_dir_mask]
    vv = vv_full[valid_dir_mask]

    uu_c = uu - width / 2.0
    vv_c = vv - height / 2.0
    proxy_rays_cam = np.stack([uu_c, vv_c, np.full_like(uu_c, f_prior)], axis=1)
    proxy_rays_cam = proxy_rays_cam / (
        np.linalg.norm(proxy_rays_cam, axis=1, keepdims=True) + 1e-6
    )

    r_i = _compute_optimal_rotation_alignment_np(proxy_rays_cam, valid_rays)
    rs_list.append(r_i)

    rays_cam = (r_i.T @ valid_rays.T).T
    all_x_over_z.append(rays_cam[:, 0] / (rays_cam[:, 2] + 1e-8))
    all_y_over_z.append(rays_cam[:, 1] / (rays_cam[:, 2] + 1e-8))
    all_u.append(uu)
    all_v.append(vv)

  x_over_z = np.concatenate(all_x_over_z)
  y_over_z = np.concatenate(all_y_over_z)
  u_vec = np.concatenate(all_u)
  v_vec = np.concatenate(all_v)

  a_u = np.stack([x_over_z, np.ones_like(x_over_z)], axis=1)
  fx, cx = _robust_lstsq(a_u, u_vec)

  a_v = np.stack([y_over_z, np.ones_like(y_over_z)], axis=1)
  fy, cy = _robust_lstsq(a_v, v_vec)

  k_mat = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

  ts_array = np.stack(ts_list)
  rs_array = np.stack(rs_list)

  ts_smooth = _smooth_trajectory(ts_array, window_size=5)
  rs_smooth = _smooth_rotations_svd(rs_array, window_size=5)

  k_seq = np.repeat(k_mat[None, ...], t_frames, axis=0)
  r_smooth = np.concatenate((rs_smooth, ts_smooth[..., None]), axis=-1)
  r_non_smooth = np.concatenate((rs_array, ts_array[..., None]), axis=-1)

  return k_seq, r_smooth, r_non_smooth


def _align_to_first_frame(extrinsics: np.ndarray) -> np.ndarray:
  """Transforms camera-to-world (N, 3, 4) to relative pose w.r.t. frame 0."""
  num_frames = extrinsics.shape[0]
  homo_pads = np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (num_frames, 1, 1))
  t_wc = np.concatenate([extrinsics[:, :3, :4], homo_pads], axis=1)
  t_cw = np.linalg.inv(t_wc)
  t_c0_ct = t_cw[:1] @ t_wc
  return t_c0_ct[:, :3, :4]


def _ensure_4x4(mat: np.ndarray) -> np.ndarray:
  """Ensures extrinsics matrix has homogeneous shape (..., 4, 4)."""
  if mat.shape[-2:] == (3, 4):
    row = np.ones_like(mat[..., :1, :]) * np.array([0.0, 0.0, 0.0, 1.0])
    mat = np.concatenate((mat, row), axis=-2)
  return mat


def _align_trajectories(
    model: np.ndarray, data: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray]:
  """Aligns two trajectories using the closed-form method of Horn."""
  model_mean = model.mean(axis=1, keepdims=True)
  data_mean = data.mean(axis=1, keepdims=True)
  model_zerocentered = model - model_mean
  data_zerocentered = data - data_mean

  w_mat = np.zeros((3, 3))
  for column in range(model.shape[1]):
    w_mat += np.outer(
        model_zerocentered[:, column], data_zerocentered[:, column]
    )
  u_mat, _, vh_mat = np.linalg.svd(w_mat.transpose())
  s_mat = np.identity(3)
  if np.linalg.det(u_mat) * np.linalg.det(vh_mat) < 0:
    s_mat[2, 2] = -1.0
  rot = u_mat @ s_mat @ vh_mat

  rotmodel = rot @ model_zerocentered
  dots = 0.0
  norms = 0.0
  for column in range(data_zerocentered.shape[1]):
    dots += np.dot(
        data_zerocentered[:, column].transpose(), rotmodel[:, column]
    )
    normi = np.linalg.norm(model_zerocentered[:, column])
    norms += normi * normi

  scale = float(dots / (norms + 1e-12))
  trans = data_mean - scale * (rot @ model_mean)
  model_aligned = scale * (rot @ model) + trans
  alignment_error = model_aligned - data
  trans_error = np.sqrt(np.sum(alignment_error * alignment_error, axis=0))
  return rot, trans, trans_error, scale, model_aligned


def _ominus(a_mat: np.ndarray, b_mat: np.ndarray) -> np.ndarray:
  return np.dot(np.linalg.inv(a_mat), b_mat)


def _compute_distance(transform: np.ndarray) -> float:
  return float(np.linalg.norm(transform[0:3, 3]))


def _compute_angle(transform: np.ndarray) -> float:
  return float(
      np.arccos(
          min(1.0, max(-1.0, (np.trace(transform[0:3, 0:3]) - 1.0) / 2.0))
      )
  )


def _evaluate_trajectory(
    traj_gt: list[np.ndarray],
    traj_est: list[np.ndarray],
    param_fixed_delta: bool = True,
    param_delta: int = 1,
) -> list[list[float]]:
  """Computes frame-to-frame relative pose translation and rotation errors."""
  pairs = []
  if param_fixed_delta:
    for i in range(len(traj_est)):
      j = int(i + param_delta)
      if j < len(traj_est):
        pairs.append((i, j))
  else:
    for i in range(len(traj_est)):
      for j in range(len(traj_est)):
        pairs.append((i, j))

  result = []
  for i, j in pairs:
    error44 = _ominus(
        _ominus(traj_est[j], traj_est[i]), _ominus(traj_gt[j], traj_gt[i])
    )
    trans = _compute_distance(error44)
    rot = _compute_angle(error44)
    result.append([float(i), float(j), trans, rot])
  return result


def _calculate_ate_rpe_megasam(
    poses: np.ndarray,
    gt_cam2w: np.ndarray,
) -> Tuple[float, float, float]:
  """Calculates ATE, RPE translation, and RPE rotation following MegaSaM."""
  est_cam2w = _ensure_4x4(poses.copy())
  gt_cam2w = _ensure_4x4(gt_cam2w.copy())
  num_cams = gt_cam2w.shape[0]
  assert gt_cam2w.shape[0] == est_cam2w.shape[0]

  full_t = np.dot(np.linalg.inv(gt_cam2w[-1]), gt_cam2w[0])
  normalize_scale = np.linalg.norm(full_t[:3, 3]) + 1e-8
  gt_cam2w[:, :3, 3] /= normalize_scale

  rot, trans, trans_error, scale, _ = _align_trajectories(
      est_cam2w[:, :3, 3].transpose(1, 0), gt_cam2w[:, :3, 3].transpose(1, 0)
  )

  est_cam2w[:, :3, 3] = (
      scale * (rot @ est_cam2w[:, :3, 3].transpose(1, 0)) + trans
  ).transpose(1, 0)

  for k in range(num_cams):
    est_cam2w[k, :3, :3] = rot @ est_cam2w[k, :3, :3]

  traj_est_dict = [est_cam2w[i, ...] for i in range(est_cam2w.shape[0])]
  traj_gt_dict = [gt_cam2w[i, ...] for i in range(gt_cam2w.shape[0])]
  rpe_result = _evaluate_trajectory(
      traj_gt_dict, traj_est_dict, param_fixed_delta=True, param_delta=1
  )

  rte_error = np.array(rpe_result)[:, 2]
  rre_error = np.array(rpe_result)[:, 3]

  trans_error_mean = np.sqrt(np.mean(rte_error**2))
  rot_error_mean = np.sqrt(np.mean(rre_error**2))

  ate = float(np.sqrt(np.dot(trans_error, trans_error) / len(trans_error)))
  rte = float(trans_error_mean)
  rre = float(np.rad2deg(rot_error_mean))
  return ate, rte, rre


def _load_pose_gt_video_sintel() -> (
    Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]
):
  """Loads ground-truth camera extrinsics and intrinsics for SINTEL videos."""
  gt_dir = os.path.join(eval_common.RELEASE_DATA_DIR, 'sintel/camera')
  video_pose_dict = {}
  video_int_dict = {}
  for file_name in sorted(os.listdir(gt_dir)):
    if not file_name.endswith('.npz'):
      continue
    video_name = file_name.replace('_camera.npz', '')
    with open(os.path.join(gt_dir, file_name), 'rb') as f:
      camera_data = np.load(f)
      raw_ext = np.array(camera_data['extrinsics'])
      raw_int = np.array(camera_data['intrinsics'])

      t_wc = _ensure_4x4(raw_ext)
      camera_extrinsics_inverse = np.linalg.inv(t_wc)[:, :3, :4]
      aligned_gt_c2w = _align_to_first_frame(camera_extrinsics_inverse)
      video_pose_dict[video_name] = aligned_gt_c2w
      video_int_dict[video_name] = raw_int
  return video_pose_dict, video_int_dict


def evaluate_sintel_camera_pose(
    pipe: GenCeptionPipeline,
    raymap_embeds: np.ndarray,
    output_dir: str = '',
) -> Dict[str, Any]:
  """Evaluates GenCeption camera pose on SINTEL."""
  vid_dir = os.path.join(eval_common.RELEASE_DATA_DIR, 'sintel/normal/resized2')
  video_pose_dict, _ = _load_pose_gt_video_sintel()
  origin_mask, dir_mask = _create_raymap_masks()
  spatial_zoom_fn = getattr(
      eval_common, 'spatial_process_frame_zoom', _spatial_process_frame_zoom
  )

  ate_list1, rpe_trans_list1, rpe_rot_list1 = [], [], []
  ate_raw_list1, rpe_trans_raw_list1, rpe_rot_raw_list1 = [], [], []
  ate_list2, rpe_trans_list2, rpe_rot_list2 = [], [], []
  ate_raw_list2, rpe_trans_raw_list2, rpe_rot_raw_list2 = [], [], []
  per_scene: Dict[str, Dict[str, float]] = {}

  for video_name in SINTEL_VIDEOS:
    rgb_path = os.path.join(vid_dir, f'{video_name}_rgb.mp4')
    raw_rgb = eval_common.read_video(rgb_path)
    # Strip 70px top/bottom black letterbox bars present in resized2 Sintel MP4s
    if raw_rgb.shape[1] == 480:
      raw_rgb = raw_rgb[:, 70:-70]
    video_81_raw = eval_common.pad_or_truncate_to_81(raw_rgb, 81)
    video_81 = np.stack(
        [
            spatial_zoom_fn(f, 832, 480)
            for f in video_81_raw
        ],
        axis=0,
    )

    pred_video_float = eval_common.run_single_81_video(
        pipe, video_81, raymap_embeds
    )
    unscaled_raymap = _unscale_raymap_origin(pred_video_float)

    gt_extrinsics = video_pose_dict[video_name]
    gt_frame_num = len(gt_extrinsics)

    _, r_robust2_smooth, r_robust2_raw = (
        _estimate_global_params_with_mask_robust2(
            unscaled_raymap, origin_mask=origin_mask, dir_mask=dir_mask
        )
    )
    ate1, rpe_t1, rpe_r1 = _calculate_ate_rpe_megasam(
        r_robust2_smooth[:gt_frame_num], gt_extrinsics
    )
    ate1_raw, rpe_t1_raw, rpe_r1_raw = _calculate_ate_rpe_megasam(
        r_robust2_raw[:gt_frame_num], gt_extrinsics
    )
    ate_list1.append(ate1)
    rpe_trans_list1.append(rpe_t1)
    rpe_rot_list1.append(rpe_r1)
    ate_raw_list1.append(ate1_raw)
    rpe_trans_raw_list1.append(rpe_t1_raw)
    rpe_rot_raw_list1.append(rpe_r1_raw)

    _, r_svd_smooth, r_svd_raw = _estimate_global_params_with_mask_robust_svd(
        unscaled_raymap, origin_mask=origin_mask, dir_mask=dir_mask
    )
    ate2, rpe_t2, rpe_r2 = _calculate_ate_rpe_megasam(
        r_svd_smooth[:gt_frame_num], gt_extrinsics
    )
    ate2_raw, rpe_t2_raw, rpe_r2_raw = _calculate_ate_rpe_megasam(
        r_svd_raw[:gt_frame_num], gt_extrinsics
    )
    ate_list2.append(ate2)
    rpe_trans_list2.append(rpe_t2)
    rpe_rot_list2.append(rpe_r2)
    ate_raw_list2.append(ate2_raw)
    rpe_trans_raw_list2.append(rpe_t2_raw)
    rpe_rot_raw_list2.append(rpe_r2_raw)

    per_scene[video_name] = {
        'ate_robust2': ate1,
        'rpe_trans_robust2': rpe_t1,
        'rpe_rot_robust2': rpe_r1,
        'ate_robust2_raw': ate1_raw,
        'rpe_trans_robust2_raw': rpe_t1_raw,
        'rpe_rot_robust2_raw': rpe_r1_raw,
        'ate_svd': ate2,
        'rpe_trans_svd': rpe_t2,
        'rpe_rot_svd': rpe_r2,
        'ate_svd_raw': ate2_raw,
        'rpe_trans_svd_raw': rpe_t2_raw,
        'rpe_rot_svd_raw': rpe_r2_raw,
    }
    logging.info(
        '[SINTEL Camera Pose] %s: [robust2] ATE=%.4f, RPE_t=%.4f, RPE_r=%.4f |'
        ' [svd] ATE=%.4f, RPE_t=%.4f, RPE_r=%.4f',
        video_name,
        ate1,
        rpe_t1,
        rpe_r1,
        ate2,
        rpe_t2,
        rpe_r2,
    )

  results = {
      'task': 'camera_pose',
      'dataset': 'sintel',
      'ate_robust2': float(np.mean(ate_list1)),
      'rpe_trans_robust2': float(np.mean(rpe_trans_list1)),
      'rpe_rot_robust2': float(np.mean(rpe_rot_list1)),
      'ate_robust2_raw': float(np.mean(ate_raw_list1)),
      'rpe_trans_robust2_raw': float(np.mean(rpe_trans_raw_list1)),
      'rpe_rot_robust2_raw': float(np.mean(rpe_rot_raw_list1)),
      'ate_svd': float(np.mean(ate_list2)),
      'rpe_trans_svd': float(np.mean(rpe_trans_list2)),
      'rpe_rot_svd': float(np.mean(rpe_rot_list2)),
      'ate_svd_raw': float(np.mean(ate_raw_list2)),
      'rpe_trans_svd_raw': float(np.mean(rpe_trans_raw_list2)),
      'rpe_rot_svd_raw': float(np.mean(rpe_rot_raw_list2)),
      'num_scenes': len(SINTEL_VIDEOS),
      'per_scene': per_scene,
  }
  logging.info(
      '[SINTEL Camera Pose Summary] [robust2] ATE=%.4f, RPE_t=%.4f, RPE_r=%.4f'
      ' | [svd] ATE=%.4f, RPE_t=%.4f, RPE_r=%.4f',
      results['ate_robust2'],
      results['rpe_trans_robust2'],
      results['rpe_rot_robust2'],
      results['ate_svd'],
      results['rpe_trans_svd'],
      results['rpe_rot_svd'],
  )
  eval_common.save_metrics_json(output_dir, 'camera_pose_sintel.json', results)
  return results
