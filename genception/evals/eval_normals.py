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

"""Surface normals evaluation on SINTEL and Hi4D for GenCeption."""

from __future__ import annotations

import os
from typing import Any, Dict, List

from absl import logging
import numpy as np

from representations4d.genception.evals import eval_common
from representations4d.genception.pipeline import GenCeptionPipeline


def _angle_between_unit_vectors(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
  dot_product = np.clip(np.sum(v1 * v2, axis=-1), -1.0, 1.0)
  return np.arccos(dot_product) / np.pi * 180.0


def _compute_normal_metrics(
    pred_video: np.ndarray,
    gt_video: np.ndarray,
    mask_video: np.ndarray,
    frame_len: int,
    pred_transform: np.ndarray,
) -> Dict[str, float]:
  """Compute angular error metrics between predicted and GT surface normals."""
  pred_normal = (pred_video[:frame_len] * 2.0 - 1.0) * pred_transform
  gt_normal = gt_video[:frame_len] * 2.0 - 1.0
  gt_mask = np.mean(mask_video[:frame_len], axis=-1, keepdims=True) > 0.5

  pred_normal = pred_normal / (
      np.linalg.norm(pred_normal, axis=-1, keepdims=True) + 1e-6
  )
  gt_normal = gt_normal / (
      np.linalg.norm(gt_normal, axis=-1, keepdims=True) + 1e-6
  )

  error = _angle_between_unit_vectors(pred_normal, gt_normal)
  valid_error = error[gt_mask[..., 0]]

  return {
      'mAE': float(np.mean(valid_error)),
      'median_error': float(np.median(valid_error)),
      'pct_11_25': float(np.mean(valid_error < 11.25) * 100.0),
      'pct_22_5': float(np.mean(valid_error < 22.5) * 100.0),
      'pct_30': float(np.mean(valid_error < 30.0) * 100.0),
  }


def evaluate_sintel_normals(
    pipe: GenCeptionPipeline,
    normals_embeds: np.ndarray,
    output_dir: str = '',
) -> Dict[str, Any]:
  """Evaluate GenCeption surface normals on SINTEL."""
  vid_dir = os.path.join(eval_common.RELEASE_DATA_DIR, 'sintel/normal/resized2')
  raw_vid_dir = os.path.join(eval_common.RELEASE_DATA_DIR, 'sintel/normal/raw2')

  video_name_list = sorted([
      f.replace('_rgb.mp4', '')
      for f in os.listdir(vid_dir)
      if f.endswith('_rgb.mp4')
  ])

  pred_transform = np.array([-1.0, -1.0, -1.0])
  per_scene: Dict[str, Dict[str, float]] = {}
  metric_keys = ['mAE', 'median_error', 'pct_11_25', 'pct_22_5', 'pct_30']
  accum: Dict[str, List[float]] = {k: [] for k in metric_keys}

  for video_name in video_name_list:
    rgb_path = os.path.join(vid_dir, f'{video_name}_rgb.mp4')
    raw_rgb = eval_common.read_video(rgb_path)
    video_81_raw = eval_common.pad_or_truncate_to_81(raw_rgb, 81)
    video_81 = np.stack(
        [eval_common.spatial_process_frame(f, 832, 480) for f in video_81_raw],
        axis=0,
    )

    pred_video_float = eval_common.run_single_81_video(
        pipe, video_81, normals_embeds
    )

    gt_normal_raw = eval_common.read_video(
        os.path.join(vid_dir, f'{video_name}_normal.mp4')
    )
    gt_normal = np.stack(
        [eval_common.spatial_process_frame(f, 832, 480) for f in gt_normal_raw],
        axis=0,
    )
    gt_mask_raw = eval_common.read_video(
        os.path.join(vid_dir, f'{video_name}_mask.mp4')
    )
    gt_mask = np.stack(
        [eval_common.spatial_process_frame(f, 832, 480) for f in gt_mask_raw],
        axis=0,
    )
    raw_normal_for_len = eval_common.read_video(
        os.path.join(raw_vid_dir, f'{video_name}_normal.mp4')
    )
    frame_len = min(
        len(raw_normal_for_len), len(gt_normal), len(pred_video_float)
    )

    metrics = _compute_normal_metrics(
        pred_video_float, gt_normal, gt_mask, frame_len, pred_transform
    )
    per_scene[video_name] = metrics
    for k in metric_keys:
      accum[k].append(metrics[k])
    logging.info(
        '[SINTEL Normals] %s: mAE=%.2f, Med=%.2f, '
        '<11.25=%.2f%%, <22.5=%.2f%%, <30=%.2f%%',
        video_name,
        metrics['mAE'],
        metrics['median_error'],
        metrics['pct_11_25'],
        metrics['pct_22_5'],
        metrics['pct_30'],
    )

  results = {
      'task': 'normals',
      'dataset': 'sintel',
      'mAE': float(np.mean(accum['mAE'])),
      'median_error': float(np.mean(accum['median_error'])),
      'pct_11_25': float(np.mean(accum['pct_11_25'])),
      'pct_22_5': float(np.mean(accum['pct_22_5'])),
      'pct_30': float(np.mean(accum['pct_30'])),
      'num_scenes': len(video_name_list),
      'per_scene': per_scene,
  }
  logging.info(
      '[SINTEL Normals Summary] mAE=%.2f, Med=%.2f, '
      '<11.25=%.2f%%, <22.5=%.2f%%, <30=%.2f%%',
      results['mAE'],
      results['median_error'],
      results['pct_11_25'],
      results['pct_22_5'],
      results['pct_30'],
  )
  eval_common.save_metrics_json(output_dir, 'normals_sintel.json', results)
  return results


def evaluate_hi4d_normals(
    pipe: GenCeptionPipeline,
    normals_embeds: np.ndarray,
    output_dir: str = '',
) -> Dict[str, Any]:
  """Evaluate GenCeption surface normals on Hi4D."""
  vid_dir = os.path.join(
      eval_common.RELEASE_DATA_DIR, 'hi4d/videos_pad_and_crop_gl_cc'
  )

  video_name_list = sorted([
      f.replace('_color.mp4', '')
      for f in os.listdir(vid_dir)
      if f.endswith('_color.mp4')
  ])

  pred_transform = np.array([1.0, -1.0, -1.0])
  per_scene: Dict[str, Dict[str, float]] = {}
  metric_keys = ['mAE', 'median_error', 'pct_11_25', 'pct_22_5', 'pct_30']
  accum: Dict[str, List[float]] = {k: [] for k in metric_keys}

  for video_name in video_name_list:
    rgb_path = os.path.join(vid_dir, f'{video_name}_color.mp4')
    raw_rgb = eval_common.read_video(rgb_path)
    resized_rgb = np.stack(
        [eval_common.spatial_process_frame(f, 832, 480) for f in raw_rgb],
        axis=0,
    )

    pred_video_float = eval_common.run_chunked_81_video(
        pipe, resized_rgb, normals_embeds, chunk_size=81
    )

    gt_normal_raw = eval_common.read_video(
        os.path.join(vid_dir, f'{video_name}_normal.mp4')
    )
    gt_normal = np.stack(
        [eval_common.spatial_process_frame(f, 832, 480) for f in gt_normal_raw],
        axis=0,
    )
    gt_mask_raw = eval_common.read_video(
        os.path.join(vid_dir, f'{video_name}_mask.mp4')
    )
    gt_mask = np.stack(
        [eval_common.spatial_process_frame(f, 832, 480) for f in gt_mask_raw],
        axis=0,
    )
    frame_len = min(len(gt_normal), len(pred_video_float))

    metrics = _compute_normal_metrics(
        pred_video_float, gt_normal, gt_mask, frame_len, pred_transform
    )
    per_scene[video_name] = metrics
    for k in metric_keys:
      accum[k].append(metrics[k])
    logging.info(
        '[Hi4D Normals] %s: mAE=%.2f, Med=%.2f, '
        '<11.25=%.2f%%, <22.5=%.2f%%, <30=%.2f%%',
        video_name,
        metrics['mAE'],
        metrics['median_error'],
        metrics['pct_11_25'],
        metrics['pct_22_5'],
        metrics['pct_30'],
    )

  results = {
      'task': 'normals',
      'dataset': 'hi4d',
      'mAE': float(np.mean(accum['mAE'])),
      'median_error': float(np.mean(accum['median_error'])),
      'pct_11_25': float(np.mean(accum['pct_11_25'])),
      'pct_22_5': float(np.mean(accum['pct_22_5'])),
      'pct_30': float(np.mean(accum['pct_30'])),
      'num_scenes': len(video_name_list),
      'per_scene': per_scene,
  }
  logging.info(
      '[Hi4D Normals Summary] mAE=%.2f, Med=%.2f, '
      '<11.25=%.2f%%, <22.5=%.2f%%, <30=%.2f%%',
      results['mAE'],
      results['median_error'],
      results['pct_11_25'],
      results['pct_22_5'],
      results['pct_30'],
  )
  eval_common.save_metrics_json(output_dir, 'normals_hi4d.json', results)
  return results
