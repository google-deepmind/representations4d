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

"""Depth evaluation on SINTEL, KITTI, and ETH3D for GenCeption."""

from __future__ import annotations

import os
import struct
from typing import Any, Dict, List

from absl import logging
import cv2
import numpy as np
from PIL import Image

from representations4d.genception.evals import eval_common
from representations4d.genception.pipeline import GenCeptionPipeline
from representations4d.genception.utils import colored_depth_to_relative_depth
from representations4d.genception.utils import compute_median_scale_and_shift

TAG_FLOAT = 202021.25

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


def _depth_read_sintel(filename: str) -> np.ndarray:
  """Read depth data from a SINTEL .dpt file."""
  with open(filename, 'rb') as f:
    check = struct.unpack('f', f.read(4))[0]
    assert check == TAG_FLOAT, (
        f'depth_read:: Wrong tag in {filename} '
        f'(should be {TAG_FLOAT}, is {check})'
    )
    width = struct.unpack('i', f.read(4))[0]
    height = struct.unpack('i', f.read(4))[0]
    assert width > 0 and height > 0 and width < 100000 and height < 100000
    depth = (
        np.frombuffer(f.read(), dtype=np.float32)
        .copy()
        .reshape((height, width))
    )
  return depth


def _compute_absrel_with_mask(
    gt_depth: np.ndarray,
    pred_depth: np.ndarray,
    mask: np.ndarray,
) -> float:
  """Compute absolute relative error after median scale-shift alignment."""
  scale, shift = compute_median_scale_and_shift(pred_depth, gt_depth, mask)
  aligned_pred = scale * pred_depth + shift
  safe_gt = np.where(mask, gt_depth, 1.0)
  safe_pred = np.where(mask, aligned_pred, 1.0)
  absrel = np.abs(safe_pred - safe_gt) / safe_gt
  return float(np.sum(absrel * mask) / np.sum(mask))


def _compute_delta1_with_mask(
    gt_depth: np.ndarray,
    pred_depth: np.ndarray,
    mask: np.ndarray,
    threshold: float = 1.25,
    epsilon: float = 1e-6,
) -> float:
  """Compute delta1 accuracy (<1.25) after median scale-shift alignment."""
  scale, shift = compute_median_scale_and_shift(pred_depth, gt_depth, mask)
  aligned_pred = np.clip(scale * pred_depth + shift, a_min=0.0, a_max=None)
  safe_gt = np.where(mask, gt_depth, 1.0)
  safe_pred = np.where(mask, aligned_pred, 1.0)
  ratio = np.maximum(safe_pred, safe_gt) / (
      np.minimum(safe_pred, safe_gt) + epsilon
  )
  correct = ratio < threshold
  return float(np.sum(correct * mask) / np.sum(mask))


def evaluate_sintel_depth(
    pipe: GenCeptionPipeline,
    depth_embeds: np.ndarray,
    output_dir: str = '',
) -> Dict[str, Any]:
  """Evaluate GenCeption depth on SINTEL."""
  raw_vid_dir = os.path.join(eval_common.RELEASE_DATA_DIR, 'sintel/normal/raw2')
  gt_depth_dir = os.path.join(eval_common.RELEASE_DATA_DIR, 'sintel/depth')

  absrel_list: List[float] = []
  delta1_list: List[float] = []
  per_scene: Dict[str, Dict[str, float]] = {}

  for video_name in SINTEL_VIDEOS:
    rgb_path = os.path.join(raw_vid_dir, f'{video_name}.mp4')
    raw_video = eval_common.read_video(rgb_path)
    video_81_raw = eval_common.pad_or_truncate_to_81(raw_video, 81)
    video_81 = np.stack(
        [
            eval_common.resize_exact_process_frame(f, 832, 480, is_depth=False)
            for f in video_81_raw
        ],
        axis=0,
    )

    pred_video_float = eval_common.run_single_81_video(
        pipe, video_81, depth_embeds
    )
    pred_depth = colored_depth_to_relative_depth(
        pred_video_float, depth_scaling_factor=0.4
    )

    video_depth_dir = os.path.join(gt_depth_dir, video_name)
    dpt_files = sorted(
        [f for f in os.listdir(video_depth_dir) if f.endswith('.dpt')]
    )
    gt_frames = []
    for dpt_file in dpt_files:
      d = _depth_read_sintel(os.path.join(video_depth_dir, dpt_file))
      gt_frames.append(
          eval_common.resize_exact_process_frame(d, 832, 480, is_depth=True)
      )
    gt_depth = np.stack(gt_frames, axis=0)

    valid_frames = min(gt_depth.shape[0], pred_depth.shape[0])
    gt_valid = gt_depth[:valid_frames]
    pred_valid = pred_depth[:valid_frames]

    mask = gt_valid < 70.0
    absrel = _compute_absrel_with_mask(gt_valid, pred_valid, mask)
    delta1 = _compute_delta1_with_mask(gt_valid, pred_valid, mask)
    absrel_list.append(absrel)
    delta1_list.append(delta1)
    per_scene[video_name] = {'absrel': absrel, 'delta1': delta1}
    logging.info(
        '[SINTEL Depth] %s: AbsRel=%.4f, Delta1=%.4f',
        video_name,
        absrel,
        delta1,
    )

  results = {
      'task': 'depth',
      'dataset': 'sintel',
      'absrel': float(np.mean(absrel_list)),
      'delta1': float(np.mean(delta1_list)),
      'num_scenes': len(absrel_list),
      'per_scene': per_scene,
  }
  logging.info(
      '[SINTEL Depth Summary] AbsRel=%.4f, Delta1=%.4f',
      results['absrel'],
      results['delta1'],
  )
  eval_common.save_metrics_json(output_dir, 'depth_sintel.json', results)
  return results


def evaluate_kitti_depth(
    pipe: GenCeptionPipeline,
    depth_embeds: np.ndarray,
    output_dir: str = '',
) -> Dict[str, Any]:
  """Evaluate GenCeption depth on KITTI."""
  raw_vid_dir = os.path.join(eval_common.RELEASE_DATA_DIR, 'kitti/rgb')
  gt_depth_dir = os.path.join(eval_common.RELEASE_DATA_DIR, 'kitti/val')

  video_name_list = sorted([
      f.replace('.mp4', '')
      for f in os.listdir(raw_vid_dir)
      if f.endswith('.mp4')
      and 'mask' not in f
      and 'depth' not in f
      and 'normal' not in f
  ])

  absrel_list: List[float] = []
  delta1_list: List[float] = []
  per_scene: Dict[str, Dict[str, float]] = {}

  for video_name in video_name_list:
    rgb_path = os.path.join(raw_vid_dir, f'{video_name}.mp4')
    raw_video = eval_common.read_video(rgb_path)
    resized_video = np.stack(
        [
            eval_common.resize_exact_process_frame(f, 832, 480, is_depth=False)
            for f in raw_video
        ],
        axis=0,
    )

    pred_video_float = eval_common.run_chunked_81_video(
        pipe, resized_video, depth_embeds, chunk_size=81
    )
    pred_depth = colored_depth_to_relative_depth(
        pred_video_float, depth_scaling_factor=0.4
    )

    video_depth_dir = os.path.join(
        gt_depth_dir, video_name, 'proj_depth/groundtruth/image_02'
    )
    png_files = sorted(
        [f for f in os.listdir(video_depth_dir) if f.endswith('.png')]
    )
    gt_frames = []
    for png_file in png_files:
      with open(os.path.join(video_depth_dir, png_file), 'rb') as f:
        d = np.array(Image.open(f), dtype=np.float32) / 256.0
      gt_frames.append(
          eval_common.resize_exact_process_frame(d, 832, 480, is_depth=True)
      )
    gt_depth = np.stack(gt_frames, axis=0)

    valid_frames = min(gt_depth.shape[0], pred_depth.shape[0])
    gt_valid = gt_depth[:valid_frames]
    pred_valid = pred_depth[:valid_frames]

    scene_absrels = []
    scene_delta1s = []
    for frame_start in range(0, len(gt_valid), 81):
      gt_seg = gt_valid[frame_start : frame_start + 81]
      pred_seg = pred_valid[frame_start : frame_start + len(gt_seg)]
      mask = (gt_seg < 80.0) & (gt_seg > 0.01)
      absrel = _compute_absrel_with_mask(gt_seg, pred_seg, mask)
      delta1 = _compute_delta1_with_mask(gt_seg, pred_seg, mask)
      absrel_list.append(absrel)
      delta1_list.append(delta1)
      scene_absrels.append(absrel)
      scene_delta1s.append(delta1)

    per_scene[video_name] = {
        'absrel': float(np.mean(scene_absrels)),
        'delta1': float(np.mean(scene_delta1s)),
    }
    logging.info(
        '[KITTI Depth] %s: AbsRel=%.4f, Delta1=%.4f',
        video_name,
        per_scene[video_name]['absrel'],
        per_scene[video_name]['delta1'],
    )

  results = {
      'task': 'depth',
      'dataset': 'kitti',
      'absrel': float(np.mean(absrel_list)),
      'delta1': float(np.mean(delta1_list)),
      'num_segments': len(absrel_list),
      'per_scene': per_scene,
  }
  logging.info(
      '[KITTI Depth Summary] AbsRel=%.4f, Delta1=%.4f',
      results['absrel'],
      results['delta1'],
  )
  eval_common.save_metrics_json(output_dir, 'depth_kitti.json', results)
  return results


def evaluate_eth3d_depth(
    pipe: GenCeptionPipeline,
    depth_embeds: np.ndarray,
    output_dir: str = '',
) -> Dict[str, Any]:
  """Evaluate GenCeption depth on ETH3D."""
  eth3d_dir = os.path.join(eval_common.RELEASE_DATA_DIR, 'eth3d/extracted')

  absrel_list: List[float] = []
  delta1_list: List[float] = []
  per_frame: Dict[str, Dict[str, float]] = {}

  for scene_name in sorted(os.listdir(eth3d_dir)):
    scene_img_dir = os.path.join(eth3d_dir, scene_name, 'images', 'dslr_images')
    if not os.path.exists(scene_img_dir):
      continue

    for frame_file in sorted(os.listdir(scene_img_dir)):
      if not frame_file.endswith('.JPG'):
        continue

      rgb_path = os.path.join(scene_img_dir, frame_file)
      gt_path = os.path.join(
          eth3d_dir, scene_name, 'ground_truth_depth', 'dslr_images', frame_file
      )

      with open(rgb_path, 'rb') as f:
        img = Image.open(f)
        orig_w, orig_h = img.size
        rgb_np = np.array(img.convert('RGB'), dtype=np.float32) / 255.0

      rgb_resized = eval_common.resize_exact_process_frame(
          rgb_np, 832, 480, is_depth=False
      )
      video_81 = np.repeat(rgb_resized[np.newaxis, ...], 81, axis=0)

      with open(gt_path, 'rb') as f:
        raw_data = f.read()
        gt_depth_map = (
            np.frombuffer(raw_data, dtype=np.float32)
            .copy()
            .reshape((orig_h, orig_w))
        )
      gt_depth_resized = cv2.resize(
          gt_depth_map, (832, 480), interpolation=cv2.INTER_NEAREST
      )[np.newaxis, ...]

      pred_video_float = eval_common.run_single_81_video(
          pipe, video_81, depth_embeds
      )
      pred_mean = np.mean(pred_video_float, axis=0, keepdims=True)
      pred_depth = colored_depth_to_relative_depth(
          pred_mean, depth_scaling_factor=0.4
      )

      mask = (
          np.isfinite(gt_depth_resized)
          & (gt_depth_resized > 0.0)
          & (gt_depth_resized < 70.0)
      )
      if mask.sum() < 100:
        continue
      absrel = _compute_absrel_with_mask(gt_depth_resized, pred_depth, mask)
      delta1 = _compute_delta1_with_mask(gt_depth_resized, pred_depth, mask)
      absrel_list.append(absrel)
      delta1_list.append(delta1)
      key = f'{scene_name}_{frame_file[:-4]}'
      per_frame[key] = {'absrel': absrel, 'delta1': delta1}
      logging.info(
          '[ETH3D Depth] %s: AbsRel=%.4f, Delta1=%.4f '
          '(running avg AbsRel=%.4f)',
          key,
          absrel,
          delta1,
          float(np.mean(absrel_list)),
      )

  results = {
      'task': 'depth',
      'dataset': 'eth3d',
      'absrel': float(np.mean(absrel_list)),
      'delta1': float(np.mean(delta1_list)),
      'num_frames': len(absrel_list),
      'per_frame': per_frame,
  }
  logging.info(
      '[ETH3D Depth Summary] AbsRel=%.4f, Delta1=%.4f',
      results['absrel'],
      results['delta1'],
  )
  eval_common.save_metrics_json(output_dir, 'depth_eth3d.json', results)
  return results
