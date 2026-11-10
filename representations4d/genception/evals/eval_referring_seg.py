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

"""Expression-referring video segmentation evaluation for GenCeption."""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from absl import logging
import cv2
import numpy as np

from representations4d.genception.evals import eval_common
from representations4d.genception.pipeline import GenCeptionPipeline


def _get_test_video_list(
    dataset_name: str,
    dataset_stride: int = 1,
) -> List[Tuple[str, List[Tuple[str, str, str]]]]:
  """Parse the dataset imgrid file into grouped (vid_id, [(rgb, gt, exp)])."""
  refvos_dir = os.path.join(eval_common.RELEASE_DATA_DIR, 'refvos')
  mp4_dir = os.path.join(refvos_dir, 'mp4s', dataset_name)
  imgrid_path = os.path.join(refvos_dir, f'{dataset_name}-all.imgrid')
  with open(imgrid_path, 'r') as f:
    html_content = f.read()

  match = re.search(r'const items = (\[.*?\]);', html_content, re.DOTALL)
  video_dict: Dict[str, List[Tuple[str, str, str]]] = {}
  if match:
    data = json.loads(match.group(1))
    for item in data:
      rgb_path = os.path.join(mp4_dir, os.path.basename(item['rgb_video']))
      gt_path = os.path.join(mp4_dir, os.path.basename(item['gt_mask_video']))
      exp_str = item['exp_str'].replace('Prompt: ', '')
      vid_id = os.path.basename(rgb_path)
      if vid_id not in video_dict:
        video_dict[vid_id] = []
      video_dict[vid_id].append((rgb_path, gt_path, exp_str))
  else:
    current_rgb = None
    for line in html_content.strip().splitlines():
      line = line.strip()
      if not line:
        continue
      if '_cap_' not in line:
        current_rgb = os.path.join(mp4_dir, os.path.basename(line))
        vid_id = os.path.basename(current_rgb)
        if vid_id not in video_dict:
          video_dict[vid_id] = []
      else:
        gt_path = os.path.join(mp4_dir, os.path.basename(line))
        cap_part = line.split('_cap_', 1)[1]
        if cap_part.endswith('..mp4'):
          exp_str = cap_part[:-5].replace('_', ' ')
        elif cap_part.endswith('.mp4'):
          exp_str = cap_part[:-4].replace('_', ' ')
        else:
          exp_str = cap_part.replace('_', ' ')
        vid_id = (
            os.path.basename(current_rgb)
            if current_rgb
            else os.path.basename(line).split('_cap_')[0] + '.mp4'
        )
        if vid_id not in video_dict:
          video_dict[vid_id] = []
        video_dict[vid_id].append((current_rgb, gt_path, exp_str))

  return list(video_dict.items())[::dataset_stride]


_EXPRESSION_ALIASES: Dict[str, str] = {
    'bike-packing_cap_a_man_wearing_a_cap_and_shorts..npy': (
        'bike-packing_cap_and_shorts..npy'
    ),
}


def _load_expression_embedding(
    dataset_name: str, gt_path: str
) -> Optional[np.ndarray]:
  """Load precomputed UMT5 embedding for a referring expression.

  Looks under ``EMBEDDINGS_DIR/refvos/`` in both the full dataset-name layout
  (e.g. ``davis_val/``) and the published asset layout (``davis/`` or
  ``mevis/``).

  Args:
    dataset_name: Dataset name, e.g. ``'davis_val'`` or ``'mevis_valid_u'``.
    gt_path: Path to the ground-truth mask video for the expression.

  Returns:
    The embedding array, or None if not found.
  """
  npy_name = os.path.basename(gt_path).replace('.mp4', '.npy')
  npy_name = _EXPRESSION_ALIASES.get(npy_name, npy_name)
  short_ds = 'davis' if 'davis' in dataset_name else 'mevis'
  candidates = []
  if eval_common.EMBEDDINGS_DIR:
    for subdir in (dataset_name, short_ds):
      candidates.append(
          os.path.join(eval_common.EMBEDDINGS_DIR, 'refvos', subdir, npy_name)
      )
  for path in candidates:
    if os.path.exists(path):
      with open(path, 'rb') as f:
        return np.load(f)
  logging.warning(
      'Referring expression embedding not found for %s in %s',
      gt_path,
      candidates,
  )
  return None


def _compute_jaccard(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
  """Compute mean region similarity (J / IoU) across frames."""
  ious = []
  for pred, gt in zip(pred_mask, gt_mask):
    pred_b = pred.astype(bool)
    gt_b = gt.astype(bool)
    intersection = np.logical_and(pred_b, gt_b).sum()
    union = np.logical_or(pred_b, gt_b).sum()
    if union == 0:
      ious.append(1.0)
    else:
      ious.append(float(intersection) / float(union))
  return float(np.mean(ious))


def _mask_to_boundary(
    mask: np.ndarray, dilation_ratio: float = 0.02
) -> np.ndarray:
  """Convert binary mask to boundary mask using morphological erosion."""
  h, w = mask.shape
  img_diag = np.sqrt(h**2 + w**2)
  dilation = max(1, int(round(dilation_ratio * img_diag)))
  kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
  mask_uint8 = mask.astype(np.uint8)
  eroded = cv2.erode(mask_uint8, kernel, iterations=dilation)
  return mask_uint8 - eroded


def _compute_f_score(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
  """Compute mean contour accuracy (F-measure) across frames."""
  f_scores = []
  for pred, gt in zip(pred_mask, gt_mask):
    pred_b = pred.astype(bool)
    gt_b = gt.astype(bool)
    if not np.any(gt_b) and not np.any(pred_b):
      f_scores.append(1.0)
      continue
    if not np.any(gt_b) or not np.any(pred_b):
      f_scores.append(0.0)
      continue

    pred_bound = _mask_to_boundary(pred_b)
    gt_bound = _mask_to_boundary(gt_b)

    h, w = pred.shape
    img_diag = np.sqrt(h**2 + w**2)
    dilation = max(1, int(round(0.02 * img_diag)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    pred_dilated = cv2.dilate(pred_bound, kernel, iterations=dilation)
    gt_dilated = cv2.dilate(gt_bound, kernel, iterations=dilation)

    precision = np.sum(pred_bound & gt_dilated) / (np.sum(pred_bound) + 1e-8)
    recall = np.sum(gt_bound & pred_dilated) / (np.sum(gt_bound) + 1e-8)
    if precision + recall == 0:
      f_scores.append(0.0)
    else:
      f_scores.append(float(2.0 * precision * recall / (precision + recall)))
  return float(np.mean(f_scores))


def evaluate_referring_segmentation(
    pipe: GenCeptionPipeline,
    dataset_name: str = 'davis_val',
    output_dir: str = '',
    worker_id: int = 0,
    num_workers: int = 1,
) -> Dict[str, Any]:
  """Evaluate GenCeption referring video segmentation on Ref-DAVIS or MEVIS."""
  video_list = _get_test_video_list(dataset_name, dataset_stride=1)
  sharded_list = video_list[worker_id::num_workers]
  logging.info(
      '[Referring Seg %s] Worker %d/%d evaluating %d/%d videos',
      dataset_name,
      worker_id,
      num_workers,
      len(sharded_list),
      len(video_list),
  )

  j_list: List[float] = []
  f_list: List[float] = []
  jf_list: List[float] = []
  per_expression: Dict[str, Dict[str, Any]] = {}

  for vid_idx, (vid_id, vid_items) in enumerate(sharded_list):
    rgb_path = vid_items[0][0]
    raw_rgb = eval_common.read_video(rgb_path)
    video_81_raw = eval_common.pad_or_truncate_to_81(raw_rgb, 81)
    video_81 = np.stack(
        [
            eval_common.spatial_process_frame_old(f, 832, 480)
            for f in video_81_raw
        ],
        axis=0,
    )

    latents = pipe.encode_video(video_81)

    for rgb_p, gt_path, exp_str in vid_items:
      del rgb_p
      prompt_embeds = _load_expression_embedding(dataset_name, gt_path)
      if prompt_embeds is None:
        logging.warning('Skipping %s due to missing prompt embedding', gt_path)
        continue
      result = pipe(prompt_embeds=prompt_embeds, latents=latents)
      pred_video_float = eval_common.get_last_video_float(pipe, result)

      gt_raw = eval_common.read_video(gt_path)
      gt_video = np.stack(
          [eval_common.spatial_process_frame_old(f, 832, 480) for f in gt_raw],
          axis=0,
      )
      gt_mask = (np.mean(gt_video, axis=-1) > 0.5).astype(np.uint8)
      valid_len = min(len(gt_mask), len(pred_video_float))
      gt_mask = gt_mask[:valid_len]
      pred_mask = (np.mean(pred_video_float[:valid_len], axis=-1) > 0.5).astype(
          np.uint8
      )

      j_score = _compute_jaccard(pred_mask, gt_mask) * 100.0
      f_score = _compute_f_score(pred_mask, gt_mask) * 100.0
      jf_score = 0.5 * (j_score + f_score)

      j_list.append(j_score)
      f_list.append(f_score)
      jf_list.append(jf_score)
      exp_key = os.path.basename(gt_path).replace('.mp4', '')
      per_expression[exp_key] = {
          'J&F': jf_score,
          'J': j_score,
          'F': f_score,
          'prompt': exp_str,
      }

    logging.info(
        '[Referring Seg %s] (%d/%d) %s (%d exps): running J&F=%.2f, J=%.2f,'
        ' F=%.2f',
        dataset_name,
        vid_idx + 1,
        len(sharded_list),
        vid_id,
        len(vid_items),
        float(np.mean(jf_list)),
        float(np.mean(j_list)),
        float(np.mean(f_list)),
    )

  results = {
      'task': 'referring_seg',
      'dataset': dataset_name,
      'worker_id': worker_id,
      'num_workers': num_workers,
      'J&F': float(np.mean(jf_list)) if jf_list else 0.0,
      'J': float(np.mean(j_list)) if j_list else 0.0,
      'F': float(np.mean(f_list)) if f_list else 0.0,
      'num_expressions': len(jf_list),
      'per_expression': per_expression,
  }
  logging.info(
      '[Referring Seg %s Summary (worker %d/%d)] J&F=%.2f, J=%.2f, F=%.2f'
      ' (%d expressions)',
      dataset_name,
      worker_id,
      num_workers,
      results['J&F'],
      results['J'],
      results['F'],
      len(jf_list),
  )

  if num_workers > 1:
    fname = f'referring_seg_{dataset_name}_w{worker_id}_of_{num_workers}.json'
    eval_common.save_metrics_json(output_dir, fname, results)
    # Check if all workers have finished and aggregate if so.
    all_exps: Dict[str, Dict[str, Any]] = {}
    all_present = True
    for w in range(num_workers):
      w_path = os.path.join(
          output_dir, f'referring_seg_{dataset_name}_w{w}_of_{num_workers}.json'
      )
      if not os.path.exists(w_path):
        all_present = False
        break
      with open(w_path, 'r') as f:
        w_data = json.load(f)
      all_exps.update(w_data.get('per_expression', {}))
    if all_present and all_exps:
      agg_jf = float(np.mean([v['J&F'] for v in all_exps.values()]))
      agg_j = float(np.mean([v['J'] for v in all_exps.values()]))
      agg_f = float(np.mean([v['F'] for v in all_exps.values()]))
      agg_results = {
          'task': 'referring_seg',
          'dataset': dataset_name,
          'J&F': agg_jf,
          'J': agg_j,
          'F': agg_f,
          'num_expressions': len(all_exps),
          'per_expression': all_exps,
      }
      logging.info(
          '[Referring Seg %s FULL AGGREGATED Summary] J&F=%.2f, J=%.2f, F=%.2f'
          ' (%d expressions)',
          dataset_name,
          agg_jf,
          agg_j,
          agg_f,
          len(all_exps),
      )
      eval_common.save_metrics_json(
          output_dir, f'referring_seg_{dataset_name}.json', agg_results
      )
      return agg_results
  else:
    eval_common.save_metrics_json(
        output_dir, f'referring_seg_{dataset_name}.json', results
    )

  return results
