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

"""Unified CLI entry point for running GenCeption evaluations."""

from __future__ import annotations

from collections.abc import Sequence
import json

from absl import app
from absl import flags
from absl import logging
import jax

from representations4d.genception.evals import eval_camera_pose
from representations4d.genception.evals import eval_common
from representations4d.genception.evals import eval_depth
from representations4d.genception.evals import eval_normals
from representations4d.genception.evals import eval_referring_seg

_TASK = flags.DEFINE_enum(
    'task',
    'depth',
    ['depth', 'normals', 'camera_pose', 'referring_seg'],
    'Evaluation task to run.',
)
_DATASET = flags.DEFINE_enum(
    'dataset',
    'sintel',
    ['sintel', 'kitti', 'eth3d', 'hi4d', 'davis_val', 'mevis_valid_u'],
    'Dataset to evaluate on.',
)
_CHECKPOINT_DIR = flags.DEFINE_string(
    'checkpoint_dir',
    '',
    'Path to the converted GenCeption checkpoint directory.',
)
_MODEL_SIZE = flags.DEFINE_enum(
    'model_size',
    '14b',
    ['14b', '1.3b'],
    'Model size variant.',
)
_OUTPUT_DIR = flags.DEFINE_string(
    'output_dir',
    '',
    'Directory to write evaluation JSON metrics.',
)
_DATA_DIR = flags.DEFINE_string(
    'data_dir',
    '',
    'Root directory containing evaluation datasets.',
)
_EMBEDDINGS_DIR = flags.DEFINE_string(
    'embeddings_dir',
    '',
    'Directory containing precomputed prompt embeddings.',
)
_WORKER_ID = flags.DEFINE_integer(
    'worker_id',
    0,
    'Worker shard ID (0-indexed) for sharded evaluations.',
)
_NUM_WORKERS = flags.DEFINE_integer(
    'num_workers',
    1,
    'Total number of workers for sharded evaluations.',
)


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  # Configure eval_common module-level paths from flags.
  if _DATA_DIR.value:
    eval_common.RELEASE_DATA_DIR = _DATA_DIR.value
  if _CHECKPOINT_DIR.value:
    eval_common.DEFAULT_CHECKPOINT_DIR = _CHECKPOINT_DIR.value
  if _EMBEDDINGS_DIR.value:
    eval_common.EMBEDDINGS_DIR = _EMBEDDINGS_DIR.value

  logging.info(
      'Starting GenCeption evaluation: task=%s, dataset=%s, checkpoint_dir=%s,'
      ' model_size=%s, jax.devices()=%s',
      _TASK.value,
      _DATASET.value,
      _CHECKPOINT_DIR.value,
      _MODEL_SIZE.value,
      jax.devices(),
  )

  # 1. Load or compute fixed prompt embeddings first (before loading model).
  prompt_embeds = None
  if _TASK.value in ('depth', 'normals'):
    prompt_embeds = eval_common.get_or_compute_prompt_embeds(_TASK.value)
  elif _TASK.value == 'camera_pose':
    prompt_embeds = eval_common.get_or_compute_prompt_embeds('raymap')

  # 2. Load the converted GenCeptionPipeline.
  pipe = eval_common.load_pipeline(
      checkpoint_dir=_CHECKPOINT_DIR.value,
      model_size=_MODEL_SIZE.value,
  )

  # 3. Run the requested evaluation.
  if _TASK.value == 'depth':
    assert prompt_embeds is not None
    if _DATASET.value == 'sintel':
      results = eval_depth.evaluate_sintel_depth(
          pipe, prompt_embeds, output_dir=_OUTPUT_DIR.value
      )
    elif _DATASET.value == 'kitti':
      results = eval_depth.evaluate_kitti_depth(
          pipe, prompt_embeds, output_dir=_OUTPUT_DIR.value
      )
    elif _DATASET.value == 'eth3d':
      results = eval_depth.evaluate_eth3d_depth(
          pipe, prompt_embeds, output_dir=_OUTPUT_DIR.value
      )
    else:
      raise ValueError(f'Unsupported dataset for depth: {_DATASET.value}')

  elif _TASK.value == 'normals':
    assert prompt_embeds is not None
    if _DATASET.value == 'sintel':
      results = eval_normals.evaluate_sintel_normals(
          pipe, prompt_embeds, output_dir=_OUTPUT_DIR.value
      )
    elif _DATASET.value == 'hi4d':
      results = eval_normals.evaluate_hi4d_normals(
          pipe, prompt_embeds, output_dir=_OUTPUT_DIR.value
      )
    else:
      raise ValueError(f'Unsupported dataset for normals: {_DATASET.value}')

  elif _TASK.value == 'camera_pose':
    assert prompt_embeds is not None
    if _DATASET.value == 'sintel':
      results = eval_camera_pose.evaluate_sintel_camera_pose(
          pipe, prompt_embeds, output_dir=_OUTPUT_DIR.value
      )
    else:
      raise ValueError(f'Unsupported dataset for camera_pose: {_DATASET.value}')

  elif _TASK.value == 'referring_seg':
    if _DATASET.value in ('davis_val', 'mevis_valid_u'):
      results = eval_referring_seg.evaluate_referring_segmentation(
          pipe,
          dataset_name=_DATASET.value,
          output_dir=_OUTPUT_DIR.value,
          worker_id=_WORKER_ID.value,
          num_workers=_NUM_WORKERS.value,
      )
    else:
      raise ValueError(
          f'Unsupported dataset for referring_seg: {_DATASET.value}'
      )
  else:
    raise ValueError(f'Unknown task: {_TASK.value}')

  summary = {
      k: v
      for k, v in results.items()
      if k not in ('per_scene', 'per_frame', 'per_expression')
  }
  logging.info('========================================')
  logging.info('FINAL EVALUATION SUMMARY: %s', json.dumps(summary, indent=2))
  logging.info('========================================')


if __name__ == '__main__':
  app.run(main)
