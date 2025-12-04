# Copyright 2025 Google LLC
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


from kauldron import konfig

# pylint: disable=g-import-not-at-top
with konfig.imports():
  from kauldron import kd
  from moog import modules
  from moog import attention
  from moog import conv
  from moog import pos_embeddings
  from moog import initializers
  from flax import linen as nn
  import optax
  import tensorflow.experimental.numpy as tnp


def get_config():
  """Get the default hyperparameter configuration."""
  cfg = kd.train.Trainer()
  cfg.seed = 1
  cfg.num_train_steps = 1100  # 1_000_000

  cfg.aux = {"num_tokens": 1024, "token_size": 512}

  # Full Model
  # TODO(svansteenkiste): 128x config and main table changes, like bs
  cfg.model = modules.Moog(
      encoder=modules.SRTEncoder(
          backbone=conv.ConvNet(
              features=(64, 128, 128, 256, 256, 512),
              kernel_sizes=((3, 3),) * 6,
              strides=((1, 1), (1, 1), (1, 1), (2, 2), (1, 1), (2, 2)),
              normalize_output=True,
          ),
          transformer=attention.ImprovedTransformer(
              qkv_size=64 * 8,
              num_heads=8,
              mlp_size=2048,
              hidden_size=512,
              num_layers=0,  # NOTE: this just adds a linear + norm
          ),
          pos_embedding=pos_embeddings.FourierEmbedding(
              num_fourier_bases=20,
              axes=(-3, -2),
              update_type="project_add",
          ),
      ),
      initializer=initializers.RandomStateInit(
          shape=(cfg.ref.aux["num_tokens"], cfg.ref.aux["token_size"]),
          random_init_scale=1e-4,
      ),
      state_grad_projector=nn.Dense(cfg.ref.aux["token_size"]),
      correction_layer_norm=nn.LayerNorm(
          epsilon=1e-4, use_scale=True, use_bias=False
      ),
      corrector=attention.ImprovedTransformer(
          qkv_size=64 * 8,
          num_heads=8,
          mlp_size=2048,
          # hidden_size=512,  # NOTE: Activate when using small slot_size.
          num_layers=2,
      ),
      predictor=attention.ImprovedTransformer(
          qkv_size=64 * 4,
          num_heads=4,
          mlp_size=2048,
          # hidden_size=512,  # NOTE: Activate when using small slot_size.
          num_layers=3,
      ),
      decoder=modules.SRTDecoder(
          # TODO(tkipf): Make initial decoder #feats configurable.
          transformer=attention.ImprovedTransformer(
              qkv_size=64 * 2,
              num_heads=2,
              mlp_size=2048,
              # hidden_size=512,  # NOTE: Activate if small slot_size.
              # output_size=512,  # NOTE: Activate if small slot_size.
              num_layers=6,
              cross_attn_only=True,
          ),
          pos_embedding=pos_embeddings.SampleFourierEmbedding(
              num_fourier_bases=16,
              update_type="concat",
          ),
          subsample_factor=8,
          eval_subsample_factor=4,
      ),
      video="batch.video",
      stop_gradient_prob=None,
      use_loss_gradient=False,
  )

  # RNGs.
  cfg.rng_streams = kd.train.RngStreams([
      kd.train.RngStream("default", eval=True),
      kd.train.RngStream("state_init", eval=False),
  ])

  # Losses
  cfg.train_losses = {
      "recon_predicted": kd.losses.L2(
          preds="preds.video_predicted",
          targets="preds.subsampled_targets_predicted",
      ),
  }

  # Optimizer
  cfg.schedules = {
      "learning_rate": optax.warmup_cosine_decay_schedule(
          init_value=0.0,
          peak_value=1e-4,
          warmup_steps=1000,
          decay_steps=cfg.ref.num_train_steps,
          end_value=1e-7,
      )
  }

  cfg.optimizer = optax.chain(
      optax.clip_by_global_norm(1.0),
      optax.adamw(
          learning_rate=cfg.ref.schedules["learning_rate"],
          b2=0.95,
      ),
  )

  cfg.train_metrics = {
      "psnr": kd.metrics.Psnr(
          pred="preds.video_predicted",
          target="preds.subsampled_targets_predicted",
      ),
  }

  cfg.log_metrics_every = 100
  cfg.log_summaries_every = 1000

  # Dataset
  shared_ds_kwargs = dict(
      batch_size=128,
      num_frames=8,
  )
  cfg.train_ds = _make_ds(train=True, shuffle=True, **shared_ds_kwargs)
  cfg.eval_ds = _make_ds(train=False, shuffle=False, **shared_ds_kwargs)

  # add readout modules and losses
  finetune = False
  r_heads, r_inputs, r_losses, r_metrics = {}, {}, {}, {}

  r_heads.update({
      "points": modules.ConditionalAutoregressiveReadout(
          backbone=modules.TransformerReadout(
              transformer=attention.ImprovedTransformer(
                  qkv_size=64 * 8,
                  num_heads=8,
                  mlp_size=2048,
                  hidden_size=512,
                  output_size=512,
                  num_layers=3,
                  cross_attn_only=True,
              ),
          ),
          predictor=modules.MLP(
              hidden_size=512,
              output_size=512,  # Should be same as transformer.
          ),
          initializer=kd.nn.Sequential(
              layers=[
                  pos_embeddings.SampleFourierEmbedding(
                      num_fourier_bases=16,
                      update_type="replace",
                  ),
                  modules.MLP(
                      hidden_size=512,
                      output_size=512,  # Should be same as transformer.
                  ),
              ]
          ),
          finetune=finetune,
          state_stop_gradient_prob=None,
          state_layer_norm=nn.LayerNorm(
              epsilon=1e-4,
              use_scale=True,
              use_bias=False,
          ),
          output_value_size=2,  # Keypoint readout.
          output_logit_size=2,  # Visible prediction and uncertainity
          output_value_activation=nn.sigmoid,
          readout_state_key="readout_state_points",
      ),
  })

  r_inputs.update({
      "points": {
          "features": "preds.states_concat",
          "conditioning": "batch.query_points_video",
          "conditioning_mask": "batch.query_points_frame_mask",
          # Workaround for passing (optional) state: pass entire batch.
          # Kauldron paths do not (yet) support optional features.
          "batch": "batch",
      }
  })

  r_heads.update({
      "depth": modules.DenseReadout(
          backbone=modules.SRTDecoder(
              transformer=attention.ImprovedTransformer(
                  qkv_size=64 * 8,
                  num_heads=8,
                  mlp_size=2048,
                  num_layers=3,
                  cross_attn_only=True,
              ),
              pos_embedding=pos_embeddings.SampleFourierEmbedding(
                  num_fourier_bases=16,
                  update_type="concat",
              ),
              subsample_factor=8,
              eval_subsample_factor=4,
          ),
          finetune=finetune,
      ),
  })

  r_inputs.update({
      "depth": {
          "features": "preds.states_concat",
          "target": "batch.depth_video",
          "mask": "batch.depth_mask",
      },
  })

  # configure readout wrapper.
  # TODO(svansteenkiste): import from kd.contrib.nn instead
  cfg.model = modules.ReadoutWrapper(
      model=cfg.model,
      readout_heads=r_heads,
      readout_inputs=r_inputs,
      model_inputs={"video": "batch.video", "batch": "batch"},
      finetune=True,  # NOTE: Configure fine-tuning in respective readouts.
  )

  # update losses
  cfg.train_losses.update(r_losses)

  # update metrics
  eval_metrics = cfg.train_metrics
  eval_metrics.update(r_metrics)

  # create evaluator
  cfg.evals = {
      "eval": kd.evals.Evaluator(
          run=kd.evals.EveryNSteps(5000),
          ds=_make_ds(train=False, shuffle=False, **shared_ds_kwargs),
          num_batches=2,
          metrics=eval_metrics,
      )
  }

  cfg.writer = kd.train.metric_writer.NoopWriter()

  return cfg


def _make_ds(
    # name,
    train,
    shuffle,
    num_frames,
    # data_dir=None,
    batch_size=None,
    # shuffle_buffer_size=10_000,
    num_point_tracks=64,
    num_box_tracks=25,
    # add_constants_values=None,
    # random_crop=None,
    # color_augmentation=False,
    # resolution=(128, 128),
    # custom_transforms=None,
):
  """Create MOVI dataset supporting training and eval."""
  # TODO(svansteenkiste): re-enable this.
  del train, shuffle, num_frames, batch_size, num_box_tracks  # unused.

  # if random_crop is None:
  #   random_crop = True if train else False
  # if custom_transforms is None:
  #   custom_transforms = []
  # transforms = [
  #     preprocessing.AddPointTracksKubric(
  #         tracks_to_sample=num_point_tracks,
  #         # TODO(svansteenkiste): make dynamic for other MOVI datasets.
  #         max_seg_id=25,
  #         mask_background_tracks=False,
  #         mask_occluded=True,
  #         random_crop=random_crop,
  #         resolution=resolution,
  #     )
  # ]

  # if color_augmentation:  # color augmentation occur before kd.data.ValueRange
  #   transforms.append(preprocessing.TapNetColorAugmentation())

  # transforms += [
  #     preprocessing.TapNetRecursiveStack(),  # NOTE: don't do this in eval
  #     preprocessing.Unbatch(),  # NOTE: don't do this in eval
  #     kd.contrib.data.AddConstants(
  #         values={
  #             "depth_mask": tnp.full(
  #                 (num_frames, resolution[0], resolution[1], 1),
  #                 fill_value=1,  # depth is defined everywhere.
  #                 dtype=tnp.float32,
  #             )
  #         }
  #     ),
  #     preprocessing.MoviDepthRangeProcessor(),
  #     kd.data.Elements(
  #         keep=[
  #             "video",
  #             "points_mask",
  #             "depth_mask",
  #         ],
  #         rename={
  #             "boxes": "boxes_video",
  #             "target_points": "target_points_video",
  #             "visible": "target_points_visible_video",
  #             "depth": "depth_video",
  #         },
  #     ),
  #     # scale video back to 0..1 as Kubric tracks change to -1.0 to 1.0
  #     kd.data.ValueRange(key="video", in_vrange=(-1, 1), vrange=(0, 1)),
  #     kd.contrib.data.TemporalRandomWindow(
  #         key=[
  #             "video",
  #             "depth_video",
  #             "depth_mask",
  #             "target_points_video",
  #             "target_points_visible_video",
  #             "boxes_video",
  #         ],
  #         length=num_frames,
  #     ),
  #     preprocessing.MaskLateOnsetTracks(
  #         tracks_key="target_points_video",
  #         visible_key="target_points_visible_video",
  #         mask_key="points_mask",
  #     ),
  #     kd.data.Rearrange(
  #         key=["target_points_visible_video"],
  #         pattern="... T K -> ... T K 1",
  #     ),
  #     preprocessing.GetTrackQueries(
  #         tracks_key="target_points_video",
  #         queries_key="query_points_video",
  #         query_timestep=0,
  #     ),
  #     preprocessing.GetTrackQueries(
  #         tracks_key="boxes_video",
  #         queries_key="query_boxes_video",
  #         query_timestep=0,
  #     ),
  #     # Integrate track / dense masks for loss.
  #     preprocessing.CombineTrackMasksForLoss(
  #         dense_mask_key="target_points_visible_video",
  #         track_mask_key="points_mask",
  #         mask_key="dense_points_mask",
  #     ),
  #     preprocessing.AddBoxesMask(key="boxes_video", mask_value=0.0),
  # ]

  final_resolution = 256
  num_frames = 24
  dummy_values = {
      # "target_points_video": tnp.full(
      #     (num_frames, num_point_tracks, 2),
      #     fill_value=-1,
      #     dtype=tnp.float32,
      # ),
      # "dense_points_mask": tnp.full(
      #     (num_frames, num_point_tracks, 1),
      #     fill_value=0,
      #     dtype=tnp.float32,
      # ),
      # "target_points_visible_video": tnp.full(
      #     (num_frames, num_point_tracks, 1),
      #     fill_value=0,
      #     dtype=tnp.float32,
      # ),
      # "points_mask": tnp.full(
      #     (num_point_tracks, 1), fill_value=0, dtype=tnp.float32
      # ),
      "query_points_video": tnp.full(
          (num_point_tracks, 2), fill_value=0, dtype=tnp.float32
      ),
      "query_points_frame_mask": tnp.full(
          (num_frames, num_point_tracks), fill_value=0, dtype=tnp.bool_
      ),
      # "points_evaluation_mask": tnp.full(
      #     (num_frames, num_point_tracks), fill_value=0, dtype=tnp.float32
      # ),
      # "boxes_video": tnp.full(
      #     (num_frames, num_box_tracks, 4), fill_value=0, dtype=tnp.float32
      # ),
      # "query_boxes_video": tnp.full(
      #     (num_box_tracks, 4), fill_value=0, dtype=tnp.float32
      # ),
      # "boxes_video_mask": tnp.full(
      #     (num_frames, num_box_tracks, 1), fill_value=0, dtype=tnp.bool_
      # ),
      "depth_video": tnp.full(
          (num_frames, final_resolution, final_resolution, 1),
          fill_value=0,
          dtype=tnp.float32,
      ),
      "depth_mask": tnp.full(
          (num_frames, final_resolution, final_resolution, 1),
          fill_value=0,
          dtype=tnp.float32,
      ),
  }

  transforms = (
      kd.data.Elements(
          keep=[
              "video",
          ]
      ),
      kd.data.Rearrange(key="video", pattern="T H W C -> T H W C"),
      # kd.data.Stack(key="video", axis=0),
      kd.data.ValueRange(key="video", vrange=(0, 1)),
      # dummy values for readouts
      kd.data.AddConstants(values=dummy_values),
  )

  # shuffle_buffer_size = 1
  batch_size = 1
  return kd.data.py.Tfds(  # pytype: disable=wrong-keyword-args
      name="movi_e/256x256:1.0.0",  # TODO(svansteenkiste): 256x and crop
      split="train",
      shuffle=False,
      # shuffle_buffer_size=shuffle_buffer_size,
      data_dir="/tmp/kubric-public/grain",
      transforms=transforms,
      batch_size=batch_size,
      num_workers=0,
  )


def sweep():
  for seed in [6, 8, 223]:
    yield {"seed": seed}
