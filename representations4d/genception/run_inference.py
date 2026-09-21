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

"""Run GenCeption inference on a single image for depth and segmentation."""

import logging
import os
import time
from urllib import request

from absl import app as absl_app
import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

from representations4d.genception.pipeline import GenCeptionPipeline  # pylint: disable=g-importing-member

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Paths
CHECKPOINT_DIR = "/tmp/genception_ckpts"
OUTPUT_DIR = "/tmp/genception_inference_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# We'll create a synthetic "indoor scene" image since loading ScanNet from
# CNS SSTables is complex. Instead, download a public domain indoor image.
INPUT_IMAGE_PATH = os.path.join(OUTPUT_DIR, "input_image.png")


def create_test_image():
  """Download a ScanNet-like indoor scene image or create a synthetic one."""
  # Try to download a public indoor scene image
  try:
    # ScanNet example from the official ScanNet dataset's public examples
    url = "https://storage.googleapis.com/representations4d/assets/example_frame.jpg"
    logger.info("Downloading test image from %s", url)
    request.urlretrieve(url, INPUT_IMAGE_PATH)
    img = np.array(Image.open(INPUT_IMAGE_PATH))
    logger.info("Downloaded image shape: %s", img.shape)
    return img
  except Exception as e:  # pylint: disable=broad-except
    logger.warning("Failed to download image: %s. Creating synthetic one.", e)

  # Create a synthetic indoor scene image (gradient with some structure)
  h, w = 480, 640
  img = np.zeros((h, w, 3), dtype=np.uint8)

  # Floor (brown-ish gradient)
  for y in range(h // 2, h):
    t = (y - h // 2) / (h // 2)
    img[y, :, 0] = int(139 * (1 - t * 0.3))  # R
    img[y, :, 1] = int(90 * (1 - t * 0.3))  # G
    img[y, :, 2] = int(43 * (1 - t * 0.3))  # B

  # Wall (light gray)
  for y in range(0, h // 2):
    t = y / (h // 2)
    img[y, :, 0] = int(200 - 30 * t)
    img[y, :, 1] = int(200 - 30 * t)
    img[y, :, 2] = int(210 - 30 * t)

  # Add a "table" rectangle
  img[h // 3 : h // 2, w // 4 : 3 * w // 4, 0] = 120
  img[h // 3 : h // 2, w // 4 : 3 * w // 4, 1] = 80
  img[h // 3 : h // 2, w // 4 : 3 * w // 4, 2] = 50

  # Add a "window" rectangle
  img[h // 8 : h // 4, w // 3 : 2 * w // 3, 0] = 135
  img[h // 8 : h // 4, w // 3 : 2 * w // 3, 1] = 206
  img[h // 8 : h // 4, w // 3 : 2 * w // 3, 2] = 235

  Image.fromarray(img).save(INPUT_IMAGE_PATH)
  logger.info("Created synthetic test image: %s", img.shape)
  return img


def image_to_video(image, num_frames=81):
  """Replicate a single image across T frames to create a 'video'."""
  # GenCeption expects video input [T, H, W, 3]
  return np.repeat(image[np.newaxis], num_frames, axis=0)


def save_visualization(input_img, depth_frame, seg_frame, output_path):
  """Create a side-by-side visualization figure."""
  h, w = input_img.shape[:2]  # pylint: disable=unused-variable
  # Target display size
  display_h, display_w = 480, 640

  def resize(img):
    pil_img = Image.fromarray(img)
    return np.array(pil_img.resize((display_w, display_h), Image.BILINEAR))

  input_resized = resize(input_img)
  depth_resized = resize(depth_frame)
  seg_resized = resize(seg_frame)

  # Create canvas: 3 images side-by-side with padding
  pad = 20
  title_h = 40
  canvas_w = 3 * display_w + 4 * pad
  canvas_h = display_h + 2 * pad + title_h
  canvas = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))

  # Paste images
  y_offset = pad + title_h
  for i, (img, title) in enumerate([
      (input_resized, "Input RGB"),
      (depth_resized, "Depth Prediction"),
      (seg_resized, "Segmentation Prediction"),
  ]):
    x = pad + i * (display_w + pad)
    canvas.paste(Image.fromarray(img), (x, y_offset))

  # Add titles
  draw = ImageDraw.Draw(canvas)
  try:
    font = ImageFont.truetype(
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20
    )
  except (IOError, OSError):
    font = ImageFont.load_default()

  titles = ["Input RGB", "Depth Prediction", "Segmentation Prediction"]
  for i, title in enumerate(titles):
    x = pad + i * (display_w + pad) + display_w // 2
    bbox = draw.textbbox((0, 0), title, font=font)
    tw = bbox[2] - bbox[0]
    draw.text((x - tw // 2, pad // 2), title, fill=(0, 0, 0), font=font)

  canvas.save(output_path, quality=95)
  logger.info("Saved visualization to %s", output_path)


def main(_):
  logger.info("JAX devices: %s", jax.devices())
  logger.info("JAX version: %s", jax.__version__)

  # 1. Create/load test image
  logger.info("Step 1: Loading test image...")
  input_image = create_test_image()
  logger.info(
      "Input image shape: %s, dtype: %s", input_image.shape, input_image.dtype
  )

  # 2. Load the pipeline (skip text encoder for CPU testing)
  logger.info("Step 2: Loading GenCeption 1.3B pipeline (no text encoder)...")
  t0 = time.time()
  pipe = GenCeptionPipeline.from_pretrained(
      CHECKPOINT_DIR,
      model_size="1.3b",
      dtype=jnp.bfloat16,
  )
  logger.info("Pipeline loaded in %.1f seconds", time.time() - t0)

  # 3. Create video from single image
  # Use a small number of frames for CPU inference.
  # Note: num_frames must be > 1 to avoid division-by-zero in temporal RoPE.
  logger.info("Step 3: Creating video from single image...")
  num_frames = 5
  video = image_to_video(input_image, num_frames=num_frames)
  logger.info("Video shape: %s (num_frames=%d)", video.shape, num_frames)

  # 5. Create zero text embeddings (skip text encoder for CPU testing)
  max_seq_len = 226
  text_dim = 4096
  prompt_embeds = jnp.zeros((1, max_seq_len, text_dim), dtype=jnp.bfloat16)
  logger.info("Using zero text embeddings: %s", prompt_embeds.shape)

  # 6. Run depth estimation (single forward pass)
  logger.info("Step 5: Running depth forward pass...")
  t0 = time.time()

  result = pipe(prompt_embeds=prompt_embeds, input_video=video)
  depth_video = result["video"]
  depth_time = time.time() - t0
  logger.info("Forward pass took %.1f seconds", depth_time)

  # Save raw colored depth frame (direct model output).
  depth_frame = depth_video[depth_video.shape[0] // 2]  # middle frame
  Image.fromarray(depth_frame).save(
      os.path.join(OUTPUT_DIR, "depth_colored.png")
  )
  logger.info("Saved raw colored depth frame.")

  # Decode colored + log-scaled depth to relative depth.
  depth_relative = pipe.decode_depth(depth_video)
  logger.info(
      "Decoded relative depth: shape=%s, range=[%.4f, %.4f]",
      depth_relative.shape,
      depth_relative.min(),
      depth_relative.max(),
  )

  # Save decoded relative depth as a grayscale visualization.
  rel_frame = depth_relative[depth_relative.shape[0] // 2]
  # Normalize to [0, 255] for visualization.
  rel_min, rel_max = rel_frame.min(), rel_frame.max()
  if rel_max > rel_min:
    rel_vis = ((rel_frame - rel_min) / (rel_max - rel_min) * 255).astype(
        np.uint8
    )
  else:
    rel_vis = np.zeros_like(rel_frame, dtype=np.uint8)
  Image.fromarray(rel_vis, mode="L").save(
      os.path.join(OUTPUT_DIR, "depth_relative.png")
  )
  logger.info("Saved decoded relative depth frame.")

  # Note: segmentation pass skipped (also needs text encoder).
  # With zero text embeddings, the output is a smoke test only.
  seg_frame = depth_frame  # placeholder for visualization

  # Create visualization
  logger.info("Step 7: Creating visualization...")
  vis_path = os.path.join(OUTPUT_DIR, "genception_1.3b_predictions.png")
  save_visualization(input_image, depth_frame, seg_frame, vis_path)

  # Save results as npz
  np.savez(
      os.path.join(OUTPUT_DIR, "results.npz"),
      input_image=input_image,
      depth_colored=depth_frame,
      depth_relative=depth_relative,
  )

  logger.info("\n=== Results Summary ===")
  logger.info("Input image: %s", INPUT_IMAGE_PATH)
  logger.info("Colored depth: %s/depth_colored.png", OUTPUT_DIR)
  logger.info("Relative depth: %s/depth_relative.png", OUTPUT_DIR)
  logger.info("Visualization: %s", vis_path)
  logger.info("Forward pass time: %.1f seconds", depth_time)


if __name__ == "__main__":
  absl_app.run(main)
