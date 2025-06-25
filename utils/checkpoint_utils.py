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

"""Utilities for loading checkpoints."""

from typing import Any

import numpy as np


def npload(fname: str) -> dict[str, np.ndarray]:
  loaded = np.load(fname, allow_pickle=False)
  return dict(loaded)


def recover_tree(flat_dict: dict[str, np.ndarray]) -> dict[str, Any]:
  """Recovers a tree structure from a flattened dictionary.

  Args:
    flat_dict: A dictionary with keys representing paths in the tree,
      separated by "/".

  Returns:
    A nested dictionary representing the recovered tree structure.
  """
  tree = (
      {}
  )  # Initialize an empty dictionary to store the resulting tree structure
  for (
      k,
      v,
  ) in (
      flat_dict.items()
  ):  # Iterate over each key-value pair in the flat dictionary
    parts = k.split(
        "/"
    )  # Split the key into parts using "/" to build the tree structure
    node = tree  # Start at the root of the tree
    for part in parts[
        :-1
    ]:  # Loop through each part of the key, except the last one
      if (
          part not in node
      ):  # If the current part doesn't exist as a key in the node,
        # create an empty dictionary for it
        node[part] = {}
      node = node[part]  # Move down the tree to the next level
    node[parts[-1]] = v  # Set the value at the final part of the key
  return tree  # Return the reconstructed tree
