# 4D Representations

Welcome to the official Google DeepMind repository for 4D Representations.

* [Scaling 4D Representations](https://arxiv.org/abs/2412.15212) focuses on evaluating self-supervised learning on non-semantic vision tasks that are more spatial (3D) and temporal (+1D = 4D), such as camera pose estimation, point and object tracking, and depth estimation. We show that by learning from very large video datasets, masked auto-encoding (MAE) with transformer video models actually scales, consistently improving performance on these 4D tasks, as model size increases from 20M all the way to the largest by far reported self-supervised video model 22B parameters.

![scaling results](./assets/scaling_20M_20B.png)

## Installation

```bash
git clone https://github.com/google-deepmind/representations4d.git
cd representations4d

python3 -m venv representations4d_env
source representations4d_env/bin/activate
pip install .
```

## Demo

* [![Open In
Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google-deepmind/representations4d/blob/main/colabs/scaling4d_depth_demo.ipynb) Depth estimation with 4DS-dist-B backbone

## Checkpoints

We release the following checkpoints

| Name | Model | # Params | File Size | Checkpoint |
| -------- | ------- | :-------: | :-------: | :-------: |
| 4DS-dist-B ScanNet depth | Backbone (Vit-B) + Readout | 105M | 420MB | [link](https://storage.googleapis.com/representations4d/checkpoints/scaling4d_dist_b_depth.npz) |

## Citing this work

```
@article{carreira2024scaling,
  title={Scaling 4D Representations},
  author={João Carreira and Dilara Gokay and Michael King and Chuhan Zhang and Ignacio Rocco and Aravindh Mahendran and Thomas Albert Keck and Joseph Heyward and Skanda Koppula and Etienne Pot and Goker Erdogan and Yana Hasson and Yi Yang and Klaus Greff and Guillaume Le Moing and Sjoerd van Steenkiste and Daniel Zoran and Drew A. Hudson and Pedro Vélez and Luisa Polanía and Luke Friedman and Chris Duvarney and Ross Goroshin and Kelsey Allen and Jacob Walker and Rishabh Kabra and Eric Aboussouan and Jennifer Sun and Thomas Kipf and Carl Doersch and Viorica Pătrăucean and Dima Damen and Pauline Luc and Mehdi S. M. Sajjadi and Andrew Zisserman},
  journal={arXiv preprint arXiv:2412.15212},
  year={2024}
}
```

## License and disclaimer

Copyright 2025 Google LLC

All software is licensed under the Apache License, Version 2.0 (Apache 2.0);
you may not use this file except in compliance with the Apache 2.0 license.
You may obtain a copy of the Apache 2.0 license at:
https://www.apache.org/licenses/LICENSE-2.0

All other materials are licensed under the Creative Commons Attribution 4.0
International License (CC-BY). You may obtain a copy of the CC-BY license at:
https://creativecommons.org/licenses/by/4.0/legalcode

Unless required by applicable law or agreed to in writing, all software and
materials distributed here under the Apache 2.0 or CC-BY licenses are
distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
either express or implied. See the licenses for the specific language governing
permissions and limitations under those licenses.

This is not an official Google product.
