# 4D Representations

Welcome to the official Google DeepMind repository for 4D Representations.

* [Scaling 4D Representations](https://arxiv.org/abs/2412.15212) focuses on evaluating self-supervised learning on non-semantic vision tasks that are more spatial (3D) and temporal (+1D = 4D), such as camera pose estimation, point and object tracking, and depth estimation. We show that by learning from very large video datasets, masked auto-encoding (MAE) with transformer video models actually scales, consistently improving performance on these 4D tasks, as model size increases from 20M all the way to the largest by far reported self-supervised video model 22B parameters.

![scaling results](./assets/scaling_20M_20B.png)

## Installation

```bash
git clone https://github.com/google-deepmind/representations4d.git
cd representations4d

pip install .
```

## Citing this work

```
@article{carreira2024scaling,
  title={Scaling 4D Representations},
  author={Carreira, Jo{\~a}o and Gokay, Dilara and King, Michael and Zhang, Chuhan and Rocco, Ignacio and Mahendran, Aravindh and Keck, Thomas Albert and Heyward, Joseph and Koppula, Skanda and Pot, Etienne and others},
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
