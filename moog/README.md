# Moving Off-the-Grid (MooG)

[Moving Off-the-Grid (MooG)](https://openreview.net/pdf?id=rjSPDVdUaw) introduces a self-supervised video representation model that departs from conventional “on-the-grid’’ methods by allowing latent tokens to move freely across space and time, enabling them to stay aligned with scene elements as they shift on the image plane. By combining cross-attention with positional embeddings, MooG disentangles representation structure from image structure, allowing tokens to bind to meaningful scene components rather than fixed pixel locations. Trained with a simple next-frame prediction objective on raw video data, MooG naturally learns tokens that track objects and structures over time, and demonstrates strong performance on a variety of downstream tasks when lightweight readouts are applied. Overall, MooG provides a powerful and flexible off-the-grid representation, outperforming traditional grid-based baselines and establishing a strong foundation for diverse 4D vision applications.

![moog architecture](../assets/moog.png)

## Installation and Training

### 1. Install Environment

```bash
sudo apt-get install python3-venv
git clone https://github.com/google-deepmind/representations4d.git
cd representations4d
python3 -m venv representations4d_env
source representations4d_env/bin/activate
pip install .
```

### 2. Download Dataset

```bash
curl -O https://storage.googleapis.com/pub/gsutil.tar.gz
tar -xzf gsutil.tar.gz
mkdir kubric-public
./gsutil/gsutil -m cp -r gs://kubric-public/grain/ kubric-public
```

### 3. Train

```bash
cd representations4d/moog/configs
python -m kauldron.main --cfg=movi.py --cfg.workdir=moog
```
