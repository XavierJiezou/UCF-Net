<div align="center">

<img src="docs/banner.jpg" alt="UCF-Net Logo Emblem" width="150" />

# UCF-Net

<b>Harnessing CLIP and DINO: </b>
<br>
An Uncertainty-Aware Cascaded Fusion Network for Generalizable Deepfake Image Detection

[![Project Page](https://img.shields.io/badge/Project%20Page-UCF--Net-blue)](https://xavierjiezou.github.io/UCF-Net/)
[![arXiv](https://img.shields.io/badge/arXiv-2609.07670-b31b1b.svg)](https://arxiv.org/abs/2609.07670)
<br>
[![HuggingFace Paper](https://img.shields.io/badge/%F0%9F%A4%97HuggingFace-Daily%20Paper-blue)](https://huggingface.co/papers/2609.07670)
[![HuggingFace Space](https://img.shields.io/badge/%F0%9F%A4%97HuggingFace-Space-yellow)](https://huggingface.co/spaces/csYiZhou/UCF-Net)
[![HuggingFace Models](https://img.shields.io/badge/%F0%9F%A4%97HuggingFace-Models-green)](https://huggingface.co/XavierJiezou/ucfnet-models)
[![HuggingFace Datasets](https://img.shields.io/badge/%F0%9F%A4%97HuggingFace-Datasets-orange)](https://huggingface.co/datasets/XavierJiezou/ucfnet-datasets)
<!--[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)-->

![framework](docs/framework.png)

</div>

## 1. Install the environment

Create a Python 3.7 environment and install the pinned dependencies (torch 1.12.0+cu113):

```bash
conda create -n ucfnet python=3.7 -y
conda activate ucfnet
pip install -U pip
pip install -r requirements.txt
```

## 2. Prepare the dataset

![dataset overview](docs/dataset-overview.png)

Download the complete dataset from [Hugging Face](https://huggingface.co/datasets/XavierJiezou/ucfnet-datasets)
into the repository root:

```bash
python - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="XavierJiezou/ucfnet-datasets",
    repo_type="dataset",
    local_dir=".",
    resume_download=True,
)
PY
```

Merge the archive parts and extract all dataset archives into
`UCF-Net-dataset/`:

```bash
# Merge the split archives.
for dataset in DF40 "FaceForensics++" MFFI; do
  cat "UCF-Net-dataset/${dataset}.zip.part-"* \
    > "UCF-Net-dataset/${dataset}.zip"
done

# Extract all archives.
for archive in UCF-Net-dataset/*.zip; do
  unzip -q "$archive" -d UCF-Net-dataset
done
```

Place the prepared dataset at `UCF-Net-dataset/` with the following layout:

```text
UCF-Net/                          # repository root
└── UCF-Net-dataset/
    ├── splits/
    ├── AIGC-datasets/
    ├── CelebA/
    ├── CelebA-HQ/
    ├── Celeb-DF-v1/
    ├── Celeb-DF-v2/
    ├── DeepFakeFace/
    ├── DF40/
    ├── DFDC/
    ├── DFDCP/
    ├── DFFD/
    ├── FaceForensics++/
    ├── FFHQ/
    ├── MFFI/
    ├── TIMIT/
    └── UADFV/
```

## 3. Download the weights (CLIP and DINO)

Download the following pretrained weights from their official sources:

- [CLIP ViT-L/14](https://huggingface.co/openai/clip-vit-large-patch14)
- [DINOv2 ViT-L/14](https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_reg4_pretrain.pth)

Alternatively, download the released CLIP and DINO files from [UCF-Net Models](https://huggingface.co/XavierJiezou/ucfnet-models):

```bash
python - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="XavierJiezou/ucfnet-models",
    local_dir=".",
    allow_patterns=[
        "pretrained/clip-vit-large-patch14-local/*",
        "pretrained/dinov2_vitl14_reg4_pretrain.pth",
    ],
    resume_download=True,
)
PY
```

After downloading, the files should be arranged as follows:

```text
UCF-Net/                          # repository root
└── pretrained/
    ├── clip-vit-large-patch14-local/
    └── dinov2_vitl14_reg4_pretrain.pth
```

## 4. Training

Run training with the default configuration:

```bash
python train.py \
  --config PATH/TO/CONFIG \       # e.g., DeepfakeBench/training/config/detector/ucfnet.yaml
  --batch-size BATCH_SIZE \       # e.g., 20
  --cuda CUDA_DEVICE_IDS          # e.g., 0,1,2,3
```

## 5. Evaluation

You can evaluate a locally trained checkpoint, or directly use the released UCF-Net checkpoint. Download the `ucfnet/` directory from [UCF-Net Models](https://huggingface.co/XavierJiezou/ucfnet-models)

```bash
python - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="XavierJiezou/ucfnet-models",
    local_dir="pretrained",
    allow_patterns="ucfnet/*",
    resume_download=True,
)
PY
```

The released checkpoint is then available at `pretrained/ucfnet/model_best.pth`. Evaluate it on all test splits:

```bash
python test.py \
  --config PATH/TO/CONFIG \          # e.g., DeepfakeBench/training/config/detector/ucfnet.yaml
  --bestpth PATH/TO/BEST_PTH \       # e.g., pretrained/ucfnet/model_best.pth
  --test-type TEST_TYPE \            # e.g., all (or indomain, crossdomain)
  --batch-size BATCH_SIZE \          # e.g., 96
  --cuda CUDA_DEVICE_IDS             # e.g., 0,1,2,3
```

## 6. Inference

- Web demo (Gradio / Hugging Face Space)

Run the demo locally with the files in [`hf_space/`](hf_space) or deploy them directly to a Hugging Face Space:

```bash
cd hf_space
pip install -r requirements.txt
python app.py
```
<!--[Open the UCF-Net Online Demo](https://huggingface.co/spaces/XavierJiezou/UCF-Net)-->

![UCF-Net Hugging Face Space inference demo](docs/HF-space.png)

- Command-line inference

Run inference on a single image or on every supported image in a directory with a
trained UCF-Net checkpoint. The command reports the predicted label together with
the probabilities of the real and fake classes.

```bash
python inference.py \
  --image PATH/TO/IMAGE_OR_DIRECTORY \ # e.g., samples/example.jpg (or samples/)
  --config PATH/TO/CONFIG \             # e.g., DeepfakeBench/training/config/detector/ucfnet.yaml
  --checkpoint PATH/TO/CHECKPOINT \     # e.g., pretrained/ucfnet/model_best.pth
  --cuda CUDA_DEVICE_ID                 # e.g., 0 (or cpu)
```

Use `--cuda cpu` for CPU inference. Directory inputs are searched recursively for
`.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`, `.tif`, and `.tiff` files. Add `--json` to
print one JSON object for a single image or a JSON array for a directory.

## 7. Visualizations

### Grad-CAM

The Grad-CAM examples compare fake-class saliency maps for the CLIP-based branch,
the DINO-based branch, several fusion strategies, and UCF-Net. The value shown below
each heatmap is $p_{\mathrm{fake}}$, the predicted probability of the fake class.
Green and red borders denote correct and incorrect predictions, respectively.

![Grad-CAM visualization](docs/gradcam.png)

### t-SNE

The t-SNE figure visualizes detection features from UCF-Net and seven representative
detectors on UADFV, DFF, DFDC, and DF40-Test. Rows correspond to models, while columns
show the summary view and the four held-out domains. Green and red markers denote real
and fake samples, respectively. The reported percentages measure linear separability
in the two-dimensional t-SNE embedding.

![t-SNE visualization](docs/tsne.png)

## Acknowledgement

We would like to thank [Tongzhou Ji](https://github.com/aiyeno1) for the contribution to the [Project Page](https://xavierjiezou.github.io/UCF-Net/).

## License

This project is licensed under the Creative Commons Attribution-NonCommercial 4.0
International Public License (CC BY-NC 4.0). See [LICENSE](LICENSE) for the SPDX
identifier and the official license text link.
