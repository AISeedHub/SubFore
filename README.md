# SubFore: Subvolume Foreground Masking for Medical Image Modeling
Offical implementation of paper [Subvolume-based Foreground Masking for Medical Masked Image Modeling]()

![alt text](assets/framework.jpg "Framework")

## Contributions
We focus on improving the pre-text task by exploring two-step
masking strategy: 

- (1) Subvolume partition, which leverages 3D subvolumes to incorporate richer spatial context than standard full volume patchified methods.
- (2) Foreground Masking, which considers the semantic distribution of anatomy, aiming to exclude background air regions that lack meaningful features.
- Dice score (BTCV: `84.64%`, Flare22: `92.43%`, MM-WHS: `90.67%`, Amos22: `88.64%`, BraTS: `78.55%`)

## Main Results
![alt text](assets/segmentation.png "Segmentation")


| Methods         | Spleen  | Rkid   | Lkid   | Gall   | Eso    | Liv    | Sto    | Aor    | IVC    | Veins  | Pan    | Rag    | Lag    | **AVG** |
|-----------------|---------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|
| UNETR backbone | 
| UNETR          | 92.30%  | 93.76% | 92.87% | 61.24% | 70.95% | 96.29% | 81.28% | 88.41% | 81.82% | 70.40% | 75.86% | 64.85% | 56.99% | **79.00%** |
| MAE            | 93.90%  | 93.79% | 93.66% | 62.10% | 72.33% | 96.55% | 80.81% | 89.44% | 84.10% | 71.44% | 76.70% | 66.55% | 62.00% | **80.26%** |
| SimMIM         | 94.43%  | 94.28% | 94.14% | 59.19% | 71.54% | 96.83% | 81.94% | 90.57% | 84.78% | 72.57% | 81.12% | 66.41% | 64.59% | **80.95%** |
| Ours           | 95.66%  | 94.66% | 94.48% | 64.48% | 71.99% | 96.99% | 82.85% | 90.42% | 84.67% | 73.97% | 81.62% | 67.35% | 63.87% | **81.77%** |
| SwinUNETR backbone |
| Swin UNETR     | 95.67%  | 93.91% | 93.81% | 65.92% | 75.18% | 96.65% | 81.35% | 90.04% | 85.89% | 74.15% | 78.92% | 70.14% | 65.17% | **82.06%** |
| SimMIM         | 95.91%  | 94.70% | 94.58% | 66.46% | 75.85% | 97.17% | 89.41% | 90.56% | 86.96% | 77.34% | 81.89% | 71.01% | 69.92% | **83.98%** | 
| Tang et al.    | 96.48%  | 95.01% | 94.43% | 63.31% | 77.42% | 97.04% | 86.32% | 90.62% | 85.91% | 75.37% | 83.54% | 71.38% | 68.34% | **83.47%** |
| SwinMM         | 96.48%  | 94.93% | 94.69% | 61.89% | 76.29% | 97.07% | 85.10% | 90.24% | 86.27% | 76.19% | 84.18% | 72.86% | 70.38% | **83.58%** |
| DAE            | 96.41%  | 94.93% | 94.75% | 65.79% | 75.68% | 97.10% | 85.57% | 91.02% | 86.19% | 75.72% | 85.17% | 71.07% | 68.02% | **83.65%** |
| SDSL           | 96.47%  | 94.64% | 94.56% | 66.23% | 78.02% | 96.93% | 89.10% | 90.04% | 87.36% | 76.34% | 82.69% | 71.14% | 70.67% | **84.17%** |
| VoCo           | 96.39%  | 94.78% | 94.59% | 67.91% | 76.08% | 97.08% | 89.94% | 90.53% | 87.01% | 75.12% | 85.60% | 72.54% | 72.17% | **84.60%** |
| Ours           | 96.45%  | 94.83% | 94.63% | 66.52% | 78.57% | 97.22% | 87.57% | 91.02% | 87.42% | 77.77% | 84.95% | 71.39% | 71.96% | **84.64%** |


## Settings

### Dataset

| **Dataset**          | **Modality**  | **Class**                  | **Train** | **Valid.** |
|-----------------------|--------------|----------------------------|-----------|------------|
| **Pretraining**       |              |                            |           |            |
| BTCV             | CT           | 13 organs                 | 24        | 6          |
| TCIA Covid19         | CT           | Binary                    | 722       | 49         |
| LUNA                 | CT           | -                         | 843       | 45         |
| **Downstream**        |              |                            |           |            |
| BTCV            | CT           | 13 organs                 | 24        | 6          |
| Flare22              | CT           | 13 organs                 | 100       | 31         |
| Amos22               | CT & MRI     | 15 abdominal organs       | 32        | 9          |
| MM-WHS               | CT           | 14                        | 6         | -          |
| MSD BraTs            | MRI          | 3 tumors                  | 387       | 97         |


### Configurations

- Pre-processing

| **Parameter**          | **Values**          |
|-------------------------|---------------------|
| Spacing                | [1.5, 1.5, 1.5]    |
| Norm [amin, amax]      | [-175.0, 250.0]    |
| Norm [bmin, bmax]      | [0.1, 1.0]         |
| ROI Size               | 96×96×96           |
| Subvolume size         | 16×16×16           |

- Pre-training
  
| **Parameter**          | **Values**          |
|-------------------------|---------------------|
| Pre-training Steps     | 1600               |
| Optimizer              | AdamW              |
| Optimization LR        | 1e-4               |
| LR Schedule            | Warmup cosine      |
| Warmup Steps           | 100                |
| Momentum               | 0.9                |
| Regularization Weight  | 1e-2               |
| Batch Size             | 4                  |

- Downstream
  
| **Parameter**          | **Values**          |
|-------------------------|---------------------|
| Optimizer              | AdamW              |
| Optimization LR        | 3e-4               |
| LR Schedule            | Warmup cosine      |
| Warmup Steps           | 100                |
| Momentum               | 0.9                |
| Regularization Weight  | 1e-5               |
| Batch Size             | 2                  |
| Inference              | Sliding window     |
| ROI Size               | 96×96×96           |


## Reproducibility

- Installation

```bash
git clone 
cd SubFore
pip install -r requirements.txt
```

- Run pretraining

```bash
python train.py --config configs/pretrain.yaml
```

- Run downstream task

```bash
python train.py --config configs/downstream.yaml
```

# Citation
```bibtex
@article{subfore,
  title={Subvolume-based Foreground Masking for Medical Masked Image Modeling},
  author={},
  journal={},
  year={2025}
}
```