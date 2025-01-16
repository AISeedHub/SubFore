# SubFore
Offical implementation of paper [Subvolume-based Foreground Masking for Medical Masked Image Modeling]()

![alt text](assets/framework.pdf "Framework")

## Contributions
We focus on improving the pre-text task by exploring two-step
masking strategy: 

- (1) Subvolume partition, which leverages 3D subvolumes to incorporate richer spatial context than standard full volume patchified methods.
- (2) Foreground Masking, which considers the semantic distribution of anatomy, aiming to exclude background air regions that lack meaningful features.
- Dice score (BTCV: `84.64%`, Flare22: `92.43%`, MM-WHS: `90.67%`, Amos22: `88.64%`, BraTS: `78.55%`)

## Main Results

![alt text](assets/segmentation.png "Segmentation")

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