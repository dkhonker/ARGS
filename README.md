# Sparse but Strong: Crafting Adversarially Robust Graph Lottery Tickets

《Sparse but Strong: Crafting Adversarially Robust Graph Lottery Tickets》的非官方复现。
This is the unofficial repository for Sparse but Strong: Crafting Adversarially Robust Graph Lottery Tickets which has been accepted at the NeurIPS 2023 GLFrontiers

## Table of Contents
1. [安装/Installation](#installation)
2. [使用/Usage](#usage)
3. [Baseline Results](#baseline-results)

<a name="installation"></a>
## Installation

（本人使用的是python 3.10)
This project requires Python 3.8 and the following Python libraries installed:

- torch==1.13.1
- dgl==1.0.4 -cu116
- scikit-learn
- networkx
- scipy
- matplotlib
- pickle-mixin
- argparse
- numpy
- warnings

要安装这些依赖项，cd到项目目录并运行以下命令:
To install these dependencies, navigate to your project's directory and run the following command:

```bash
pip install -r requirements.txt
```
<a name="usage"></a>
## Usage

You can test the performance of the trained adversarially robust GLTs by executing these commands on the terminal:

### Evaluating the performance of ARGLTs for Cora and Citeseer Datasets attacked by PGD attack
```
python main_test.py --dataset cora --embedding-dim 1433 512 7 --attack_name pgd --ptb_rate 0.05
python main_test.py --dataset cora --embedding-dim 1433 512 7 --attack_name pgd --ptb_rate 0.1
python main_test.py --dataset cora --embedding-dim 1433 512 7 --attack_name pgd --ptb_rate 0.15
python main_test.py --dataset cora --embedding-dim 1433 512 7 --attack_name pgd --ptb_rate 0.2

python main_test.py --dataset citeseer --embedding-dim 3703 512 6 --attack_name pgd --ptb_rate 0.05
python main_test.py --dataset citeseer --embedding-dim 3703 512 6 --attack_name pgd --ptb_rate 0.1
python main_test.py --dataset citeseer --embedding-dim 3703 512 6 --attack_name pgd --ptb_rate 0.15
python main_test.py --dataset citeseer --embedding-dim 3703 512 6 --attack_name pgd --ptb_rate 0.2
```
<a name="baseline-results"></a>
## Baseline Results

Here are the baseline accuracies for GCN on the Cora and Citeseer datasets under PGD attack for comparison:

### GCN with PGD Attack on Cora Dataset

| Perturbation Rate | Accuracy |
|-------------------|----------|
| 0.05              | 78.81    |
| 0.1               | 78.46    |
| 0.15              | 77.93    |
| 0.2               | 77.99    |

### GCN with PGD Attack on Citeseer Dataset

| Perturbation Rate | Accuracy |
|-------------------|----------|
| 0.05              | 73.28    |
| 0.1               | 73.96    |
| 0.15              | 74.62    |
| 0.2               | 73.52    |

You can also train the models by executing the following commands on the terminal:

### Training ARGLT for Cora Dataset
```
 python main_train.py --dataset cora --embedding-dim 1433 512 7 --lr 0.008 --weight-decay 8e-5 --pruning_percent_wei 0.2 --pruning_percent_adj 0.05 --total_epoch 10 --s1 1e-2 --s2 1e-2 --init_soft_mask_type all_one --attack_name pgd --ptb_rate 0.05 --k 30 --alpha 1 --beta 0.01 --gamma 1 --alpha_fix_mask 1 --gamma_fix_mask 1


```

作者只提供了PGD的数据集。