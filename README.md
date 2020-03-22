## pytorch-light-GCN

*still testing*

This is our Pytorch implementation for the paper:

>Xiangnan He, Kuan Deng ,Xiang Wang, Yan Li, Yongdong Zhang, Meng Wang(2020). LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation, [Paper in arXiv](https://arxiv.org/abs/2002.02126).

Author: Prof. Xiangnan He (staff.ustc.edu.cn/~hexn/)

(Also see Tensorflow [implementation](https://github.com/kuandeng/LightGCN))

## Introduction

In this work, we aim to simplify the design of GCN to make it more concise and appropriate for recommendation. We propose a new model named LightGCN,including only the most essential component in GCN—neighborhood aggregation—for collaborative filtering



## Enviroment Requirement

`pip install requirements.txt`



## Dataset

We provide three processed datasets: Gowalla, Yelp2018 and Amazon-book and one small dataset LastFM.

see more in `dataloader.py`

## usage

```shell
Go lightGCN

optional arguments:
  -h, --help            show this help message and exit
  --bpr_batch BPR_BATCH
                        the batch size for bpr loss training procedure
  --recdim RECDIM       the embedding size of lightGCN
  --layer LAYER         the layer num of lightGCN
  --lr LR               the learning rate
  --decay DECAY         the weight decay for l2 normalizaton
  --dropout DROPOUT     using the dropout or not
  --keepprob KEEPPROB   the batch size for bpr loss training procedure
  --a_fold A_FOLD       the fold num used to split large adj matrix, like
                        gowalla
  --testbatch TESTBATCH
                        the batch size of users for testing
  --dataset DATASET     available datasets: [lastfm, gowalla]
  --path PATH           path to save weights
  --topks [TOPKS]       @k test list
  --tensorboard TENSORBOARD
                        enable tensorboard
  --comment COMMENT
  --load LOAD
  --epochs EPOCHS
  --multicore MULTICORE
                        whether we use multiprocessing or not in test
  --pretrain PRETRAIN   whether we use pretrained weight or not
```

## notes:

code structure is below.

```shell
code
├── parse.py
├── Procedure.py
├── dataloader.py
├── main.py
├── model.py
├── utils.py
└── world.py
```

if you want to run lightGCN on your own dataset, you should go to `dataloader.py`, and implement a dataloader.

## Results

gowalla:

|             | Recall in paper | Recall in `torch` |
| ----------- | --------------- | ----------------- |
| **layer=1** | 0.1726          | 0.1692            |
| **layer=2** | 0.1786          | 0.1783            |
| **layer=3** | 0.1809          | 0.1807            |

