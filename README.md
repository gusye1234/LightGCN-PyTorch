## pytorch-light-GCN

*still testing*

refer to [lightGCN](https://arxiv.org/abs/2002.02126)

This is our Pytorch implementation for the paper:

>Xiangnan He, Kuan Deng ,Xiang Wang, Yan Li, Yongdong Zhang, Meng Wang(2020). LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation, [Paper in arXiv](https://arxiv.org/abs/2002.02126).

Author: Prof. Xiangnan He (staff.ustc.edu.cn/~hexn/)

(Also see Tensorflow [implementation](https://github.com/kuandeng/LightGCN))







**usage**:

```shell
usage: main.py [-h] [--bpr_batch BPR_BATCH] [--recdim RECDIM] [--layer LAYER]
               [--dropout DROPOUT] [--keepprob KEEPPROB] [--a_fold A_FOLD]
               [--testbatch TESTBATCH] [--dataset DATASET] [--path PATH]
               [--topks [TOPKS]] [--tensorboard TENSORBOARD]
               [--comment COMMENT] [--load LOAD] [--epochs EPOCHS]

Go lightGCN

optional arguments:
  -h, --help            show this help message and exit
  --bpr_batch BPR_BATCH
                        the batch size for bpr loss training procedure
  --recdim RECDIM       the embedding size of lightGCN
  --layer LAYER         the layer num of lightGCN
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
```



**notes:**

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
