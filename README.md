## pytorch-light-GCN

refer to [lightGCN](https://arxiv.org/abs/2002.02126)

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









notes:

code structure is below.

```shell
code
├── parser.py
├── Procedure.py
├── dataloader.py
├── main.py
├── model.py
├── utils.py
└── world.py
```

if you want to run lightGCN on your own dataset, you should go to `dataloader.py`, and implement a dataloader.

初始化具体的dataset, models, 以及训练参数的地方, 并且管理训练的流程, 良好的设计应该让训练时所有的输出有`main.py`控制, `main.py`应当是尽可能model-irrelevant, dataset-irrelevant的, 

所以一个想法的实现应当对应:

* 在 `dataloader.py` 中 实现具体的dataset 读取, 继承自`BasicDataset`, 满足一定的属性
* 在`model.py`中注册相关的模型
* 在`world.py` and  `parse.py`中注册相关的可能需要的超参数便于统一管理
* 在`utils.py`中实现sampling 流程
* 在`Procedure.py`中实现 batch train的过程.
* 在`main.py`中更换相关的初始化函数, 以及训练函数