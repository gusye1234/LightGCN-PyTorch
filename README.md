## pytorch-light-GCN

refer to [lightGCN](https://arxiv.org/abs/2002.02126)

```shell
code
├── Procedure.py
├── dataloader.py
├── main.py
├── model.py
├── utils.py
└── world.py
```



初始化具体的dataset, models, 以及训练参数的地方, 并且管理训练的流程, 良好的设计应该让训练时所有的输出有`main.py`控制, `main.py`应当是尽可能model-irrelevant, dataset-irrelevant的, 

所以一个想法的实现应当对应:

* 在 `dataloader.py` 中 实现具体的dataset 读取, 继承自`BasicDataset`, 满足一定的属性
* 在`model.py`中注册相关的模型
* 在`world.py`中注册相关的可能需要的超参数便于统一管理
* 在`utils.py`中实现sampling 流程
* 在`Procedure.py`中实现 batch train的过程.
* 在`main.py`中更换相关的初始化函数, 以及训练函数