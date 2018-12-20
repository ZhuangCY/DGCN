# DGCN

## Introduction

This is a Theano implementation of DGCN, a Dual Graph Convolutional Networks method for graph-based semi-supervised classification proposed in the following paper:

[Dual Graph Convolutional Networks for Graph-Based Semi-Supervised Classification](https://www.researchgate.net/publication/324514333_Dual_Graph_Convolutional_Networks_for_Graph-Based_Semi-Supervised_Classification).
Chenyi Zhuang, Qiang Ma.
WWW 2018.

Please cite our paper if you use this code in your own work.

## Requirements

* python 2.7
* theano
* networkx
* scipy
* Lasagne

## Run the demo

```bash
python Test.py [dataset]
```

[dataset] could be strings: "citeseer", "cora", and "pubmed".

## Models

The DGCN model is mainly implemented in `Model.py`. In `layers.py`, the dense layer, diffusion layer and dropout function are defined. In `LossCalculation.py`, the loss calculation functions and evaluation metric function (i.e., accuracy) are defined. In `utilities.py`, the random walk functions and temporal weight function are defined. 

## Prepare the raw data

In order to run the code on your own dataset, you need to prepare:
* an n by n adjacency matrix (n is the number of nodes), 
* an n by k feature matrix (k is the number of features per node), and
* an n by c binary label matrix (c is the number of classes).

Please refer to our paper and the files `DataPreparation.py` and `utilities.py` for detailed data pre-processing information.

For testing, the "citeseer", "cora", and "pubmed" datasets are available in the directory `data`. Due to the file size limitation, for the "nell_full" dataset, you could find at [http://www.cs.cmu.edu/~zhiliny/data/nell_data.tar.gz](http://www.cs.cmu.edu/~zhiliny/data/nell_data.tar.gz) or [our pre-processed version](https://www.dropbox.com/s/bxrvf1syyuzmcqq/DGCN.zip?dl=0).

Since we used the exactly same datasets for testing, for detailed information about these four datasets, you may refer to the [Planetoid repository](https://github.com/kimiyoung/planetoid)

## Related work

Our work is inspired by the following papers:

* Thomas N. Kipf, Max Welling, [Semi-Supervised Classification with Graph Convolutional Networks.](http://arxiv.org/abs/1609.02907)
* David I Shuman, Sunil K. Narang, Pascal Frossard, Antonio Ortega, Pierre Vandergheynst, [The Emerging Field of Signal Processing on Graphs: Extending High-Dimensional Data Analysis to Networks and Other Irregular Domains.](https://arxiv.org/abs/1211.0053)

The testing datasets were provided by:
* Zhilin Yang, William W. Cohen, Ruslan Salakhutdinov, [Revisiting Semi-Supervised Learning with Graph Embeddings.](https://arxiv.org/abs/1603.08861)
