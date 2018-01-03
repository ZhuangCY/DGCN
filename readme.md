# DGCN

## Introduction

This is a Theano implementation of DGCN, a Dual Graph Convolutional Networks method for graph-based semi-supervised classification proposed in the following paper:

[Dual Graph Convolutional Networks for Graph-Based Semi-Supervised Classification]().
Chenyi Zhuang, Qiang Ma.
WWW 2018.

Please cite our paper if you use this code in your own work.

## Run the demo

```bash
python Test.py [dataset]
```

[dataset] could be strings: "citeseer", "cora", "pubmed" and "nell_full".

## Models

The DGCN model is mainly implemented in `Model.py`. In `layers.py`, the dense layer, diffusion layer and dropout function are defined. In `LossCalculation.py`, the loss calculation functions and evaluation metric function (i.e., accuracy) are defined. In `utilities.py`, the random walk functions and temporal weight function are defined. 

## Prepare the raw data

In order to run the code on your own dataset, you need to provide:
* an n by n adjacency matrix (n is the number of nodes), 
* an n by k feature matrix (k is the number of features per node), and
* an n by c binary label matrix (c is the number of classes).

For testing, the "citeseer", "cora", and "pubmed" datasets are available in the directory `data`. Due to the file size limitation, for the "nell_full" dataset, you could find at [http://www.cs.cmu.edu/~zhiliny/data/nell_data.tar.gz](http://www.cs.cmu.edu/~zhiliny/data/nell_data.tar.gz) or [our pre-processed version](https://www.dropbox.com/s/bxrvf1syyuzmcqq/DGCN.zip?dl=0).

Since we used the exactly same datasets for testing, for detailed information about these four datasets, you may refer to the [Planetoid repository](https://github.com/kimiyoung/planetoid)
