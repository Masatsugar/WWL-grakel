# Wasserstein Weisfeiler-Lehman Graph Kernels
Reimplementation of NeurIPS 2019 paper *Wasserstein Weisfeiler-Lehman Graph Kernels by Matteo Togninalli et al*. 

- [Paper](https://proceedings.neurips.cc/paper/2019/hash/73fed7fd472e502d8908794430511f4d-Abstract.html)
- [original](https://github.com/BorgwardtLab/WWL)

## Installation

```shell
pip install git+https://github.com/Masatsugar/WWL-grakel.git
```

## Examples:
Graph data is based on Grakel Graph. If you want to know the usages, see `examples/mutag.py`.

```shell
$ python examples/mutag.py --cv 10 
WWL(C=1, gamma=10): Mean 10-fold accuracy: 86.67 +- 6.88 %
WL(C=1): Mean 10-fold accuracy: 81.87 +- 6.06 %
```


## Requirements
- [Grakel](https://ysig.github.io/GraKeL/0.1a8/)
- [POT: Python Optimal Transport](https://pythonot.github.io/)

## 解説

[Weisfeiler-Lehman Graph Kernel](https://www.jmlr.org/papers/volume12/shervashidze11a/shervashidze11a.pdf)を用いて得られた
ノードラベルに対してWasserstein距離を求め，カーネルを作成したものです．ノード同士の関係性も考慮したWLカーネルになります．

Wasserstein距離は最適輸送行列とコスト関数行列のフロベニウス積 から求められます．
最適輸送行列はshinkhornアルゴリズムを用いると近似的に得られます． 



