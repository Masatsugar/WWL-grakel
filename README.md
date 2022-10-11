# Wasserstein Weisfeiler-Lehman Graph Kernels
Reimplementation of NeurIPS 2019 paper Wasserstein Weisfeiler-Lehman Graph Kernels. 

- [Paper](https://proceedings.neurips.cc/paper/2019/hash/73fed7fd472e502d8908794430511f4d-Abstract.html)
- [original](https://github.com/BorgwardtLab/WWL)

## Installation
```shell
pip install git+https://github.com/Masatsugar/WWL-grakel.git
```

- Examples:

```python
from wwl_grakel import WassersteinWeisfeilerLehman

wwl = WassersteinWeisfeilerLehman(n_iter=2)
M = wwl.compute_wasserstein_dictance(G)
```


## Requirements
- [Grakel](https://ysig.github.io/GraKeL/0.1a8/)
- [POT: Python Optimal Transport](https://pythonot.github.io/)

## 解説

[Weisfeiler-Lehman Graph Kernel](https://www.jmlr.org/papers/volume12/shervashidze11a/shervashidze11a.pdf)を用いて得られた
ノードラベルに対してWasserstein距離を求め，カーネルを作成したものです．ノード同士の関係性も考慮したWLカーネルになります．

Wasserstein距離は最適輸送行列とコスト関数行列のフロベニウス積 から求められます．
最適輸送行列はshinkhornアルゴリズムを用いると近似的に得られます． 



