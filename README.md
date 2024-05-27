# A Sparsity Principle for Partially Observable Causal Representation Learning (ICML 2024)
This repository contains the official implementation of the ICML 2024 paper: [A Sparsity Principle for Partially Observable Causal Representation Learning](https://arxiv.org/abs/2403.08335). This work performs by Danru Xu, Dingling Yao, Sébastien Lachapelle, Perouz Taslakian, Julius von Kügelgen, Francesco Locatello and Sara Magliacane. Please [cite](#bibtex) us when making use of our code or ideas.

![]()
## Environment
Tested on python 3.10.

```
pip install -r requirements.txt
```

## Numerical experiment
```
# linear mixing function
python Numerical/main_linear.py

# piecewise-linear mixing function
python Numerical/main_pw.py
```

## Image-based experiment
Download the dataset [Causal3DIdent](https://zenodo.org/records/4784282#.YgWo0PXMKbg) [von Kügelgen et al., 2021]
```
# multi-balls with missing balls
python ball/main_balls_missing.py

# multi-balls with fixed positions
python ball/main_balls_fixed_position.py

# partialCausal3DIdent
python Causal3DIdent/main_3dident.py --dataroot "$PATH_TO_DATA" 
```

## Acknowledgements
This implementation is built upon the code open-sourced by [Lachapelle et al. (2022)](https://github.com/slachapelle/disentanglement_via_mechanism_sparsity) (define invertible mixing functions); [Ahuja et al. (2022)](https://github.com/ahujak/WSRL) (generate multi-balls dataset); [von Kügelgen et al. (2021)](https://github.com/ysharma1126/ssl_identifiability) (Causal3DIdent dataset); [Zheng et al. (2018)](https://github.com/xunzheng/notears) (define structure causal models).

## BibTeX
```
@article{xu2024sparsity,
  title={A Sparsity Principle for Partially Observable Causal Representation Learning},
  author={Xu, Danru and Yao, Dingling and Lachapelle, S{\'e}bastien and Taslakian, Perouz and von K{\"u}gelgen, Julius and Locatello, Francesco and Magliacane, Sara},
  journal={International Conference on Machine Learning (ICML)},
  year={2024}
}
```
