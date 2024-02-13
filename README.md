# Attention-based Iterative Decomposition for Tensor Product Representation


Source code for the ICLR'24 paper: [Attention-based Iterative Decomposition for Tensor Product Representation](https://openreview.net/forum?id=FDb2JQZsFH)

**TL;DR:** Slot-based competitive mechanism that effectively binds sequential features to the structured representations (roles and fillers) of TPR

<br>

## Requirement

- Python 3.8
- PyTorch 1.13

```setup
pip install -r requirements.txt
```

<br>


## Acknowledement

We've developed our code using open-source implementations for various tasks, as below.

- **[bAbI task (FWM)](https://github.com/ischlag/Fast-Weight-Memory-public)** 

- **[bAbI task (TPR-RNN)](https://github.com/APodolskiy/TPR-RNN-Torch)** 

- **[Sort-of-CLEVR task](https://github.com/sarthmit/Compositional-Attention)** 

- **[WikiText-103 task](https://github.com/IDSIA/lmtool-fwp)** 

<br>

## Citation
```
@inproceedings{
  park2024attentionbased,
  title={Attention-based Iterative Decomposition for Tensor Product Representation},
  author={Taewon Park and Inchul Choi and Minho Lee},
  booktitle={The Twelfth International Conference on Learning Representations},
  year={2024},
  url={https://openreview.net/forum?id=FDb2JQZsFH}
}
```