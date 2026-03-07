`rictr` is a knowledge distillation library for PyTorch, to make compressing and transferring knowledge between models easy.

#### Documentation

refer to [rictr.in](https://rictr.in)

#### Implemented Papers

- **Logit Distillation (Soft Targets):** Based on Hinton et al. (2015), [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531).

- **Attention Transfer:** Based on Zagoruyko & Komodakis (2016), [Paying More Attention to Attention: Improving the Performance of Convolutional Neural Networks via Attention Transfer](https://arxiv.org/abs/1612.03928).

- **Hidden State Distillation:** Feature-based distillation by matching intermediate layer activations between teacher and student.

#### Installation

```bash
pip install rictr
```