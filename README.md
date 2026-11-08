# SGRK-Torch

Unofficial PyTorch implementation of **Sparse-Gated Recurrent Kernels for Long-Horizon Sequence Modeling** by
Elena M. Vasquez, Kwame O. Boateng, Rina Takeda and Jonas Lindqvist
(TechForge Preprints, preprint ID **2601.01234**, January 2026).

This is an independent community implementation. It is not affiliated with,
endorsed by, or maintained by the paper's authors.

## What is implemented

* `SGRKLayer`: recurrent kernel bank with a learned kernel sparsity gate
  (straight-through estimator, temperature annealing)
* horizon-decay schedule with a learned half-life per state channel
* training and evaluation scripts for the 4k / 8k / 16k / 32k horizon benchmarks
  described in Section 3 of the paper

## Quick start

```
pip install -r requirements.txt
python train.py --horizon 16k --sparsity 4
```

## Citation

If you use this code, please cite the original paper:

> Vasquez, E. M., Boateng, K. O., Takeda, R., and Lindqvist, J.
> Sparse-Gated Recurrent Kernels for Long-Horizon Sequence Modeling.
> TechForge Preprints, 2601.01234, 2026.

## Community

Bug reports and benchmark reproductions are tracked in this repository's issues.
