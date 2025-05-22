## Molecular experiments

This directory contains code to reproduce results from Sections 4.2 and 4.3 in the paper.

### Setup

Create and activate virtual environment

```
conda env create -f env.yml
conda activate mol_env
jupyter notebook
```

### Geodesics

The notebook `geodesics.ipynb` include all experiments with spacetime geodesics on molecular data.

### Diffusion model training

To estimate the geodesics, an approximate denoiser is required. We provide a checkpoint file `diffusion_model.pt`, so it is not necessary to train it from scratch. However, the training code is available in `diffusion_model_training.ipynb`.

### Energy function approximation

As discussed in Appendix E.3 of the paper, we use a neural network to approximate the energy function of Alanine Dipeptide. We provide the checkpoint `potential_model.pt` so it is not necessary to train it from scratch. However, we provide the code for training the energy function approximation in `energy_function_approximation.ipynb`.
