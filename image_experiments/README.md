## Image experiments

### Setup

Experiments in this section depend on the original [EDM2 codebase](https://github.com/NVlabs/edm2), which is included as a Git submodule.

#### Step 1: Initialize submodules

```bash
git submodule update --init --recursive
```

#### Step 2: Copy required files to the edm2 repository

```bash
cp geodesic.py edm2/
cp ../curves.py edm2/
```

#### Step 3: Setup enviroment

Follow instructions in the [EDM2 codebase](https://github.com/NVlabs/edm2). We recommend using Docker.


### Hardware

All our examples on image data were run on a single NVIDIA A100 GPU, which allowed using `batch_size=16` in the geodesic experiments, where we used the largest available pretrained model: `edm2-img512-xxl-fid`.

### Example usage

The interpolation results in the paper were generated with the following commands (we do not include the results due to their size).

```bash
python geodesic.py --preset=edm2-img512-xxl-fid --outdir=results/geodesic_167 --num_nodes 8 --class 167 --seeds 0 --batch_size 16 --n_opt_steps_warmup 350 --n_opt_steps_finetune 250 --img1 interpolation_images/dog_interp1.png --img2 interpolation_images/dog_interp2.png

python geodesic.py --preset=edm2-img512-xxl-fid --outdir=results/geodesic_179 --num_nodes 8 --class 179 --seeds 0 --batch_size 16 --n_opt_steps_warmup 350 --n_opt_steps_finetune 250 --img1 interpolation_images/similar_dog_interp1.png --img2 interpolation_images/similar_dog_interp2.png

python geodesic.py --preset=edm2-img512-xxl-fid --outdir=results/geodesic_483 --num_nodes 8 --class 483 --seeds 0 --batch_size 16 --n_opt_steps_warmup 350 --n_opt_steps_finetune 250 --img1 interpolation_images/similar_castle_interp1.png --img2 interpolation_images/similar_castle_interp2.png

python geodesic.py --preset=edm2-img512-xxl-fid --outdir=results/geodesic_895 --num_nodes 8 --class 895 --seeds 0 --batch_size 16 --n_opt_steps_warmup 350 --n_opt_steps_finetune 250 --img1 interpolation_images/plane_interp1.png --img2 interpolation_images/plane_interp2.png
```
