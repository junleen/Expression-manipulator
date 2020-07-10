# Toward Fine-grained Facial Expression Manipulation (ECCV 2020)

![Python 3.6](https://img.shields.io/badge/python-3.5-green.svg?style=plastic)
![Pytorch 0.4.1](https://img.shields.io/badge/pytorch-0.4.1-green.svg?style=plastic)
![Pytorch 1.3.1](https://img.shields.io/badge/pytorch-1.3.1-green.svg?style=plastic)

![cover](.docs/cover.png)
**Figure:** *Real image editing using the proposed In-Domain GAN inversion with a fixed GAN generator.*

In the repository, we propose an in-domain GAN inversion method, which not only faithfully reconstructs the input image but also ensures the inverted code to be **semantically meaningful** for editing. Basically, the in-domain GAN inversion contains two steps:

1. Training **domain-guided** encoder.
2. Performing **domain-regularized** optimization.
