# Toward Fine-grained Facial Expression Manipulation (ECCV 2020, [Paper](https://arxiv.org/pdf/2004.03132.pdf))

![Python 3.6](https://img.shields.io/badge/python-3.5-green.svg?style=plastic)
![Pytorch 0.4.1](https://img.shields.io/badge/pytorch-0.4.1-green.svg?style=plastic)
![Pytorch 1.3.1](https://img.shields.io/badge/pytorch-1.3.1-green.svg?style=plastic)

![cover](./docs/cover.png)
Figure1. **Arbitrary Facial Expression Manipulation**. Our model can 1) perform continuous editing between two expressions (top); 2) learn to only modify one facial component(middle); 3) transform expression in paintings (bottom). From left to right, the emotion intensity is set to 0, 0.5, 0.75, 1, and 1.25.

## Single AU Editing
![single-au](./docs/single-au.png)
Figure2. **Single-au Editing**. AU4: Brow Lowerer; AU5: Upper Lid Raiser; AU7: Lid Tightener; AU12: Lip Corner Puller; AU15: Lip Corner Depressor; AU20: Lip Stretcher. The legend below the images are relative AUs intensity. The higher (lower) AUs value means to strengthen (weaken) the corresponding facial action unit in input image.

## Arbitrary Facial Expression Manipulation
![arbitrary](./docs/arbitrary-edition.png)
Figure3. **Arbitrary Facial Expresson Manipulation**. The top-left image with blue box is input, the images in odd rows are image with target expression, the images in even rows are animated results.

## Code coming soon...
