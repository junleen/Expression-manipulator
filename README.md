# Toward Fine-grained Facial Expression Manipulation (ECCV 2020, [Paper](https://arxiv.org/pdf/2004.03132.pdf))

![Python 3.6](https://img.shields.io/badge/python-3.5-green.svg?style=plastic)
![Pytorch 0.4.1](https://img.shields.io/badge/pytorch-0.4.1-green.svg?style=plastic)
![Pytorch 1.3.1](https://img.shields.io/badge/pytorch-1.3.1-green.svg?style=plastic)

![cover](./docs/cover.png)
**Arbitrary Facial Expression Manipulation**. Our model can 1) perform continuous editing between two expressions (top); 2) learn to only modify one facial component(middle); 3) transform expression in paintings (bottom). From left to right, the emotion intensity is set to 0, 0.5, 0.75, 1, and 1.25.


### Single/Multiple AU Editing
![single-au](./docs/single-au.png)
**Single/multiple au Editing**. AU4: Brow Lowerer; AU5: Upper Lid Raiser; AU7: Lid Tightener; AU12: Lip Corner Puller; AU15: Lip Corner Depressor; AU20: Lip Stretcher. The legend below the images are relative AUs intensity. The higher (lower) AUs value means to strengthen (weaken) the corresponding facial action unit in input image.

### Expression Transfer

![arbitrary](./docs/arbitrary-expression.png)
**Arbitrary Facial Expresson Manipulation**. The top-left image with blue box is input, the images in odd rows are image with target expression, the images in even rows are animated results.

![gif](./docs/gif.gif)

## Resources
Here are some links for you to know Action Units better.
- [OpenFace](https://github.com/TadasBaltrusaitis/OpenFace/wiki/Action-Units)
- [Imotions](https://imotions.com/blog/facial-action-coding-system/)
- [Paul Ekman](https://www.paulekman.com/facial-action-coding-system/)
- [FACS](https://www.cs.cmu.edu/~face/facs.htm)
- [Wiki](https://en.wikipedia.org/wiki/Facial_Action_Coding_System)

## Prerequisites
- Install PyTorch(version==0.4.1 or  >= 1.3.0), torchvision

- Install `requirements.txt`

  ```txt
  numpy
  matplotlib
  tqdm
  pickle
  opencv-python
  tensorboardX
  face_alignment
  ```

## Getting Started

### 1. Data Preparation

- prepare your images ([EmotionNet](http://cbcsl.ece.ohio-state.edu/EmotionNetChallenge/index.html), or [AffectNet](http://mohammadmahoor.com/affectnet/), etc.)

- Extract the Action Units with [OpenFace](https://github.com/TadasBaltrusaitis/OpenFace/wiki/Action-Units), and generate `aus_dataset.pkl` which contains  a list of dict, e.g., `[{'file_path': <path of image1>, 'aus':<extracted aus of image1>}, {'file_path': <path of image2>, 'aus':<extracted aus of image2>}]`

- Please refer to `src/samples/aus_dataset.pkl`

- You may use the function of pickle to save `pkl` file

  ```python
  with open('aus_dataset.pkl', 'wb') as f:
      pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
  ```

### 2. Training

To train, please modify the parameters in `launch/train.sh` and run:

```bash
bash launch/train.sh
```


## Citation
If you find this repository helpful, use this code or adopt ideas from the paper for your research, please cite:

```bibtex
@inProceedings{ling2020finegrained,
  title     = {Toward Fine-grained Facial Expression Manipulation},
  author    = {Ling, Jun and Xue, Han and Song, Li and Yang, Shuhui and Xie, Rong and Gu, Xiao},
  booktitle = {Proceedings of European Conference on Computer Vision (ECCV)},
  year      = {2020}
}
```


## Contact
Please contact lingjun@sjtu.edu.cn or open an issue for any questions or suggestions.


## Acknowledgement
- Thanks [Albert Pumarola](https://www.albertpumarola.com/) for sharing his work [GANimation](https://github.com/albertpumarola/GANimation).
- Thanks the authors of [AffectNet](http://mohammadmahoor.com/affectnet/) for his work on large scale emotion database.
