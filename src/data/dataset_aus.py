import os
import time
import pickle
import random
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from utils import cv_utils
from data.dataset import DatasetBase


class AusDataset(DatasetBase):
    def __init__(self, opt, is_for_train):
        super(AusDataset, self).__init__(opt, is_for_train)
        self._name = "AusDataset"
        self._read_dataset()
        self._cond_nc = opt.cond_nc

    def __len__(self):
        return self._dataset_size
    
    def __getitem__(self, idx):
        assert (idx < self._dataset_size)
        real_img = None
        real_cond = None
        real_img_path = None
        while real_img is None or real_cond is None:
            # get sample data
            sample_id = self._get_id(idx)
    
            real_img, real_img_path = self._get_img_by_id(idx)
            real_cond = self._get_cond_by_id(idx)
            if real_img is None:
                print('error reading image %s, skipping sample' % real_img_path)
                idx = random.randint(0, self._dataset_size - 1)
            if real_cond is None:
                print('error reading aus %s, skipping sample' % sample_id)
                idx = random.randint(0, self._dataset_size - 1)
        real_cond += np.random.uniform(-0.02, 0.02, real_cond.shape)
        desired_img, desired_cond, noise = self._generate_random_cond()

        # transform data
        real_img = self._transform(Image.fromarray(real_img))
        desired_img = self._transform(Image.fromarray(desired_img))

        # pack data
        sample = {'real_img': real_img,
                  'real_cond': real_cond,
                  'desired_img': desired_img,
                  'desired_cond': desired_cond,
                  'cond_diff': desired_cond - real_cond,
                  }
        return sample

    def _create_transform(self):
        if self._is_for_train:
            transform_list = [transforms.RandomHorizontalFlip(),
                              transforms.Resize(self._image_size),
                              transforms.Pad(self._image_size // 16),
                              transforms.RandomCrop(self._image_size),
                              transforms.ToTensor(),
                              transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]
            
        else:
            transform_list = [transforms.Resize(self._image_size),
                              transforms.ToTensor(),
                              transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]
        
        self._transform = transforms.Compose(transform_list)
    
    def _read_dataset(self):
        self._root = self._opt.data_dir
        self._imgs_dir = os.path.join(self._root, self._opt.images_folder)

        annotations_file = self._opt.train_annotations_file if self._is_for_train else self._opt.test_annotations_file
        pkl_path = os.path.join(self._root, annotations_file)
        self._info = self._read_pkl(pkl_path)
        self._image_size = self._opt.image_size
        # dataset size
        self._dataset_size = len(self._info)
        
    def _read_pkl(self, file_path):
        assert os.path.exists(file_path) and file_path.endswith('.pkl'), 'Read pkl file error. Cannot open %s' % file_path
        with open(file_path, 'rb') as f:
            return pickle.load(f)

    def _get_id(self, idx):
        id = self._info[idx]['file_path']
        return os.path.splitext(id)[0]
    
    def _get_cond_by_id(self, idx):
        cond = None
        if idx < self._dataset_size:
            cond = self._info[idx]['aus'] / 5.0
        return cond


    def _get_img_by_id(self, idx):
        if idx < self._dataset_size:
            img_path = os.path.join(self._imgs_dir, self._info[idx]['file_path'])
            img = cv_utils.read_cv2_img(img_path)
            return img, self._info[idx]['file_path']
        else:
            print('You input idx： ', idx)
            return None, None

    def _generate_random_cond(self):
        cond = None
        rand_sample_id = -1
        while cond is None:
            rand_sample_id = random.randint(0, self._dataset_size - 1)
            cond = self._get_cond_by_id(rand_sample_id)
        img, _ = self._get_img_by_id(rand_sample_id)
        noise = np.random.uniform(-0.1, 0.1, cond.shape)
        if img is None:
            img, cond, noise = self._generate_random_cond()
        cond += noise
        return img, cond, noise

