import torch.utils.data
import torchvision.transforms as transforms
import os


class CustomDatasetDataLoader:
    def __init__(self, opt, is_for_train=True):
        self._opt = opt
        self._is_for_train = is_for_train
        self._num_threds = opt.n_threads_train if is_for_train else opt.n_threads_test
        self._create_dataset()
        
    def _create_dataset(self):
        self._dataset = DatasetFactory.get_by_name(self._opt.dataset_mode, self._opt, self._is_for_train)

        
    def load_data(self):
        self._dataloader = torch.utils.data.DataLoader(
                self._dataset,
                batch_size=self._opt.batch_size,
                shuffle=not self._opt.serial_batches,
                num_workers=int(self._num_threds),
                drop_last=True)
        return self._dataloader
    
    def __len__(self):
        return len(self._dataset)
    

class DatasetFactory:
    def __init__(self):
        pass
    
    @staticmethod
    def get_by_name(dataset_name, opt, is_for_train):
        if dataset_name == 'aus':
            from data.dataset_aus import AusDataset
            dataset = AusDataset(opt, is_for_train)
        else:
            raise ValueError("Dataset [%s] not find." % (dataset_name))
        
        print('Dataset {} was created.'.format(dataset.name))
        return dataset


class DatasetBase(torch.utils.data.Dataset):
    def __init__(self, opt, is_for_train):
        super(DatasetBase, self).__init__()
        self._name = 'BaseDataset'
        self._root = None
        self._opt = opt
        self._is_for_train = is_for_train
        self._image_size = self._opt.image_size
        self._create_transform()
        
        self._IMG_EXTENSIONS = [
            '.jpg', '.JPG', '.jpeg', '.JPEG',
            '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
        ]
    
    @property
    def name(self):
        return self._name
    
    @property
    def path(self):
        return self._root
    
    def _create_transform(self):
        self._transform = transforms.Compose([])
        
    def get_transform(self):
        return self._transform
    
    def _is_image_file(self, filename):
        return any(filename.endswith(extention) for extention in self._IMG_EXTENSIONS)
    
    def _is_csv_file(self, filename):
        return filename.endswith('.csv')
    
    def _get_all_files_in_subfolders(self, dir, is_file):
        '''
        to get all files paths in dir
        Args:
            dir: input dataset directory
            is_file: a function which judge is a file is image or csv document.
        return:
            a list contains files paths
        '''
        
        images = []
        
        assert os.path.isdir(dir), '%s is not a valid directory' % dir
        
        for root, _, fnames in sorted(os.walk(dir)):
            for fname in fnames:
                if is_file(fname):
                    path = os.path.join(root, fname)
                    images.append(path)
        
        return images
