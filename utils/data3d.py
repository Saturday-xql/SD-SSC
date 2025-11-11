import os
import logging
import numpy as np
from torch.utils.data import Dataset
import cv2
import imageio
import random
from skimage import io

class NYU(Dataset):
    classes = ['empty', 'ceiling', 'floor', 'wall', 'window', 'chair', 'bed', 'sofa', 'table', 'tvs', 'furn', 'objs']
    num_classes = 12
    class2color = {'empty': [22, 191, 206],
                   'ceiling': [214, 38, 40],
                   'floor': [43, 160, 4],
                   'wall': [158, 216, 229],
                   'window': [114, 158, 206],
                   'chair': [204, 204, 91],
                   'bed': [255, 186, 119],
                   'sofa': [147, 102, 188],
                   'table': [30, 119, 181],
                   'tvs': [188, 188, 33],
                   'furn': [255, 127, 12],
                   'objects': [196, 175, 214],
                   'ignore': [153, 153, 153]}
    cmap = [*class2color.values()]

    def __init__(self,
                 data_root: str = '/path-to-data/SSC/NYU',
                 img_H: int = 480,
                 img_W: int = 640,
                 split: str = 'train',
                 tsdf_resolution='lr',
                 label_resolution='lr',
                 data_augmentation=False,
                 CAD=False,
                 ):

        super().__init__()
        self.split, self.data_root = \
            split,  data_root
        self.data_augmentation = data_augmentation
        self.img_H = img_H
        self.img_W = img_W
        self.CAD=CAD
        self.tsdf_resolution=tsdf_resolution
        self.label_resolution = label_resolution

        self.data_list =self.get_file_names(self.split,self.data_root)
        # print(self.data_list)
        self.data_idx = np.arange(len(self.data_list))

        assert len(self.data_idx) > 0
        logging.info(f"\nTotally {len(self.data_idx)} samples in {split} set")

    @staticmethod
    def get_file_names(split: str, data_root: str):
        """
        Load file names from a text file.
        """
        s = split.replace("valid", "test")
        path = os.path.join(data_root, "{}.txt".format(s))
        with open(path, "r") as file:
            lines = file.readlines()
        file_names = [line.strip() for line in lines]
        return file_names



    def load_tsdf(self, item):
        tsdf_path = os.path.join(self.data_root, 'TSDF', '{}.npz'.format(item))

        if self.CAD:
            tsdf_path = tsdf_path.replace('NYU', 'NYUCAD')

        if self.tsdf_resolution=='hr':
            tsdf = np.load(tsdf_path)['hr']
            tsdf = tsdf.astype(np.float32).reshape(1, 240, 144, 240)
        else:
            tsdf = np.load(tsdf_path)['lr']
            tsdf=tsdf.astype(np.float32).reshape(1, 60, 36, 60)
        return tsdf

    def load_label_weight(self, item):
        label_weight_path = os.path.join(self.data_root, 'label_weight', '{}.npz'.format(item))

        if self.label_resolution=='hr':
            label_weight = {'1': np.load(label_weight_path)['hr'].astype(np.float32),
                            '2': np.load(label_weight_path)['hr_2'].astype(np.float32),
                            '4': np.load(label_weight_path)['hr_4'].astype(np.float32)}
        else:
            label_weight = {'1': np.load(label_weight_path)['lr'].astype(np.float32),
                            '2': np.load(label_weight_path)['lr_2'].astype(np.float32),
                            '4': np.load(label_weight_path)['lr_4'].astype(np.float32)}

        return label_weight

    def load_label(self, item):
        label3d_path = os.path.join(self.data_root, 'Label', '{}.npz'.format(item))

        if self.label_resolution == 'hr':
            label3d = {'1': np.load(label3d_path)['hr'].astype(np.int64),
                       '2': np.load(label3d_path)['hr_2'].astype(np.int64),
                       '4': np.load(label3d_path)['hr_4'].astype(np.int64)}
        else:
            label3d = {'1': np.load(label3d_path)['lr'].astype(np.int64),
                       '2': np.load(label3d_path)['lr_2'].astype(np.int64),
                       '4': np.load(label3d_path)['lr_4'].astype(np.int64)}
        return label3d


    def load_depth_prior_3d(self,item):
        path=os.path.join(self.data_root, 'depth_prior', '{}.npz'.format(item))

        if self.CAD:
            path = path.replace('NYU', 'NYUCAD')

        if self.label_resolution == 'hr':
            depth_prior = np.load(path)['hr']
            depth_prior = depth_prior.astype(np.float32).reshape(12, 240, 144, 240)
        else:
            depth_prior = np.load(path)['lr']
            depth_prior = depth_prior.astype(np.float32).reshape(12, 60, 36, 60)

        return depth_prior

    def __getitem__(self, idx):
        data_idx = self.data_idx[idx % len(self.data_idx)]

        item = self.data_list[data_idx]

        label3d = self.load_label(item)

        tsdf = self.load_tsdf(item)
        label_weight=self.load_label_weight(item)
        depth_prior=self.load_depth_prior_3d(item)

        if self.data_augmentation:

            if random.random() >= .5:
                tsdf = np.swapaxes(tsdf, axis1=1, axis2=3).copy()
                depth_prior = np.swapaxes(depth_prior, axis1=1, axis2=3).copy()
                for k, v in label_weight.items():
                    label_weight[k] = np.swapaxes(v, axis1=0, axis2=2).copy()
                for k, v in label3d.items():
                    label3d[k] = np.swapaxes(v, axis1=0, axis2=2).copy()


            if random.random() >= .5:
                tsdf = np.flip(tsdf, axis=1).copy()
                depth_prior = np.flip(depth_prior, axis=1).copy()
                for k, v in label_weight.items():
                    label_weight[k] = np.flip(v, axis=0).copy()
                for k, v in label3d.items():
                    label3d[k] = np.flip(v, axis=0).copy()


            if random.random() >= .5:
                tsdf = np.flip(tsdf, axis=3).copy()
                depth_prior = np.flip(depth_prior, axis=3).copy()
                for k, v in label_weight.items():
                    label_weight[k] = np.flip(v, axis=2).copy()
                for k, v in label3d.items():
                    label3d[k] = np.flip(v, axis=2).copy()


        data={'label3d': label3d,
                'file': item,
              'tsdf': tsdf,
              'label_weight':label_weight,
              'depth_prior':depth_prior
              }

        # pre-process.

        return data

    def __len__(self):
        return len(self.data_idx)
