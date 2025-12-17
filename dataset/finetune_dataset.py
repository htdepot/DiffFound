from torch.utils.data import Dataset
import numpy as np
import torch
from glob import glob
import os.path as osp

class Dataset_Dmri(Dataset):
    def __init__(self, data_path, data_gt_path, mask_path=None, is_train=False):
        super(Dataset_Dmri, self).__init__()
        data_dir = osp.join(osp.join(data_path), '*')
        self.dataset_path = glob(data_dir)
        self.dataset_path = np.sort(self.dataset_path)
        data_gt_dir = osp.join(osp.join(data_gt_path), '*')
        self.dataset_gt_path = glob(data_gt_dir)
        self.dataset_gt_path = np.sort(self.dataset_gt_path)
        self.is_train = is_train
        if mask_path is not None:
            m_dir = osp.join(osp.join(mask_path), '*')
            self.mask_path = glob(m_dir)
            self.mask_path = np.sort(self.mask_path)
        if is_train:
            print('fetch {} samples for training'.format(len(self.dataset_path)))
        else:
            print('fetch {} samples for testing'.format(len(self.dataset_path)))
    def __getitem__(self, index):
        # fetch image
        if osp.basename(self.dataset_path[index])[:-4] == osp.basename(self.dataset_gt_path[index])[:-12]:
            data = torch.from_numpy(np.load(self.dataset_path[index])).to(torch.float32)
            gt = torch.from_numpy(np.load(self.dataset_gt_path[index])).to(torch.float32)
            if self.is_train:
                return data, gt
            else:
                if osp.basename(self.mask_path[index])[:-12] == osp.basename(self.dataset_path[index])[:-4]:
                    return data, gt, self.mask_path[index]
                else:
                    print('mask data no match')
        else:
            print('data no match')

    def __len__(self):
        return len(self.dataset_path)