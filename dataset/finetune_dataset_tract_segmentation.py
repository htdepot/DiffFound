from torch.utils.data import Dataset
import numpy as np
import torch
import os.path as osp
from glob import glob
class Dataset_Dmri(Dataset):
    def __init__(self, data_path, data_gt_path, mask_path=None, is_train=False):
        super(Dataset_Dmri, self).__init__()
        self.dataset_path= np.loadtxt(data_path, dtype=str)
        self.dataset_gt_path = np.loadtxt(data_gt_path, dtype=str)
        self.is_train = is_train
        if mask_path is not None:
            self.mask_path = np.loadtxt(mask_path, dtype=str)
            # self.mask_path = glob(m_dir)
            # self.mask_path = np.sort(self.mask_path)
        print('fetch {} samples for training'.format(len(self.dataset_path)))

    def __getitem__(self, index):
        # fetch image
        if osp.basename(self.dataset_path[index]).split('.')[0].split('_')[0] == osp.basename(self.dataset_gt_path[index]).split('.')[0].split('_')[0]:
            data = torch.from_numpy(np.load(self.dataset_path[index])).to(torch.float32)
            gt = torch.from_numpy(np.load(self.dataset_gt_path[index])).to(torch.float32)
            if self.is_train:
                return data, gt
            else:
                if osp.basename(self.mask_path[index])[:-12].split('_')[0] == osp.basename(self.dataset_path[index])[:-4].split('_')[0]:
                    return data, gt, self.mask_path[index]
                else:
                    print('mask data no match')
            return data, gt
        else:
            print('data not match')


    def __len__(self):
        return len(self.dataset_path)