from torch.utils.data import Dataset
import numpy as np
import torch
from collections import OrderedDict


class CachedMMapDataset(Dataset):
    def __init__(self, data_path, warmup_cache=20000, max_cache_size=60000, max_open_files=500):
        self.paths = np.loadtxt(data_path, dtype=str)
        print(f'Loaded {len(self.paths)} samples')

        # 内存缓存系统
        self.cache = OrderedDict()
        self.mmap_refs = OrderedDict()  # 改用有序字典管理
        self.max_cache = max_cache_size
        self.max_open_files = max_open_files  # 新增最大文件打开数限制

        # 预加载第一批数据到内存
        self._warmup_cache(warmup_cache)

    def _warmup_cache(self, num_samples):
        """预加载高频访问数据到内存"""
        indices = np.random.choice(len(self.paths), min(num_samples, len(self.paths)), replace=False)
        for idx in indices:
            self._load_to_cache(idx)

    def _close_old_mmap(self):
        """关闭最久未使用的内存映射文件"""
        while len(self.mmap_refs) >= self.max_open_files:
            oldest_idx, _ = self.mmap_refs.popitem(last=False)
            if oldest_idx in self.cache:
                del self.cache[oldest_idx]  # 同步清理缓存

    def _load_to_cache(self, idx):
        """将单个样本加载到缓存"""
        if idx not in self.cache:
            # 1. 清理旧文件
            self._close_old_mmap()

            # 2. 加载新数据
            if idx not in self.mmap_refs:
                mmap_data = np.load(self.paths[idx], mmap_mode='r')
                self.mmap_refs[idx] = mmap_data  # 记录新映射
                self.mmap_refs.move_to_end(idx)  # 标记为最近使用
            else:
                mmap_data = self.mmap_refs[idx]

            # 3. 缓存数据并管理内存
            data = torch.from_numpy(mmap_data[...].copy()).float()
            self.cache[idx] = data

            # 4. 缓存淘汰
            if len(self.cache) > self.max_cache:
                oldest_idx, _ = self.cache.popitem(last=False)
                if oldest_idx in self.mmap_refs:
                    del self.mmap_refs[oldest_idx]

    def __getitem__(self, index):
        # 动态缓存管理
        self._load_to_cache(index)
        return self.cache[index]

    def __len__(self):
        return len(self.paths)

    def __del__(self):
        """对象销毁时显式关闭所有资源"""
        for mmap in self.mmap_refs.values():
            if hasattr(mmap, '_mmap'):
                mmap._mmap.close()
        self.mmap_refs.clear()
        self.cache.clear()