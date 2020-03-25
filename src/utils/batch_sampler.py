#!/usr/bin/python
# -*- encoding: utf-8 -*-


import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler
import cv2
import numpy as np
import random
import logging
import sys


class BatchSampler(Sampler):
    '''
    sampler used in dataloader. method __iter__ should output the indices each time it is called
    '''
    def __init__(self, dataset, n_class, n_num, *args, **kwargs):
        super(BatchSampler, self).__init__(dataset, *args, **kwargs)
        self.n_class = n_class  # person num of each batch
        self.n_num = n_num  # image num of each person
        self.batch_size = n_class * n_num  # batch size
        self.dataset = dataset  # dataset
        self.classes = np.array(dataset.classes)  # person classes
        self.len = len(dataset) // self.batch_size  # num of batches
        self.class_dict = dataset.class_dict  # person class dict
        self.iter_num = len(self.classes) // self.n_class  # num of iterations
        # self.labels = np.array(dataset.lb_ids)  # seem unuse

    def __iter__(self):
        curr_p = 0
        np.random.shuffle(self.classes)
        for k, v in self.class_dict.items():
            random.shuffle(self.class_dict[k])
        for i in range(self.iter_num):
            label_batch = self.classes[curr_p: curr_p + self.n_class]
            curr_p += self.n_class
            idx = []
            for lb in label_batch:
                if len(self.class_dict[lb]) > self.n_num:
                    idx_smp = np.random.choice(self.class_dict[lb],
                            self.n_num, replace = False)
                else:
                    idx_smp = np.random.choice(self.class_dict[lb],
                            self.n_num, replace = True)
                idx.extend(idx_smp.tolist())
            yield idx


    def __len__(self):
        return self.iter_num


if __name__ == "__main__":
    import dataset
    import itertools

    print("-----MARKET-----")
    data_dir = '/home/b604/a_tyy/Datasets/Market-1501-v15.09.15'
    data = 'MARKET'
    height = 384
    width = 128
    ds = dataset.__dict__[data](root=data_dir, part='train', size=(height, width),
                                require_path=True, true_pair=True)
    sampler = BatchSampler(ds, 16, 4)
    dl = DataLoader(ds, batch_sampler = sampler, num_workers = 4)

    diter = itertools.cycle(dl)

    while True:
        ims, lbs, _ = next(diter)
        print(lbs.shape)
    print(len(list(ds.classes)))

    print("----------")
    print("-----DUKE-----")
    data_dir = '/home/b604/a_tyy/Datasets/DukeMTMC-reID/DukeMTMC-reID/'
    data = 'DUKE'
    height = 384
    width = 128
    ds = dataset.__dict__[data](root=data_dir, part='train', size=(height, width),
                                require_path=True, true_pair=True)
    sampler = BatchSampler(ds, 16, 4)
    dl = DataLoader(ds, batch_sampler = sampler, num_workers = 4)

    diter = itertools.cycle(dl)

    while True:
        ims, lbs, _ = next(diter)
        print(lbs.shape)
    print(len(list(ds.classes)))
