# -*- encoding: utf-8 -*-
'''
@File    :   generator.py
@Time    :   2021/07/13 20:08:59
@Author  :   Liu Qidong
@Version :   1.0
@Contact :   dong_liuqi@163.com
'''

# here put the import lib
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch


class MF_Data(Dataset):

    def __init__(self, user, item, label):

        self.user = user.long()
        self.item = item.long()
        self.label = label.float()

    def __len__(self):

        return len(self.user)

    def __getitem__(self, index):

        return self.user[index], self.item[index], self.label[index]


class IPS_Data(Dataset):

    def __init__(self, user, item, label, ps):

        self.user = user.long()
        self.item = item.long()
        self.label = label.float()
        self.ps = ps.float()

    def __len__(self):

        return len(self.user)

    def __getitem__(self, index):

        return self.user[index], self.item[index], self.label[index], self.ps[index]


class DataGenerator():

    def __init__(self, opt, ps_model=None):
        
        self.opt = opt
        self._load_data()
        self.items_num = int(1720)
        self.users_num = int(1000)
        self.ps_model = ps_model


    def _load_data(self):

        self.train_data = np.loadtxt(self.opt.train_data)
        self.test_data = np.loadtxt(self.opt.test_data)
        self.validation_data = np.loadtxt(self.opt.val_all_data)


    def make_train_loader(self):

        user = self.train_data[:, 0]
        item = self.train_data[:, 1]
        label = self.train_data[:, 2]

        traindata = MF_Data(torch.tensor(user, device=torch.device(self.opt.device)), 
                            torch.tensor(item, device=torch.device(self.opt.device)), 
                            torch.tensor(label, device=torch.device(self.opt.device)))

        return DataLoader(traindata, batch_size=self.opt.batch_size, shuffle=True)

    
    def make_ips_loader(self):

        user = self.train_data[:, 0]
        item = self.train_data[:, 1]
        label = self.train_data[:, 2]
        ps = self.ps_model.predict(user, item)

        traindata = IPS_Data(torch.tensor(user, device=torch.device(self.opt.device)), 
                            torch.tensor(item, device=torch.device(self.opt.device)), 
                            torch.tensor(label, device=torch.device(self.opt.device)),
                            torch.tensor(ps, device=torch.device(self.opt.device)))

        return DataLoader(traindata, batch_size=self.opt.batch_size, shuffle=True)

    
    def make_test_loader(self, validation=False):

        if validation:
            user = self.validation_data[:, 0]
            item = self.validation_data[:, 1]
            label = self.validation_data[:, 2]
        else:
            user = self.test_data[:, 0]
            item = self.test_data[:, 1]
            label = self.test_data[:, 2]

        testdata = MF_Data(torch.tensor(user, device=torch.device(self.opt.device)), 
                           torch.tensor(item, device=torch.device(self.opt.device)), 
                           torch.tensor(label, device=torch.device(self.opt.device)))

        return DataLoader(testdata, batch_size=testdata.__len__(), shuffle=True)



