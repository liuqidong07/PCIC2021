import os
from torch.utils.data import DataLoader
from torch.utils import data
from tqdm import tqdm
from time import time
import numpy as np
import argparse
import random
import torch
import torch.nn as nn
from torch.nn.init import normal_
from sklearn.metrics import roc_auc_score
import copy

from naie.datasets import get_data_reference
from naie.context import Context
import moxing as mox

class DefaultConfig(object):
    model = 'MF_Naive'

    data_dir = '/cache'

    train_data = data_dir + '/extract_alldata.txt'
    val_all_data = data_dir + '/datasets/DatasetService/infer_valid/validation.txt'
    test_data = data_dir + '/datasets/DatasetService/infer_test/test.txt'

    reg_c = 0.001
    reg_t = 0.001
    reg_tc = 0.001

    metric = 'auc'
    verbose = 10

    device = 'cpu'
    batch_size = 512
    embedding_size = 16

    max_epoch = 50
    lr = 0.001 
    weight_decay = 1e-5


opt = DefaultConfig()


class MF_Naive(nn.Module):
    def __init__(self, num_users, num_items, embedding_size, device='cpu'):
        super(MF_Naive, self).__init__()

        self.num_users = num_users
        self.num_items = num_items

        self.user_e = nn.Embedding(self.num_users, embedding_size)
        self.item_e = nn.Embedding(self.num_items, embedding_size)
        self.user_b = nn.Embedding(self.num_users, 1)
        self.item_b = nn.Embedding(self.num_items, 1)

        self.apply(self._init_weights)

        self.loss = nn.MSELoss()

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            normal_(module.weight.data, mean=0.0, std=0.1)

    def forward(self, user, item):
        user_embedding = self.user_e(user)
        item_embedding = self.item_e(item)

        preds = self.user_b(user)
        preds += self.item_b(item)
        preds += (user_embedding * item_embedding).sum(dim=1, keepdim=True)

        return preds.squeeze()

    def calculate_loss(self, user_list, item_list, label_list):
        return self.loss(self.forward(user_list, item_list), label_list)

    def predict(self, user, item):
        return self.forward(user, item)

    def get_optimizer(self, lr, weight_decay):
        return torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)

    def get_embedding(self):
        return self.user_e, self.item_e


class MF_DATA(data.Dataset):
    def __init__(self, filename):
        raw_matrix = np.loadtxt(filename)
        self.users_num = int(1000)
        self.items_num = int(1720)
        self.data = raw_matrix

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.data.shape[0]


def MSE(preds, true):
    squaredError = []
    for i in range(len(preds)):
        dis = true[i] - preds[i]
        squaredError.append(dis * dis)  
    return sum(squaredError) / len(squaredError)


def MAE(preds, true):
    absError = []
    for i in range(len(preds)):
        dis = true[i] - preds[i]
        absError.append(abs(dis)) 
    return sum(absError) / len(absError)


def RMSE(preds, true):
    squaredError = []
    absError = []
    for i in range(len(preds)):
        dis = true[i] - preds[i]
        squaredError.append(dis * dis)
        absError.append(abs(dis))
    from math import sqrt
    return sqrt(sum(squaredError) / len(squaredError))


def AUC(true, preds):
    return roc_auc_score(true, preds)


def evaluate_model(model, val_data, opt):
    true = val_data[:, 2]
    user = torch.LongTensor(val_data[:, 0]).to(opt.device)
    item = torch.LongTensor(val_data[:, 1]).to(opt.device)
    preds = model.predict(user, item).to(opt.device)

    mae = MAE(preds, true)
    mse = MSE(preds, true)
    rmse = RMSE(preds, true)
    auc = AUC(true, preds.detach().cpu().numpy())

    return mae, mse, rmse, auc



seed_num = 2021
print("seed_num:", seed_num)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


setup_seed(seed_num)


def extract_data():
 # 将数据集下载到镜像本地，对应的本地目录分别是：
 # /cache/datasets/DatasetService/infer_recommendation_train
 # /cache/datasets/DatasetService/infer_valid
 # /cache/datasets/DatasetService/infer_test
    data_reference1 = get_data_reference(dataset="DatasetService", dataset_entity="infer_recommendation_train", enable_local_cache=True)
    data_reference2 = get_data_reference(dataset="DatasetService", dataset_entity="infer_valid", enable_local_cache=True)
    data_reference3 = get_data_reference(dataset="DatasetService", dataset_entity="infer_test", enable_local_cache=True)
 
    bigtag = np.loadtxt('/cache/datasets/DatasetService/infer_recommendation_train/bigtag.txt',dtype=int)
    choicetag = np.loadtxt('/cache/datasets/DatasetService/infer_recommendation_train/choicetag.txt',dtype=int)
    movie_data = np.loadtxt('/cache/datasets/DatasetService/infer_recommendation_train/movie.txt',dtype=int)
    movie = []
    for i in range(movie_data.shape[0]):
        tmp = movie_data[i,1:]
        movie.append(tmp)

    tag_num = np.max(movie)

    mat = np.zeros((1000,tag_num+1))
    all_data_array = []
    bigtag_array = []
    choicetag_array = []

    # extract deterministic data from bigtag
    for i in range(bigtag.shape[0]):
        if bigtag[i][2] != -1:
            mat[bigtag[i][0]][bigtag[i][2]] = 1
            all_data_array.append([bigtag[i][0],bigtag[i][2],1])
            bigtag_array.append([bigtag[i][0],bigtag[i][2],1])
        if bigtag[i][2] == -1:
            for tag in movie[bigtag[i][1]]:
                mat[bigtag[i][0]][tag] = -1
                all_data_array.append([bigtag[i][0],tag,0])
                bigtag_array.append([bigtag[i][0],tag,0])

 # extract deterministic data from choicetag
    for i in range(choicetag.shape[0]):
        if choicetag[i][2] != -1:
            mat[choicetag[i][0]][choicetag[i][2]] = 1
            all_data_array.append([choicetag[i][0],choicetag[i][2],1])
            choicetag_array.append([choicetag[i][0],choicetag[i][2],1])
        if choicetag[i][2] == -1:
            for tag in movie[choicetag[i][1]]:
                mat[choicetag[i][0]][tag] = -1
                all_data_array.append([choicetag[i][0],tag,0])
                choicetag_array.append([choicetag[i][0],tag,0])
    for i in range(choicetag.shape[0]):
        if choicetag[i][2] != -1:
            for tag in movie[choicetag[i][1]]:
                if mat[choicetag[i][0]][tag] == 0:
                    mat[choicetag[i][0]][tag] = -1
                    all_data_array.append([choicetag[i][0],tag,0])
                    choicetag_array.append([choicetag[i][0],tag,0])

 # Unique
    all_data_array = np.array(all_data_array)
    print(all_data_array.shape[0])
    print(np.count_nonzero(all_data_array[:,2]))
    all_data_array = [tuple(row) for row in all_data_array]
    all_data_array = np.unique(all_data_array, axis=0)
    print(all_data_array.shape[0])
    print(np.count_nonzero(all_data_array[:,2]))

 # Unique
    bigtag_array = np.array(bigtag_array)
    print(bigtag_array.shape[0])
    print(np.count_nonzero(bigtag_array[:,2]))
    bigtag_array = [tuple(row) for row in bigtag_array]
    bigtag_array = np.unique(bigtag_array, axis=0)
    print(bigtag_array.shape[0])
    print(np.count_nonzero(bigtag_array[:,2]))

 # Unique
    choicetag_array = np.array(choicetag_array)
    print(choicetag_array.shape[0])
    print(np.count_nonzero(choicetag_array[:,2]))
    choicetag_array = [tuple(row) for row in choicetag_array]
    choicetag_array = np.unique(choicetag_array, axis=0)
    print(choicetag_array.shape[0])
    print(np.count_nonzero(choicetag_array[:,2]))

    np.savetxt("/cache/extract_bigtag.txt",np.array(bigtag_array),fmt="%d")
    np.savetxt("/cache/extract_choicetag.txt",np.array(choicetag_array),fmt="%d")
    np.savetxt("/cache/extract_alldata.txt",np.array(all_data_array),fmt="%d")


def train():
    print('train begin')

    train_all_data = MF_DATA(opt.train_data)
    train_data = copy.deepcopy(train_all_data)
    val_data = MF_DATA(opt.val_all_data)

    train_dataloader = DataLoader(train_data, opt.batch_size, shuffle=True)

 # get model
    model = MF_Naive(train_all_data.users_num, train_all_data.items_num, opt.embedding_size, opt.device)
    model.to(opt.device)
    optimizer = model.get_optimizer(opt.lr, opt.weight_decay)

    best_mse = 10000000.
    best_mae = 10000000.
    best_auc = 0
    best_iter = 0

 # train
    model.train()
    for epoch in range(opt.max_epoch):
        t1 = time()
        for i, data in tqdm(enumerate(train_dataloader)):
            user = data[:, 0].to(opt.device)
            item = data[:, 1].to(opt.device)
            label = data[:, 2].to(opt.device)

            loss = model.calculate_loss(user.long(), item.long(), label.float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        t2 = time()

        (mae, mse, rmse, auc) = evaluate_model(model, val_data, opt)
        if opt.metric == 'mae':
            if mae < best_mae:
                best_mae, best_mse, best_auc, best_iter = mae, mse, auc, epoch
                torch.save(model.state_dict(), "./checkpoint/ci-mae-model.pth")
        elif opt.metric == 'mse':
            if mse < best_mse:
                best_mae, best_mse, best_auc, best_iter = mae, mse, auc, epoch
                torch.save(model.state_dict(), "/cache/ci-mse-model.pth")
        elif opt.metric == 'auc':
            if auc > best_auc:
                best_mae, best_mse, best_auc, best_iter = mae, mse, auc, epoch
                torch.save(model.state_dict(), "./checkpoint/ci-auc-model.pth")

        if epoch % opt.verbose == 0:
            print('Epoch %d [%.1f s]:', epoch, t2 - t1)
            print('Train Loss = ', loss.item())
            print('Val MAE = %.4f, MSE = %.4f, RMSE = %.4f, AUC = %.4f [%.1f s]' % (mae, mse, rmse, auc, time() - t2))
            print("------------------------------------------")

    print("train end\nBest Epoch %d:  MAE = %.4f, MSE = %.4f, AUC = %.4f" % (best_iter, best_mae, best_mse, best_auc))

    best_model = MF_Naive(train_all_data.users_num, train_all_data.items_num, opt.embedding_size, opt.device)
    best_model.to(opt.device)

    if opt.metric == 'mae':
        best_model.load_state_dict(torch.load("./checkpoint/ci-mae-model.pth"))
    elif opt.metric == 'mse':
        best_model.load_state_dict(torch.load("/cache/ci-mse-model.pth"))
    elif opt.metric == 'auc':
        best_model.load_state_dict(torch.load("./checkpoint/ci-auc-model.pth"))


    print("\n====================== best model ======================")
    mae, mse, rmse, auc = evaluate_model(best_model, train_data, opt)
    print('Train MAE = %.4f, MSE = %.4f, RMSE = %.4f, AUC = %.4f' % (mae, mse, rmse, auc))
    mae, mse, rmse, auc = evaluate_model(best_model, val_data, opt)
    print('Val MAE = %.4f, MSE = %.4f, RMSE = %.4f, AUC = %.4f' % (mae, mse, rmse, auc))
    print("=========================================================\n")

    return best_model


def generate_submit(model):
    test_data = np.loadtxt(opt.test_data, dtype=int)
    user = torch.LongTensor(test_data[:, 0]).to(opt.device)
    item = torch.LongTensor(test_data[:, 1]).to(opt.device)
    pred = model.predict(user, item).to(opt.device)
    pred = pred.detach().cpu().numpy()
 # normalize
    pred = (pred-np.min(pred))/(np.max(pred) - np.min(pred))
    pred = pred.reshape(-1,1)
    submit = np.hstack((test_data, pred))
    np.savetxt("/cache/submit.csv", submit, fmt = ('%d','%d','%f'))
 
 # 将结果保存到output目录，最后在比赛界面提交的时候选择对应的训练任务就可以
    mox.file.copy('/cache/submit.csv', os.path.join(Context.get_output_path(), 'submit.csv'))



if __name__ == '__main__':   
    Context.set("model", "MF_Naive")
    Context.set("batch_size", "512")
    Context.set("epoch", "50")
    Context.set("lr", "0.001")
    Context.set("metric", "mse")

    opt.model = Context.get("model")
    opt.batch_size = int(Context.get("batch_size"))
    opt.max_epoch = int(Context.get("epoch"))
    opt.lr = float(Context.get("lr"))
    opt.metric = Context.get("metric")

    print('\n'.join(['%s:%s' % item for item in opt.__dict__.items()]))

    extract_data()

    if opt.model == 'MF_Naive':
        best_model = train()
        generate_submit(best_model)

    print('end')