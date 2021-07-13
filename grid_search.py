# -*- encoding: utf-8 -*-
'''
@File    :   grid_search.py
@Time    :   2021/07/13 21:15:13
@Author  :   Liu Qidong
@Version :   1.0
@Contact :   dong_liuqi@163.com
'''

# here put the import lib
from main import train
from utils import get_propensity_score
from config import opt
from generator import DataGenerator
import os
import logging


if __name__ == '__main__':

    ps_model = get_propensity_score()

    opt.model = 'MF_IPS'

    log_path = r'./log/' + opt.model + '/'
    if not os.path.exists(log_path):
        os.mkdir(log_path)

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    sh = logging.StreamHandler()
    fh = logging.FileHandler(filename=log_path+'log.txt')
    logger.addHandler(sh)
    logger.addHandler(fh)

    l = []

    best_model = {'auc': 0}
    for bs in [16, 32, 64, 128]:
        for lr in [0.001, 0.0005, 0.0001]:
            for em in [8, 16, 32]:
                opt.batch_size = bs
                opt.lr = lr
                opt.embedding_size = em
                DG = DataGenerator(opt, ps_model)
                _, auc, best_iter = train(DG, opt)
                logger.info(str({'batch_size': bs, 'lr': lr, 'embedding_size': em, 'best_iter': best_iter, 'auc': auc}))
                if auc > best_model['auc']:
                    best_model['auc'] = auc
                    best_model['batch_size'] = bs
                    best_model['lr'] = lr
                    best_model['embedding_size'] = em

    print(str(best_model))





