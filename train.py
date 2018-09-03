import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import utlis
import datasetv2
import CDNet
from tensorboardX import SummaryWriter
import time

train_data_dir = '../../data-v2/20180615/alldata/train'
labelencoder_dir = '../ctrad/labelencoder.pkl'

train_data = datasetv2.DataSetV2(train_data_dir, labelencoder_dir, True)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=8192, num_workers=1, drop_last=False)

gpu_device = torch.device("cuda")
cpu_device = torch.device("cpu")

def main():
    # instantiate model and initialize weights
    model = CDNet.CDNet(embedding_index=[0, 1, 2, 3, 4, 5, 6, [7, 8, 9]], embedding_size=[13, 3, 1001, 2, 10, 10, 7, 11], dense_feature_num=75-6, cross_layer_num=2,deep_layer=[256, 128, 32])
    model = model.to(gpu_device)

    optimizer = optim.SGD(model.parameters(), lr=0.0003, momentum=0.9)

    criterion = nn.BCELoss()

    logloss = []

    summary = SummaryWriter('/cephfs/group/teg-qboss-teg-qboss-ocr-shixi/elijahzhang/log/cdnet/'+time.strftime("%Y-%m-%d", time.localtime()))
    
    epoch_num = 5

    iter_num = 0

    for epoch in range(epoch_num):
        print('starit epoch [{}/{}]'.format(epoch + 1, 5))
        model.train()
        for sparse_feature, dense_feature, label in train_loader:
            iter_num += 1
            begin_time = time.time()
            sparse_feature, dense_feature, label = sparse_feature.to(gpu_device), dense_feature.to(gpu_device), label.to(gpu_device)
            pctr = model(sparse_feature, dense_feature)
            loss = criterion(pctr, label)
            iter_loss = loss.item()
            logloss.append(iter_loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            end_time = time.time()
            print("epoch {}/{}, total_iter is {}, logloss is {:.2f}, cost time is {:.2f}s".format(epoch+1, epoch_num, iter_num, iter_loss, end_time-begin_time))
            if iter_num % 20 == 0:
                total_loss = np.mean(logloss)
                logloss = []
                summary.add_scalar('logloss', total_loss, iter_num)
            if iter_num % 2000 ==  0:
                save_dir = '../../model/cdnet'+str(iter_num)+'.pkl'
                torch.save(model.state_dict(), save_dir)
                auc_score, bias_score = test()
                summary.add_scalar('auc', auc_score, iter_num)
                summary.add_scalar('bias', bias_score, iter_num)
                model.train()

if __name__ == '__main__':
    main()
