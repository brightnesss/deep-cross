import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import utlis
import math
import pdb


class DeepNet(nn.Module):
    """
    Deep part of Cross and Deep Network
    All of the layer in this module are full-connection layers
    """

    def __init__(self, input_feature_num, deep_layer: list):
        """
        :param input_feature_num: total num of input_feature, including of the embedding feature and dense feature
        :param deep_layer: a list contains the num of each hidden layer's units
        """
        super(DeepNet, self).__init__()
        fc_layer_list = []
        fc_layer_list.append(nn.Linear(input_feature_num, deep_layer[0]))
        fc_layer_list.append(nn.BatchNorm1d(deep_layer[0], affine=False))
        fc_layer_list.append(nn.ReLU(inplace=True))
        for i in range(1, len(deep_layer)):
            fc_layer_list.append(nn.Linear(deep_layer[i - 1], deep_layer[i]))
            fc_layer_list.append(nn.BatchNorm1d(deep_layer[i], affine=False))
            fc_layer_list.append(nn.ReLU(inplace=True))
        self.deep = nn.Sequential(*fc_layer_list)

    def forward(self, x):
        dense_output = self.deep(x)
        return dense_output


class CrossNet(nn.Module):
    """
    Cross layer part in Cross and Deep Network
    The ops in this module is x_0 * x_l^T * w_l + x_l + b_l for each layer l, and x_0 is the init input of this module
    """

    def __init__(self, input_feature_num, cross_layer: int):
        """
        :param input_feature_num: total num of input_feature, including of the embedding feature and dense feature
        :param cross_layer: the number of layer in this module expect of init op
        """
        super(CrossNet, self).__init__()
        self.cross_layer = cross_layer + 1  # add the first calculate
        weight_w = []
        weight_b = []
        batchnorm = []
        for i in range(self.cross_layer):
            weight_w.append(nn.Parameter(torch.nn.init.normal_(torch.empty(input_feature_num))))
            weight_b.append(nn.Parameter(torch.nn.init.normal_(torch.empty(input_feature_num))))
            batchnorm.append(nn.BatchNorm1d(input_feature_num, affine=False))
        self.weight_w = nn.ParameterList(weight_w)
        self.weight_b = nn.ParameterList(weight_b)
        self.batchnorm = nn.ModuleList(batchnorm)

    def forward(self, x):
        output = x
        x = x.reshape(x.shape[0], -1, 1)
        for i in range(self.cross_layer):
            output = torch.matmul(torch.bmm(x, torch.transpose(output.reshape(output.shape[0], -1, 1), 1, 2)),self.weight_w[i]) + self.weight_b[i] + output
            output = self.batchnorm[i](output)
        return output


class CDNet(nn.Module):
    """
    Cross and Deep Network in Deep & Cross Network for Ad Click Predictions
    """

    def __init__(self, embedding_index: list, embedding_size: list, dense_feature_num: int, cross_layer_num: int,
                 deep_layer: list):
        """
        :param embedding_index: a list to show the index of the embedding_feature.
                                [0,1,2] shows that index 0, index 1 and index 2 are different category feature
                                [0,[1,2]] shows that index 0 is a category feature and [1,2] is another category feature
        :param embedding_size: a list to show the num of classes for each category feature
        :param dense_feature_num: the dim of dense feature
        :param cross_layer_num: the num of cross layer in CrossNet
        :param deep_layer: a list contains the num of each hidden layer's units
        """
        super(CDNet, self).__init__()
        if len(embedding_index) != len(embedding_size):
            raise ValueError("embedding_index length is {}, embedding_size lenght is {} and they two must have same length")
        self.embedding_index = embedding_index
        self.embedding_size = embedding_size
        # For categorical features, we embed the features in dense vectors of dimension of 6 * category cardinality^1/4
        embedding_num = list(map(lambda x: int(6 * pow(x, 0.25)), self.embedding_size))
        input_feature_num = np.sum(embedding_num) + dense_feature_num
        embedding_list = []
        for i in range(len(embedding_size)):
            embedding_list.append(nn.Embedding(embedding_size[i], embedding_num[i], scale_grad_by_freq=True))
        self.embedding_layer = nn.ModuleList(embedding_list)
        self.batchnorm = nn.BatchNorm1d(input_feature_num, affine=False)
        self.CrossNet = CrossNet(input_feature_num, cross_layer_num)
        self.DeepNet = DeepNet(input_feature_num, deep_layer)
        last_layer_feature_num = input_feature_num + deep_layer[-1]  # the dim of feature in last layer
        self.output_layer = nn.Linear(last_layer_feature_num, 1)  # 0, 1 classification

    def forward(self, sparse_feature, dense_feature):
        num_sample = sparse_feature.shape[0]
        if isinstance(self.embedding_index[0], list):
            embedding_feature = torch.mean(self.embedding_layer[0](sparse_feature[:, self.embedding_index[0]].to(torch.long)), dim=1)
        else:
            embedding_feature = torch.mean(self.embedding_layer[0](sparse_feature[:, self.embedding_index[0]].to(torch.long).reshape(num_sample, 1)), dim=1)
        for i in range(1, len(self.embedding_index)):
            if isinstance(self.embedding_index[i], list):
                embedding_feature = torch.cat((embedding_feature, torch.mean(self.embedding_layer[i](sparse_feature[:, self.embedding_index[i]].to(torch.long)), dim=1)), dim=1)
            else:
                embedding_feature = torch.cat((embedding_feature, torch.mean(self.embedding_layer[i](sparse_feature[:, self.embedding_index[i]].to(torch.long).reshape(num_sample, 1)), dim=1)), dim=1)
        input_feature = torch.cat((embedding_feature, dense_feature), 1)
        input_feature = self.batchnorm(input_feature)
        out_cross = self.CrossNet(input_feature)
        out_deep = self.DeepNet(input_feature)
        final_feature = torch.cat((out_cross, out_deep), dim=1)
        pctr = self.output_layer(final_feature)
        pctr = pctr.view(-1)
        pctr = torch.sigmoid(pctr)
        return pctr

