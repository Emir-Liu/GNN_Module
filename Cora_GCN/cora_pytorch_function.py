# -*- coding: utf-8 -*-
"""
Created on Sat Jul 25 10:36:12 2020
Cora数据集：
机器学习领域的paper,被分为7个类别：
+ Case_Based
+ Genetic_Algorithms
+ Neural_Networks
+ Probabilistic_Methods
+ Reinforcement_Learning
+ Rule_Learning
+ Theory

连通图
2708paper (vertex)
1433关键词 (feat)
5429边    (edge)

@author: EGlym
"""
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F


'''
node_num, feat_dim, stat_dim, num_class, T
feat_Matrix, X_Node, X_Neis, dg_list
'''
content_path = "./cora/cora.content"
cite_path = "./cora/cora.cites"

# 读取文本内容
with open(content_path, "r") as fp:
    contents = fp.readlines()
with open(cite_path, "r") as fp:
    cites = fp.readlines()

'''
contents:返回列表，列表中每个元素格式
paper_id \t feat \t label \n
其中：
feat：通过\t分割的1433个0或1，代表词汇存在与否

cites:列表，每个元素格式：
paper_id paper_id
'''
contents = np.array([np.array(l.strip().split("\t")) for l in contents])
paper_list, feat_list, label_list = np.split(contents, [1,-1], axis=1)
paper_list, label_list = np.squeeze(paper_list), np.squeeze(label_list)
# Paper -> Index dict
# paper_dict (paper_id ,val)
paper_dict = dict([(key, val) for val, key in enumerate(paper_list)])

# Label -> Index 字典
# label_dict (label,val)
labels = list(set(label_list))

#将其转变为集合，这样就没有重复的了
label_dict = dict([(key, val) for val, key in enumerate(labels)])

# Edge_index
cites = [i.strip().split("\t") for i in cites]
cites = np.array([[paper_dict[i[0]], paper_dict[i[1]]] for i in cites], 
                 np.int64).T   # (2, edge)

cites = np.concatenate((cites, cites[::-1, :]), axis=1)  # (2, 2*edge) or (2, E)
#concatenate连接函数，将引用上下颠倒，和原引用拼接到一起
# Degree
#下面——，得到了一个节点编号和度的向量
_, degree_list = np.unique(cites[0,:], return_counts=True)
#去除重复的元素，并将元素按照从大到小排序

# Input
node_num = len(paper_list)
feat_dim = feat_list.shape[1]
stat_dim = 32
num_class = len(labels)
T = 2
#状态迭代2次
feat_Matrix = torch.Tensor(feat_list.astype(np.float32))
#改变特征向量的元素类型，形成矩阵
X_Node, X_Neis = np.split(cites, 2, axis=0)
#X_Node:一条边的源节点的序号
#X_Neis:一条边的终节点的序号
X_Node, X_Neis = torch.from_numpy(np.squeeze(X_Node)), \
                 torch.from_numpy(np.squeeze(X_Neis))
dg_list = degree_list[X_Node]
label_list = np.array([label_dict[i] for i in label_list])
label_list = torch.from_numpy(label_list)

print("{}Data Process Info{}".format("*"*20, "*"*20))
print("==> Number of node : {}".format(node_num))
print("==> Number of edges : {}/2={}".format(cites.shape[1], int(cites.shape[1]/2)))
print("==> Number of classes : {}".format(num_class))
print("==> Dimension of node features : {}".format(feat_dim))
print("==> Dimension of node state : {}".format(stat_dim))
print("==> T : {}".format(T))
print("==> Shape of feat_Matrix : {}".format(feat_Matrix.shape))
print("==> Shape of X_Node : {}".format(X_Node.shape))
print("==> Shape of X_Neis : {}".format(X_Neis.shape))
print("==> Length of dg_list : {}".format(len(dg_list)))
'''
上面的数据处理部分:
    
首先得到paper序号：paper_dict和label序号：label_dict和特征向量:feat_list
然后计算边的连接矩阵cites 和节点的度向量degree_list
然后，将连接矩阵分为两个源节点和终节点向量X_Node，X_Neis，以及对应的度向量dg_list

将上面的都转换为Tensor形式
'''
# Split dataset
train_mask = torch.zeros(node_num, dtype=torch.uint8).bool()
train_mask[:node_num - 1000] = 1                  # 1700左右training
val_mask = None                                    # 0valid
test_mask = torch.zeros(node_num, dtype=torch.uint8).bool()
test_mask[node_num - 500:] = 1                    # 500test
x = feat_Matrix
edge_index = torch.from_numpy(cites)

'''
同Linear GNN
'''
class AggrSum(nn.Module):
    def __init__(self, node_num):
        super(AggrSum, self).__init__()
        self.V = node_num
    
    def forward(self, H, X_node):
        # H : (N, s) -> (V, s)
        # X_node : (N, )
        mask = torch.stack([X_node] * self.V, 0)
        mask = mask.float() - torch.unsqueeze(torch.arange(0,self.V).float(), 1)
        mask = (mask == 0).float()
        # (V, N) * (N, s) -> (V, s)
        return torch.mm(mask, H)

'''
用于实现GCN的卷积块。
Initialize :
Input :
    in_channel : (int)输入的节点特征维度
    out_channel : (int)输出的节点特征维度
Forward :
Input :
    x : (Tensor)节点的特征矩阵，shape为(N, in_channel)，N为节点个数
    edge_index : (Tensor)边矩阵，shape为(2, E)，E为边个数。
Output :
    out : (Tensor)新的特征矩阵，shape为(N, out_channel)
'''
class GCNConv(nn.Module):
    def __init__(self, in_channel, out_channel, node_num):
        super(GCNConv, self).__init__()
        self.linear = nn.Linear(in_channel, out_channel)
        self.aggregation = AggrSum(node_num)
        
    def forward(self, x, edge_index):
        # Add self-connect edges
        edge_index = self.addSelfConnect(edge_index, x.shape[0])
        
        # Apply linear transform
        x = self.linear(x)
        
        # Normalize message
        row, col = edge_index
        deg = self.calDegree(row, x.shape[0]).float()
        deg_sqrt = deg.pow(-0.5)  # (N, )
        norm = deg_sqrt[row] * deg_sqrt[col]
        
        # Node feature matrix
        tar_matrix = torch.index_select(x, dim=0, index=col)
        tar_matrix = norm.view(-1, 1) * tar_matrix  # (E, out_channel)
        # Aggregate information
        aggr =  self.aggregation(tar_matrix, row)  # (N, out_channel)
        return aggr
        
    
    def calDegree(self, edges, num_nodes):
        ind, deg = np.unique(edges.cpu().numpy(), return_counts=True)
        deg_tensor = torch.zeros((num_nodes, ), dtype=torch.long)
        deg_tensor[ind] = torch.from_numpy(deg)
        return deg_tensor.to(edges.device)
    
    def addSelfConnect(self, edge_index, num_nodes):
        selfconn = torch.stack([torch.arange(0, num_nodes, dtype=torch.long)]*2,
                               dim=0).to(edge_index.device)
        return torch.cat(tensors=[edge_index, selfconn],
                         dim=1)
    
        
'''
构建模型，使用两层GCN，第一层GCN使得节点特征矩阵
    (N, in_channel) -> (N, out_channel)
第二层GCN直接输出
    (N, out_channel) -> (N, num_class)
激活函数使用relu函数，网络最后对节点的各个类别score使用softmax归一化。
'''
class Net(nn.Module):
    def __init__(self, feat_dim, num_class, num_node):
        super(Net, self).__init__()
        self.conv1 = GCNConv(feat_dim, 16, num_node)
        self.conv2 = GCNConv(16, num_class, num_node)
    
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        
        return F.log_softmax(x, dim=1)

'''
开始训练模型
'''
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net(feat_dim, num_class, node_num).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
x = x.to(device)
edge_index = edge_index.to(device)

for epoch in range(200):
    model.train()
    optimizer.zero_grad()
    
    # Get output
    out = model(x, edge_index)
    
    # Get loss
    loss = F.nll_loss(out[train_mask], label_list[train_mask].type(torch.LongTensor))
    _, pred = out.max(dim=1)
    
    # Get predictions and calculate training accuracy
    correct = float(pred[train_mask].eq(label_list[train_mask]).sum().item())
    acc = correct / train_mask.sum().item()
    print('[Epoch {}/200] Loss {:.4f}, train acc {:.4f}'.format(epoch, loss.cpu().detach().data.item(), acc))
    
    # Backward
    loss.backward()
    optimizer.step()
    
    # Evaluation on test data every 10 epochs
    if (epoch+1) % 10 == 0:
        model.eval()
        _, pred = model(x, edge_index).max(dim=1)
        correct = float(pred[test_mask].eq(label_list[test_mask]).sum().item())
        acc = correct / test_mask.sum().item()
        print('Accuracy: {:.4f}'.format(acc))
