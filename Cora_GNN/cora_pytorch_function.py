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

'''
实现论文中的Xi函数，作为Hw函数的转换矩阵A，根据节点对(i,j)的特征向量
生成A矩阵，其中ln是特征向量维度，s为状态向量维度。
Initialization :
Input :
    ln : (int)特征向量维度
    s : (int)状态向量维度
Forward :
Input :
    x : (Tensor)节点对(i,j)的特征向量拼接起来，shape为(N, 2*ln)
Output :
    out : (Tensor)A矩阵，shape为(N, s, s)
'''
class Xi(nn.Module):
    def __init__(self, ln, s):
        super(Xi, self).__init__()
        self.ln = ln   # 节点特征向量的维度
        self.s = s     # 节点的个数
        
        # 线性网络层
        self.linear = nn.Linear(in_features=2 * ln,
                                out_features=s ** 2,
                                bias=True)
        # 激活函数
        self.tanh = nn.Tanh()
        
    def forward(self, X):
        bs = X.size()[0]
        out = self.linear(X)
        out = self.tanh(out)
        return out.view(bs, self.s, self.s)


'''
实现论文中的Rou函数，作为Hw函数的偏置项b
Initialization :
Input :
    ln : (int)特征向量维度
    s : (int)状态向量维度
Forward :
Input :
    x : (Tensor)节点的特征向量矩阵，shape(N, ln)
Output :
    out : (Tensor)偏置矩阵，shape(N, s)
'''
class Rou(nn.Module):
    def __init__(self, ln, s):
        super(Rou, self).__init__()
        self.linear = nn.Linear(in_features=ln,
                                out_features=s,
                                bias=True)
        self.tanh = nn.Tanh()
    def forward(self, X):
        return self.tanh(self.linear(X))

'''
实现Hw函数，即信息生成函数
Initialize :
Input :
    ln : (int)节点特征向量维度
    s : (int)节点状态向量维度
    mu : (int)设定的压缩映射的压缩系数
Forward :
Input :
    X : (Tensor)每一行为一条边的两个节点特征向量连接起来得到的向量，shape为(N, 2*ln)
    H : (Tensor)与X每行对应的source节点的状态向量
    dg_list : (list or Tensor)与X每行对应的source节点的度向量
Output :
    out : (Tensor)Hw函数的输出
'''
class Hw(nn.Module):
    def __init__(self, ln, s, mu=0.9):
        super(Hw, self).__init__()
        self.ln = ln
        self.s = s
        self.mu = mu
        
        # 初始化网络层
        self.Xi = Xi(ln, s)
        self.Rou = Rou(ln, s)
    
    def forward(self, X, H, dg_list):
        if isinstance(dg_list, list) or isinstance(dg_list, np.ndarray):
            dg_list = torch.Tensor(dg_list).to(X.device)
        elif isinstance(dg_list, torch.Tensor):
            pass
        else:
            raise TypeError("==> dg_list should be list or tensor, not {}".format(type(dg_list)))
        A = (self.Xi(X) * self.mu / self.s) / dg_list.view(-1, 1, 1)# (N, S, S)
        b = self.Rou(torch.chunk(X, chunks=2, dim=1)[0])# (N, S)
        out = torch.squeeze(torch.matmul(A, torch.unsqueeze(H, 2)),-1) + b  # (N, s, s) * (N, s) + (N, s)
        return out    # (N, s)
    
'''
实现信息聚合函数，将前面使用Hw函数得到的信息按照每一个source节点进行聚合，
之后用于更新每一个节点的状态向量。
Initialize :
Input :
    node_num : (int)节点的数量
Forward :
Input :
    H : (Tensor)Hw的输出，shape为(N, s)
    X_node : (Tensor)H每一行对应source节点的索引，shape为(N, )
Output :
    out : (Tensor)求和式聚合之后的新的节点状态向量，shape为(V, s)，V为节点个数
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
实现Linear GNN模型，循环迭代计算T次，达到不动点之后，使用线性函数得到输出，进行
分类。
Initialize :
Input :
    node_num : (int)节点个数
    feat_dim : (int)节点特征向量维度
    stat_dim : (int)节点状态向量维度
    T : (int)迭代计算的次数
Forward :
Input :
    feat_Matrix : (Tensor)节点的特征矩阵，shape为(V, ln)
    X_Node : (Tensor)每条边的source节点对应的索引，shape为(N, )，比如`节点i->节点j`，source节点是`节点i`
    X_Neis : (Tensor)每条边的target节点对应的索引，shape为(N, )，比如`节点i->节点j`，target节点是`节点j`
    dg_list : (list or Tensor)与X_Node对应节点的度列表，shape为(N, )
Output :
    out : (Tensor)每个节点的类别概率，shape为(V, num_class)
'''
class OriLinearGNN(nn.Module):
    def __init__(self, node_num, feat_dim, stat_dim, num_class, T):
        super(OriLinearGNN, self).__init__()
        self.embed_dim = feat_dim
        self.stat_dim = stat_dim
        self.T = T
        # 输出层
        '''
        self.out_layer = nn.Sequential(
            nn.Linear(stat_dim, 16),   # ln+s -> hidden_layer
            nn.Tanh(),
            nn.Dropout(p=0.5),
            nn.Linear(16, num_class)   # hidden_layer -> logits
        )
        '''
        self.out_layer = nn.Linear(stat_dim, num_class)
        self.dropout = nn.Dropout()
        self.log_softmax = nn.LogSoftmax(dim=-1)
        # 实现Fw
        self.Hw = Hw(feat_dim, stat_dim)
        # 实现H的分组求和
        self.Aggr = AggrSum(node_num)
        
    def forward(self, feat_Matrix, X_Node, X_Neis, dg_list):
        node_embeds = torch.index_select(input=feat_Matrix,
                                         dim=0,
                                         index=X_Node)  # (N, ln)
        neis_embeds = torch.index_select(input=feat_Matrix,
                                         dim=0,
                                         index=X_Neis)  # (N, ln)
        X = torch.cat((node_embeds, neis_embeds), 1)    # (N, 2 * ln)
        H = torch.zeros((feat_Matrix.shape[0], self.stat_dim), dtype=torch.float32)  # (V, s)
        H = H.to(feat_Matrix.device)
        # 循环T次计算
        for t in range(self.T):
            # (V, s) -> (N, s)
            H = torch.index_select(H, 0, X_Node)
            # (N, s) -> (N, s)
            H = self.Hw(X, H, dg_list)
            # (N, s) -> (V, s)
            H = self.Aggr(H, X_Node)
            # print(H[1])
        # out = torch.cat((feat_Matrix, H), 1)   # (V, ln+s)
        out = self.log_softmax(self.dropout(self.out_layer(H)))
        return out  # (V, num_class)
    
    
# Split dataset
train_mask = torch.zeros(node_num, dtype=torch.uint8).bool()
train_mask[:node_num - 1000] = 1                  # 1700左右training
val_mask = None                                    # 0valid
test_mask = torch.zeros(node_num, dtype=torch.uint8).bool()
test_mask[node_num - 500:] = 1                    # 500test

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = OriLinearGNN(node_num, feat_dim, stat_dim, num_class, T).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-3)
feat_Matrix = feat_Matrix.to(device)
X_Node = X_Node.to(device)
X_Neis = X_Neis.to(device)
train_loss_list= []
train_acc_list = []
test_acc_list  = []
for epoch in range(200):
    model.train()
    optimizer.zero_grad()
    
    # Get output
    out = model(feat_Matrix, X_Node, X_Neis, dg_list)
    '''
    train_data = label_list[train_mask].long()
    pre = out[train_mask].long()
    loss = F.nll_loss(out[train_mask].long(), label_list[train_mask].long())
    '''
    # Get loss
    loss = F.nll_loss(out[train_mask], label_list[train_mask].type(torch.LongTensor))
    _, pred = out.max(dim=1)
    
    # Get predictions and calculate training accuracy
    correct = float(pred[train_mask].eq(label_list[train_mask]).sum().item())
    acc = correct / train_mask.sum().item()
    print('[Epoch {}/200] Loss {:.4f}, train acc {:.4f}'.format(epoch, loss.cpu().detach().data.item(), acc))
    
    train_loss_list.append(loss.cpu().detach().data.item())
    train_acc_list.append(acc)
    # Backward
    loss.backward()
    optimizer.step()
    
    # Evaluation on test data every 10 epochs
    if (epoch+1) % 10 == 0:
        model.eval()
        _, pred = model(feat_Matrix, X_Node, X_Neis, dg_list).max(dim=1)
        correct = float(pred[test_mask].eq(label_list[test_mask]).sum().item())
        acc = correct / test_mask.sum().item()
        print('Accuracy: {:.4f}'.format(acc))
        test_acc_list.append(acc)

        
epoch = len(train_loss_list)
# 画出train_loss
plt.subplot(3,1,1)
plt.plot(range(epoch), train_loss_list)
plt.xlim((0,epoch))    # 设置x轴的范围
plt.ylabel('loss')      # 设置y周标签
plt.yticks(np.arange(0,1.5,0.2))  # 设置y轴刻度

# 画出train_acc
plt.subplot(3,1,2)
plt.plot(range(epoch), train_acc_list)
plt.xlim((0,epoch))
plt.ylim((0,1))
plt.ylabel('train_acc')
plt.yticks(np.arange(0,1,0.1))

# 画出test_acc
epoch = len(test_acc_list)
plt.subplot(3,1,3)
plt.plot(range(epoch), test_acc_list)
plt.xlim((0,epoch))
plt.ylim((0,1))
plt.xlabel('epoch')
plt.ylabel('test_acc')
plt.yticks(np.arange(0,1,0.1))
plt.show()
# 保存图片
# plt.savefig("./images/result.png")
    
    
    
    
    