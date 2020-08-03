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

#print('contents:\n',contents,'\n')
#print('cites:\n',cites,'\n')
'''
contents:返回列表，列表中每个元素格式
paper_id \t feat \t label \n
其中：
feat：通过\t分割的1433个0或1，代表词汇存在与否

cites:列表，每个元素格式：
paper_id paper_id
'''
contents = np.array([np.array(l.strip().split("\t")) for l in contents])
#print('contents:\n',contents,'\n')
paper_list, feat_list, label_list = np.split(contents, [1,-1], axis=1)
#print('paper_list:\n',paper_list,'\n')
#print('feat_list:\n',feat_list,'\n')
#print('label_list:\n',label_list,'\n')
paper_list, label_list = np.squeeze(paper_list), np.squeeze(label_list)
#print('paper_list:\n',paper_list,'\n')
#print('label_list:\n',label_list,'\n')

# Paper -> Index dict
# paper_dict (paper_id ,val)
paper_dict = dict([(key, val) for val, key in enumerate(paper_list)])
#print('paper_dict:\n',paper_dict,'\n')
# Label -> Index 字典
# label_dict (label,val)
labels = list(set(label_list))
#print('labels:\n',labels,'\n')
#将其转变为集合，这样就没有重复的了
label_dict = dict([(key, val) for val, key in enumerate(labels)])

#得到paper序号：paper_dict和label序号：label_dict和特征向量:feat_list

#print('label_dict:\n',label_dict,'\n')
# Edge_index
cites = [i.strip().split("\t") for i in cites]
#print('cites:\n',cites,'\n')
cites = np.array([[paper_dict[i[0]], paper_dict[i[1]]] for i in cites], 
                 np.int64).T   # (2, edge) 这里转置了
#print('cites:\n',cites,'\n')
cites = np.concatenate((cites, cites[::-1, :]), axis=1)  # (2, 2*edge) or (2, E)
#concatenate连接函数，将引用上下颠倒，和原引用拼接到一起
#print('cites:\n',cites,'\n')
# Degree
#下面——，应该是得到了一个节点编号和度的向量

i_dont_care, degree_list = np.unique(cites[0,:], return_counts=True)
#print('i_dont_care:\n',i_dont_care,'\n')
#print('degree_list:\n',degree_list,'\n')
#去除重复的元素，并将元素按照从大到小排序

#得到边矩阵cites 和 节点序号的度degree_list

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
print('label_dict:\n',label_dict,'\n')
X_Node, X_Neis = torch.from_numpy(np.squeeze(X_Node)), \
                 torch.from_numpy(np.squeeze(X_Neis))
dg_list = degree_list[X_Node]
#print('label_list:\n',label_list,'\n')
label_list = np.array([label_dict[i] for i in label_list])
#print('label_list:\n',label_list,'\n',label_list.shape,'\n')
label_list = torch.from_numpy(label_list)
#print('label_list:\n',label_list,'\n',label_list.shape,'\n')

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
        #这里不同，将直接将ln和s作为隐藏层的内容
        '''
        '''
        下面的有些变动，为之前的代码：
        下面的由于有节点特征向量所以不用初始化，而且为了防止过拟合用了dropout
        # 初始化节点的embedding，即节点特征向量 (V, ln)
        self.node_features = nn.Embedding(node_num, feat_dim)
        
        之前数据里面已经有状态向量了，骗人，没有初始化
        # 初始化节点的状态向量 (V, s)
        self.node_states = torch.zeros((node_num, stat_dim))
        
        # 这次输出层有点奇怪，直接将状态作为输出层，不对，是将特征和状态作为中间层
        # 输出层
        self.linear = nn.Linear(feat_dim+stat_dim, 3)
        self.softmax = nn.Softmax(dim = 1)
        
        
        '''
        self.out_layer = nn.Linear(stat_dim, num_class)
        '''
        nn.Dropout()函数是用来防止或者减轻过拟合而使用的函数，一般用在全连接层
        全连接层：每个节点都与上一层的所有的节点相连接，用来把前面提取到的特征综合起来
        其参数最多
        '''
        self.dropout = nn.Dropout()
        '''

        log_softmax = log(softmax)
        虽然在数学上等价，但是单独做这两个操作速度较慢，而且数值不稳定
        这个函数用另一种公式来正确计算梯度。
        
        crossentropyloss和nullloss:
        NLLLoss的输入是一个对数概率向量和一个目标标签，
        适合网络的最后一层为log_softmax.
        crossentropyloss
        '''
        self.log_softmax = nn.LogSoftmax(dim=-1)
        
        
        '''
        上面的有修改
        '''        
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
    #print('out[train_mask]',type(out[train_mask].type(torch.LongTensor)),'\n')
    #print('label_list[train_mask]',type(label_list[train_mask].type(torch.LongTensor)),'\n')
    #loss = F.nll_loss(out[train_mask], label_list[train_mask])
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
        _, pred = model(feat_Matrix, X_Node, X_Neis, dg_list).max(dim=1)
        correct = float(pred[test_mask].eq(label_list[test_mask]).sum().item())
        acc = correct / test_mask.sum().item()
        print('Accuracy: {:.4f}'.format(acc))
