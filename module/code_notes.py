import os
import json
#import scipy
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
'''
V 节点的数量
N 边的数量乘以2，节点的度之和
X 节点的特征向量
H 节点的状态向量，初始化状态向量为0

使用T时刻节点的状态向量和特征向量来得到节点的输出
T 对模型中节点的状态向量进行迭代更新的次数
ln 特征向量维度
s 状态向量维度

Hw函数，使用了线性模型，需要Xi和Rou函数来构成A和b
mu : (int)设定的压缩映射的压缩系数

'''
torch.set_printoptions(precision=None, threshold=None, edgeitems=30, linewidth=30000, profile=None)
# 数据输入
# (node, label)数据集N和边集E
N = [("n{}".format(i), 0) for i in range(1,7)] + \
    [("n{}".format(i), 1) for i in range(7,13)] + \
    [("n{}".format(i), 2) for i in range(13,19)]
E = [("n1","n2"), ("n1","n3"), ("n1","n5"),
     ("n2","n4"),
     ("n3","n6"), ("n3","n9"),
     ("n4","n5"), ("n4","n6"), ("n4","n8"),
     ("n5","n14"),
     ("n7","n8"), ("n7","n9"), ("n7","n11"),
     ("n8","n10"), ("n8","n11"), ("n8", "n12"),
     ("n9","n10"), ("n9","n14"),
     ("n10","n12"),
     ("n11","n18"),
     ("n13","n15"), ("n13","n16"), ("n13","n18"),
     ("n14","n16"), ("n14","n18"),
     ("n15","n16"), ("n15","n18"),
     ("n17","n18")]


'''
node_list 节点名称的列表
label_list 节点分类的列表
#print('node_list:',node_list,'\n')
#print('label_list:',label_list,'\n')
node_list: ['n1', 'n2', 'n3', 'n4', 'n5', 'n6', 'n7', 'n8', 'n9', 'n10', 'n11', 'n12', 'n13', 'n14', 'n15', 'n16', 'n17', 'n18'] 
label_list: [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2] 
'''
node_list = list(map(lambda x:x[0],N))
label_list= list(map(lambda x:x[1],N))

# 图显示
G = nx.Graph()
G.add_nodes_from(node_list)
G.add_edges_from(E)

ncolor = ['r'] * 6 + ['b'] * 6 + ['g'] * 6
nsize = [700] * 6 + [700] * 6 + [700] * 6
plt.figure(1)
nx.draw(G, with_labels=True, font_weight='bold', 
        node_color=ncolor, node_size=nsize)
# plt.savefig("./images/graph.png")

'''
实现论文中的Xi函数，作为Hw函数的转换矩阵A，根据节点对(i,j)的特征向量生成A矩阵。
Forward :
Input :
    x : (Tensor)节点对(i,j)的特征向量拼接起来，shape为(N, 2*ln)
Output :
    out : (Tensor)A矩阵，shape为(N, s, s)
'''
class Xi(nn.Module):
    with torch.autograd.set_detect_anomaly(True):
            
        def __init__(self, ln, s):
            super(Xi, self).__init__()
            self.ln = ln   
            self.s = s     
            
            # 线性网络层
            '''
            输入:一条边的两个节点对的特征向量，维度为2*ln
            输出:为状态向量维度s**2
            '''
            self.linear = nn.Linear(in_features=2 * ln,
                                    out_features=s ** 2,
                                    bias=True)
            # 激活函数
            '''
            激活函数为nn.Tanh()双曲正切曲线
            Tanh(x)= (exp(x)-exp(-x))/(exp(x)+exp(-x))
            '''
            self.tanh = nn.Tanh()
    
            '''
            将X的第一行进行线性变换和激活函数之后，将其转换格式
            '''
        def forward(self, X):
            bs = X.size()[0]
            out = self.linear(X)
            out = self.tanh(out)
            return out.view(bs, self.s, self.s)

'''
实现论文中的Rou函数，作为Hw函数的偏置项b
Forward :
Input :
    x : (Tensor)节点的特征向量矩阵，shape(N, ln)
Output :
    out : (Tensor)偏置矩阵，shape(N, s)
'''
class Rou(nn.Module):
    with torch.autograd.set_detect_anomaly(True):
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
Forward :
Input :
    X : (Tensor)每一行为一条边的两个节点特征向量连接起来得到的向量，shape为(N, 2*ln)
    H : (Tensor)与X每行对应的source节点的状态向量
    dg_list : (list or Tensor)与X每行对应的source节点的度向量
Output :
    out : (Tensor)Hw函数的输出
'''
class Hw(nn.Module):
    with torch.autograd.set_detect_anomaly(True):
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
            out = torch.squeeze(torch.matmul(A, torch.unsqueeze(H, 2)),-1) + b # (N, s, s) * (N, s) + (N, s)
            
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
#H = self.Hw(X, H, dg_list) # (N, s) -> (N, s)
class AggrSum(nn.Module):
    with torch.autograd.set_detect_anomaly(True):
        def __init__(self, node_num):
            super(AggrSum, self).__init__()
            self.V = node_num

        def forward(self, H, X_node):
            #print('H:\n',H,'\n')
            #print('X_node',X_node,'\n')
            # H : (N, s) -> (V, s)
            # X_node : (N, )
            mask = torch.stack([X_node] * self.V, 0)
            #print(mask,'\n')
            mask = mask.float() - torch.unsqueeze(torch.arange(0,self.V).float(), 1)
            #print('torch.arange(0,self.V).float():\n',torch.arange(0,self.V).float(),'\n')
            #print('torch.unsqueeze(torch.arange(0,self.V).float(), 1)',torch.unsqueeze(torch.arange(0,self.V).float(), 1),'\n')
            #print(mask,'\n')
            mask = (mask == 0).float()
            #print(mask,'\n')
            #print('H:\n',H,'\n')
            #print('Node_state:\n',torch.mm(mask, H),'\n')
    
            # (V, N) * (N, s) -> (V, s)
            return torch.mm(mask, H)

'''
实现GNN模型
Initialize :
Input :
    node_num : (int)节点个数
    feat_dim : (int)节点特征向量维度
    stat_dim : (int)节点状态向量维度
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
    with torch.autograd.set_detect_anomaly(True):
        def __init__(self, node_num, feat_dim, stat_dim, T):
            super().__init__()
            self.embed_dim = feat_dim
            self.stat_dim = stat_dim
            self.T = T
            # 初始化节点特征向量 (V, ln)，初始值为正态分布
            self.node_features = nn.Embedding(node_num, feat_dim)
    
            # 初始化节点的状态向量 (V, s)，初始值为0
            self.node_states = torch.zeros((node_num, stat_dim))
            # 输出层，线性变换
            self.linear = nn.Linear(feat_dim+stat_dim, 3)
    
            self.softmax = nn.Softmax(dim=1)
    
            '''
            实现Fw,因为是非位置图，所以采用了线性模型Hw
            '''
            self.Hw = Hw(feat_dim, stat_dim)
            # 实现H的分组求和
            self.Aggr = AggrSum(node_num)
    
        # Input : 
        #    X_Node : (N, )
        #    X_Neis : (N, )
        #    H      : (N, s)
        #    dg_list: (N, )
        def forward(self, X_Node, X_Neis, dg_list):
            '''
    X_Node:
     tensor([ 0,  0,  0,  1,  1,  2,  2,  2,  3,  3,  3,  3,  4,  4,  4,  5,  5,  6,
             6,  6,  7,  7,  7,  7,  7,  8,  8,  8,  8,  9,  9,  9, 10, 10, 10, 11,
            11, 12, 12, 12, 13, 13, 13, 13, 14, 14, 14, 15, 15, 15, 16, 17, 17, 17,
            17, 17]) 
    
    X_Neis:
     tensor([ 1,  4,  2,  3,  0,  5,  8,  0,  1,  4,  5,  7,  3,  0, 13,  2,  3,  8,
            10,  7,  6,  3, 10,  9, 11,  9,  6,  2, 13, 11,  8,  7,  6, 17,  7,  9,
             7, 15, 14, 17, 15,  4,  8, 17, 15, 12, 17, 14, 12, 13, 17, 14, 12, 13,
            10, 16]) 
    
    dg_list:
     [3, 3, 3, 2, 2, 3, 3, 3, 4, 4, 4, 4, 3, 3, 3, 2, 2, 3, 3, 3, 5, 5, 5, 5, 5, 4, 4, 4, 4, 3, 3, 3, 3, 3, 3, 2, 2, 3, 3, 3, 4, 4, 4, 4, 3, 3, 3, 3, 3, 3, 1, 5, 5, 5, 5, 5] 
            '''      
            #将特征向量进行抽取(V,L) -> (N,L)
            node_embeds = self.node_features(X_Node)  # 一条边的原节点特征值(N, ln)
            neis_embeds = self.node_features(X_Neis)  # 一条边的近节点特征值(N, ln)
            # 将一条边的两个节点的特征值合并为一行
            X = torch.cat((node_embeds, neis_embeds), 1)  # (N, 2 * ln) 
            '''
            X:
             tensor([[-1.6099,  0.8003,  0.9821,  0.5156],
                    [-1.6099,  0.8003,  0.1617,  0.6496],
                    [-1.6099,  0.8003, -0.2284,  1.5709],
                    [ 0.9821,  0.5156,  1.0583, -0.5877],
                    [ 0.9821,  0.5156, -1.6099,  0.8003],
                    [-0.2284,  1.5709,  0.4187, -0.0264],
                    [-0.2284,  1.5709, -1.4442, -0.0358],
                    [-0.2284,  1.5709, -1.6099,  0.8003],
                    [ 1.0583, -0.5877,  0.9821,  0.5156],
                    [ 1.0583, -0.5877,  0.1617,  0.6496],
                    [ 1.0583, -0.5877,  0.4187, -0.0264],
                    [ 1.0583, -0.5877,  1.7424,  0.7404],
                    [ 0.1617,  0.6496,  1.0583, -0.5877],
                    [ 0.1617,  0.6496, -1.6099,  0.8003],
                    [ 0.1617,  0.6496, -0.8059, -0.3895],
                    [ 0.4187, -0.0264, -0.2284,  1.5709],
                    [ 0.4187, -0.0264,  1.0583, -0.5877],
                    [-0.4789, -0.2107, -1.4442, -0.0358],
                    [-0.4789, -0.2107, -0.7170,  0.6958],
                    [-0.4789, -0.2107,  1.7424,  0.7404],
                    [ 1.7424,  0.7404, -0.4789, -0.2107],
                    [ 1.7424,  0.7404,  1.0583, -0.5877],
                    [ 1.7424,  0.7404, -0.7170,  0.6958],
                    [ 1.7424,  0.7404,  2.6017, -1.3521],
                    [ 1.7424,  0.7404, -0.2113,  0.4059],
                    [-1.4442, -0.0358,  2.6017, -1.3521],
                    [-1.4442, -0.0358, -0.4789, -0.2107],
                    [-1.4442, -0.0358, -0.2284,  1.5709],
                    [-1.4442, -0.0358, -0.8059, -0.3895],
                    [ 2.6017, -1.3521, -0.2113,  0.4059],
                    [ 2.6017, -1.3521, -1.4442, -0.0358],
                    [ 2.6017, -1.3521,  1.7424,  0.7404],
                    [-0.7170,  0.6958, -0.4789, -0.2107],
                    [-0.7170,  0.6958, -0.9262,  1.1642],
                    [-0.7170,  0.6958,  1.7424,  0.7404],
                    [-0.2113,  0.4059,  2.6017, -1.3521],
                    [-0.2113,  0.4059,  1.7424,  0.7404],
                    [-1.3676,  1.3421, -1.2686, -1.1046],
                    [-1.3676,  1.3421, -0.8206,  0.6743],
                    [-1.3676,  1.3421, -0.9262,  1.1642],
                    [-0.8059, -0.3895, -1.2686, -1.1046],
                    [-0.8059, -0.3895,  0.1617,  0.6496],
                    [-0.8059, -0.3895, -1.4442, -0.0358],
                    [-0.8059, -0.3895, -0.9262,  1.1642],
                    [-0.8206,  0.6743, -1.2686, -1.1046],
                    [-0.8206,  0.6743, -1.3676,  1.3421],
                    [-0.8206,  0.6743, -0.9262,  1.1642],
                    [-1.2686, -1.1046, -0.8206,  0.6743],
                    [-1.2686, -1.1046, -1.3676,  1.3421],
                    [-1.2686, -1.1046, -0.8059, -0.3895],
                    [ 0.7874,  0.0993, -0.9262,  1.1642],
                    [-0.9262,  1.1642, -0.8206,  0.6743],
                    [-0.9262,  1.1642, -1.3676,  1.3421],
                    [-0.9262,  1.1642, -0.8059, -0.3895],
                    [-0.9262,  1.1642, -0.7170,  0.6958],
                    [-0.9262,  1.1642,  0.7874,  0.0993]], grad_fn=<CatBackward>) 
            '''
            #print('X:\n',X,'\n')
            #循环T次计算，求其稳定的状态
            for t in range(self.T):
                '''
                #print('node_states:\n',self.node_states,'\n')
                #print('X_Node:\n',X_Node,'\n')
                #print('H:\n',H,'\n')
    node_states:
     tensor([[0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.]]) 
    
    X_Node:
     tensor([ 0,  0,  0,  1,  1,  2,  2,  2,  3,  3,  3,  3,  4,  4,  4,  5,  5,  6,
             6,  6,  7,  7,  7,  7,  7,  8,  8,  8,  8,  9,  9,  9, 10, 10, 10, 11,
            11, 12, 12, 12, 13, 13, 13, 13, 14, 14, 14, 15, 15, 15, 16, 17, 17, 17,
            17, 17]) 
    
    H:
     tensor([[0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.]]) 
                '''
                H = torch.index_select(self.node_states, 0, X_Node) # (V, s) -> (N, s)
                H = self.Hw(X, H, dg_list) # (N, s) -> (N, s)
                self.node_states = self.Aggr(H, X_Node) # (N, s) -> (V, s)
            #将节点特征向量和节点状态向量结合，进行线性变换得到输出
            out = self.linear(torch.cat((self.node_features.weight, self.node_states), 1))
            #print('self.node_features.weight:\n',self.node_features.weight,'\n')
            #print('self.node_states:\n',self.node_states,'\n')
            #print('torch.cat((self.node_features.weight, self.node_states), 1):\n',torch.cat((self.node_features.weight, self.node_states), 1),'\n')
            #将输出变为每一片之和为0的概率，作为结果输出
            out = self.softmax(out)
            return out  # (V, 3)
        
def CalAccuracy(output, label):
    with torch.autograd.set_detect_anomaly(True):
        # output : (N, C)
        # label : (N)
    
        out = np.argmax(output, axis=1)
        res = out - label
        '''    
        print('label:\n',label,'\n')
        print('output:\n',output,'\n')
        print('out:\n',out,'\n')
        print('res:\n',res,'\n')
        print('ret:\n',list(res).count(0) / len(res),'\n')
        '''
        return list(res).count(0) / len(res)
 
        

# 开始训练模型
def train(node_list, edge_list, label_list, T, ndict_path="./node_dict.json"):
    with torch.autograd.set_detect_anomaly(True):
        # 生成node-index字典和点集
        '''
        node_dict 
        {'stoi': {'n1': 0, 'n2': 1, 'n3': 2, 'n4': 3, 'n5': 4, 'n6': 5, 'n7': 6, 'n8': 7, 'n9': 8, 'n10': 9, 'n11': 10, 'n12': 11, 'n13': 12, 'n14': 13, 'n15': 14, 'n16': 15, 'n17': 16, 'n18': 17}, 
        'itos': ['n1', 'n2', 'n3', 'n4', 'n5', 'n6', 'n7', 'n8', 'n9', 'n10', 'n11', 'n12', 'n13', 'n14', 'n15', 'n16', 'n17', 'n18']} 
        '''
        if os.path.exists(ndict_path):
            with open(ndict_path, "r") as fp:
                node_dict = json.load(fp)
        else:
            node_dict = dict([(node, ind) for ind, node in enumerate(node_list)])
            node_dict = {"stoi" : node_dict,
                         "itos" : node_list}
            with open(ndict_path, "w") as fp:
                json.dump(node_dict, fp)       
        #print('node_dict',node_dict,'\n')
    
        # 统计节点的相邻的节点,用字典来表示
        '''
        Degree:
            {'n1': {'n3', 'n5', 'n2'}, 'n2': {'n4', 'n1'}, 'n3': {'n9', 'n6', 'n1'}, 'n5': {'n4', 'n14', 'n1'}, 'n4': {'n8', 'n6', 'n5', 'n2'}, 'n6': {'n4', 'n3'}, 'n9': {'n7', 'n14', 'n10', 'n3'}, 'n8': {'n7', 'n10', 'n4', 'n12', 'n11'}, 'n14': {'n16', 'n9', 'n5', 'n18'}, 'n7': {'n8', 'n9', 'n11'}, 'n11': {'n7', 'n8', 'n18'}, 'n10': {'n8', 'n9', 'n12'}, 'n12': {'n8', 'n10'}, 'n18': {'n15', 'n14', 'n13', 'n11', 'n17'}, 'n13': {'n16', 'n15', 'n18'}, 'n15': {'n16', 'n18', 'n13'}, 'n16': {'n14', 'n15', 'n13'}, 'n17': {'n18'}} 
        '''
        Degree = dict()
        for n1, n2 in edge_list:
            if n1 in Degree:
                Degree[n1].add(n2)
            else:
                Degree[n1] = {n2}
            if n2 in Degree:
                Degree[n2].add(n1)
            else:
                Degree[n2] = {n1}
        #print('Degree:','\n',Degree,'\n')
        
        '''
        node_inds和每个节点的度相关，该节点有几个度就占几位该节点的序号
        node_neis和每个节点的联系有关，和上面的数组相对应上面序号的节点的相邻节点的序号
        dg_list和每个结点的联系有关，和上面的对应显示度
        node_inds:
            [0, 0, 0, 1, 1, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 5, 5, 6, 6, 6, 7, 7, 7, 7, 7, 8, 8, 8, 8, 9, 9, 9, 10, 10, 10, 11, 11, 12, 12, 12, 13, 13, 13, 13, 14, 14, 14, 15, 15, 15, 16, 17, 17, 17, 17, 17] 
    
        node_neis:
            [2, 4, 1, 3, 0, 8, 5, 0, 7, 5, 4, 1, 3, 13, 0, 3, 2, 7, 8, 10, 6, 9, 3, 11, 10, 6, 13, 9, 2, 7, 8, 11, 6, 7, 17, 7, 9, 15, 14, 17, 15, 8, 4, 17, 15, 17, 12, 13, 14, 12, 17, 14, 13, 12, 10, 16] 
        '''
        node_inds = []
        node_neis = []
    
        for n in node_list:
            node_inds += [node_dict["stoi"][n]] * len(Degree[n])
            node_neis += list(map(lambda x: node_dict["stoi"][x],list(Degree[n])))
        # 生成度向量
        dg_list = list(map(lambda x: len(Degree[node_dict["itos"][x]]), node_inds))
        '''
        print('node_inds:','\n',node_inds,'\n')
        print('node_neis:','\n',node_neis,'\n')
        print('dg_list:','\n',dg_list,'\n')
        '''
        # 准备训练集和测试集
        train_node_list = [0,1,2,6,7,8,12,13,14]
        train_node_label = [0,0,0,1,1,1,2,2,2]
        test_node_list = [3,4,5,9,10,11,15,16,17]
        test_node_label = [0,0,0,1,1,1,2,2,2]
        
        #建立模型
        model = OriLinearGNN(node_num=len(node_list),
                             feat_dim=2,
                             stat_dim=2,
                             T=T)
        '''
        CLASS torch.optim.Adam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
        这是一种优于随机梯度下降的算法，Adam随机算法，来源：Adam:A Method for Stochastic Optimization
        优化函数，
        params=model_parameters()为其中可以优化的参数，
        lr学习速率
        weight_decay 权重衰减
        '''
        '''
        #显示可以优化的参数
        for name,param in model.named_parameters():
            if param.requires_grad:
                print(name)
        '''
        '''
        可以优化的参数：
        node_features.weight
        linear.weight
        linear.bias
        Hw.Xi.linear.weight
        Hw.Xi.linear.bias
        Hw.Rou.linear.weight
        Hw.Rou.linear.bias
        '''
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.01)
        #criterion = nn.CrossEntropyLoss(size_average=True)
        criterion = nn.CrossEntropyLoss(reduction='mean')
        #训练误差显示
        '''
        0.4以后tensor和variable没有区别了，下面的不用看。
        这里需要注意两种数量类型tensor和variable
        pytorch的操作和numpy类似，但是因为其能够在GPU上运行，所以比numpy快。
        variable是对tensor的封装，
        Variable和Tensor之间的区别：
    1. Variable是可更改的，而Tensor是不可更改的。
    2. Variable用于存储网络中的权重矩阵等变量，而Tensor更多的是中间结果等。
    3. Variable是会显示分配内存空间的，需要初始化操作（assign一个tensor），由Session管理，可以进行存储、读取、更改等操作。相反地，诸如Const, Zeros等操作创造的Tensor，是记录在Graph中，所以没有单独的内存空间；而其他未知的由其他Tensor操作得来的Tensor则是只会在程序运行中间出现。
    4. Tensor可以使用的地方，几乎都可以使用Variable。
        '''
        min_loss = float('inf')
        train_loss_list = []
        train_acc_list = []
        test_acc_list = []
    
        node_inds_tensor = torch.Tensor(node_inds).long()
        node_neis_tensor = torch.Tensor(node_neis).long()
        train_label = torch.Tensor(train_node_label).long()
    
        '''
        node_inds_tensor = Variable(torch.Tensor(node_inds).long())
        node_neis_tensor = Variable(torch.Tensor(node_neis).long())
        train_label = Variable(torch.Tensor(train_node_label).long())
        '''
        for ep in range(500):
            # 运行模型得到结果
            '''
            print('node_inds_tensor:\n',node_inds_tensor,'\n')
            print('node_neis_tensor:\n',node_neis_tensor,'\n')
            print('dg_list:\n',dg_list,'\n')
            '''
            res = model(node_inds_tensor, node_neis_tensor, dg_list) # (V, 3)
            # 备注：因为图中的点被分为3类所有三维
            #print('res:\n',res,'\n')
            #抽取被标记的点的结果和训练的点的结果
            train_res = torch.index_select(res, 0, torch.Tensor(train_node_list).long())
            test_res = torch.index_select(res, 0, torch.Tensor(test_node_list).long())
            
            #用交叉熵法来计算误差
            #    criterion = nn.CrossEntropyLoss()
            # train_label:训练样本的标签 [0,0,0,1,1,1,2,2,2]
            # train_res: 训练的结果
            loss = criterion(input=train_res,target=train_label)
            loss_val = loss.item()
            #print(loss_val)
            #计算训练精确度和测试精确度
            train_acc = CalAccuracy(train_res.cpu().detach().numpy(), np.array(train_node_label))
            test_acc = CalAccuracy(test_res.cpu().detach().numpy(), np.array(test_node_label))
        
            # 更新梯度
            #    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.01)
            #将梯度初始化清0
            optimizer.zero_grad()
            print(ep,'\n')
            loss.backward(retain_graph=True)
            print(ep,'end\n')
            optimizer.step()
            # 上面的是对模型进行更新
                
            # 保存loss和acc
            train_loss_list.append(loss_val)
            test_acc_list.append(test_acc)
            train_acc_list.append(train_acc)
            if loss_val < min_loss:
                min_loss = loss_val
            print("==> [Epoch {}] : loss {:.4f}, min_loss {:.4f}, train_acc {:.3f}, test_acc {:.3f}".format(ep, loss_val, min_loss, train_acc, test_acc))
            #break


train(node_list,E, label_list,5)
