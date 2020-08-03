# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 14:32:47 2020
Describe:
    这部分主要是关于torch等模块中相关函数的样例，以免忘记，样例大多来自官方手册。
@author: EGlym
"""

import torch
import torch.nn as nn
import numpy as np

def function():
    V = 5
    X_node = torch.Tensor([0,1,2,3,4])
    mask = torch.stack([X_node] * V, 0)
    print('mask:\n',mask,'\n')
    a = torch.unsqueeze(torch.arange(0,V).float(), 1)
    print(a)
    
def count_function():
    # 返回某个字符出现的次数
    a = np.array([3, 1, 2, 4, 6, 1])
    num1 = list(a).count(1)
    print(num1)
    
def argmax_function():
    # 返回最大的一个数字的索引，当数据为多维的时候可以选择方向
    # axis = 0 列方向 1 行方向
    a = np.array([3, 1, 2, 4, 6, 1])
    b=np.argmax(a)
    print('input:\n',a,'\n')
    print('output:\n',b,'\n')
    a = np.array([[1, 5, 5, 2],
                  [9, 6, 2, 8],
                  [3, 7, 9, 1]])
    b0 = np.argmax(a,axis= 0)
    b1 = np.argmax(a,axis= 1)
    print('input:\n',a,'\n')
    print('b0:\n',b0,'\n')
    print('b1:\n',b1,'\n')
    
def crossentropyloss_function():
    #这个怎么用没看懂,待定
    loss = nn.CrossEntropyLoss()
    input = torch.randn(3, 5, requires_grad=True)
    target = torch.empty(3, dtype=torch.long).random_(5)
    output = loss(input, target)
    output.backward()

def arange_function():
    a = torch.arange(1,6)
    print("a:\n",a,'\n')

def nn_softmax_function():
    '''
    这一部分用了softmax函数，这个函数主要功能就是将向量归一化为（0，1）之间的数值，
    采用指数运算，使得向量中较大的量更加明显。
    torch.nn.Softmax(dim: Optional[int] = None)
    dim:用来计算softmax尺寸，因此，沿着dim的每个切片的综合为0
    '''
    m = nn.Softmax(dim=1)
    input = torch.randn(2, 3)
    output = m(input)
    print('input:\n',input,'\n')
    print('output:\n',output,'\n')
    
def stack_function():
    a = torch.Tensor([[1,2,3],[4,5,6],[7,8,9]])
    b = torch.Tensor([[10,20,30],[40,50,60],[70,80,90]])
    c = torch.Tensor([[100,200,300],[400,500,600],[700,800,900]])
    d = torch.stack((a,b,c),dim = 0)
    print(a)
    print(b)
    print(c)
    print(d)

def matmul_function():
    '''
    tensor乘法，输入可以是高维的
    '''
    a = torch.ones(3,4)
    b = torch.ones(4,2)
    c = torch.matmul(a,b)
    print(c.shape)

def unsqueeze_function():
    a=torch.randn(1,3)
    b=torch.unsqueeze(a,0)
    c=torch.unsqueeze(a,1)
    d=torch.unsqueeze(a,2)
    print(a.shape)
    print(b.shape)
    print(c.shape)
    print(d.shape)

def squeeze_function():
    '''
    对数据维度进行压缩，去掉维度为1的数据
    和unsqueeze正好相反
    torch.squeeze(input, dim=None, out=None)
    当给定dim时候，那么挤压操作只会在给定维度上
    '''
    a = torch.randn(1,1,3)
    b = torch.squeeze(a)
    print(a.shape)
    print(b.shape)
    

def chunk_function():
    '''
    将tensor分割为tensor元组，
    torch.chunk(tensor,chunks,dim=0)
    '''
    a = torch.Tensor([[1,2,4,5],[4,5,7,9],[3,5,9,8],[3,6,9,8]])
    c = torch.chunk(a,2,dim=1)
    d = torch.chunk(a,4,dim=1)
    print('a:\n',a,'\n')
    print('c:\n',c,'\n')
    print('d:\n',d,'\n')   

def view_function():
    '''
    返回一个有相同数据但大小不同的张量，返回的张量必须有与原张量相同的数据和相同数目的元素，但是可以有不同的大小。
    '''
    x = torch.randn(4,4)
    y0 = x.view(16)
    y1 = x.view(-1,8)
    print('x:\n',x,'\n')
    print('y0:\n',y0,'\n')
    print('y1:\n',y1,'\n')
    
def nn_linear_function():
    '''
    nn.Linear function 进行线性代数运算
    class torch.nn.Linear(in_features, out_features, bias=True)
    y = x*W^(T)+b
    x (N,in_features)
    y (N,out_features)
    w (out_features,in_features)
    bia (1,out_features)
    '''
    m = nn.Linear(2,3,bias=True)
    input = torch.randn(10,2)
    output = m(input)
    print('weight:\n',m.weight.shape,'\n')
    print('bias:\n',m.bias.shape,'\n')
    print('input:\n',input.shape,'\n')
    print('output:\n',output.shape,'\n')
    
def forward_function():
    #forward function
    class model(nn.Module):
        def __init__(self,x):
            super().__init__()
            self.stat = x
            print('init:state=',self.stat,'\n')
        def forward(self,y):
            self.stat = self.stat + y
            print('state:',self.stat,'\n')
            
    tmp = model(1)
    test = tmp(1)

def embedding_function():
    #Embedding字典抽取
    embedding = nn.Embedding(10,3)
    print('embedding:\n',embedding.weight,'\n')
    
    input = torch.LongTensor([[1,2,4,5],[4,3,2,9]])
    print('input:\n',input);
    
    ans = embedding(input)
    print('ans:\n',ans,'\n')

def cat_function():
    #cat函数,用来拼接数组
    x = torch.randn(2,3)
    print('x:\n',x,'\n')
    tmp0 = torch.cat((x, x, x), 0)
    tmp1 = torch.cat((x, x, x), 1)
    print('tmp0:\n',tmp0,'\n')
    print('tmp1:\n',tmp1,'\n')
    
def index_select_function():
    x = torch.randn(3,4)
    indices = torch.tensor([0,2])
    tmp0 = torch.index_select(x,0,indices)
    tmp1 = torch.index_select(x,1,indices)
    print('x:\n',x,'\n')
    print('indices:\n',indices,'\n')
    print('tmp0:\n',tmp0,'\n')
    print('tmp1:\n',tmp1,'\n')