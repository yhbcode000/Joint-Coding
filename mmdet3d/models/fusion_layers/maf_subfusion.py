from typing import Tuple
import torchvision.transforms as transforms
import torch
from torch import nn
import cv2
from PIL import Image
import numpy as np
import torchvision
from mmcv.runner import BaseModule

from ..builder import FUSION_LAYERS

# Usecases
# attention = MultiheadAttention(hid_dim=32, n_heads=2, dropout=0.1)
# output1= attention(F_in_1,F_in_2)
# output2= attention(F_in_2,F_in_1)

@FUSION_LAYERS.register_module()
class MultiheadAttentionSubFusion(BaseModule):
    # n_heads：多头注意力的数量
    # hid_dim：每个词输出的向量维度
    def __init__(self, hid_dim, n_heads, dropout,num_h=100):
        super(MultiheadAttentionSubFusion, self).__init__()
        self.LN=nn.LayerNorm(hid_dim,eps=0,elementwise_affine=True)
        self.hid_dim = hid_dim
        self.n_heads = n_heads

        # 强制 hid_dim 必须整除 h
        assert hid_dim % n_heads == 0
        # 定义 W_q 矩阵
        self.w_q = nn.Linear(hid_dim, hid_dim)
        # 定义 W_k 矩阵
        self.w_k = nn.Linear(hid_dim, hid_dim)
        # 定义 W_v 矩阵
        self.w_v = nn.Linear(hid_dim, hid_dim)
        # self.fc = nn.Linear(hid_dim, hid_dim)
        self.do = nn.Dropout(dropout)
        # 缩放
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim // n_heads])).cuda()
        self.MLp=nn.Sequential(
            nn.Linear(hid_dim,num_h),
            nn.ReLU(),
            nn.Linear(num_h,num_h),
            nn.ReLU(),
            nn.Linear(num_h,hid_dim)
        )
    def forward(self, F_in_1, F_in_2,mask=None):
        # K: [64,10,300], batch_size 为 64，有 12 个词，每个词的 Query 向量是 300 维
        # V: [64,10,300], batch_size 为 64，有 10 个词，每个词的 Query 向量是 300 维
        # Q: [64,12,300], batch_size 为 64，有 10 个词，每个词的 Query 向量是 300 维
        F_in_1=self.LN(F_in_1)
        F_in_2=self.LN(F_in_2)
        F_in_Q=torch.cat((F_in_1, F_in_2),0)
        # device = torch.device("cuda:0")
        bsz = F_in_Q.shape[0]
        Q = self.w_q(F_in_Q)
        K = self.w_k(F_in_1)
        V = self.w_v(F_in_1)
        
        attention = torch.matmul(Q, K.permute(1,0)) / self.scale
       
        

        # 把 mask 不为空，那么就把 mask 为 0 的位置的 attention 分数设置为 -1e10
        if mask is not None:
            attention = attention.masked_fill(mask == 0, -1e10)

        # 第 2 步：计算上一步结果的 softmax，再经过 dropout，得到 attention。
        # 注意，这里是对最后一维做 softmax，也就是在输入序列的维度做 softmax
        # attention: [64,6,12,10]
        attention = self.do(torch.softmax(attention, dim=-1))

        # 第三步，attention结果与V相乘，得到多头注意力的结果
        # [64,6,12,10] * [64,6,10,50] = [64,6,12,50]
        # x: [64,6,12,50]
        # x = torch.matmul(attention, V)
        x = torch.matmul(attention, V)
        

        # 因为 query 有 12 个词，所以把 12 放到前面，把 5 和 60 放到后面，方便下面拼接多组的结果
        # x: [64,6,12,50] 转置-> [64,12,6,50]
        # yuan x = x.permute(0, 2, 1, 3).contiguous()
        # 这里的矩阵转换就是：把多组注意力的结果拼接起来
        # 最终结果就是 [64,12,300]
        # x: [64,12,6,50] -> [64,12,300]
        # x = x.view(bsz, -1, self.n_heads * (self.hid_dim // self.n_heads))
        x1 = self.LN(x)
        x1 = self.MLp(x1)
        x = x+x1


        return x

            




