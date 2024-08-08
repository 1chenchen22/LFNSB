
import torch
import torch.nn as nn
import torch.nn.functional as F

class AffinityLoss(nn.Module):
    def __init__(self, device, num_class=7, feat_dim=256):
        super(AffinityLoss, self).__init__()
        self.num_class = num_class  # 类别数量
        self.feat_dim = feat_dim  # 特征维度
        self.gap = nn.AdaptiveAvgPool2d(1)  # 全局平均池化层，将特征图转换为特征向量
        self.device = device  # 设备（CPU或GPU）
        self.centers = nn.Parameter(torch.randn(self.num_class, self.feat_dim).to(device))  # 可学习的类别中心

    def compute_adjusted_cosine_distance(self, x, centers):
        # 计算输入特征向量和类别中心的均值
        x_mean = x.mean(dim=1, keepdim=True)
        centers_mean = centers.mean(dim=1, keepdim=True)

        # 调整向量
        x_adj = x - x_mean
        centers_adj = centers - centers_mean

        # 计算调整后的余弦相似度
        x_norm = F.normalize(x_adj, p=2, dim=1)
        centers_norm = F.normalize(centers_adj, p=2, dim=1)
        return 1 - torch.mm(x_norm, centers_norm.t())
    def forward(self, x, labels):
        x = self.gap(x).view(x.size(0), -1)  # 全局平均池化后将特征图展平为1维特征向量
        batch_size = x.size(0)
        # # 计算余弦距离矩阵
        #distmat = self.compute_cosine_distance(x, self.centers)
        #
        # # 计算调整后的余弦距离矩阵
        distmat = self.compute_adjusted_cosine_distance(x, self.centers)

        classes = torch.arange(self.num_class).long().to(self.device)  # 生成类别索引
        labels = labels.unsqueeze(1).expand(batch_size, self.num_class)  # 扩展标签以便比较
        mask = labels.eq(classes.expand(batch_size, self.num_class))  # 创建用于计算类内距离的掩码
        # 计算特征向量类别内的方差
        class_variances = torch.var(x, dim=0)
        # 计算类内距离和类间距离
        dist = distmat * mask.float()
        intra_class_distance = dist.sum() / batch_size
        inter_class_distance = dist.sum() / (batch_size * (self.num_class - 1))
        #权衡类内方差和类间距离的加权平衡
        alpha = 0.1 # 根据任务调整权重
        loss = alpha*intra_class_distance/class_variances.sum() +(1-alpha) *inter_class_distance
        return loss

