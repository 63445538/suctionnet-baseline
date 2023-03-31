""" 3DSuctionNet model definition.
    Author: Rui Cao
"""

# import os
# import sys
# import numpy as np
# import torch
import torch.nn as nn
import MinkowskiEngine as ME
import torch.nn.functional as F

# from models.resunet14 import MinkUNet50
from models.resunet14 import MinkUNet14D


class SuctionNet(nn.Module):
    def __init__(self, feature_dim=256, is_training=True):
        super().__init__()
        self.feature_dim = feature_dim
        self.backbone = MinkUNet14D(in_channels=3, out_channels=self.feature_dim, D=3)
        self.suction_scoring = SuctionScoringNet(feature_dim=self.feature_dim)

    def forward(self, end_points):
        seed_xyz = end_points['point_clouds']  # use all sampled point cloud, B*Ns*3
        B, point_num, _ = seed_xyz.shape  # batch _size

        # point-wise features
        coordinates_batch = end_points['coors']
        features_batch = end_points['feats']
        mink_input = ME.SparseTensor(features_batch, coordinates=coordinates_batch)
        seed_features = self.backbone(mink_input).F
        seed_features = seed_features[end_points['quantize2original']].view(B, point_num, -1).transpose(1, 2)
        end_points = self.suction_scoring(seed_features, end_points)
        return end_points


class SuctionScoringNet(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.in_dim = feature_dim
        self.conv_scoring = nn.Conv1d(self.in_dim, 2, 1)
        # self.conv1 = nn.Conv1d(self.in_dim, 128, 1)
        # self.bn1 = nn.BatchNorm1d(128)
        # self.seal_scoring = nn.Conv1d(128, 1, 1)
        # self.wrench_scoring = nn.Conv1d(128, 1, 1)
        
        # v0.2.4
        # self.conv1 = nn.Conv1d(self.in_dim, 256, 1)
        # self.bn1 = nn.BatchNorm1d(256)
        # self.conv2 = nn.Conv1d(256, 128, 1)
        # self.bn2 = nn.BatchNorm1d(128)
        # self.seal_scoring = nn.Conv1d(128, 1, 1)
        # self.wrench_scoring = nn.Conv1d(128, 1, 1)
        
        # v0.2.5
        self.conv1 = nn.Conv1d(self.in_dim, 256, 1)
        self.bn1 = nn.BatchNorm1d(256)
        self.conv2 = nn.Conv1d(256, 128, 1)
        self.bn2 = nn.BatchNorm1d(128)
        self.seal_scoring = nn.Conv1d(128, 10, 1)
        self.wrench_scoring = nn.Conv1d(128, 10, 1)

    def forward(self, seed_features, end_points):
        # v0.2.1
        # suction_score = self.conv_scoring(seed_features)  # (B, 3, num_seed)
        # end_points['seal_score_pred'] = suction_score[:, 0]
        # end_points['wrench_score_pred'] = suction_score[:, 1]
                
        # scoring_feat = F.relu(self.bn1(self.conv1(seed_features)))  # (B, 3, num_seed)
        # v0.2.4
        x = F.relu(self.bn1(self.conv1(seed_features)))
        x = F.relu(self.bn2(self.conv2(x)))
        end_points['seal_score_pred'] = self.seal_scoring(x).squeeze(1)
        end_points['wrench_score_pred'] = self.wrench_scoring(x).squeeze(1)

        return end_points