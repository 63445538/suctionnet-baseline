import torch
import torch.nn as nn


def get_loss(end_points):
    seal_loss, end_points = compute_seal_score_loss(end_points)
    wrench_loss, end_points = compute_wrench_score_loss(end_points)

    loss = seal_loss + wrench_loss
    end_points['loss/overall_loss'] = loss
    return loss, end_points


def compute_seal_score_loss(end_points):
    # criterion = nn.SmoothL1Loss(reduction='mean')
    # seal_score = end_points['seal_score_pred']
    # seal_score_label = end_points['seal_score_label']

    criterion = nn.CrossEntropyLoss(reduction='mean', label_smoothing=0.0)
    seal_score = end_points['seal_score_pred']
    seal_score_label = end_points['seal_score_idx_label']
    loss = criterion(seal_score, seal_score_label)

    end_points['loss/seal_score_loss'] = loss
    seal_score_pred = torch.argmax(seal_score, 1)
    end_points['seal_score_acc'] = (seal_score_pred == seal_score_label.long()).float().mean()

    return loss, end_points


def compute_wrench_score_loss(end_points):
    # criterion = nn.SmoothL1Loss(reduction='mean')
    # wrench_score = end_points['wrench_score_pred']
    # wrench_score_label = end_points['wrench_score_label']

    criterion = nn.CrossEntropyLoss(reduction='mean', label_smoothing=0.0)
    wrench_score = end_points['wrench_score_pred']
    wrench_score_label = end_points['wrench_score_idx_label']
    loss = criterion(wrench_score, wrench_score_label)

    end_points['loss/wrench_score_loss'] = loss
    wrench_score_pred = torch.argmax(wrench_score, 1)
    end_points['wrench_score_acc'] = (wrench_score_pred == wrench_score_label.long()).float().mean()
    
    return loss, end_points

