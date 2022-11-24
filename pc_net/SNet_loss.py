import torch.nn as nn


def get_loss(end_points):
    seal_loss, end_points = compute_seal_score_loss(end_points)
    wrench_loss, end_points = compute_wrench_score_loss(end_points)

    loss = seal_loss + wrench_loss
    end_points['loss/overall_loss'] = loss
    return loss, end_points


def compute_seal_score_loss(end_points):
    criterion = nn.SmoothL1Loss(reduction='mean')
    seal_score_pred = end_points['seal_score_pred']
    seal_score_label = end_points['seal_score_label']
    loss = criterion(seal_score_pred, seal_score_label)

    end_points['loss/seal_score_loss'] = loss
    return loss, end_points


def compute_wrench_score_loss(end_points):
    criterion = nn.SmoothL1Loss(reduction='mean')
    wrench_score_pred = end_points['wrench_score_pred']
    wrench_score_label = end_points['wrench_score_label']
    loss = criterion(wrench_score_pred, wrench_score_label)

    end_points['loss/wrench_score_loss'] = loss
    return loss, end_points

