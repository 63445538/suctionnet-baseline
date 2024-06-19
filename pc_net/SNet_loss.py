import torch
import torch.nn as nn
import MinkowskiEngine as ME
# from metric_loss import MetricLoss
from aleatoric_loss import AleatoricLoss

aleatoric_criterion = AleatoricLoss(is_log_sigma=False, res_loss='l1', nb_samples=10)
# metric_criterion = MetricLoss(nsample=18, kl_scale_factor=1e-4, epsilon=0.1)

# def get_loss(end_points):
#     seal_loss, end_points = compute_seal_score_loss(end_points)
#     wrench_loss, end_points = compute_wrench_score_loss(end_points)

#     loss = seal_loss + wrench_loss
#     end_points['loss/overall_loss'] = loss
#     return loss, end_points


# v0.2.6
# def get_loss(end_points):
#     # label_dense = torch.cat((label_dense), 0)
#     end_points['score_label'] = end_points['seal_score_label'] * end_points['wrench_score_label']
#     end_points['score_label'] = end_points['score_label'].view(-1)
    
#     score_loss, end_points = compute_score_loss(end_points)
#     metric_loss, end_points = compute_metric_loss(end_points)
#     # wrench_loss, end_points = compute_wrench_score_loss(end_points)

#     loss = score_loss + metric_loss
#     end_points['loss/overall_loss'] = loss
#     return loss, end_points

# v0.2.7
def get_loss(end_points):
    # label_dense = torch.cat((label_dense), 0)
    end_points['score_label'] = end_points['seal_score_label'] * end_points['wrench_score_label']
    # end_points['score_label'] = end_points['score_label'].view(-1, 1)
    
    loss = aleatoric_criterion(end_points['score_pred'], end_points['sigma_pred'], end_points['score_label'])
    end_points['loss/overall_loss'] = loss
    return loss, end_points


# v0.2.1.1
# def get_loss(end_points):
#     # label_dense = torch.cat((label_dense), 0)
#     end_points['score_label'] = end_points['seal_score_label'] * end_points['wrench_score_label']
#     # end_points['score_label'] = end_points['score_label'].view(-1, 1)
    
#     criterion = nn.SmoothL1Loss(reduction='mean')
#     score = end_points['score_pred']
#     score_label = end_points['score_label']
#     loss = criterion(score, score_label)
#     end_points['loss/overall_loss'] = loss
#     return loss, end_points


def compute_score_loss(end_points):
    criterion = nn.SmoothL1Loss(reduction='mean')
    score = end_points['score_pred']
    score_label = end_points['score_label']

    # criterion = nn.CrossEntropyLoss(reduction='mean', label_smoothing=0.1)
    # seal_score = end_points['seal_score_pred']
    # seal_score_label = end_points['seal_score_idx_label']
    loss = criterion(score.view(-1), score_label)

    end_points['loss/score_loss'] = loss
    # seal_score_pred = torch.argmax(seal_score, 1)
    # end_points['seal_score_acc'] = (seal_score_pred == seal_score_label.long()).float().mean()

    return loss, end_points


# def compute_metric_loss(end_points):
#     emb_mu_dense = end_points['emb_mu_dense']
#     emb_sigma_dense = end_points['emb_sigma_dense']

#     label_dense = end_points['score_label']
#     xyz_dense = end_points['coors']
    
#     xyz_sparse, unique_map = ME.utils.sparse_quantize(xyz_dense, return_index=True)
#     labels_sparse = label_dense[unique_map]
#     emb_mu_sparse = emb_mu_dense.F[unique_map]
#     emb_sigma_sparse = emb_sigma_dense.F[unique_map]

#     loss, _ = metric_criterion(emb_mu_sparse, emb_sigma_sparse, xyz_sparse, labels_sparse.view(-1, 1))
#     end_points['loss/metric_loss'] = loss
    
#     return loss, end_points


def compute_seal_score_loss(end_points):
    # criterion = nn.SmoothL1Loss(reduction='mean')
    # seal_score = end_points['seal_score_pred']
    # seal_score_label = end_points['seal_score_label']

    criterion = nn.CrossEntropyLoss(reduction='mean', label_smoothing=0.1)
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

    criterion = nn.CrossEntropyLoss(reduction='mean', label_smoothing=0.1)
    wrench_score = end_points['wrench_score_pred']
    wrench_score_label = end_points['wrench_score_idx_label']
    loss = criterion(wrench_score, wrench_score_label)

    end_points['loss/wrench_score_loss'] = loss
    wrench_score_pred = torch.argmax(wrench_score, 1)
    end_points['wrench_score_acc'] = (wrench_score_pred == wrench_score_label.long()).float().mean()
    
    return loss, end_points

