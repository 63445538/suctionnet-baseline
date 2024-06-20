""" Training routine for GraspNet baseline model. """

import os
import numpy as np
from datetime import datetime
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ExponentialLR, MultiStepLR, CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
import MinkowskiEngine as ME

import resource

# RuntimeError: received 0 items of ancdata. Issue: pytorch/pytorch#973
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
hard_limit = rlimit[1]
soft_limit = min(500000, hard_limit)
print("soft limit: ", soft_limit, "hard limit: ", hard_limit)
resource.setrlimit(resource.RLIMIT_NOFILE, (soft_limit, hard_limit))

import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

import sys
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(ROOT_DIR))


import random
def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
# 设置随机数种子
setup_seed(0)

# ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(os.path.join(ROOT_DIR, 'utils'))
# sys.path.append(os.path.join(ROOT_DIR, 'pointnet2'))
# sys.path.append(os.path.join(ROOT_DIR, 'models'))
# sys.path.append(os.path.join(ROOT_DIR, 'dataset'))

# from graspnet import GraspNet, get_loss
# from GSNet import GraspNet
# from GSNet_loss import get_loss
# from pytorch_utils import BNMomentumScheduler

from dataset.dataset import SuctionDataset, collate_fn, minkowski_collate_fn, load_obj_list
# from SNet import SuctionNet
# from label_generation import process_grasp_labels


parser = argparse.ArgumentParser()
parser.add_argument('--dataset_root', default='/media/user/data1/rcao/graspnet', help='Dataset root')
parser.add_argument('--camera', default='realsense', help='Camera split [realsense/kinect]')
parser.add_argument('--resume_checkpoint', default=None, help='Model checkpoint path [default: None]')
parser.add_argument('--method_id', default='snet_v1.1', help='Dump dir to save model checkpoint [default: log]')
parser.add_argument('--ckpt_root', default='/media/user/data1/rcao/result/uisn', help='Checkpoint dir to save model [default: log]')
parser.add_argument('--num_point', type=int, default=1024, help='Point Number [default: 20000]')
parser.add_argument('--seed_feat_dim', default=512, type=int, help='Point wise feature dim')
parser.add_argument('--worker_num', type=int, default=32, help='Worker number for dataloader [default: 4]')
parser.add_argument('--ckpt_save_interval', type=int, default=5, help='Number for save checkpoint[default: 5]')
parser.add_argument('--max_epoch', type=int, default=61, help='Epoch to run [default: 18]')
parser.add_argument('--batch_size', type=int, default=64, help='Batch Size during training [default: 2]')
parser.add_argument('--learning_rate', type=float, default=0.002, help='Initial learning rate [default: 0.001]')
parser.add_argument('--weight_decay', type=float, default=0.001, help='Optimization L2 weight decay [default: 0]')
parser.add_argument('--voxel_size', type=float, default=0.002, help='Voxel Size for Quantize [default: 0.005]')
parser.add_argument('--visib_threshold', type=float, default=0.5, help='Visibility Threshold [default: 0.5]')
parser.add_argument('--lr_sched', default=True, action='store_true')
parser.add_argument('--lr_sched_period', type=int, default=16, help='T_max of cosine learing rate scheduler [default: 16]')
# parser.add_argument('--bn_decay_step', type=int, default=2, help='Period of BN decay (in epochs) [default: 2]')
# parser.add_argument('--bn_decay_rate', type=float, default=0.5, help='Decay rate for BN decay [default: 0.5]')
# parser.add_argument('--lr_decay_steps', default='8,12,16',
#                     help='When to decay the learning rate (in epochs) [default: 8,12,16]')
# parser.add_argument('--lr_decay_rates', default='0.1,0.1,0.1', help='Decay rates for lr decay [default: 0.1,0.1,0.1]')
cfgs = parser.parse_args()

cfgs.ckpt_dir = os.path.join(cfgs.ckpt_root, cfgs.method_id, cfgs.camera)
cfgs.log_dir = os.path.join('log', cfgs.method_id, cfgs.camera)
os.makedirs(cfgs.ckpt_dir, exist_ok=True)
os.makedirs(cfgs.log_dir, exist_ok=True)

EPOCH_CNT = 0
DEFAULT_CHECKPOINT_PATH = os.path.join(cfgs.ckpt_dir, 'checkpoint.tar')
CHECKPOINT_PATH = cfgs.resume_checkpoint if cfgs.resume_checkpoint is not None \
    else DEFAULT_CHECKPOINT_PATH

if not os.path.exists(cfgs.log_dir):
    os.makedirs(cfgs.log_dir)

LOG_FOUT = open(os.path.join(cfgs.log_dir, 'log_train.txt'), 'a')
LOG_FOUT.write(str(cfgs) + '\n')


def log_string(out_str):
    LOG_FOUT.write(out_str + '\n')
    LOG_FOUT.flush()
    print(out_str)


# Init datasets and dataloaders
def my_worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)
    pass


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(device)

# Create Dataset and Dataloader
valid_obj_idxs = load_obj_list()
TRAIN_DATASET = SuctionDataset(cfgs.dataset_root, valid_obj_idxs, camera=cfgs.camera, split='train', num_points=cfgs.num_point, 
                               remove_outlier=False, real_data=True, syn_data=True, augment=False, visib_threshold=cfgs.visib_threshold, voxel_size=cfgs.voxel_size)
TEST_DATASET = SuctionDataset(cfgs.dataset_root, valid_obj_idxs, camera=cfgs.camera, split='test_seen', num_points=cfgs.num_point, 
                              remove_outlier=False, real_data=True, syn_data=False, augment=False, visib_threshold=cfgs.visib_threshold, voxel_size=cfgs.voxel_size)

print(len(TRAIN_DATASET), len(TEST_DATASET))
# TRAIN_DATALOADER = DataLoader(TRAIN_DATASET, batch_size=cfgs.batch_size, shuffle=True,
#     num_workers=4, worker_init_fn=my_worker_init_fn, collate_fn=collate_fn)
# TEST_DATALOADER = DataLoader(TEST_DATASET, batch_size=cfgs.batch_size, shuffle=False,
#     num_workers=4, worker_init_fn=my_worker_init_fn, collate_fn=collate_fn)
TRAIN_DATALOADER = DataLoader(TRAIN_DATASET, batch_size=cfgs.batch_size, shuffle=True,
                              num_workers=cfgs.worker_num, worker_init_fn=my_worker_init_fn, collate_fn=minkowski_collate_fn)
TEST_DATALOADER = DataLoader(TEST_DATASET, batch_size=cfgs.batch_size, shuffle=False,
                             num_workers=cfgs.worker_num, worker_init_fn=my_worker_init_fn, collate_fn=minkowski_collate_fn)
print(len(TRAIN_DATALOADER), len(TEST_DATALOADER))
# Init the model and optimzier
# net = GraspNet(input_feature_dim=0, num_view=cfgs.num_view, num_angle=12, num_depth=4,
#                         cylinder_radius=0.05, hmin=-0.02, hmax_list=[0.01,0.02,0.03,0.04])

# net = SuctionNet(feature_dim=512)
# net.to(device)

# v0.2.6
# from models.resunet import Res16UNet34CProbMG
# net = Res16UNet34CProbMG(in_channels=3, out_channels=1, max_t=-1, logit_norm=False)
# net.to(device)

# v0.2.7
# from models.resunet import Res16UNet34CAleatoric
# from SNet_loss import get_loss
# net = Res16UNet34CAleatoric(in_channels=3, out_channels=1)
# net.to(device)

# v0.2.7.2
from SNet import SuctionNet_prob
from SNet_loss import get_loss
net = SuctionNet_prob(feature_dim=cfgs.seed_feat_dim)
net.to(device)

# dropout_prob = 0.1
# # v0.2.7.1 0.5 v0.2.7.3 0.1
# import torch.nn.functional as F
# def dropout_hook_wrapper(module, sinput, soutput):
#     input = soutput.F
#     output = F.dropout(input, p=dropout_prob, training=True)
#     soutput_new = ME.SparseTensor(output, coordinate_map_key=soutput.coordinate_map_key, coordinate_manager=soutput.coordinate_manager)
#     return soutput_new
# for module in net.modules():
#     if isinstance(module, ME.MinkowskiConvolution):
#         module.register_forward_hook(dropout_hook_wrapper)
        
# v0.2.1.1
# from SNet import SuctionNet
# from SNet_loss import get_loss
# net = SuctionNet(feature_dim=512)
# net.to(device)

# Load the Adam optimizer
optimizer = optim.AdamW(net.parameters(), lr=cfgs.learning_rate, weight_decay=cfgs.weight_decay)
if cfgs.lr_sched:
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=cfgs.lr_sched_period, eta_min=1e-4)
    
# Load checkpoint if there is any
it = -1  # for the initialize value of `LambdaLR` and `BNMomentumScheduler`
start_epoch = 0
if CHECKPOINT_PATH is not None and os.path.isfile(CHECKPOINT_PATH):
    checkpoint = torch.load(CHECKPOINT_PATH)
    net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if cfgs.lr_sched:
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
    start_epoch = checkpoint['epoch']
    log_string("-> loaded checkpoint %s (epoch: %d)" % (CHECKPOINT_PATH, start_epoch))
# Decay Batchnorm momentum from 0.5 to 0.999
# note: pytorch's BN momentum (default 0.1)= 1 - tensorflow's BN momentum
# BN_MOMENTUM_INIT = 0.5
# BN_MOMENTUM_MAX = 0.001
# bn_lbmd = lambda it: max(BN_MOMENTUM_INIT * cfgs.bn_decay_rate**(int(it / cfgs.bn_decay_step)), BN_MOMENTUM_MAX)
# bnm_scheduler = BNMomentumScheduler(net, bn_lambda=bn_lbmd, last_epoch=start_epoch-1)


# def get_current_lr(epoch):
#     lr = cfgs.learning_rate
#     for i, lr_decay_epoch in enumerate(LR_DECAY_STEPS):
#         if epoch >= lr_decay_epoch:
#             lr *= LR_DECAY_RATES[i]
#     return lr


# def adjust_learning_rate(optimizer, epoch):
#     lr = get_current_lr(epoch)
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr


# TensorBoard Visualizers
log_writer = SummaryWriter(os.path.join(cfgs.log_dir))
# ------------------------------------------------------------------------- GLOBAL CONFIG END

def train_one_epoch():
    stat_dict = {}  # collect statistics
    # adjust_learning_rate(optimizer, EPOCH_CNT)
    # bnm_scheduler.step() # decay BN momentum
    # set model to training mode
    overall_loss = 0
    net.train()
    for batch_idx, batch_data_label in enumerate(TRAIN_DATALOADER):
        for key in batch_data_label:
            if 'list' in key:
                for i in range(len(batch_data_label[key])):
                    for j in range(len(batch_data_label[key][i])):
                        batch_data_label[key][i][j] = batch_data_label[key][i][j].to(device)
            else:
                batch_data_label[key] = batch_data_label[key].to(device)

        # # Forward pass v0.2.7.2
        end_points = net(batch_data_label)

        # v0.2.6
        # in_data = ME.TensorField(features=batch_data_label['feats'], coordinates=batch_data_label['coors'],
        #                         quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE)

        # end_points = batch_data_label
        
        # v0.2.6
        # score, emb_mu, emb_sigma = net(in_data)
        # end_points['score_pred'] = score
        # end_points['emb_mu_dense'] = emb_mu.slice(in_data)
        # end_points['emb_sigma_dense'] = emb_sigma.slice(in_data)
        
        # v0.2.7
        # score, sigma = net(in_data)
        # end_points['score_pred'] = score
        # end_points['sigma_pred'] = sigma
        
        # Compute loss and gradients, update parameters.
        loss, end_points = get_loss(end_points)
        loss.backward()
        if (batch_idx + 1) % 1 == 0:
            optimizer.step()
            optimizer.zero_grad()

        # Accumulate statistics and print out
        for key in end_points:
            if 'loss' in key or 'acc' in key or 'prec' in key or 'recall' in key or 'count' in key:
                if key not in stat_dict: stat_dict[key] = 0
                stat_dict[key] += end_points[key].item()

        overall_loss += stat_dict['loss/overall_loss']
        batch_interval = 10
        if (batch_idx + 1) % batch_interval == 0:
            log_string(' ---- batch: %03d ----' % (batch_idx + 1))
            for key in sorted(stat_dict.keys()):
                log_writer.add_scalar('train_' + key, stat_dict[key] / batch_interval,
                                        (EPOCH_CNT * len(TRAIN_DATALOADER) + batch_idx) * cfgs.batch_size)
                log_string('mean %s: %f' % (key, stat_dict[key] / batch_interval))
                stat_dict[key] = 0

    log_string('overall loss:{}, batch num:{}'.format(overall_loss, batch_idx+1))
    mean_loss = overall_loss/float(batch_idx+1)
    return mean_loss

def evaluate_one_epoch():
    stat_dict = {}  # collect statistics
    # set model to eval mode (for bn and dp)
    net.eval()
    overall_loss = 0
    for batch_idx, batch_data_label in enumerate(TEST_DATALOADER):
        if batch_idx % 10 == 0:
            print('Eval batch: %d' % batch_idx)
        for key in batch_data_label:
            if 'list' in key:
                for i in range(len(batch_data_label[key])):
                    for j in range(len(batch_data_label[key][i])):
                        batch_data_label[key][i][j] = batch_data_label[key][i][j].to(device)
            else:
                batch_data_label[key] = batch_data_label[key].to(device)

        # Forward pass v0.2.7.2
        with torch.no_grad():
            end_points = net(batch_data_label)

            # in_data = ME.TensorField(features=batch_data_label['feats'], coordinates=batch_data_label['coors'],
            #                         quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE)

            # end_points = batch_data_label
            
            # Forward pass v0.2.6
            # score, emb_mu, emb_sigma = net(in_data)
            # end_points['score_pred'] = score
            # end_points['emb_mu_dense'] = emb_mu.slice(in_data)
            # end_points['emb_sigma_dense'] = emb_sigma.slice(in_data)
            
            # score, sigma = net(in_data)
            # end_points['score_pred'] = score
            # end_points['sigma_pred'] = sigma
        
        # Compute loss
        loss, end_points = get_loss(end_points)

        # Accumulate statistics and print out
        for key in end_points:
            if 'loss' in key or 'acc' in key or 'prec' in key or 'recall' in key or 'count' in key:
                if key not in stat_dict: stat_dict[key] = 0
                stat_dict[key] += end_points[key].item()

        overall_loss += stat_dict['loss/overall_loss']
        
    for key in sorted(stat_dict.keys()):
        log_writer.add_scalar('test_' + key, stat_dict[key] / float(batch_idx + 1),
                               (EPOCH_CNT + 1) * len(TRAIN_DATALOADER) * cfgs.batch_size)
        log_string('eval mean %s: %f' % (key, stat_dict[key] / (float(batch_idx + 1))))

    log_string('overall loss:{}, batch num:{}'.format(overall_loss, batch_idx+1))
    mean_loss = overall_loss/float(batch_idx+1)
    return mean_loss


def train(start_epoch):
    global EPOCH_CNT
    min_loss = 1e10
    loss = 0
    for epoch in range(start_epoch, cfgs.max_epoch):
        EPOCH_CNT = epoch
        log_string('**** EPOCH %03d ****' % epoch)
        current_lr = optimizer.param_groups[0]['lr']
        log_string('Current learning rate: %f' % (current_lr))
        # log_string('Current BN decay momentum: %f'%(bnm_scheduler.lmbd(bnm_scheduler.last_epoch)))
        log_string(str(datetime.now()))
        # Reset numpy seed.
        # REF: https://github.com/pytorch/pytorch/issues/5059
        np.random.seed()
        train_loss = train_one_epoch()
        log_writer.add_scalar('training/learning_rate', current_lr, epoch)
        eval_loss = evaluate_one_epoch()
        
        # Save checkpoint
        save_dict = {'epoch': epoch+1, # after training one epoch, the start_epoch should be epoch+1
                    'optimizer_state_dict': optimizer.state_dict()}
        
        if cfgs.lr_sched:
            lr_scheduler.step()
            save_dict['lr_scheduler'] = lr_scheduler.state_dict()
        try: # with nn.DataParallel() the net is added as a submodule of DataParallel
            save_dict['model_state_dict'] = net.module.state_dict()
        except:
            save_dict['model_state_dict'] = net.state_dict()
        
        if eval_loss < min_loss:
            min_loss = eval_loss
            best_epoch = epoch
            ckpt_name = "epoch_" + str(best_epoch) \
                        + "_train_" + str(train_loss) \
                        + "_val_" + str(eval_loss)
            torch.save(save_dict['model_state_dict'], os.path.join(cfgs.ckpt_dir, ckpt_name + '.tar'))
        elif not EPOCH_CNT % cfgs.ckpt_save_interval:
            torch.save(save_dict, os.path.join(cfgs.ckpt_dir, 'checkpoint_{}.tar'.format(EPOCH_CNT)))
        torch.save(save_dict, os.path.join(cfgs.ckpt_dir, 'checkpoint.tar'))
        log_string("best_epoch:{}".format(best_epoch))


if __name__ == '__main__':
    train(start_epoch)
