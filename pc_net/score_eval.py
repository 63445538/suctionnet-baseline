import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import scipy.io as scio

import pickle
import sys
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, '..', 'utils'))

from data_utils import CameraInfo, create_point_cloud_from_depth_image, get_workspace_mask
import MinkowskiEngine as ME

# from suctionnetAPI.utils.rotation import viewpoint_to_matrix
# from suctionnetAPI.utils.utils import plot_sucker
# from suctionnetAPI.suction import SuctionGroup

import open3d as o3d
import matplotlib.pyplot as plt


import random
def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
# 设置随机数种子
setup_seed(0)


minimum_num_pt = 50
num_pt = 1024
width = 1280
height = 720
voxel_size = 0.002
suction_height = 0.1
suction_radius = 0.01
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


parser = argparse.ArgumentParser()
parser.add_argument('--split', default='test', help='dataset split [default: test_seen]')
parser.add_argument('--camera', default='realsense', help='camera to use [default: kinect | realsense]')
parser.add_argument('--save_root', default='save', help='where to save')
parser.add_argument('--dataset_root', default='/data/rcao/dataset/graspnet', help='where dataset is')
FLAGS = parser.parse_args()

network_ver = 'v0.2.7.2'
trained_epoch = 40
save_ver = '{}_{}'.format(network_ver, trained_epoch)

split = FLAGS.split
camera = FLAGS.camera
dataset_root = FLAGS.dataset_root
save_root = os.path.join(FLAGS.save_root, save_ver)
torch.cuda.set_device(device)

# from SNet import SuctionNet
# net = SuctionNet(feature_dim=512, is_training=False)
# net.to(device)
# # checkpoint = torch.load(os.path.join('log', 'snet_'+network_ver, camera, 'checkpoint.tar'), map_location=device)
# checkpoint = torch.load(os.path.join('log', 'snet_'+network_ver, camera, 'checkpoint_{}.tar'.format(trained_epoch)), map_location=device)
# net.load_state_dict(checkpoint['model_state_dict'])
# net.eval()

# 0.2.7.2
from SNet import SuctionNet_prob
net = SuctionNet_prob(feature_dim=512)
net.to(device)
checkpoint = torch.load(os.path.join('log', 'snet_'+network_ver, camera, 'checkpoint_{}.tar'.format(trained_epoch)), map_location=device)
# checkpoint = torch.load(os.path.join('log', 'snet_'+network_ver, camera, 'checkpoint.tar'), map_location=device)
net.load_state_dict(checkpoint['model_state_dict'])
net.eval()

# # v0.2.6
# from models.resunet import Res16UNet34CProbMG
# net = Res16UNet34CProbMG(in_channels=3, out_channels=1, max_t=-1, logit_norm=False)
# net.to(device)
# checkpoint = torch.load(os.path.join('log', 'snet_'+network_ver, camera, 'checkpoint.tar'), map_location=device)
# # checkpoint = torch.load(os.path.join('log', 'snet_'+network_ver, camera, 'checkpoint_{}.tar'.format(trained_epoch)), map_location=device)
# net.load_state_dict(checkpoint['model_state_dict'])
# net.eval()

# v0.2.7
# from models.resunet import Res16UNet34CAleatoric
# net = Res16UNet34CAleatoric(in_channels=3, out_channels=1)
# net.to(device)
# checkpoint = torch.load(os.path.join('log', 'snet_'+network_ver, camera, 'checkpoint_{}.tar'.format(trained_epoch)), map_location=device)
# # checkpoint = torch.load(os.path.join('log', 'snet_'+network_ver, camera, 'checkpoint.tar'), map_location=device)
# net.load_state_dict(checkpoint['model_state_dict'])
# net.eval()

epoch_num = checkpoint['epoch']
print("network version:{}, epoch:{}".format(network_ver, epoch_num))

eps = 1e-12
def normalize(array):
    max = np.max(array)
    min = np.min(array)
    array = (array - min) / (max - min + eps)
    return array


def normalize_tensor(tensor):
    max = tensor.max()
    min = tensor.min()
    tensor = (tensor - min) / (max - min + eps)
    return tensor


mae_criterion = torch.nn.L1Loss(reduction='mean')
mse_criterion = torch.nn.MSELoss()

scene_num = 90
anno_num = 256
# def inference(scene_idx):
mae_dict = np.zeros((scene_num, anno_num))
mse_dict = np.zeros((scene_num, anno_num))

scene_list = []
if split == 'test':
    for i in range(100, 190):
        scene_list.append(i)
elif split == 'test_seen':
    for i in range(100, 130):
        scene_list.append(i)
elif split == 'test_similar':
    for i in range(130, 160):
        scene_list.append(i)
elif split == 'test_novel':
    for i in range(169, 190):
        scene_list.append(i)
else:
    print('invalid split')
    
        
for scene_idx in scene_list:
    for anno_idx in range(256):

        rgb_path = os.path.join(dataset_root, 'scenes/scene_{:04d}/{}/rgb/{:04d}.png'.format(scene_idx, camera, anno_idx))
        depth_path = os.path.join(dataset_root, 'scenes/scene_{:04d}/{}/depth/{:04d}.png'.format(scene_idx, camera, anno_idx))
        gt_mask_path = os.path.join(dataset_root, 'scenes/scene_{:04d}/{}/label/{:04d}.png'.format(scene_idx, camera, anno_idx))
        seg_mask_path = os.path.join(dataset_root, 'seg_mask/scene_{:04d}/{}/{:04d}.png'.format(scene_idx, camera, anno_idx))
        meta_path = os.path.join(dataset_root, 'scenes/scene_{:04d}/{}/meta/{:04d}.mat'.format(scene_idx, camera, anno_idx))
        normal_path = os.path.join(dataset_root, 'normals/scene_{:04d}/{}/{:04d}.npy'.format(scene_idx, camera, anno_idx))
        suction_score_path = os.path.join(dataset_root, 'suction/scene_{:04d}/{}/{:04d}.npz'.format(scene_idx, camera, anno_idx))

        # depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
        # seg = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED).astype(np.bool)

        color = np.array(Image.open(rgb_path), dtype=np.float32) / 255.0
        depth = np.array(Image.open(depth_path))
        gt_seg = np.array(Image.open(gt_mask_path))
        net_seg = np.array(Image.open(seg_mask_path))
        normal = np.load(normal_path)

        suction_score_gt = np.load(suction_score_path)
        seal_score_gt = suction_score_gt['seal_score'][:, 0]
        wrench_score_gt = suction_score_gt['wrench_score'][:, 0]

        meta = scio.loadmat(meta_path)

        obj_idxs = meta['cls_indexes'].flatten().astype(np.int32)
        intrinsics = meta['intrinsic_matrix']
        factor_depth = meta['factor_depth']
        camera_info = CameraInfo(width, height, intrinsics[0][0], intrinsics[1][1], intrinsics[0][2], intrinsics[1][2], factor_depth)
        cloud = create_point_cloud_from_depth_image(depth, camera_info, organized=True)

        depth_mask = (depth > 0)
        camera_poses = np.load(os.path.join(dataset_root, 'scenes/scene_{:04d}/{}/camera_poses.npy'.format(scene_idx, camera)))
        align_mat = np.load(os.path.join(dataset_root, 'scenes/scene_{:04d}/{}/cam0_wrt_table.npy'.format(scene_idx, camera)))
        trans = np.dot(align_mat, camera_poses[anno_idx])
        workspace_mask = get_workspace_mask(cloud, gt_seg, trans=trans, organized=True, outlier=0.02)
        mask = (depth_mask & workspace_mask)

        cloud_masked = cloud[mask]
        color_masked = color[mask]
        seg_masked = net_seg[mask]
        normal_masked = normal

        scene = o3d.geometry.PointCloud()
        scene.points = o3d.utility.Vector3dVector(cloud_masked)
        scene.colors = o3d.utility.Vector3dVector(color_masked)
        # scene.estimate_normals(o3d.geometry.KDTreeSearchParamRadius(0.015), fast_normal_computation=True)
        # scene.orient_normals_to_align_with_direction(np.array([0., 0., -1.]))
        # scene.normalize_normals()
        # normal_masked = np.asarray(scene.normals).astype(np.float32)

        # print(seg_masked.shape)
        # print(normal_masked.shape)
        inst_cloud_list = []
        inst_color_list = []
        inst_coors_list = []
        inst_feats_list = []
        inst_normals_list = []
        inst_seal_score_list = []
        inst_wrench_score_list = []

        seg_idxs = np.unique(net_seg)
        for obj_idx in seg_idxs:
            if obj_idx == 0:
                continue
            inst_mask = seg_masked == obj_idx

            inst_mask_len = inst_mask.sum()
            if inst_mask_len < minimum_num_pt:
                continue
            if inst_mask_len >= num_pt:
                idxs = np.random.choice(inst_mask_len, num_pt, replace=False)
            else:
                idxs1 = np.arange(inst_mask_len)
                idxs2 = np.random.choice(inst_mask_len, num_pt - inst_mask_len, replace=True)
                idxs = np.concatenate([idxs1, idxs2], axis=0)

            inst_cloud_list.append(cloud_masked[inst_mask][idxs].astype(np.float32))
            inst_color_list.append(color_masked[inst_mask][idxs].astype(np.float32))
            inst_coors_list.append(cloud_masked[inst_mask][idxs].astype(np.float32) / voxel_size)
            inst_feats_list.append(color_masked[inst_mask][idxs].astype(np.float32))
            inst_normals_list.append(normal_masked[inst_mask][idxs].astype(np.float32))
            inst_seal_score_list.append(seal_score_gt[inst_mask][idxs].astype(np.float32))
            inst_wrench_score_list.append(wrench_score_gt[inst_mask][idxs].astype(np.float32))

        inst_cloud_tensor = torch.tensor(np.array(inst_cloud_list), dtype=torch.float32, device=device)
        inst_colors_tensor = torch.tensor(np.array(inst_color_list), dtype=torch.float32, device=device)
        inst_normals_tensor = torch.tensor(np.array(inst_normals_list), dtype=torch.float32, device=device)
        inst_seal_score_tensor = torch.tensor(np.array(inst_seal_score_list), dtype=torch.float32, device=device)
        inst_wrench_score_tensor = torch.tensor(np.array(inst_wrench_score_list), dtype=torch.float32, device=device)

        coordinates_batch, features_batch = ME.utils.sparse_collate(inst_coors_list, inst_feats_list,
                                                                    dtype=torch.float32)
        coordinates_batch, features_batch, _, quantize2original = ME.utils.sparse_quantize(
            coordinates_batch, features_batch, return_index=True, return_inverse=True)

        score_gt = inst_seal_score_tensor * inst_wrench_score_tensor
        batch_data_label = {"point_clouds": inst_cloud_tensor,
                            "cloud_colors": inst_colors_tensor,
                            "coors": coordinates_batch.to(device),
                            "feats": features_batch.to(device),
                            "quantize2original": quantize2original.to(device)
                            }

        # Forward pass
        # with torch.no_grad():
        #     end_points = net(batch_data_label)
        #
        # seal_score_logits = end_points['seal_score_pred']
        # wrench_score_logits = end_points['wrench_score_pred']
        #
        # # directly prediction
        # _, seal_score_pred = seal_score_logits.max(1)
        # _, wrench_score_pred = wrench_score_logits.max(1)
        # seal_score = bins[seal_score_pred + 1]
        # wrench_score = bins[wrench_score_pred + 1]
        # score = seal_score * wrench_score

        # # weighted prediction
        # seal_score_prob = F.softmax(seal_score_logits, 1)
        # wrench_score_prob = F.softmax(wrench_score_logits, 1)
        # seal_score = torch.sum(torch.ones_like(seal_score_prob) * seal_score_prob, 1)
        # wrench_score = torch.sum(torch.ones_like(wrench_score_prob) * wrench_score_prob, 1)
        # score = seal_score * wrench_score

        with torch.no_grad():
            end_points = net(batch_data_label)
        
        # seal_score_pred = end_points['seal_score_pred']
        # wrench_score_pred = end_points['wrench_score_pred']
        # score = seal_score_pred * wrench_score_pred
        
        score = end_points['score_pred']
        
        # seal_score_pred = normalize_tensor(seal_score_pred)
        # wrench_score_pred = normalize_tensor(wrench_score_pred)

        # Forward pass v0.2.6
        # with torch.no_grad():
        #     in_data = ME.TensorField(features=batch_data_label['feats'], coordinates=batch_data_label['coors'],
        #                             quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE)

        #     end_points = batch_data_label

        #     score, emb_mu, emb_sigma = net(in_data)
        #     end_points['score_pred'] = score
        #     end_points['emb_mu_dense'] = emb_mu.slice(in_data)
        #     end_points['emb_sigma_dense'] = emb_sigma.slice(in_data)
            
        #     B, point_num, _ = inst_cloud_tensor.shape  # batch _size
        #     score = score.view(B, point_num)
        #     uncertainty = emb_sigma.slice(in_data).F.view(B, point_num)

            # score, sigma = net(in_data)
            # end_points['score_pred'] = score
            # end_points['sigma_pred'] = sigma

            # B, point_num, _ = inst_cloud_tensor.shape  # batch _size
            # score = score.view(B, point_num)
            # uncertainty = sigma.view(B, point_num)

        mae_dict[scene_idx - 100, anno_idx] = mae_criterion(score, score_gt)
        mse_dict[scene_idx - 100, anno_idx] = mse_criterion(score, score_gt)
    
    print("Scene index:{}, Mean average error:{}, Mean square error:{}".format(scene_idx, 
                                                                               np.mean(mae_dict[scene_idx - 100, :]), 
                                                                               np.mean(mse_dict[scene_idx - 100, :])))

print("{}: Mean average error:{}, Mean square error:{}".format(split, np.mean(mae_dict[:, :]), np.mean(mse_dict[:, :])))

result_dict = {'mae': mae_dict, 'mse': mse_dict}
with open('result_{}_{}.pkl'.format(network_ver, epoch_num), 'wb') as file:
    pickle.dump(result_dict, file)

# Scene index:100, Mean average error:0.11768608872080222, Mean square error:0.033818830670497846 # network version:v0.2.1, epoch:40
# Scene index:100, Mean average error:0.1270925800781697  # network version:v0.2.4, epoch:40
# Scene index:100, Mean average error:0.22136519674677402  # network version: v0.2.5 epoch:36 directly prediction
# Scene index:100, Mean average error:0.810388406040147 # network version: v0.2.5 epoch:36 weighted prediction
# Scene index:100, Mean average error:0.20686332881450653 # network version: v0.2.5.1 epoch:60 directly prediction

# Scene index:100, Mean average error:0.15200641195406206  # network version:v0.2.6, epoch:26
# Scene index:100, Mean average error:0.16349999501835555  # network version:v0.2.6.2, epoch:7
# Scene index:100, Mean average error:0.12661055580247194, Mean square error:0.0428966131148627  # network version:v0.2.7, epoch:16