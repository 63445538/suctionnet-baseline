import copy
import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import scipy.io as scio
import time
import pickle

# import sys
# ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(os.path.join(ROOT_DIR, '..', 'utils'))

from ..utils.data_utils import CameraInfo, create_point_cloud_from_depth_image, get_workspace_mask
import MinkowskiEngine as ME

from suctionnetAPI.utils.rotation import viewpoint_to_matrix
from suctionnetAPI.utils.utils import plot_sucker
from suctionnetAPI.suction import SuctionGroup

import open3d as o3d
from SNet import SuctionNet
# import matplotlib.pyplot as plt

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
# 0.005 large degradation 
# Top-50 AP: 37.99 -> 12.26 (seen) 37.73 -> 15.60 (similar) 9.5 -> 3.16 (novel)
suction_height = 0.1
suction_radius = 0.01

parser = argparse.ArgumentParser()
parser.add_argument('--split', default='test_similar', help='dataset split [default: test_seen]')
parser.add_argument('--camera', default='kinect', help='camera to use [default: kinect]')
parser.add_argument('--sample_time', default=20, help='sample times for uncertainty estimation')
parser.add_argument('--dump_dir', default='save', help='where to save')
parser.add_argument('--gpu_id', default='0', help='GPU index')
parser.add_argument('--network_ver', default='v0.2.7.4', help='where to save')
parser.add_argument('--seg_model', default='uoais', help='where to save')
parser.add_argument('--epoch_num', default=40, help='where to save')
parser.add_argument('--dataset_root', default='/media/gpuadmin/rcao/dataset/graspnet', help='where dataset is')
parser.add_argument('--checkpoint_root', default='/media/gpuadmin/rcao/result/snet/', help='where dataset is')
cfgs = parser.parse_args()
print(cfgs)

network_ver = cfgs.network_ver
trained_epoch = cfgs.epoch_num
sample_time = int(cfgs.sample_time)
save_ver = '{}_{}_{}'.format(network_ver+'.p', trained_epoch, sample_time)
device = torch.device("cuda:{}".format(cfgs.gpu_id) if torch.cuda.is_available() else "cpu")

split = cfgs.split
camera = cfgs.camera
dataset_root = cfgs.dataset_root
checkpoint_root = cfgs.checkpoint_root
seg_model = cfgs.seg_model # 'uoais' 'uois'
if seg_model not in ['uoais', 'uois']:
    raise ValueError('unsupported segmentation model: ' + seg_model)
dump_dir = cfgs.dump_dir
torch.cuda.set_device(device)

# net = SuctionNet(feature_dim=512, is_training=False)
# net.to(device)
# net.eval()
# checkpoint = torch.load(os.path.join('log', 'snet_'+network_ver, camera, 'checkpoint_{}.tar'.format(trained_epoch)), map_location=device)
# # checkpoint = torch.load(os.path.join('log', 'snet_'+network_ver, camera, 'checkpoint.tar'), map_location=device)
# net.load_state_dict(checkpoint['model_state_dict'])

# v0.2.7
# from models.resunet import Res16UNet34CAleatoric
# net = Res16UNet34CAleatoric(in_channels=3, out_channels=1)
# net.to(device)
# # checkpoint = torch.load(os.path.join('log', 'snet_'+network_ver, camera, 'checkpoint.tar'), map_location=device)
# checkpoint = torch.load(os.path.join('log', 'snet_'+network_ver, camera, 'checkpoint_{}.tar'.format(trained_epoch)), map_location=device)
# net.load_state_dict(checkpoint['model_state_dict'])
# net.eval()

from SNet import SuctionNet_prob
net = SuctionNet_prob(feature_dim=512)
net.to(device)
checkpoint = torch.load(os.path.join(checkpoint_root, 'log', 'snet_'+network_ver, camera, 
                                     'checkpoint_{}.tar'.format(trained_epoch)), map_location=device)
net.load_state_dict(checkpoint['model_state_dict'])
net.eval()

dropout_prob = 0.1
# v0.2.7.1 0.5 v0.2.7.3 0.1
import torch.nn.functional as F
def dropout_hook_wrapper(module, sinput, soutput):
    input = soutput.F
    output = F.dropout(input, p=dropout_prob, training=True)
    soutput_new = ME.SparseTensor(output, coordinate_map_key=soutput.coordinate_map_key, coordinate_manager=soutput.coordinate_manager)
    return soutput_new
for module in net.modules():
    if isinstance(module, ME.MinkowskiConvolution):
        module.register_forward_hook(dropout_hook_wrapper)


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


def inference(scene_idx):
    infer_time_list = []
    for anno_idx in range(256):

        rgb_path = os.path.join(dataset_root, 'scenes/scene_{:04d}/{}/rgb/{:04d}.png'.format(scene_idx, camera, anno_idx))
        depth_path = os.path.join(dataset_root, 'scenes/scene_{:04d}/{}/depth/{:04d}.png'.format(scene_idx, camera, anno_idx))
        gt_mask_path = os.path.join(dataset_root, 'scenes/scene_{:04d}/{}/label/{:04d}.png'.format(scene_idx, camera, anno_idx))
        seg_mask_path = os.path.join(dataset_root, '{}_mask/scene_{:04d}/{}/{:04d}.png'.format(seg_model, scene_idx, camera, anno_idx))
        meta_path = os.path.join(dataset_root, 'scenes/scene_{:04d}/{}/meta/{:04d}.mat'.format(scene_idx, camera, anno_idx))
        normal_path = os.path.join(dataset_root, 'normals/scene_{:04d}/{}/{:04d}.npy'.format(scene_idx, camera, anno_idx))
        # suction_score_path = os.path.join(dataset_root, 'suction/scene_{:04d}/{}/{:04d}.npz'.format(scene_idx, camera, anno_idx))

        # depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
        # seg = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED).astype(np.bool)

        color = np.array(Image.open(rgb_path), dtype=np.float32) / 255.0
        depth = np.array(Image.open(depth_path))
        gt_seg = np.array(Image.open(gt_mask_path))
        net_seg = np.array(Image.open(seg_mask_path))
        normal = np.load(normal_path)
        
        # suction_score_gt = np.load(suction_score_path)
        # seal_score_gt = suction_score_gt['seal_score'][:, 0]
        # wrench_score_gt = suction_score_gt['wrench_score'][:, 0]

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
        
        # scene = o3d.geometry.PointCloud()
        # scene.points = o3d.utility.Vector3dVector(cloud_masked)
        # scene.colors = o3d.utility.Vector3dVector(color_masked)
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
        # inst_seal_score_list = []
        # inst_wrench_score_list = []

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
            # inst_seal_score_list.append(seal_score_gt[inst_mask][idxs].astype(np.float32))
            # inst_wrench_score_list.append(wrench_score_gt[inst_mask][idxs].astype(np.float32))

        inst_cloud_tensor = torch.tensor(np.array(inst_cloud_list), dtype=torch.float32, device=device)
        inst_colors_tensor = torch.tensor(np.array(inst_color_list), dtype=torch.float32, device=device)
        inst_normals_tensor = torch.tensor(np.array(inst_normals_list), dtype=torch.float32, device=device)

        coordinates_batch, features_batch = ME.utils.sparse_collate(inst_coors_list, inst_feats_list,
                                                                    dtype=torch.float32)
        coordinates_batch, features_batch, _, quantize2original = ME.utils.sparse_quantize(
            coordinates_batch, features_batch, return_index=True, return_inverse=True)

        batch_data_label = {"point_clouds": inst_cloud_tensor,
                            "cloud_colors": inst_colors_tensor,
                            "coors": coordinates_batch.to(device),
                            "feats": features_batch.to(device),
                            "quantize2original": quantize2original.to(device)
                            }

        # Forward pass
        # with torch.no_grad():
        #     end_points = net(batch_data_label)

        # seal_score_pred = end_points['seal_score_pred']
        # wrench_score_pred = end_points['wrench_score_pred']
        
        # # seal_score_pred = normalize_tensor(seal_score_pred)
        # # wrench_score_pred = normalize_tensor(wrench_score_pred)
        # suction_score = seal_score_pred * wrench_score_pred
        
        # seal_score_logits = end_points['seal_score_pred']
        # wrench_score_logits = end_points['wrench_score_pred']

        # _, seal_score_pred = seal_score_logits.max(1)
        # _, wrench_score_pred = wrench_score_logits.max(1)
        # seal_score = bins[seal_score_pred + 1]
        # wrench_score = bins[wrench_score_pred + 1]
        # suction_score = seal_score * wrench_score
        
        # Forward pass v0.2.7
        # with torch.no_grad():
        #     in_data = ME.TensorField(features=batch_data_label['feats'], coordinates=batch_data_label['coors'],
        #                     quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE)
        #     end_points = batch_data_label

        #     score, sigma = net(in_data)
        #     end_points['score_pred'] = score
        #     end_points['sigma_pred'] = sigma
            
        #     B, point_num = inst_cloud_tensor.shape[:2]
        #     score = score.view(B, point_num)
        #     sigma = sigma.view(B, point_num)
        
        # score = end_points['score_pred']
        # top_suction_scores, top_suction_indices = torch.topk(score, k=200, dim=1)
        
        # Bs, _ = top_suction_scores.size()
        # top_suction_scores = top_suction_scores.detach().cpu().numpy()
        # top_suction_indices = top_suction_indices.detach().cpu().numpy()
        torch.cuda.synchronize()
        start = time.time()
        with torch.no_grad():
            Sample_T = sample_time
            Bs, point_num = inst_cloud_tensor.shape[:2]
            score_sample = torch.zeros(Sample_T, Bs, point_num).to(device)
            sigma_sample = torch.zeros(Sample_T, Bs, point_num).to(device)
            for i in range(Sample_T):
                # in_data = ME.TensorField(features=batch_data_label['feats'], coordinates=batch_data_label['coors'],
                #                 quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE)
                # end_points = batch_data_label
                # score, sigma = net(in_data)
                # end_points['score_pred'] = score
                # end_points['sigma_pred'] = sigma
                
                end_points = net(batch_data_label)
                score = end_points['score_pred']
                sigma = end_points['sigma_pred']

                score = score.view(Bs, point_num)
                sigma = sigma.view(Bs, point_num)

                sigma = torch.exp(sigma)
                score_sample[i, :, :] = score
                sigma_sample[i, :, :] = sigma

            # uncertainty = torch.square(score)
            scores = torch.mean(score_sample, dim=0)
            uncertainty = torch.mean(torch.square(score_sample), dim=0) - torch.square(torch.mean(score_sample, dim=0)) + torch.mean(sigma_sample, dim=0)
            uncertainty = uncertainty / uncertainty.max(dim=1, keepdim=True)[0]
            # min-max normalization
            # uncertainty = (uncertainty - uncertainty.min(dim=1, keepdim=True)[0]) \
            #     / (uncertainty.max(dim=1, keepdim=True)[0] - uncertainty.min(dim=1, keepdim=True)[0])
        
        torch.cuda.synchronize()
        infer_time = time.time() - start
        infer_time_list.append(infer_time)
        
        _, top_suction_indices = torch.topk(scores * (1 - uncertainty), k=200, dim=1)
    
        scores = scores.detach().cpu().numpy()
        top_suction_indices = top_suction_indices.detach().cpu().numpy()

        inst_normals = inst_normals_tensor.detach().cpu().numpy()
        inst_cloud = inst_cloud_tensor.detach().cpu().numpy()

        suction_scores = []
        suction_directions = []
        suction_translations = []
        # inst_vis_pc_list = []
        # inst_vis_copy_list = []
        for i in range(Bs):
            top_50_suction_idx = top_suction_indices[i]
            suction_scores.append(scores[i][top_50_suction_idx])
            # suction_scores.append(top_suction_scores[i])
            suction_directions.append(inst_normals[i][top_50_suction_idx])
            suction_translations.append(inst_cloud[i][top_50_suction_idx])

            # inst_pc = o3d.geometry.PointCloud()
            # inst_pc.points = o3d.utility.Vector3dVector(inst_cloud_list[i])
            # # inst_pc.colors = o3d.utility.Vector3dVector(inst_color_list[i])
            # inst_pc.normals = o3d.utility.Vector3dVector(inst_normals_list[i])
            #
            # cmap = plt.get_cmap('viridis')
            # cmap_rgb = cmap(normalize(seal_score_vis[i]))[:, :3]  # wrench_score_vis  seal_score_vis
            # # cmap_rgb = cmap(normalize(inst_seal_score_list[i]))[:, :3]
            # # cmap_rgb = cmap(inst_wrench_score_list[i])[:, :3]
            #
            # inst_pc_copy = copy.deepcopy(inst_pc)
            # inst_pc_copy.points = o3d.utility.Vector3dVector(np.asarray(inst_pc_copy.points) + np.array([0, 0.5, 0]))
            # cmap_rgb_copy = cmap(normalize(inst_seal_score_list[i]))[:, :3]   # inst_wrench_score_list  inst_seal_score_list
            # inst_pc_copy.colors = o3d.utility.Vector3dVector(cmap_rgb_copy)
            #
            # inst_pc.colors = o3d.utility.Vector3dVector(cmap_rgb)
            # inst_vis_pc_list.append(inst_pc)
            # inst_vis_copy_list.append(inst_pc_copy)

        # o3d.visualization.draw_geometries(inst_vis_pc_list + inst_vis_copy_list)

        suction_scores = np.stack(suction_scores).reshape(-1)
        suction_directions = np.stack(suction_directions).reshape(-1, 3)
        suction_translations = np.stack(suction_translations).reshape(-1, 3)

        suction_arr = np.concatenate([suction_scores[..., np.newaxis], suction_directions, suction_translations], axis=-1)

        # suction_group = SuctionGroup(suction_arr)
        # suction_group = suction_group.nms(0.02, 181.0/180*np.pi)
        # suction_nms_arr = suction_group.suction_group_array
        
        suction_dir = os.path.join(dump_dir, split, 'scene_%04d'%scene_idx, camera, 'suction')
        os.makedirs(suction_dir, exist_ok=True)
        print('Saving:', suction_dir+'/%04d'%anno_idx+'.npz')
        np.savez(suction_dir+'/%04d'%anno_idx+'.npz', suction_arr)

        # downsampled_scene = scene.voxel_down_sample(voxel_size=0.005)
        # suckers = []
        # for sampled_idx, sampled_point in enumerate(suction_translations):
        #     normal = suction_directions[sampled_idx]
        #     R = viewpoint_to_matrix(normal)
        #     t = sampled_point
        #     sucker = plot_sucker(R, t, suction_scores[sampled_idx], suction_radius, suction_height)
        #     suckers.append(sucker)
        #
        # o3d.visualization.draw_geometries([downsampled_scene, *suckers], width=1536, height=864)

    result_dict = {'infer_time': np.array(infer_time_list)}
    with open(os.path.join(dump_dir, split, 'scene_%04d'%scene_idx, camera, 'infer_time.pkl'), 'wb') as file:
        pickle.dump(result_dict, file)

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
    for i in range(160, 190):
        scene_list.append(i)
else:
    print('invalid split')
# scene_list = [1]
for scene_idx in scene_list:
    inference(scene_idx)