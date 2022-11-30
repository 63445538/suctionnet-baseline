import copy
import os
import cv2
# import pcl
import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import scipy.io as scio
from utils.data_utils import CameraInfo, create_point_cloud_from_depth_image, get_workspace_mask
import MinkowskiEngine as ME

from suctionnetAPI.utils.rotation import viewpoint_to_matrix
from suctionnetAPI.utils.utils import plot_sucker

import open3d as o3d
from SNet import SuctionNet
import matplotlib.pyplot as plt

minimum_num_pt = 50
num_pt = 1024
width = 1280
height = 720
voxel_size = 0.002
suction_height = 0.1
suction_radius = 0.01
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('--split', default='test_seen', help='dataset split [default: test_seen]')
parser.add_argument('--camera', default='realsense', help='camera to use [default: kinect]')
parser.add_argument('--save_root', default='save', help='where to save')
parser.add_argument('--dataset_root', default='/media/rcao/Data/Dataset/graspnet', help='where dataset is')
parser.add_argument('--save_visu', action='store_true', default=True, help='whether to save visualization')
FLAGS = parser.parse_args()

network_ver = 'v0.1.1'
trained_epoch = 25
split = FLAGS.split
camera = FLAGS.camera
dataset_root = FLAGS.dataset_root
save_root = os.path.join(FLAGS.save_root, network_ver)

net = SuctionNet(feature_dim=512, is_training=False)
net.to(device)
net.eval()
checkpoint = torch.load(os.path.join('log', 'snet_'+network_ver, 'checkpoint_{}.tar'.format(trained_epoch)), map_location=device)
net.load_state_dict(checkpoint['model_state_dict'])

eps = 1e-12
def normalize(array):
    max = np.max(array)
    min = np.min(array)
    array = (array - min) / (max - min + eps)
    return array


def inference(scene_idx):
    for anno_idx in range(256):

        rgb_path = os.path.join(dataset_root, 'scenes/scene_{:04d}/{}/rgb/{:04d}.png'.format(scene_idx, camera, anno_idx))
        depth_path = os.path.join(dataset_root, 'scenes/scene_{:04d}/{}/depth/{:04d}.png'.format(scene_idx, camera, anno_idx))
        mask_path = os.path.join(dataset_root, 'scenes/scene_{:04d}/{}/label/{:04d}.png'.format(scene_idx, camera, anno_idx))
        meta_path = os.path.join(dataset_root, 'scenes/scene_{:04d}/{}/meta/{:04d}.mat'.format(scene_idx, camera, anno_idx))
        suction_score_path = os.path.join(dataset_root, 'suction/scene_{:04d}/{}/{:04d}.npz'.format(scene_idx, camera, anno_idx))

        # depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
        # seg = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED).astype(np.bool)

        color = np.array(Image.open(rgb_path), dtype=np.float32) / 255.0
        depth = np.array(Image.open(depth_path))
        seg = np.array(Image.open(mask_path))

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
        workspace_mask = get_workspace_mask(cloud, seg, trans=trans, organized=True, outlier=0.02)
        mask = (depth_mask & workspace_mask)

        cloud_masked = cloud[mask]
        color_masked = color[mask]
        seg_masked = seg[mask]

        scene = o3d.geometry.PointCloud()
        scene.points = o3d.utility.Vector3dVector(cloud_masked)
        scene.colors = o3d.utility.Vector3dVector(color_masked)
        scene.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(50), fast_normal_computation=False)
        scene.orient_normals_to_align_with_direction(np.array([0., 0., -1.]))
        normal_masked = np.asarray(scene.normals)

        inst_cloud_list = []
        inst_color_list = []
        inst_coors_list = []
        inst_feats_list = []
        inst_normals_list = []
        inst_seal_score_list = []
        inst_wrench_score_list = []

        for i, obj_idx in enumerate(obj_idxs):
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

        inst_cloud_tensor = torch.tensor(inst_cloud_list, dtype=torch.float32, device=device)
        # inst_colors_tensor = torch.tensor(inst_color_list, dtype=torch.float32, device=device)
        inst_colors_tensor = torch.ones_like(inst_cloud_tensor)
        inst_normals_tensor = torch.tensor(inst_normals_list, dtype=torch.float32, device=device)

        coordinates_batch, features_batch = ME.utils.sparse_collate(inst_coors_list, inst_feats_list,
                                                                    dtype=torch.float32)
        coordinates_batch, features_batch, _, quantize2original = ME.utils.sparse_quantize(
            coordinates_batch, features_batch, return_index=True, return_inverse=True)

        batch_data_label = {"point_clouds": inst_cloud_tensor,
                            "cloud_colors": inst_colors_tensor,
                            "coors": coordinates_batch.to(device),
                            "feats": features_batch.to(device),
                            "quantize2original": quantize2original.to(device)}

        end_points = net(batch_data_label)

        seal_score_pred = end_points['seal_score_pred']
        wrench_score_pred = end_points['wrench_score_pred']

        suction_score = seal_score_pred * wrench_score_pred
        # suction_score = seal_score_pred
        top_50_suction_scores, top_50_suction_indices = torch.topk(suction_score, k=50, dim=1)

        Bs, _ = top_50_suction_scores.size()
        top_50_suction_scores = top_50_suction_scores.detach().cpu().numpy()
        top_50_suction_indices = top_50_suction_indices.detach().cpu().numpy()

        inst_normals = inst_normals_tensor.detach().cpu().numpy()
        inst_cloud = inst_cloud_tensor.detach().cpu().numpy()

        seal_score_vis = seal_score_pred.detach().cpu().numpy()
        wrench_score_vis = wrench_score_pred.detach().cpu().numpy()

        suction_scores = []
        suction_directions = []
        suction_translations = []
        inst_vis_pc_list = []
        inst_vis_copy_list = []
        for i in range(Bs):
            top_50_suction_idx = top_50_suction_indices[i]
            suction_scores.append(top_50_suction_scores[i])
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

        # suction_scores = np.stack(suction_scores).reshape(-1)
        # # # test = torch.index_select(inst_normals_tensor, dim=(0, 1), top_50_suction_indices)
        # inst_normals_tensor = inst_normals_tensor.reshape(-1, 3)
        # inst_cloud_tensor = inst_cloud_tensor.reshape(-1, 3)
        # top_50_suction_indices = top_50_suction_indices.reshape(-1)
        #
        # suction_directions = inst_normals_tensor[top_50_suction_indices, :].detach().cpu().numpy()
        # suction_translations = inst_cloud_tensor[top_50_suction_indices, :].detach().cpu().numpy()

        suction_arr = np.concatenate([suction_scores[..., np.newaxis], suction_directions, suction_translations], axis=-1)

        suction_dir = os.path.join(save_root, split, 'scene_%04d'%scene_idx, camera, 'suction')
        os.makedirs(suction_dir, exist_ok=True)
        print('Saving:', suction_dir+'/%04d'%anno_idx+'.npz')
        # start_time = time.time()
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


scene_list = []
if split == 'test':
    for i in range(100, 190):
        scene_list.append(i)
elif split == 'test_seen':
    for i in range(101, 130):
        scene_list.append(i)
elif split == 'test_similiar':
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
