import os
import argparse
import numpy as np
import torch
from PIL import Image
import scipy.io as scio
import cv2
from tqdm import tqdm

import sys
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, '..', 'utils'))

from data_utils import CameraInfo, create_point_cloud_from_depth_image, get_workspace_mask

from suctionnetAPI.utils.rotation import viewpoint_to_matrix, batch_viewpoint_to_matrix
from suctionnetAPI.utils.utils import plot_sucker
from suctionnetAPI.suction import SuctionGroup

import open3d as o3d

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
suction_height = 0.1
suction_radius = 0.01
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('--split', default='test', help='dataset split [default: test_seen]')
parser.add_argument('--camera', default='kinect', help='camera to use [default: kinect]')
parser.add_argument('--save_root', default='save', help='where to save')
parser.add_argument('--dataset_root', default='/data/rcao/dataset/graspnet', help='where dataset is')
FLAGS = parser.parse_args()

topk = 200
save_ver = 'geo_v0.2'
split = FLAGS.split
camera = FLAGS.camera
dataset_root = FLAGS.dataset_root
save_root = os.path.join(FLAGS.save_root, save_ver)

torch.cuda.set_device(device)

eps = 1e-12
def normalize(array):
    max = np.max(array)
    min = np.min(array)
    array = (array - min) / (max - min + eps)
    return array


def stdFilt(img, wlen):
    '''
    cal std filter of img
    :param img:
    :param wlen:  kernal size
    :return:
    '''
    wmean, wsqrmean = (cv2.boxFilter(x, -1, (wlen, wlen), borderType=cv2.BORDER_REFLECT) for x in (img, img*img))
    # pdb.set_trace()
    return np.sqrt(abs(wsqrmean - wmean*wmean))


def estimate_suction(scene_normal, scene_mask):

    normal_map = np.zeros([height, width, 3], dtype=np.float32)
    # normal_map = normal_map.reshape((-1, 3))
    normal_map[scene_mask, :] = scene_normal
    # normal_map = normal_map.reshape((height, width, 3))
    # print('filter start')
    # tic = time.time()
    # mean_normal_std = np.mean(generic_filter(normal_map, np.std, size=25), axis=2)
    mean_normal_std = np.mean(stdFilt(normal_map, 25), axis=2)
    # toc = time.time()
    # print('filter time:', toc - tic)
    heatmap = 1 - mean_normal_std / np.max(mean_normal_std)
    # heatmap = heatmap.reshape((-1, 3))
    heatmap[~scene_mask] = 0
    # heatmap = heatmap.reshape((height, width, 3))
    return heatmap, normal_map


k = 15.6
g = 9.8
radius = 0.01
wrench_thre = k * radius * np.pi
def batch_get_wrench_score(suction_points, directions, center, g_direction):
    gravity = g_direction * g

    suction_axis = batch_viewpoint_to_matrix(directions)
    bs = suction_axis.shape[0]

    suction2center = (center[np.newaxis, :] - suction_points)[:, np.newaxis, :]
    coord = np.matmul(suction2center, suction_axis)

    gravity_proj = np.matmul(np.tile(gravity[np.newaxis, :], (bs, 1, 1)), suction_axis)

    torque_y = gravity_proj[:, 0, 0] * coord[:, 0, 2] - gravity_proj[:, 0, 2] * coord[:, 0, 0]
    torque_z = -gravity_proj[:, 0, 0] * coord[:, 0, 1] + gravity_proj[:, 0, 1] * coord[:, 0, 0]

    torque_max = np.maximum(np.abs(torque_z), np.abs(torque_y))
    score = 1 - np.minimum(torque_max / wrench_thre, 1)

    return score


def inference(scene_idx):
    for anno_idx in range(256):

        rgb_path = os.path.join(dataset_root, 'scenes/scene_{:04d}/{}/rgb/{:04d}.png'.format(scene_idx, camera, anno_idx))
        depth_path = os.path.join(dataset_root, 'scenes/scene_{:04d}/{}/depth/{:04d}.png'.format(scene_idx, camera, anno_idx))
        gt_mask_path = os.path.join(dataset_root, 'scenes/scene_{:04d}/{}/label/{:04d}.png'.format(scene_idx, camera, anno_idx))
        seg_mask_path = os.path.join(dataset_root, 'seg_mask/scene_{:04d}/{}/{:04d}.png'.format(scene_idx, camera, anno_idx))
        meta_path = os.path.join(dataset_root, 'scenes/scene_{:04d}/{}/meta/{:04d}.mat'.format(scene_idx, camera, anno_idx))
        normal_path = os.path.join(dataset_root, 'normals/scene_{:04d}/{}/{:04d}.npy'.format(scene_idx, camera, anno_idx))
        # suction_score_path = os.path.join(dataset_root, 'suction/scene_{:04d}/{}/{:04d}.npz'.format(scene_idx, camera, anno_idx))

        color = np.array(Image.open(rgb_path), dtype=np.float32) / 255.0
        depth = np.array(Image.open(depth_path))
        gt_seg = np.array(Image.open(gt_mask_path))
        net_seg = np.array(Image.open(seg_mask_path))
        normal = np.load(normal_path)

        meta = scio.loadmat(meta_path)

        # depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
        # seg = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED).astype(np.bool)
        depth_mask = (depth > 0)
        camera_poses = np.load(os.path.join(dataset_root, 'scenes/scene_{:04d}/{}/camera_poses.npy'.format(scene_idx, camera)))
        align_mat = np.load(os.path.join(dataset_root, 'scenes/scene_{:04d}/{}/cam0_wrt_table.npy'.format(scene_idx, camera)))
        camera_pose = np.dot(align_mat, camera_poses[anno_idx])

        intrinsics = meta['intrinsic_matrix']
        factor_depth = meta['factor_depth']
        camera_info = CameraInfo(width, height, intrinsics[0][0], intrinsics[1][1], intrinsics[0][2], intrinsics[1][2], factor_depth)
        cloud = create_point_cloud_from_depth_image(depth, camera_info, organized=True)

        workspace_mask = get_workspace_mask(cloud, seg=gt_seg, trans=camera_pose, organized=True, outlier=0.02)
        mask = (depth_mask & workspace_mask)

        # cloud_masked = cloud[mask]
        # color_masked = color[mask]

        # scene = o3d.geometry.PointCloud()
        # scene.points = o3d.utility.Vector3dVector(cloud_masked)
        # scene.colors = o3d.utility.Vector3dVector(color_masked)
        # scene.normals = o3d.utility.Vector3dVector(normal)

        # suction_score_gt = np.load(suction_score_path)
        # seal_score_gt = suction_score_gt['seal_score'][:, 0]
        # wrench_score_gt = suction_score_gt['wrench_score'][:, 0]

        # seg_mask = net_seg.astype(np.bool_)
        heatmap, normal = estimate_suction(normal, mask)

        suction_scores = []
        suction_directions = []
        suction_translations = []

        seg_idxs = np.unique(net_seg)
        for obj_idx in seg_idxs:
            if obj_idx == 0:
                continue
            inst_mask = net_seg == obj_idx

            inst_mask_len = inst_mask.sum()
            if inst_mask_len < minimum_num_pt:
                continue

            inst_cloud = cloud[inst_mask, :]
            inst_color = color[inst_mask, :]
            inst_heatmap = heatmap[inst_mask]
            inst_normals = normal[inst_mask]

            nonzero_mask = np.where(np.all(inst_cloud[:, :] != [0.0, 0.0, 0.0], axis=-1) == True)
            inst_cloud = inst_cloud[nonzero_mask]
            inst_color = inst_color[nonzero_mask]
            inst_heatmap = inst_heatmap[nonzero_mask]
            inst_normals = inst_normals[nonzero_mask]

            inst_pc = o3d.geometry.PointCloud()
            inst_pc.points = o3d.utility.Vector3dVector(inst_cloud.reshape(-1, 3))

            try:
                inst_hull, _ = inst_pc.compute_convex_hull()
            except:
                print('error raised when computing convex hull')
                continue
            hull_ls = o3d.geometry.LineSet.create_from_triangle_mesh(inst_hull)
            hull_ls.paint_uniform_color((1, 0, 0))
            hull_center = inst_hull.get_center()

            wrench_score = batch_get_wrench_score(inst_cloud, inst_normals, hull_center, np.array([[0, 0, -1]]))
            # seal_score = normalize(inst_heatmap)

            seal_score = inst_heatmap
            geo_score = seal_score * wrench_score

            inst_mask_len = len(inst_cloud)
            if inst_mask_len >= num_pt:
                idxs = np.random.choice(inst_mask_len, num_pt, replace=False)
            else:
                idxs1 = np.arange(inst_mask_len)
                idxs2 = np.random.choice(inst_mask_len, num_pt - inst_mask_len, replace=True)
                idxs = np.concatenate([idxs1, idxs2], axis=0)

            sampled_cloud = inst_cloud[idxs]
            sampled_score = geo_score[idxs]
            sampled_normals = inst_normals[idxs]

            topk_idx = np.argsort(sampled_score)[::-1][:topk]
            suction_scores.append(sampled_score[topk_idx])
            suction_directions.append(sampled_normals[topk_idx])
            suction_translations.append(sampled_cloud[topk_idx])

        suction_scores = np.stack(suction_scores).reshape(-1)
        suction_directions = np.stack(suction_directions).reshape(-1, 3)
        suction_translations = np.stack(suction_translations).reshape(-1, 3)

        suction_arr = np.concatenate([suction_scores[..., np.newaxis], suction_directions, suction_translations], axis=-1)

        suction_group = SuctionGroup(suction_arr)
        suction_group = suction_group.nms(0.02, 181.0/180*np.pi)
        suction_nms_arr = suction_group.suction_group_array

        suction_dir = os.path.join(save_root, split, 'scene_%04d'%scene_idx, camera, 'suction')
        os.makedirs(suction_dir, exist_ok=True)
        print('Saving:', suction_dir+'/%04d'%anno_idx+'.npz')
        np.savez(suction_dir+'/%04d'%anno_idx+'.npz', suction_nms_arr)

        # downsampled_scene = scene.voxel_down_sample(voxel_size=0.005)
        # nms_score = suction_group.scores()
        # nms_t = suction_group.translations()
        # nms_r = suction_group.directions()
        # 
        # suckers = []
        # for sampled_idx, sampled_point in enumerate(nms_t):
        #     normal = nms_r[sampled_idx]
        #     R = viewpoint_to_matrix(normal)
        #     t = sampled_point
        #     sucker = plot_sucker(R, t, nms_score[sampled_idx], suction_radius, suction_height)
        #     suckers.append(sucker)
        # 
        # o3d.visualization.draw_geometries([downsampled_scene, *suckers], width=1536, height=864)


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
for scene_idx in tqdm(scene_list):
    inference(scene_idx)