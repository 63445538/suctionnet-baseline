import os
import cv2
import math
import time
import numpy as np
from PIL import Image
import scipy.io as scio
import random
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from utils.image import get_affine_transform


class CameraInfo():
    def __init__(self, width, height, fx, fy, cx, cy, scale):
        self.width = width
        self.height = height
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.scale = scale


def drawGaussian(img, pt, score, sigma=1):
    """Draw 2d gaussian on input image.
    Parameters
    ----------
    img: torch.Tensor
        A tensor with shape: `(3, H, W)`.
    pt: list or tuple
        A point: (x, y).
    sigma: int
        Sigma of gaussian distribution.
    Returns
    -------
    torch.Tensor
        A tensor with shape: `(3, H, W)`.
    """
    # img = to_numpy(img)
    tmp_img = np.zeros([img.shape[0], img.shape[1]], dtype=np.float32)
    tmpSize = 3 * sigma
    # Check that any part of the gaussian is in-bounds
    ul = [int(pt[0] - tmpSize), int(pt[1] - tmpSize)]
    br = [int(pt[0] + tmpSize + 1), int(pt[1] + tmpSize + 1)]

    if (ul[0] >= img.shape[1] or ul[1] >= img.shape[0] or br[0] < 0 or br[1] < 0):
        # If not, just return the image as is
        return img

    # Generate gaussian
    size = 2 * tmpSize + 1
    x = np.arange(0, size, 1, float)
    # print('x:', x.shape)
    y = x[:, np.newaxis]
    # print('x:', x.shape)
    x0 = y0 = size // 2
    # The gaussian is not normalized, we want the center value to equal 1
    
    # print('g:', g.shape)
    # Usable gaussian range
    g_x = max(0, -ul[0]), min(br[0], img.shape[1]) - ul[0]
    g_y = max(0, -ul[1]), min(br[1], img.shape[0]) - ul[1]
    # Image range
    img_x = max(0, ul[0]), min(br[0], img.shape[1])
    img_y = max(0, ul[1]), min(br[1], img.shape[0])

    g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2)) * score
    g = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
    # g = np.concatenate([g[..., np.newaxis], np.zeros([g.shape[0], g.shape[1], 2], dtype=np.float32)], axis=-1)

    tmp_img[img_y[0]:img_y[1], img_x[0]:img_x[1]] = g
    # img[img_y[0]:img_y[1], img_x[0]:img_x[1]] = g
    img += tmp_img

    return img


class ARCDataset(Dataset):
    def __init__(self, dataset_root, split='train', data_aug=False, input_size=(480, 480)):
        # assert(num_points<=50000)
        self.dataset_root = dataset_root
        self.split = split
        self.dim = input_size
        self.data_rng = np.random.RandomState(123)
        self.data_aug = data_aug
        # self.remove_outlier = remove_outlier
        # self.valid_obj_idxs = valid_obj_idxs
        # self.augment = augment
        # self.crop = crop
                
        self.data_list = []
        with open(os.path.join(self.dataset_root, '{}-split.txt'.format(self.split))) as f:
            for line in f:
                self.data_list.append(line.strip())
        if split == 'train':
            random.shuffle(self.data_list)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        scene_name = self.data_list[index]

        color_dir = os.path.join(self.dataset_root, 'color-input', scene_name + '.png')
        depth_dir = os.path.join(self.dataset_root, 'depth-input',  scene_name + '.png')
        score_dir = os.path.join(self.dataset_root, 'label', scene_name + '.png')
        
        # print('color_dir:', color_dir)
        # tic = time.time()
        color = cv2.imread(color_dir, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.0
        depth = cv2.imread(depth_dir, cv2.IMREAD_UNCHANGED).astype(np.float32) / 10000.0
        score = cv2.imread(score_dir, cv2.IMREAD_UNCHANGED).astype(np.int16)
        # toc = time.time()
        # print('input read time:', toc-tic)

        if self.split == 'train':
            color, depth, score = self.crop_array(color, depth, score, self.dim)
            # color_noise = np.random.normal(scale=0.03, size=color.shape).astype(np.float32)
            depth_noise = np.random.normal(scale=0.03, size=depth.shape).astype(np.float32)
            # color = color + color_noise
            if self.data_aug:
                color = color_aug(self.data_rng, color)
                depth = depth + depth_noise

        return color, depth, score, scene_name
    
    def augment(self, img, depth, score):
        input_h, input_w = img.shape[0], img.shape[1]
        s = max(input_h, input_w) * 1.0
        c = np.array([input_w / 2., input_h / 2.], dtype=np.float32)
        
        # flipped = False
        flip = 0.3

        s = s * np.random.choice(np.arange(0.8, 1.2, 0.1))
        w_border = self._get_border(128, img.shape[1])
        h_border = self._get_border(128, img.shape[0])
        c[0] = np.random.randint(low=w_border, high=img.shape[1] - w_border)
        c[1] = np.random.randint(low=h_border, high=img.shape[0] - h_border)
        
        if np.random.random() < flip:
            # flipped = True
            img = img[:, ::-1, :]
            depth = depth[:, ::-1]
            score = score[:, ::-1]
            c[0] = input_w - c[0] - 1
        
        trans_input = get_affine_transform(c, s, 0, [input_w, input_h])
        img = cv2.warpAffine(img, trans_input, 
                            (input_w, input_h),
                            flags=cv2.INTER_LINEAR)
        depth = cv2.warpAffine(depth, trans_input, 
                            (input_w, input_h),
                            flags=cv2.INTER_LINEAR)
        score = cv2.warpAffine(score, trans_input, 
                            (input_w, input_h),
                            flags=cv2.INTER_NEAREST)

        return img, depth, score

    def _get_border(self, border, size):
        i = 1
        while size - border // i <= border // i:
            i *= 2
        return border // i
    
    # def crop_array(self, color, depth, score, t_size=(480, 480)):
    #     height, width = color.shape[0], color.shape[1]
    #     center_x = np.random.randint(t_size[1] // 2, width - t_size[1] // 2)
    #     center_y = np.random.randint(t_size[0] // 2, height - t_size[0] // 2)

    #     cropped_color = color[center_y-t_size[0]//2:center_y+t_size[0]//2, center_x-t_size[1]//2:center_x+t_size[1]//2]
    #     cropped_depth = depth[center_y-t_size[0]//2:center_y+t_size[0]//2, center_x-t_size[1]//2:center_x+t_size[1]//2]
    #     cropped_score = score[center_y-t_size[0]//2:center_y+t_size[0]//2, center_x-t_size[1]//2:center_x+t_size[1]//2]
    #     return cropped_color, cropped_depth, cropped_score

    def crop_array(self, color, depth, score, t_size=(480, 480)):
        height, width = color.shape[:2]
        # 高度方向上不随机选择中心，因为裁剪大小等于原图大小
        center_y = t_size[0] // 2
        
        # 宽度方向上进行随机选择中心点
        if width > t_size[1]:
            center_x = np.random.randint(t_size[1] // 2, width - t_size[1] // 2)
        else:
            center_x = width // 2  # 如果宽度小于等于目标裁剪尺寸，则直接使用中心
        
        # 计算裁剪的起始和结束位置
        start_x = max(center_x - t_size[1] // 2, 0)
        end_x = start_x + t_size[1]
        start_y = max(center_y - t_size[0] // 2, 0)
        end_y = start_y + t_size[0]
        
        # 根据计算的起始和结束位置进行裁剪
        cropped_color = color[start_y:end_y, start_x:end_x]
        cropped_depth = depth[start_y:end_y, start_x:end_x]
        cropped_score = score[start_y:end_y, start_x:end_x]

        return cropped_color, cropped_depth, cropped_score

def grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def lighting_(data_rng, image, alphastd, eigval, eigvec):
    alpha = data_rng.normal(scale=alphastd, size=(3, ))
    image += np.dot(eigvec, eigval * alpha)
    return image

def blend_(alpha, image1, image2):
    image1 *= alpha
    image2 *= (1 - alpha)
    image1 += image2
    return image1

def saturation_(data_rng, image, gs, gs_mean, var):
    alpha = 1. + data_rng.uniform(low=-var, high=var)
    return blend_(alpha, image, gs[:, :, None])

def brightness_(data_rng, image, gs, gs_mean, var):
    alpha = 1. + data_rng.uniform(low=-var, high=var)
    image *= alpha
    return image

def contrast_(data_rng, image, gs, gs_mean, var):
    alpha = 1. + data_rng.uniform(low=-var, high=var)
    return blend_(alpha, image, gs_mean)

def color_aug(data_rng, image):
    functions = [brightness_, contrast_, saturation_]
    random.shuffle(functions)

    gs = grayscale(image)
    gs_mean = gs.mean()
    for f in functions:
        image = f(data_rng, image, gs, gs_mean, 0.4)
    # image = lighting_(data_rng, image, 0.1, eig_val, eig_vec)
    return image

def color_aug2(data_rng, image, eig_val, eig_vec):
    functions = [brightness_, contrast_, saturation_]
    random.shuffle(functions)

    gs = grayscale(image)
    gs_mean = gs.mean()
    for f in functions:
        image = f(data_rng, image, gs, gs_mean, 0.4)
    image = lighting_(data_rng, image, 0.1, eig_val, eig_vec)
    return image


if __name__ == "__main__":
    dataset_root = '/media/gpuadmin/rcao/dataset/ARC'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_dataset = ARCDataset(dataset_root, split='train')
    vis_root = os.path.join('vis')
    os.makedirs(vis_root, exist_ok=True)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=8,
        pin_memory=False
        )
    
    for i, data in train_loader:
        image, depth, label, scene_name = data[0], data[1], data[2], data[3]
        image = image.to(device)
        depth = depth.to(device)
        label = label.to(device)

        image_vis = image.detach().cpu().numpy()
        depth_vis = depth.detach().cpu().numpy() * 1000.0
        label_vis = label.detach().cpu().numpy()
        label_vis = label_vis / np.max(label_vis) * 255
        cv2.imwrite(os.path.join(vis_root, scene_name + '_image.png'), image_vis)
        cv2.imwrite(os.path.join(vis_root, scene_name + '_depth.png'), depth_vis)
        cv2.imwrite(os.path.join(vis_root, scene_name + '_label.png'), label_vis)
         
        if i > 0:
            break
        
    print(len(train_dataset))

    # end_points = train_dataset[233]
    # cloud = end_points['point_clouds']
    # seg = end_points['objectness_label']