import numpy as np
import timm
import json
import math
import torch
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import numpy as np
from PIL import Image
import json
import os

from torch.utils.data import Dataset, random_split
from PIL import Image
import json

from torch.utils.data import DataLoader
from torchvision import transforms



class NuScenesData(Dataset):
    def __init__(self, root_dir, max_len=None,transform_img=None, transform_bev=None):
        self.root_dir = root_dir
        self.sample_ids = sorted([
            fname.split('.')[0] for fname in os.listdir(os.path.join(root_dir, 'bev_img'))
            if fname.endswith('.npy')
        ])
        if max_len is not None:
            self.sample_ids = self.sample_ids[:max_len]
        self.transform_img = transform_img
        self.transform_bev = transform_bev

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        sid = self.sample_ids[idx]
        root_dir = self.root_dir
        # 读取文件
        bev_img = np.load(os.path.join(self.root_dir, 'bev_img', f'{sid}.npy'))        # (H, W, 2)
        bev_mask = np.load(os.path.join(self.root_dir, 'bev_mask', f'{sid}.npy'))      # (H, W)
        bev_polys = np.load(os.path.join(self.root_dir, 'bev_polys', f'{sid}.npy'), allow_pickle=True)
        trajectory = np.load(os.path.join(self.root_dir, 'trajectory', f'{sid}.npy'))  # (T, 2)
        img_transform = transforms.Compose([
                    transforms.Resize((900, 4200)),      # 调整大小到 (H, W)
                    transforms.ToTensor(),               # 转为 [0,1] 范围的 FloatTensor, shape (C,H,W)
                ])

        img_raw = Image.open(f"{root_dir}/img_raw/{sid}.jpg").convert('RGB')  # (H, W, 3)

        img_tensor = img_transform(img_raw)
        bev_transform = transforms.Compose([
        #    transforms.Resize((128,128)),
            transforms.ToTensor(),
            ])
        bev_view = Image.open(f"{root_dir}/bev_view/{sid}.jpg").convert('RGB')  # (H, W, 3
        bev_tensor = bev_transform(bev_view)
        bev_tensor = bev_tensor/225.
        with open(os.path.join(self.root_dir, 'img_boxes', f'{sid}.json'), 'r') as f:
            img_boxes = json.load(f)  # List[dict] with bbox
        trajectory = np.load(f"{root_dir}/trajectory/{sid}.npy")
        speed = np.load(f"{root_dir}/speed/{sid}.npy")   # a number
        target_image = bev_mask*0
        target_point = trajectory[-1,:]
        target_image[target_point[0],target_point[1]]=1
        target_image = np.expand_dims(target_image,axis=0)
        trajectory[:,0] = trajectory[:,0]-250
        trajectory = trajectory/10
        # print(trajectory)
        # Transform (if any)
        if self.transform_img:
            img_raw = self.transform_img(img_raw)

        if self.transform_bev:
            bev_img = self.transform_bev(bev_img)

        # 转为 tensor
        bev_img = torch.from_numpy(bev_img).permute(2, 0, 1).float()      # (2, H, W)
        bev_mask = torch.from_numpy(bev_mask).long()
        bev_tensor = bev_transform(bev_view)
        bev_tensor = bev_tensor/225.
        with open(os.path.join(self.root_dir, 'img_boxes', f'{sid}.json'), 'r') as f:
            img_boxes = json.load(f)  # List[dict] with bbox
        trajectory = np.load(f"{root_dir}/trajectory/{sid}.npy")
        speed = np.load(f"{root_dir}/speed/{sid}.npy")   # a number
        target_image = bev_mask*0
        target_point = trajectory[-1,:]
        target_image[target_point[0],target_point[1]]=1
        target_image = np.expand_dims(target_image,axis=0)
        trajectory[:,0] = trajectory[:,0]-250
        trajectory = trajectory/10
        # print(trajectory)
        # Transform (if any)
        if self.transform_img:
            img_raw = self.transform_img(img_raw)

        if self.transform_bev:
            bev_img = self.transform_bev(bev_img)

        # 转为 tensor
        bev_img = torch.from_numpy(bev_img).permute(2, 0, 1).float()      # (2, H, W)
        bev_mask = torch.from_numpy(bev_mask).long()
        #img_raw = torch.from_numpy(img_raw).permute(2, 0, 1).float()
                       # (3,H, W)
        trajectory = torch.from_numpy(trajectory).float()                # (T, 2)
        speed_value = float(speed)              # to Python float
        speed_tensor = torch.tensor([speed_value])  # shape (1, 1) tensor

        return {
            'bev_img': bev_img,
            'bev_mask': bev_mask,
            #'bev_polys': bev_polys,          # list of polygons (numpy arrays)
            'img_raw': img_tensor,              # PIL image (or transformed tensor)
            #'img_boxes': img_boxes,          # list of {"token", "bbox"}
            'trajectory': trajectory,
            'speed': speed_tensor,
            'bev_view':bev_tensor,
            'target_img': target_image,

        }
