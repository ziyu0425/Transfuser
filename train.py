import timm
import json
import math
import torch
from torch import nn
import torch.nn.functional as F
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import numpy as np
from PIL import Image
import json
import os
from model import *

from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
from tqdm.auto import tqdm
import torch
import os
import numpy as np
import torch
from torch.utils.data import Dataset, random_split
from PIL import Image
import json

from torch.utils.data import DataLoader
from torchvision import transforms



class Engine(object):
  """
  Engine that runs training.
  """

  def __init__(self, model, optimizer, dataloader_train, dataloader_val, writer, device, cur_epoch=16,wp_only=False):
      self.cur_epoch = cur_epoch
      self.bestval_epoch = cur_epoch
      self.train_loss = []
      self.val_loss = []
      self.bestval = 1e10
      self.model = model
      self.optimizer = optimizer
      self.dataloader_train = dataloader_train
      self.dataloader_val   = dataloader_val
      self.writer = writer
      self.device = device
      self.vis_save_path = r'./visualizations'
      self.detailed_losses = ['loss_wp', 'loss_bev']
      if wp_only:
          detailed_losses_weights = [1.0, 0.0]
      else:
          detailed_losses_weights = [1.0,1.0]
      self.detailed_weights = {key: detailed_losses_weights[idx] for idx, key in enumerate(self.detailed_losses)}

  def load_data_compute_loss(self, data):

        # Move data to GPU
        rgb = data['img_raw'].to(self.device, dtype=torch.float32)
        bev = data['bev_view'].to(self.device, dtype=torch.long)
        lidar = data['bev_img'].to(self.device, dtype=torch.float32)
        ego_waypoint = data['trajectory'].to(self.device, dtype=torch.float32)
        target_point = ego_waypoint[:,-1,:]

        #target_point_image = data['target_point_image'].to(self.device, dtype=torch.float32)
        target_point_image = data['target_img'].to(self.device,dtype=torch.float32)
        ego_vel = data['speed'].to(self.device, dtype=torch.float32)
        #print('ego vel shape: ',ego_vel.shape)
        #print(bev.shape)
        losses = self.model(rgb, lidar, ego_waypoint=ego_waypoint, target_point=target_point,
                            target_point_image=target_point_image,
                            ego_vel=ego_vel, bev_mask=bev)
        return losses

  def train(self):
      self.model.train()

      num_batches = 0
      loss_epoch = 0.0
      detailed_losses_epoch  = {key: 0.0 for key in self.detailed_losses}
      self.cur_epoch += 1

      # Train loop
      for data in tqdm(self.dataloader_train, dynamic_ncols=True):
          self.optimizer.zero_grad(set_to_none=True)
          losses = self.load_data_compute_loss(data)
          loss = torch.tensor(0.0).to(self.device, dtype=torch.float32)

          for key, value in losses.items():
              loss += self.detailed_weights[key] * value
              detailed_losses_epoch[key] += float(self.detailed_weights[key] * value.item())
          loss.backward()

          self.optimizer.step()
          num_batches += 1
          loss_epoch += float(loss.item())
      self.log_losses(loss_epoch, detailed_losses_epoch, num_batches, '')
      print(f'current epoch: loss{detailed_losses_epoch}')
  @torch.inference_mode() # Faster version of torch_no_grad
  def validate(self):
        self.model.eval()

        num_batches = 0
        loss_epoch = 0.0
        detailed_val_losses_epoch  = {key: 0.0 for key in self.detailed_losses}

        # Evaluation loop loop
        for data in tqdm(self.dataloader_val):
            losses = self.load_data_compute_loss(data)

            loss = torch.tensor(0.0).to(self.device, dtype=torch.float32)

            for key, value in losses.items():
                loss += self.detailed_weights[key] * value
                detailed_val_losses_epoch[key] += float(self.detailed_weights[key] * value.item())

            num_batches += 1
            loss_epoch += float(loss.item())
        self.log_losses(loss_epoch, detailed_val_losses_epoch, num_batches, 'val_')
        print(f'validation loss: {loss_epoch}')

  def log_losses(self, loss_epoch, detailed_losses_epoch, num_batches, prefix=''):
        # Average all the batches into one number
        loss_epoch = loss_epoch / num_batches
        for key, value in detailed_losses_epoch.items():
            detailed_losses_epoch[key] = value / num_batches
        gathered_detailed_losses = [None]
        gathered_loss = [None]
        gathered_detailed_losses[0] = detailed_losses_epoch
        gathered_loss[0] = loss_epoch
        aggregated_total_loss = sum(gathered_loss) / len(gathered_loss)
        self.writer.add_scalar(prefix + 'loss_total', aggregated_total_loss, self.cur_epoch)

        # Log detailed losses
        for key, value in detailed_losses_epoch.items():
            aggregated_value = 0.0
            for i in range(1):
                aggregated_value += gathered_detailed_losses[i][key]

            aggregated_value = aggregated_value / 1

            self.writer.add_scalar(prefix + key, aggregated_value, self.cur_epoch)


  def save(self):
      # NOTE saving the model with torch.save(model.module.state_dict(), PATH) if parallel processing is used would be cleaner, we keep it for backwards compatibility
      torch.save(self.model.state_dict(), os.path.join('./log/', 'model_%d.pth' % self.cur_epoch))
      torch.save(self.optimizer.state_dict(), os.path.join('./log/', 'optimizer_%d.pth' % self.cur_epoch))



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

#dataset = NuScenesData(root_dir="./data/train/", transform_img=None)
#dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=2)
#train_size = int(0.8 * len(dataset))
#val_size = len(dataset) - train_size

#train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

#train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)
#val_loader   = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=4)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logdir = './log'
lr = 0.0001

model = TransfuserNet(device, pred_len=8 )
#checkpoint = torch.load("./model_ckpt/models_2022/transfuser/model_seed3_37.pth")
checkpoint = torch.load("./log/history/model_15.pth")
pretrained_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
#for key in pretrained_dict:
#    if 'transformer' not in key and 'lidar_encoder' not in key and 'image_encoder' not in key:
#        print(key)
#lidar_encoder_dict = {
#    k.replace("module.", ""): v
#    for k, v in pretrained_dict.items()
#    if k.startswith("module._model.lidar_encoder")
#}
#image_encoder_dict = {
#    k.replace("module.", ""): v
#    for k, v in pretrained_dict.items()
#    if k.startswith("module._model.image_encoder")
#}
#transformer_dict = {
#    k.replace("module.", ""): v
#    for k, v in pretrained_dict.items()
#    if k.startswith("module._model.trans")
#}
#model.load_state_dict(lidar_encoder_dict, strict=False)
#model.load_state_dict(image_encoder_dict, strict=False)
#model.load_state_dict(transformer_dict, strict=False)
model.load_state_dict(pretrained_dict,strict=False)
#for k in unexpect:
#    print(" -",k)
for name, param in model.named_parameters():
    if "_model.lidar_encoder" in name:
        param.requires_grad = False
    elif "_model.image_encoder" in name:
        param.requires_grad = False
    elif "_model.trans" in name:
        param.requires_grad = False
   # else:
    #    print(name)
cur_epoch = 16
model.to(device=device)
optimizer = optim.AdamW(model.parameters(), lr=lr) # For single GPU training
model_parameters = filter(lambda p: p.requires_grad, model.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print ('Total trainable parameters: ', params)
#for name, _  in model.named_parameters():
 #   if 'lidar_encoder' in name:
  #      print(name)

dataset = NuScenesData(root_dir="./data/train/", max_len=1000,transform_img=None)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=2)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

del dataset

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
val_loader   = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4)

for batch_idx, data in enumerate(train_loader):
    print(f"Batch {batch_idx}:")
    for key, value in data.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape}")
        elif isinstance(value, (list, tuple)):
            print(f"  {key}: list/tuple of length {len(value)}")
        else:
            print(f"  {key}: {type(value)}")
    break
if ((not os.path.isdir(logdir)) ):
        print('Created dir:', logdir)
        os.makedirs(logdir, exist_ok=True)
writer = SummaryWriter(log_dir=logdir)

trainer = Engine( model, optimizer, train_loader, val_loader, writer=writer, device=device,cur_epoch=cur_epoch)
for epoch in range(trainer.cur_epoch, 30):
  trainer.train()
  if epoch % 1 == 0:
            trainer.validate()
  trainer.save()
