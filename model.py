import math
import torch
from torch import nn
import torch.nn.functional as F
#!pip uninstall -y sympy
#!pip install sympy==1.12
import timm
import json
from transfuser_boneneck import *
import numpy as np
class TransfuserNet(nn.Module):
    def __init__(self,device,use_velocity=True, pred_len=10,use_target_image=True):
        super().__init__()
        self.device = device
        self.pred_len = pred_len
        self.use_target_point_image = use_target_image
        self.gru_concat_target_point = True
        self.gru_hidden_size = 64
        self._model = TransfuserBackbone(use_target_point_image=self.use_target_point_image)
        channel = 64
        self.pred_bev = nn.Sequential(
                            nn.Conv2d(channel, channel, kernel_size=(3, 3), stride=1, padding=(1, 1), bias=True),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(channel, 3, kernel_size=(1, 1), stride=1, padding=0, bias=True)
        ).to(self.device)
        # waypoints prediction
        self.join = nn.Sequential(
                            nn.Linear(512, 256),
                            nn.ReLU(inplace=True),
                            nn.Linear(256, 128),
                            nn.ReLU(inplace=True),
                            nn.Linear(128, 64),
                            nn.ReLU(inplace=True),
                        ).to(self.device)

        self.decoder = nn.GRUCell(input_size=4 if self.gru_concat_target_point else 2, # 2 represents x,y coordinate
                                  hidden_size=self.gru_hidden_size).to(self.device)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.output = nn.Linear(self.gru_hidden_size, 3).to(self.device)
    
    def forward_gru(self, z, target_point):
        z = self.join(z)
        output_wp = list()
        # initial input variable to GRU
        x = torch.zeros(size=(z.shape[0], 2), dtype=z.dtype).to(z.device)
        target_point = target_point.clone()
        # autoregressive generation of output waypoints
        for _ in range(self.pred_len):
            if self.gru_concat_target_point:
                x_in = torch.cat([x, target_point], dim=1)
            else:
                x_in = x
            
            z = self.decoder(x_in, z)
            dx = self.output(z)
            
            x = dx[:,:2] + x

            output_wp.append(x[:,:2])

            pred_wp = torch.stack(output_wp, dim=1)

        
        return pred_wp



    def forward(self, rgb, bev, ego_waypoint, target_point, target_point_image, ego_vel, bev_mask, save_path=None):
        loss = {}
        if self.use_target_point_image:
          lidar_bev = torch.cat((bev,target_point_image),dim=1)
        else:
          lidar_bev = bev
        features, image_features_grid, fused_features = self._model(rgb, lidar_bev, target_point, ego_vel)

        pred_wp = self.forward_gru(fused_features, target_point)

        pred_bev = self.forward_gru(fused_features, target_point)
        pred_bev = self.pred_bev(features[0])
        pred_bev = F.interpolate(pred_bev, (bev_mask.shape[2], bev_mask.shape[3]), mode='bilinear', align_corners=True)
        weight = torch.from_numpy(np.array([1., 1., 3.])).to(dtype=torch.float32, device=pred_bev.device)
        bev_label = bev_mask.argmax(dim=1)
        #print(bev_label.shape)
        loss_bev = F.cross_entropy(pred_bev, bev_label, weight=weight).mean()

        loss_wp = torch.mean(torch.abs(pred_wp - ego_waypoint))
        loss.update({
            "loss_wp": loss_wp,
            "loss_bev": loss_bev
        })
        return loss







