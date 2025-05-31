from nuscenes.nuscenes import NuScenes
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import json
import cv2
from nuscene_data_process import *


save_dir = "/home/ubuntu/nuscenes_data/nuscenes_trainval/nuscenes"
nusc = NuScenes(version='v1.0-trainval', dataroot=save_dir, verbose=True)
my_scene = nusc.scene[0]
first_sample_token = my_scene['first_sample_token']
xrange = [-25,25]
yrange = [0,50]
bev_img, bev_mask, polys = read_lidar_from_token(nusc,first_sample_token,xrange,yrange)
img_with_box, img_raw, boxes = merge_images_and_boxes(nusc, first_sample_token)
future_wp = get_future_trajectory_matrix(nusc, first_sample_token,steps=10)
x_range = xrange
y_range =  yrange
x_pix, y_pix = project_to_bev_coords(future_wp,x_range,y_range,resolution=0.1)
speed = get_ego_speed(nusc,first_sample_token)


store_all_data(nusc, './data/train/')

