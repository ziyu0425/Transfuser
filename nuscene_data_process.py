
import os
#FOLDERNAME = 'cs231n/proposal/'
#save_dir = "/content/drive/My Drive/{}/data/sets/nuscenes".format(FOLDERNAME)
from nuscenes.nuscenes import NuScenes
#nusc = NuScenes(version='v1.0-mini', dataroot=save_dir, verbose=True)

import numpy as np
import cv2  
from PIL import Image  
from nuscenes.utils.data_classes import Box
from nuscenes.utils.geometry_utils import transform_matrix
from pyquaternion import Quaternion
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import view_points
from nuscenes.map_expansion.map_api import NuScenesMap
def read_lidar_from_token(nusc,sample_token,xrange=[-50,50], yrange=[-50,50],z_range=(-2,5)):
  sample = nusc.get('sample', sample_token)
  data_token_dic = sample['data']
  lidar_token = data_token_dic['LIDAR_TOP']
  lidar_sd = nusc.get('sample_data',lidar_token)

  lidar_path = nusc.get_sample_data_path(lidar_token)
  lidar_points = np.fromfile(lidar_path,dtype = np.float32).reshape(-1,5)
  lidarseg_label_path = f"{nusc.dataroot}/lidarseg/v1.0-trainval/{lidar_token}_lidarseg.bin"
  labels = np.fromfile(lidarseg_label_path, dtype=np.uint8)
  x, y, z, intensity = lidar_points[:, 0], lidar_points[:, 1], lidar_points[:, 2], lidar_points[:, 3]
  points = lidar_points
  resolution = 0.1
  x_range = xrange
  y_range = yrange
  mask = (
    (points[:, 0] >= x_range[0]) & (points[:, 0] <= x_range[1]) &
    (points[:, 1] >= y_range[0]) & (points[:, 1] <= y_range[1]) &
    (points[:, 2] >= z_range[0]) & (points[:, 2] <= z_range[1])
    )
  points = points[mask]
  x, y, intensity, z = x[mask], y[mask], intensity[mask], z[mask]
  x_indices = ((points[:, 0] - x_range[0]) / resolution).astype(np.int32)
  y_indices = ((points[:, 1] - y_range[0]) / resolution).astype(np.int32)

  bev_w = int((x_range[1] - x_range[0]) / resolution)
  bev_h = int((y_range[1] - y_range[0]) / resolution)
  x_img = ((x - x_range[0]) / resolution).astype(np.int32)
  y_img = ((y - y_range[0]) / resolution).astype(np.int32)
  bev_mask = np.zeros((bev_h, bev_w), dtype=np.uint8)
  bev_img = np.zeros((bev_h, bev_w,2), dtype=np.float32)
  #bev_img[y_img, x_img,0] = intensity
  #bev_img[y_img, x_img,1] = z
  #for xi, yi in zip(x_indices, y_indices):
    #bev_img[ yi, xi,0] += 1
  for i, (xi, yi) in enumerate(zip(x_indices, y_indices)):
    zi = points[i, 2]
    intense = points[i,3]
    bev_img[yi, xi,1] = max(bev_img[yi, xi,1], zi)
    bev_img[ yi, xi,0] += intense
  #bev_img[y_img, x_img,0] = intensity
  #bev_img[y_img, x_img,1] = z
  
  
  bev_mask[y_img, x_img] = labels[mask]
  
  polys = []
  cs_record = nusc.get('calibrated_sensor', lidar_sd['calibrated_sensor_token']) # find the calibrate sensor
  pose_record = nusc.get('ego_pose', lidar_sd['ego_pose_token'])
  lidar_from_car = transform_matrix(
    cs_record['translation'],
    Quaternion(cs_record['rotation']),
    inverse=True
  )

  car_from_global = transform_matrix(
      pose_record['translation'],
      Quaternion(pose_record['rotation']),
      inverse=True
  )
  for ann_token in sample['anns']:
    ann = nusc.get('sample_annotation', ann_token)

    if ann['category_name'][0:7] != "vehicle": # we only care about the vehicle part
      continue
    box = Box(ann['translation'], ann['size'], Quaternion(ann['rotation']))

    # 坐标变换：global → lidar
    #box.transform(global_to_lidar)
    box.translate(-np.array(pose_record['translation']))
    box.rotate(Quaternion(pose_record['rotation']).inverse)
    box.translate(-np.array(cs_record['translation']))
    box.rotate(Quaternion(cs_record['rotation']).inverse)
    # 获取框的 4 个底面点 (corners shape: [3, 8])
    corners = box.corners()[:2, [0, 1, 5, 4]]  # x-y 的前左后右顺序

    # 将 lidar 坐标 → BEV 像素坐标
    x_img = (corners[0] - x_range[0]) / resolution
    y_img = (corners[1] - y_range[0]) / resolution
    poly = np.stack([y_img, x_img], axis=1)
    if np.max(poly[:,0])<(x_range[1]-x_range[0]) / resolution and np.min(poly[:,0])>0:
      if np.max(poly[:,1])<(y_range[1]-y_range[0])/ resolution and np.min(poly[:,1])>0:
        poly_int = poly.astype(int)
        polys.append(poly_int)
  return bev_img, bev_mask,polys #(H,W,C), (y,x)


def read_image_from_token(nusc, sample_token, cam_name='CAM_FRONT'):
    """
    Args:
        nusc: NuScenes 
        sample_token (str): sample token
        cam_name (str):  'CAM_FRONT', 'CAM_BACK_LEFT' 

    Returns:
        img (np.ndarray):  (H, W, 3)
        intrinsics (np.ndarray):  (3, 3)
        cs_record (dict):  rotation translation
        pose_record (dict): ego pose
        cam_token (str): sample_data token
    """
    
    sample = nusc.get('sample', sample_token)
    cam_token = sample['data'][cam_name]
    cam_sd = nusc.get('sample_data', cam_token)
    
    
    img_path = nusc.get_sample_data_path(cam_token)
    image = Image.open(img_path)
    return np.array(image) 

def show_front_view_combined(nusc, sample_token, crop_margin=100):
    cam_names = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT']
    imgs = [read_image_from_token(nusc, sample_token, name) for name in cam_names]
    imgs[1] = imgs[1][:,crop_margin:-crop_margin,:]
    imgs[0] = imgs[0][:,crop_margin:-crop_margin,:]
    imgs[2] = imgs[2][:,crop_margin:-crop_margin,:]
    # Resize to same height if needed (optional, here assumes all same size)
    # Concatenate horizontally
    combined = np.concatenate(imgs, axis=1)
    return combined   


def project_lidarseg_to_image(nusc, sample_token, cam_name='CAM_FRONT', lidar_name='LIDAR_TOP'):
    sample = nusc.get('sample', sample_token)
    
    cam_token = sample['data'][cam_name]
    lidar_token = sample['data'][lidar_name]

    cam_data = nusc.get('sample_data', cam_token)
    lidar_data = nusc.get('sample_data', lidar_token)

    cam_calib = nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
    lidar_calib = nusc.get('calibrated_sensor', lidar_data['calibrated_sensor_token'])

    cam_intrinsic = np.array(cam_calib['camera_intrinsic'])

    # 读取图像大小
    img = Image.open(nusc.get_sample_data_path(cam_token)).convert("RGB")
    img_w, img_h = img.size
    mask = np.zeros((img_h, img_w), dtype=np.uint8)  # 初始化标签图

    # 加载点云
    lidar_points = LidarPointCloud.from_file(nusc.get_sample_data_path(lidar_token)).points  # (4, N)

    # 加载语义标签
    lidarseg_path = f"{nusc.dataroot}/lidarseg/v1.0-trainval/{lidar_token}_lidarseg.bin"
    labels = np.fromfile(lidarseg_path, dtype=np.uint8)

    # lidar → ego
    lidar2ego = transform_matrix(
        lidar_calib['translation'], Quaternion(lidar_calib['rotation']), inverse=False
    )
    points_ego = lidar2ego @ np.vstack((lidar_points[:3, :], np.ones(lidar_points.shape[1])))

    # ego → global
    lidar_pose = nusc.get('ego_pose', lidar_data['ego_pose_token'])
    ego2global = transform_matrix(
        lidar_pose['translation'], Quaternion(lidar_pose['rotation']), inverse=False
    )
    points_global = ego2global @ points_ego

    # global → ego(cam)
    cam_pose = nusc.get('ego_pose', cam_data['ego_pose_token'])
    global2cam_ego = transform_matrix(
        cam_pose['translation'], Quaternion(cam_pose['rotation']), inverse=True
    )
    points_cam_ego = global2cam_ego @ points_global

    # ego(cam) → cam
    ego2cam = transform_matrix(
        cam_calib['translation'], Quaternion(cam_calib['rotation']), inverse=True
    )
    points_cam = ego2cam @ points_cam_ego

    # 相机空间投影到图像平面
    depths = points_cam[2, :]
    points_cam = points_cam[:3, depths > 0]  # 只保留前方点
    labels = labels[depths > 0]
    depths = depths[depths > 0]

    proj = cam_intrinsic @ points_cam
    proj[:2, :] /= proj[2, :]

    u, v = proj[0, :], proj[1, :]
    u, v = u.astype(np.int32), v.astype(np.int32)

    valid = (u >= 0) & (u < img_w) & (v >= 0) & (v < img_h)
    u, v = u[valid], v[valid]
    labels = labels[valid]

    mask[v, u] = labels  # 注意这里是 mask[y, x] = label

    return np.array(img), mask

def draw_3d_boxes_on_image(nusc, sample_token, cam_name, box_color=(0, 255, 0), crop_margin=100):
    from nuscenes.utils.geometry_utils import view_points, transform_matrix
    from nuscenes.utils.data_classes import Box
    from pyquaternion import Quaternion

    sample = nusc.get('sample', sample_token)
    cam_data = nusc.get('sample_data', sample['data'][cam_name])
    cs_record = nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
    pose_record = nusc.get('ego_pose', cam_data['ego_pose_token'])
    cam_intrinsic = np.array(cs_record['camera_intrinsic'])
    img_path = nusc.get_sample_data_path(cam_data['token'])
    raw_img = np.array(Image.open(img_path))   # ← 原始图像保留不带框
    img = raw_img.copy()                        # ← 用于绘制框的图像副本

    boxes = []
    for ann_token in sample['anns']:
        ann = nusc.get('sample_annotation', ann_token)
        #if not ann['category_name'].startswith('vehicle'):
           # continue
        if ann['visibility_token'] == '1':
          
          continue
        box = Box(ann['translation'], ann['size'], Quaternion(ann['rotation']))
        box.translate(-np.array(pose_record['translation']))
        box.rotate(Quaternion(pose_record['rotation']).inverse)
        box.translate(-np.array(cs_record['translation']))
        box.rotate(Quaternion(cs_record['rotation']).inverse)
        corners = box.corners()
        if np.any(corners[2, :] < 1e-3):
            continue
        corners_2d = view_points(corners, cam_intrinsic, normalize=True).astype(np.int32)
        x1, y1 = np.min(corners_2d[0]), np.min(corners_2d[1])
        x2, y2 = np.max(corners_2d[0]), np.max(corners_2d[1])
        if x2 < crop_margin+50 or x1 > img.shape[1] - crop_margin-50:
            continue
        if (x2 - x1) < 50 or (y2 - y1) < 50:
            continue

        x1 = max(x1, crop_margin)
        x2 = min(x2, img.shape[1] - crop_margin)
        img = cv2.rectangle(img, (x1, y1), (x2, y2), box_color, 2)
        boxes.append({
            'token': ann_token,
            'bbox': (int(x1 - crop_margin), int(y1), int(x2 - crop_margin), int(y2))
        })

    cropped_img_with_box = img[:, crop_margin:-crop_margin]
    cropped_img_raw = raw_img[:, crop_margin:-crop_margin]

    return cropped_img_with_box, cropped_img_raw, boxes

def merge_images_and_boxes(nusc, sample_token, cam_names=['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT'], box_color=(0,255,0), crop_margin=100):
    imgs_with_box = []
    imgs_raw = []
    token_to_boxes = {}
    width_offsets = []
    total_width = 0

    for cam_name in cam_names:
        img_with_box, img_raw, boxes = draw_3d_boxes_on_image(nusc, sample_token, cam_name, box_color, crop_margin)
        imgs_with_box.append(img_with_box)
        imgs_raw.append(img_raw)
        width_offsets.append(total_width)
        total_width += img_with_box.shape[1]

        for box in boxes:
            token = box['token']
            x1, y1, x2, y2 = box['bbox']
            x1 += width_offsets[-1]
            x2 += width_offsets[-1]
            if token not in token_to_boxes:
                token_to_boxes[token] = []
            token_to_boxes[token].append((x1, y1, x2, y2))

    final_img_with_box = np.concatenate(imgs_with_box, axis=1)
    final_img_raw = np.concatenate(imgs_raw, axis=1)

    merged_boxes = []
    for token, boxes in token_to_boxes.items():
        x1 = min(b[0] for b in boxes)
        y1 = min(b[1] for b in boxes)
        x2 = max(b[2] for b in boxes)
        y2 = max(b[3] for b in boxes)
        merged_boxes.append({
            'token': token,
            'bbox': (x1, y1, x2, y2)
        })

    return final_img_with_box, final_img_raw, merged_boxes

def get_future_trajectory_matrix(nusc, sample_token, steps=6):
    trajectory = []
    current_token = sample_token

    # init ego pose
    sample = nusc.get('sample', current_token)
    sd = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
    init_pose = nusc.get('ego_pose', sd['ego_pose_token'])

    init_trans = np.array(init_pose['translation'])
    init_rot = Quaternion(init_pose['rotation'])
    global_to_ego = transform_matrix(init_trans, init_rot, inverse=True)  # 4x4

    for _ in range(steps):
        sample = nusc.get('sample', current_token)
        sd = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        pose = nusc.get('ego_pose', sd['ego_pose_token'])

        pos = np.array(pose['translation'])
        pos_homo = np.concatenate([pos, [1.0]])  # shape (4,)
        pos_ego = global_to_ego @ pos_homo  # shape (4,)
        trajectory.append(pos_ego[:2][::-1])  # x, y in ego frame

        if sample['next'] == '':
            break
        current_token = sample['next']

    return np.array(trajectory)  # shape (steps, 2)

def project_to_bev_coords(trajectory, x_range, y_range, resolution=0.1):
    
    x_pixels = ((trajectory[:, 0] - x_range[0]) / resolution).astype(int)
    y_pixels = ((trajectory[:, 1] - y_range[0]) / resolution).astype(int)
    return x_pixels, y_pixels

def get_ego_speed(nusc, sample_token):
    sample = nusc.get('sample', sample_token)
    sd = nusc.get('sample_data', sample['data']['LIDAR_TOP'])

    # current pose
    curr_pose = nusc.get('ego_pose', sd['ego_pose_token'])
    curr_time = sd['timestamp'] * 1e-6  # ms->s
    curr_pos = np.array(curr_pose['translation'])

    # 前一个 pose
    if sample['prev'] == '':
        prev_sample = nusc.get('sample', sample['next'])
    else:
        prev_sample = nusc.get('sample', sample['prev'])
    prev_sd = nusc.get('sample_data', prev_sample['data']['LIDAR_TOP'])
    prev_pose = nusc.get('ego_pose', prev_sd['ego_pose_token'])
    prev_time = prev_sd['timestamp'] * 1e-6
    prev_pos = np.array(prev_pose['translation'])

    # calculate the time
    dt = curr_time - prev_time
    if sample['prev'] == '':
      dt = -dt
    if dt == 0:
        return 0.0
    speed = np.linalg.norm(curr_pos - prev_pos) / dt
    return speed  # m/s

def get_all_sample_tokens(nusc):
    all_sample_tokens = []
    for scene in nusc.scene:
        first_sample_token = scene['first_sample_token']
        sample_token = first_sample_token
        while sample_token != '':
            all_sample_tokens.append(sample_token)
            sample = nusc.get('sample', sample_token)
            sample_token = sample['next']
    return all_sample_tokens
import gc
import json
def store_all_data(nusc, save_root, max_samples=None,x_range=[-25,25],y_range=[0,50]):
    os.makedirs(save_root, exist_ok=True)
    dirs = ['bev_img', 'bev_mask', 'bev_polys', 'img_raw', 'img_boxes', 'trajectory', 'speed','bev_view']
    for d in dirs:
        os.makedirs(os.path.join(save_root, d), exist_ok=True)

    all_samples_tokens = get_all_sample_tokens(nusc)
    print(f'Total samples: {len(all_samples_tokens)}')
    if max_samples:
        all_samples_tokens = all_samples_tokens[:max_samples]
    num_step = 8

    for idx, sample_token in enumerate(all_samples_tokens):
        print(idx)
        try:
            future_wp = get_future_trajectory_matrix(nusc, sample_token, steps=num_step)
            if future_wp.shape[0] < num_step:
                print(future_wp.shape)
                continue
            
            bev_img, bev_mask, polys = read_lidar_from_token(nusc, sample_token, x_range, y_range)
            img_with_box, img_raw, boxes = merge_images_and_boxes(nusc, sample_token)
            
            x_pix, y_pix = project_to_bev_coords(future_wp, x_range, y_range, resolution=0.1)
            trajectory = np.stack([x_pix, y_pix], axis=-1)
            speed = get_ego_speed(nusc, sample_token)
            img = bev_view(nusc,sample_token,polys,x_range,y_range)
            
            sid = f"{idx:06d}"
            np.save(os.path.join(save_root, 'bev_img', f'{sid}.npy'), bev_img)
            np.save(os.path.join(save_root, 'bev_mask', f'{sid}.npy'), bev_mask)
            np.save(os.path.join(save_root, 'bev_polys', f'{sid}.npy'), polys)
            Image.fromarray(img_raw).save(os.path.join(save_root, 'img_raw', f'{sid}.jpg'))
            with open(os.path.join(save_root, 'img_boxes', f'{sid}.json'), 'w') as f:
                json.dump(boxes, f)
            np.save(os.path.join(save_root, 'trajectory', f'{sid}.npy'), trajectory)
            np.save(os.path.join(save_root, 'speed', f'{sid}.npy'), speed)
            img.save(os.path.join(save_root, 'bev_view', f'{sid}.jpg'))
        except Exception as e:
            print(f"[{idx}] Error processing {sample_token[:8]}: {e}")
        finally:
            gc.collect()  # 手动释放内存
import math
from matplotlib.path import Path
def polygons_to_mask(polygons, height, width):
    mask = np.zeros((height, width), dtype=np.uint8)
    for poly in polygons:
        poly_np = np.array(poly)
        path = Path(poly_np)
        
        # 创建网格坐标
        x, y = np.meshgrid(np.arange(width), np.arange(height))
        points = np.vstack((x.flatten(), y.flatten())).T

        # 使用 path.contains_points 进行点内检测
        inside = path.contains_points(points).reshape((height, width))
        mask[inside] = 1
        mask = mask
    mask = mask.T
    return mask
def polygon_to_bev(nusc, sample_token, polys, x_range=[-25,25], y_range=[0,50]):
  sample = nusc.get('sample',sample_token)
  lidar_sd = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
  ego_pose = nusc.get('ego_pose', lidar_sd['ego_pose_token'])
  scene = nusc.get('scene', sample['scene_token'])
  log = nusc.get('log', scene['log_token'])
  location = log['location']
  nusc_map = NuScenesMap(dataroot=nusc.dataroot, map_name=location)
  nx = int(2*(x_range[1]-x_range[0])*10)
  ny = int(2*(y_range[1]-y_range[0])*10)
  canvas_size = (nx, ny)  # 输出图像大小，像素（0.1m/pixel）
  npix = 10
  patch_box = (
      ego_pose['translation'][0] ,  # x 向前偏移 25m，使车靠底部
      ego_pose['translation'][1],      # y
      nx/npix,  # 高度（前方距离）
      ny/npix   # 宽度（左右范围）
  )

  # -----------------------
  # 4. 计算车辆朝向角并旋转地图
  # -----------------------
  ego_quat = Quaternion(ego_pose['rotation'])
  forward = ego_quat.rotate(np.array([1, 0, 0]))  # x 轴单位向量旋转后的方向
  yaw_rad = math.atan2(forward[1], forward[0])
  yaw_deg = math.degrees(yaw_rad)
  patch_angle = yaw_deg  # 让车始终朝上



  layer_names = ['road_segment']
  mask = nusc_map.get_map_mask(
      patch_box,
      patch_angle,
      layer_names=layer_names,
      canvas_size=canvas_size
  )  # shape: (2, H, W)
  mask[0] = mask[0].T[:,::-1]
  road_mask = mask[0][ny//2+y_range[0]*npix:ny//2+y_range[1]*npix,nx//2+x_range[0]*npix:nx//2+x_range[1]*npix]
  polygon_mask = polygons_to_mask(polys, height=ny//2, width=nx//2)
  polygon_mask = np.expand_dims(polygon_mask,axis=0)
  road_mask = np.expand_dims(road_mask,axis=0)
  road_mask1 =  ((polygon_mask==0)&(road_mask==1)).astype(np.uint16)
  back_mask= ((polygon_mask==0)&(road_mask==0)).astype(np.uint16)
  return polygon_mask, road_mask1, back_mask
def bev_view(nusc,token_sample,polys, x_range=[-25,25], y_range=[0,50]):
  pp, rr, bb = polygon_to_bev(nusc,token_sample,polys,x_range=x_range,y_range=y_range)
  bev = np.concatenate((bb,rr,pp),axis=0)
  img_array = (bev.transpose(1, 2, 0) * 255).astype(np.uint8)
  img = Image.fromarray(img_array)
  return img

  
