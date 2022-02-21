# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 11:44:19 2020

@author: loocy
"""

from pixloc.pixlib.datasets.base_dataset import BaseDataset
import numpy as np
import os
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torch
import pandas as pd
import pixloc.pixlib.datasets.Kitti_utils as Kitti_utils
import time
from matplotlib import pyplot as plt
from sklearn.neighbors import NearestNeighbors
import pixloc.pixlib.datasets.Kitti_gps_coord_func as gps_func
import h5py
import random
import cv2
from glob import glob
from pixloc.pixlib.datasets.transformations import euler_from_matrix, euler_matrix, quaternion_matrix
from pixloc.pixlib.geometry import Camera, Pose
import open3d as o3d
import yaml

root_dir = "/home/shan/projects/Ford/data/Ford"
sat_dir = 'satellite_images'
drive_id = "20171026"
log_id = "2017-10-26-V2-Log1"
lidar_dir = 'lidar_blue_pointcloud'
calib_dir = 'V2'
satellite_ori_size = 1280
query_size = [1656, 860]


ToTensor = transforms.Compose([
    transforms.ToTensor()])

def read_calib_yaml(calib_folder, file_name):
    with open(os.path.join(calib_folder, file_name), 'r') as stream:
        try:
            cur_yaml = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return cur_yaml

def read_txt(root_folder, file_name):
    with open(os.path.join(root_folder, file_name)) as f:
        cur_file = f.readlines()
    return cur_file

def read_numpy(root_folder, file_name):
    with open(os.path.join(root_folder, file_name), 'rb') as f:
        cur_file = np.load(f)
    return cur_file

def convert_body_yaw_to_360(yaw_body):
    yaw_360 = 0
    # if (yaw_body >= 0.0) and (yaw_body <=90.0):
    #     yaw_360 = 90.0 - yaw_body

    if (yaw_body >90.0) and (yaw_body <=180.0):
        yaw_360 = 360.0 - yaw_body + 90.0
    else:
        yaw_360 = 90.0 - yaw_body

    # if (yaw_body >= -90) and (yaw_body <0.0):
    #     yaw_360 = 90.0 - yaw_body
    #
    # if (yaw_body >= -180) and (yaw_body < -90):
    #     yaw_360 = 90.0 - yaw_body
    return yaw_360

def quat_from_pose(trans):

    w = trans['transform']['rotation']['w']
    x = trans['transform']['rotation']['x']
    y = trans['transform']['rotation']['y']
    z = trans['transform']['rotation']['z']

    return [w,x,y,z]

def trans_from_pose(trans):

    x = trans['transform']['translation']['x']
    y = trans['transform']['translation']['y']
    z = trans['transform']['translation']['z']

    return [x,y,z]

def inverse_pose(pose):

    pose_inv = np.identity(4)
    pose_inv[:3,:3] = np.transpose(pose[:3,:3])
    pose_inv[:3, 3] = - pose_inv[:3,:3] @ pose[:3,3]

    return pose_inv

class FordAV(BaseDataset):
    default_conf = {
        'dataset_dir': '/data/Kitti',

        'two_view': True,
        'min_overlap': 0.3,
        'max_overlap': 1.,
        'min_baseline': None,
        'max_baseline': None,
        'sort_by_overlap': False,

        'grayscale': False,
        'resize': None,
        'resize_by': 'max',
        'crop': None,
        'pad': None,
        'optimal_crop': True,
        'seed': 0,

        'max_num_points3D': 15000,
        'force_num_points3D': False,
    }

    def _init(self, conf):
        pass

    def get_dataset(self, split):
        assert split != 'test', 'Not supported'
        return _Dataset(self.conf, split)

def farthest_point_sample(points, number):
    """
       Input:
           points: point data, [N, *]
           number: number of samples
       Return:
           centroids: sampled point index[N]
       """
    centroids = torch.zeros(number).int()
    distance = torch.ones_like(points[:,0]) * 1e5  # init distance

    farthest = random.randint(0,len(points)-1)  # randam select the first points

    for i in range(number):
        centroids[i] = farthest  # recoder the farthest point
        centroid = points[farthest, :]
        dist = torch.sum((points - centroid) ** 2, -1)  # cal the distance with centroid point
        mask = dist < distance
        distance[mask] = dist[mask]  # update distance
        farthest = torch.max(distance, -1)[1]

    return centroids

class _Dataset(Dataset):
    def __init__(self, conf, split):
        self.root = root_dir
        self.log_id = log_id
        self.conf = conf

        # get calib infor
        calib_folder = os.path.join(self.root, calib_dir)
        cameraFrontLeftIntrinsics = read_calib_yaml(calib_folder, "cameraFrontLeftIntrinsics.yaml")
        camera_k = np.asarray(cameraFrontLeftIntrinsics['K']).reshape(3, 3)
        self.camera_k = torch.from_numpy(np.asarray(camera_k))
        # camera projection matrix
        self.camera_p = np.asarray(cameraFrontLeftIntrinsics['P']).reshape(3, 4)

        # get left camera shift from body
        FL_cameraFrontLeft_body = read_calib_yaml(calib_folder, "cameraFrontLeft_body.yaml")
        self.offset_body = [FL_cameraFrontLeft_body['transform']['translation']['x'],
                       FL_cameraFrontLeft_body['transform']['translation']['y']]
        # self.base_line = np.abs(offset_body[1] * 2)
        FL_relPose_body = quaternion_matrix(quat_from_pose(FL_cameraFrontLeft_body))
        FL_relTrans_body = trans_from_pose(FL_cameraFrontLeft_body)
        FL_relPose_body[0, 3] = FL_relTrans_body[0]
        FL_relPose_body[1, 3] = FL_relTrans_body[1]
        FL_relPose_body[2, 3] = FL_relTrans_body[2]

        # get lidar blue shift from body
        with open(os.path.join(calib_folder, "lidarBlue_body.yaml"), 'r') as stream:
            try:
                lidarBlue_body = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
        LidarBlue_relPose_body = quaternion_matrix(quat_from_pose(lidarBlue_body))
        LidarBlue_relTrans_body = trans_from_pose(lidarBlue_body)
        LidarBlue_relPose_body[0, 3] = LidarBlue_relTrans_body[0]
        LidarBlue_relPose_body[1, 3] = LidarBlue_relTrans_body[1]
        LidarBlue_relPose_body[2, 3] = LidarBlue_relTrans_body[2]

        self.LidarBlue_relPose_FL = inverse_pose(FL_relPose_body) @ LidarBlue_relPose_body

        # get original image & location information
        log_folder = os.path.join(self.root, self.log_id )
        self.file_name = read_txt(log_folder, self.log_id  + '-FL-names.txt')
        self.file_name.pop(0)

        # get the satellite images
        satellite_folder = os.path.join(log_folder, "satellite_images")
        satellite_names = glob(satellite_folder + '/*.png')
        nb_satellite_images = len(satellite_names)
        self.satellite_dict = {}
        for i in range(nb_satellite_images):
            cur_sat = int(os.path.split(satellite_names[i])[-1].split("_")[1])
            self.satellite_dict[cur_sat] = satellite_names[i]

        self.groundview_yaws = read_numpy(log_folder, 'groundview_yaws_pose_gt.npy')  # 'groundview_yaws_pose.npy'
        self.groundview_rolls = read_numpy(log_folder, 'groundview_rolls_pose_gt.npy')  # 'groundview_yaws_pose.npy'
        self.groundview_pitchs = read_numpy(log_folder, 'groundview_pitchs_pose_gt.npy')  # 'groundview_yaws_pose.npy'
        self.groundview_gps = read_numpy(log_folder, 'groundview_gps.npy')
        self.match_pair = read_numpy(log_folder,
                                'groundview_satellite_pair.npy')  # 'groundview_satellite_pair_2.npy'# 'groundview_gps_2.npy'

        # read lidar points
        self.vel_pair = read_numpy(log_folder, 'vel_time_per_images.npy')

        if 0:  # for debug
            if split == 'train':
                self.file_name = random.sample(self.file_name, len(self.file_name)//2)

            if split == 'val':
                self.file_name = random.sample(self.file_name, len(self.file_name)//2)

    def __len__(self):
        return len(self.file_name)

    def __getitem__(self, idx):
        ###############################
        query_gps = self.groundview_gps[idx, :]
        satellite_img = os.path.split(self.satellite_dict[self.match_pair[idx]])[-1].split("_")
        satellite_gps = [float(satellite_img[3]), float(satellite_img[5])]

        # using the satellite image as the reference and calculate the offset of the ground-view query
        dx, dy = gps_func.angular_distance_to_xy_distance_v2(satellite_gps[0], satellite_gps[1], query_gps[0],
                                                             query_gps[1])

        # get the current resolution of satellite image
        # a scale at 2 when downloading the dataset
        meter_per_pixel = 156543.03392 * np.cos(satellite_gps[0] * np.pi / 180.0) / np.power(2, 20) / 2.0

        # get the pixel offsets of car gps
        # along the east direction
        dx_pixel = dx / meter_per_pixel
        # along the north direction
        dy_pixel = -dy / meter_per_pixel

        heading = convert_body_yaw_to_360(self.groundview_yaws[idx]) * np.pi / 180.0 # heading of car
        roll = convert_body_yaw_to_360(self.groundview_rolls[idx]) * np.pi / 180.0
        pitch = convert_body_yaw_to_360(self.groundview_pitchs[idx]) * np.pi / 180.0

        # add the offset between camera and body to shift the center to query camera
        tan_theta = -self.offset_body[1] / self.offset_body[0]
        length_dxy = np.sqrt(self.offset_body[0] * self.offset_body[0] + self.offset_body[1] * self.offset_body[1])
        offset_cam_theta = np.arctan(tan_theta)
        yaw_FL = offset_cam_theta + heading # camera & center yaw
        dx_cam_pixel = np.cos(yaw_FL) * length_dxy / meter_per_pixel
        dy_cam_pixel = -np.sin(yaw_FL) * length_dxy / meter_per_pixel
        # query postion (camera) on the satellite map
        cam_location_x = dx_pixel + satellite_ori_size / 2.0 + dx_cam_pixel
        cam_location_y = dy_pixel + satellite_ori_size / 2.0 + dy_cam_pixel

        # velodyne_points
        ped_file_name = os.path.join(os.path.join(self.root, self.log_id ), lidar_dir, str(int(self.vel_pair[idx,0]))+'.pcd')
        pcd = o3d.io.read_point_cloud(ped_file_name)
        pcd.transform(self.LidarBlue_relPose_FL) # move from lidar coordinate to left camera coordinate
        cam_3d = np.transpose(np.asarray(pcd.points))
        non_visible_mask = cam_3d[2, :] < 0.0
        # cam_3d = np.concatenate([cam_3d,np.ones_like(cam_3d[:,0:1])],axis=-1) # [n,3]=>[n,4] homogeneous
        # remove behind image plan
        # cam_3d = cam_3d[cam_3d[:, 0] >= 0, :]

        # project the 3D points to image
        cam_2d1 = self.camera_p[:3, :3] @ cam_3d + self.camera_p[:3, 3][:, None]
        cam_2d = cam_2d1 / cam_2d1[2, :]

        # update non0-visible mask
        non_visible_mask = non_visible_mask | (cam_2d[0, :] > query_size[1]) | (cam_2d[0, :] < 0.0)
        non_visible_mask = non_visible_mask | (cam_2d[1, :] > query_size[0]) | (cam_2d[1, :] < 0.0)
        # visible mask
        visible_mask = np.invert(non_visible_mask)
        # get visible point clouds and their projections
        cam_3d = cam_3d[:, visible_mask]

        # sat
        with Image.open(self.satellite_dict[self.match_pair[idx]], 'r') as SatMap:
            sat_map = SatMap.convert('RGB')
            sat_map = ToTensor(sat_map)

        grd2sat = np.array([[0,0,1,0],[1,0,0,0],[0,1,0,0],[0,0,0,1]]) #grd z->sat x; grd x->sat y, grd y->sat z
        grd2sat = Pose.from_4x4mat(grd2sat).float()

        camera = Camera.from_colmap(dict(
            model='SIMPLE_PINHOLE', params=(1 / meter_per_pixel, cam_location_x, cam_location_y, 0,0,0,0,np.infty),#np.infty for parallel projection
            width=int(satellite_ori_size), height=int(satellite_ori_size)))
        sat_image = {
            'image': sat_map.float(),
            'camera': camera.float(),
            'T_w2cam': Pose.from_4x4mat(np.eye(4)).float() # grd 2 sat in q2r, so just eye(4)
        }

        # project to sat and find visible mask
        cam_3d = cam_3d.T
        _, visible = camera.world2image(torch.from_numpy(cam_3d).float())
        cam_3d = cam_3d[visible]

        num_diff = self.conf.max_num_points3D - len(cam_3d)
        if num_diff < 0:
            # select max_num_points
            sample_idx = np.random.choice(range(len(cam_3d)), self.conf.max_num_points3D)
            #idx = farthest_point_sample(torch.from_numpy(cam_3d).float(), self.conf.max_num_points3D)
            cam_3d = cam_3d[sample_idx]
        elif num_diff > 0 and self.conf.force_num_points3D:
            point_add = np.ones((num_diff, 3)) * cam_3d[-1]
            cam_3d = np.vstack((cam_3d, point_add))

        # grd
        # ground images, left color camera
        log_folder = os.path.join(self.root, self.log_id)
        query_image_folder = os.path.join(log_folder, self.log_id + "-FL-rectify")
        leftname = os.path.join(query_image_folder, self.file_name[idx][:-1])
        with Image.open(leftname, 'r') as GrdImg:
            grd_left = GrdImg.convert('RGB')
            grd_left = ToTensor(grd_left)

        camera_para = (self.camera_k[0,0],self.camera_k[1,1],self.camera_k[0,2],self.camera_k[1,2])
        camera = Camera.from_colmap(dict(
            model='PINHOLE', params=camera_para,
            width=int(query_size[1]), height=int(query_size[0])))
        grd_image = {
            # to array, when have multi query
            'image': grd_left.float(),
            'camera': camera.float(),
            'T_w2cam': Pose.from_4x4mat(np.eye(4)).float(), #Pose.from_Rt(camera_R, camera_t).float(), # already consider calibration in points3D
            'points3D': torch.from_numpy(cam_3d).float() # world is camera coordinates # one for multi
        }

        # ramdom shift translation and ratation on yaw
        YawShiftRange = 6 * np.pi / 180  # in 10 degree
        yaw = 2 * YawShiftRange * np.random.random() - YawShiftRange
        R_yaw = torch.tensor([[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0], [-np.sin(yaw), 0, np.cos(yaw)]])
        TShiftRange = 3  # in 5 meter
        T = 2 * TShiftRange * np.random.rand((3)) - TShiftRange
        T[1] = 0  # no shift on height
        #print(f'in dataset: yaw:{yaw/np.pi*180},t:{T}')

        r2q_gt = Pose.from_aa( np.array([pitch,heading,-roll]),np.zeros(3)).float()
        #r2q_gt = r2q_gt@(Pose.from_Rt(camera_R, camera_t).inv().float()) # cancel camera_grd calibration
        r2q_init = Pose.from_Rt(R_yaw,T).float()
        r2q_init = r2q_gt@r2q_init

        # add grd2sat into gt and init
        q2r_gt = grd2sat @ (r2q_gt.inv())
        q2r_init = grd2sat @ (r2q_init.inv())

        data = {
            'ref': sat_image,
            'query': grd_image,
            'overlap': 0.5,
            'T_q2r_init': q2r_init,
            'T_q2r_gt': q2r_gt,
        }

        # debug
        if 1:
            fig = plt.figure(figsize=plt.figaspect(0.5))
            ax1 = fig.add_subplot(1, 2, 1)
            ax2 = fig.add_subplot(1, 2, 2)

            color_image0 = transforms.functional.to_pil_image(grd_left, mode='RGB')
            color_image0 = np.array(color_image0)
            color_image1 = transforms.functional.to_pil_image(sat_map, mode='RGB')
            color_image1 = np.array(color_image1)
            grd_2d, _ = grd_image['camera'].world2image(grd_image['points3D']) ##camera 3d to 2d
            grd_2d = grd_2d.T
            for j in range(grd_2d.shape[1]):
                cv2.circle(color_image0, (np.int32(grd_2d[0][j]), np.int32(grd_2d[1][j])), 2, (0, 255, 0),
                           -1)

            # sat gt green
            sat_3d = data['T_q2r_gt']*grd_image['points3D']
            sat_2d, _ = sat_image['camera'].world2image(sat_3d)  ##camera 3d to 2d
            sat_2d = sat_2d.T
            for j in range(sat_2d.shape[1]):
                cv2.circle(color_image1, (np.int32(sat_2d[0][j]), np.int32(sat_2d[1][j])), 2, (0, 255, 0),
                           -1)

            # sat init red
            sat_3d = data['T_q2r_init']* grd_image['points3D']
            sat_2d, _ = sat_image['camera'].world2image(sat_3d)  ##camera 3d to 2d
            sat_2d = sat_2d.T
            for j in range(sat_2d.shape[1]):
                cv2.circle(color_image1, (np.int32(sat_2d[0][j]), np.int32(sat_2d[1][j])), 2, (255, 0, 0),
                           -1)

            ax1.imshow(color_image0)
            ax2.imshow(color_image1)
            # camera position
            ax2.scatter(x=cam_location_x, y=cam_location_y, c='r', s=30)
            # plot the direction of the body frame
            length_xy = 200.0
            origin = np.array([[cam_location_x], [cam_location_y]])  # origin point
            dx_east = length_xy * np.cos(heading)
            dy_north = length_xy * np.sin(heading)
            V = np.array([[dx_east, dy_north]])
            ax2.quiver(*origin, V[:, 0], V[:, 1], color=['r'], scale=1000)
            plt.show()

        return data

if __name__ == '__main__':
    # test to load 1 data
    conf = {
        'max_num_points3D': 1500,
        'force_num_points3D': True,
        'batch_size': 1,
        'min_baseline': 1.,
        'max_baseline': 7.,

        'resize': None,
        'resize_by': 'min',
        'crop': 720,  # ?
        'optimal_crop': False,
        'seed': 1,
        'num_workers': 0,
    }
    dataset = FordAV(conf)
    loader = dataset.get_data_loader('train', shuffle=True)  # or 'train' ‘val’

    for _, data in zip(range(10), loader):
        print(data)


