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
query_size = [860, 1656]


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

        # get intrinsic of front left camera and its RT to body
        FL_K_dict = read_calib_yaml(calib_folder, "cameraFrontLeftIntrinsics.yaml")
        self.FL_k = np.asarray(FL_K_dict['K']).reshape(3, 3)

        FL2body = read_calib_yaml(calib_folder, "cameraFrontLeft_body.yaml")
        self.offset_body = [FL2body['transform']['translation']['x'],
                       FL2body['transform']['translation']['y']]
        self.FL_relPose_body = quaternion_matrix(quat_from_pose(FL2body))
        FL_relTrans_body = trans_from_pose(FL2body)
        self.FL_relPose_body[0, 3] = FL_relTrans_body[0]
        self.FL_relPose_body[1, 3] = FL_relTrans_body[1]
        self.FL_relPose_body[2, 3] = FL_relTrans_body[2]

        # RR camera to body coordinate
        RR_K_dict = read_calib_yaml(calib_folder, "cameraRearRightIntrinsics.yaml")
        self.RR_k = np.asarray(RR_K_dict['K']).reshape(3, 3)
        RR2body = read_calib_yaml(calib_folder, "cameraRearRight_body.yaml")
        self.RR_relPose_body = quaternion_matrix(quat_from_pose(RR2body))
        RR_relTrans_body = trans_from_pose(RR2body)
        self.RR_relPose_body[0, 3] = RR_relTrans_body[0]
        self.RR_relPose_body[1, 3] = RR_relTrans_body[1]
        self.RR_relPose_body[2, 3] = RR_relTrans_body[2]

        # SL camera to body coordinate
        SL_K_dict = read_calib_yaml(calib_folder, "cameraSideLeftIntrinsics.yaml")
        self.SL_k = np.asarray(SL_K_dict['K']).reshape(3, 3)
        SL2body = read_calib_yaml(calib_folder, "cameraSideLeft_body.yaml")
        self.SL_relPose_body = quaternion_matrix(quat_from_pose(SL2body))
        SL_relTrans_body = trans_from_pose(SL2body)
        self.SL_relPose_body[0, 3] = SL_relTrans_body[0]
        self.SL_relPose_body[1, 3] = SL_relTrans_body[1]
        self.SL_relPose_body[2, 3] = SL_relTrans_body[2]

        # SR camera to body coordinate
        SR_K_dict = read_calib_yaml(calib_folder, "cameraSideRightIntrinsics.yaml")
        self.SR_k = np.asarray(SR_K_dict['K']).reshape(3, 3)
        SR2body = read_calib_yaml(calib_folder, "cameraSideRight_body.yaml")
        self.SR_relPose_body = quaternion_matrix(quat_from_pose(SR2body))
        SR_relTrans_body = trans_from_pose(SR2body)
        self.SR_relPose_body[0, 3] = SR_relTrans_body[0]
        self.SR_relPose_body[1, 3] = SR_relTrans_body[1]
        self.SR_relPose_body[2, 3] = SR_relTrans_body[2]

        # get lidar blue shift from body
        with open(os.path.join(calib_folder, "lidarBlue_body.yaml"), 'r') as stream:
            try:
                lidarBlue_body = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)

        # lidar 2 body coordinate
        self.LidarBlue_relPose_body = quaternion_matrix(quat_from_pose(lidarBlue_body))
        LidarBlue_relTrans_body = trans_from_pose(lidarBlue_body)
        self.LidarBlue_relPose_body[0, 3] = LidarBlue_relTrans_body[0]
        self.LidarBlue_relPose_body[1, 3] = LidarBlue_relTrans_body[1]
        self.LidarBlue_relPose_body[2, 3] = LidarBlue_relTrans_body[2]

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
        roll = self.groundview_rolls[idx] * np.pi / 180.0
        pitch = self.groundview_pitchs[idx] * np.pi / 180.0

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
        LidarBlue2FL = inverse_pose(self.FL_relPose_body) @ self.LidarBlue_relPose_body
        pcd.transform(LidarBlue2FL) # move from lidar coordinate to left camera coordinate
        cam_3d = np.transpose(np.asarray(pcd.points))
        # non_visible_mask = cam_3d[2, :] < 0.0
        # cam_3d = np.concatenate([cam_3d,np.ones_like(cam_3d[:,0:1])],axis=-1) # [n,3]=>[n,4] homogeneous
        cam_3d = cam_3d.T

        # project the 3D points to image and get valid mask
        # cam_FL_2d1 = self.FL_k @ cam_3d
        # cam_FL_2d = cam_FL_2d1 / cam_FL_2d1[2, :]
        # # update non0-visible mask
        # non_visible_mask = (cam_FL_2d[0, :] > query_size[1]) | (cam_FL_2d[0, :] < 0.0)
        # non_visible_mask = non_visible_mask | (cam_FL_2d[1, :] > query_size[0]) | (cam_FL_2d[1, :] < 0.0)
        # visible mask
        # FL_visible_mask = np.invert(non_visible_mask)
        # get visible point clouds and their projections
        #cam_3d = cam_3d[:, visible_mask] # for debug

        # sat
        with Image.open(self.satellite_dict[self.match_pair[idx]], 'r') as SatMap:
            sat_map = SatMap.convert('RGB')
            sat_map = ToTensor(sat_map)

        # grd: moving derection == east
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
        # sat coordinate to ground(east=z/south=x) coordinate , then, to camera coordinate
        grd2cam = Pose.from_aa( np.array([pitch, heading,-roll]),np.zeros(3)).float()
        cam2sat = grd2sat @ (grd2cam.inv())
        sat_3d = cam2sat * torch.from_numpy(cam_3d).float()
        _, visible = camera.world2image(sat_3d)
        cam_3d = cam_3d[visible]

        # grd ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        log_folder = os.path.join(self.root, self.log_id)

        # ground images, rear right camera
        query_image_folder = os.path.join(log_folder, self.log_id + "-RR")
        name = os.path.join(query_image_folder, self.file_name[idx][:-1])
        with Image.open(name, 'r') as GrdImg:
            grd = GrdImg.convert('RGB')
            grd = ToTensor(grd)

        camera_para = (self.RR_k[0,0],self.RR_k[1,1],self.RR_k[0,2],self.RR_k[1,2])
        camera = Camera.from_colmap(dict(
            model='PINHOLE', params=camera_para,
            width=int(query_size[1]), height=int(query_size[0])))
        FL2RR = inverse_pose(self.RR_relPose_body) @ self.FL_relPose_body
        RR_image = {
            # to array, when have multi query
            'image': grd.float(),
            'camera': camera.float(),
            'T_w2cam': Pose.from_4x4mat(FL2RR).float(), #Pose.from_Rt(camera_R, camera_t).float(), # already consider calibration in points3D
        }
        # project the 3D points to image and get valid mask
        tran_3d = RR_image['T_w2cam'] * torch.from_numpy(cam_3d).float()
        _, RR_visible = camera.world2image(tran_3d)

        # ground images, side left camera
        query_image_folder = os.path.join(log_folder, self.log_id + "-SL")
        name = os.path.join(query_image_folder, self.file_name[idx][:-1])
        with Image.open(name, 'r') as GrdImg:
            grd = GrdImg.convert('RGB')
            grd = ToTensor(grd)

        camera_para = (self.SL_k[0,0],self.SL_k[1,1],self.SL_k[0,2],self.SL_k[1,2])
        camera = Camera.from_colmap(dict(
            model='PINHOLE', params=camera_para,
            width=int(query_size[1]), height=int(query_size[0])))
        FL2SL = inverse_pose(self.SL_relPose_body) @ self.FL_relPose_body
        SL_image = {
            # to array, when have multi query
            'image': grd.float(),
            'camera': camera.float(),
            'T_w2cam': Pose.from_4x4mat(FL2SL).float(), #Pose.from_Rt(camera_R, camera_t).float(), # already consider calibration in points3D
        }

        # project the 3D points to image and get valid mask
        tran_3d = SL_image['T_w2cam'] * torch.from_numpy(cam_3d).float()
        _, SL_visible = camera.world2image(tran_3d)

        # ground images, side right camera
        query_image_folder = os.path.join(log_folder, self.log_id + "-SR")
        name = os.path.join(query_image_folder, self.file_name[idx][:-1])
        with Image.open(name, 'r') as GrdImg:
            grd = GrdImg.convert('RGB')
            grd = ToTensor(grd)

        camera_para = (self.SR_k[0,0],self.SR_k[1,1],self.SR_k[0,2],self.SR_k[1,2])
        camera = Camera.from_colmap(dict(
            model='PINHOLE', params=camera_para,
            width=int(query_size[1]), height=int(query_size[0])))
        FL2SR = inverse_pose(self.SR_relPose_body) @ self.FL_relPose_body
        SR_image = {
            # to array, when have multi query
            'image': grd.float(),
            'camera': camera.float(),
            'T_w2cam': Pose.from_4x4mat(FL2SR).float(), #Pose.from_Rt(camera_R, camera_t).float(), # already consider calibration in points3D
        }
        # project the 3D points to image and get valid mask
        tran_3d = SR_image['T_w2cam'] * torch.from_numpy(cam_3d).float()
        _, SR_visible = camera.world2image(tran_3d)

        # ground images, front left color camera
        query_image_folder = os.path.join(log_folder, self.log_id + "-FL")
        name = os.path.join(query_image_folder, self.file_name[idx][:-1])
        with Image.open(name, 'r') as GrdImg:
            grd = GrdImg.convert('RGB')
            grd = ToTensor(grd)

        camera_para = (self.FL_k[0,0],self.FL_k[1,1],self.FL_k[0,2],self.FL_k[1,2])
        camera = Camera.from_colmap(dict(
            model='PINHOLE', params=camera_para,
            width=int(query_size[1]), height=int(query_size[0])))

        # project the 3D points to image and get valid mask
        _, FL_visible = camera.world2image(torch.from_numpy(cam_3d).float())
        visible = RR_visible | SL_visible | SR_visible | FL_visible
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

        FL_image = {
            # to array, when have multi query
            'image': grd.float(),
            'camera': camera.float(),
            'T_w2cam': Pose.from_4x4mat(np.eye(4)).float(), # already consider calibration in points3D
            'points3D': torch.from_numpy(cam_3d).float() # world is camera coordinates # one for multi
        }

        grd_image = (FL_image, RR_image, SL_image, SR_image)

        # init and gt pose~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # ramdom shift translation and ratation on yaw
        YawShiftRange = 6 * np.pi / 180  # in 10 degree
        yaw = 2 * YawShiftRange * np.random.random() - YawShiftRange
        R_yaw = torch.tensor([[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0], [-np.sin(yaw), 0, np.cos(yaw)]])
        TShiftRange = 3  # in 5 meter
        T = 2 * TShiftRange * np.random.rand((3)) - TShiftRange
        T[1] = 0  # no shift on height
        #print(f'in dataset: yaw:{yaw/np.pi*180},t:{T}')

        # add random yaw and t to init pose
        grd2cam_init = Pose.from_Rt(R_yaw,T).float()
        grd2cam_init = grd2cam@grd2cam_init
        cam2sat_init = grd2sat @ (grd2cam_init.inv())

        data = {
            'ref': sat_image,
            'query': grd_image,
            'overlap': 0.5,
            'T_q2r_init': cam2sat_init,
            'T_q2r_gt': cam2sat,
        }

        # debug
        if 1:
            color_image = transforms.functional.to_pil_image(data['ref']['image'], mode='RGB')
            color_image = np.array(color_image)
            # sat gt green
            sat_3d = data['T_q2r_gt']*data['query'][0]['points3D']
            sat_2d, _ = sat_image['camera'].world2image(sat_3d)  ##camera 3d to 2d
            sat_2d = sat_2d.T
            for j in range(sat_2d.shape[1]):
                cv2.circle(color_image, (np.int32(sat_2d[0][j]), np.int32(sat_2d[1][j])), 2, (0, 255, 0),
                           -1)

            # # sat init red
            # sat_3d = data['T_q2r_init']* grd_image['points3D']
            # sat_2d, _ = sat_image['camera'].world2image(sat_3d)  ##camera 3d to 2d
            # sat_2d = sat_2d.T
            # for j in range(sat_2d.shape[1]):
            #     cv2.circle(color_image, (np.int32(sat_2d[0][j]), np.int32(sat_2d[1][j])), 2, (255, 0, 0),
            #                -1)

            plt.imshow(color_image)
            # camera position
            plt.scatter(x=cam_location_x, y=cam_location_y, c='r', s=30)
            # plot the direction of the body frame
            length_xy = 200.0
            origin = np.array([[cam_location_x], [cam_location_y]])  # origin point
            dx_east = length_xy * np.cos(heading)
            dy_north = length_xy * np.sin(heading)
            V = np.array([[dx_east, dy_north]])
            plt.quiver(*origin, V[:, 0], V[:, 1], color=['r'], scale=1000)

            plt.show()

            fig = plt.figure(figsize=plt.figaspect(0.5))
            ax1 = fig.add_subplot(2, 2, 1)
            ax2 = fig.add_subplot(2, 2, 2)
            ax3 = fig.add_subplot(2, 2, 3)
            ax4 = fig.add_subplot(2, 2, 4)

            color_image1 = transforms.functional.to_pil_image(data['query'][0]['image'], mode='RGB')
            color_image1 = np.array(color_image1)
            grd_3d = data['query'][0]['T_w2cam']*data['query'][0]['points3D']
            grd_2d, valid = data['query'][0]['camera'].world2image(grd_3d) ##camera 3d to 2d
            grd_2d = grd_2d[valid].T
            for j in range(grd_2d.shape[1]):
                cv2.circle(color_image1, (np.int32(grd_2d[0][j]), np.int32(grd_2d[1][j])), 2, (0, 255, 0),
                           -1)
            ax1.imshow(color_image1)

            color_image2 = transforms.functional.to_pil_image(data['query'][1]['image'], mode='RGB')
            color_image2 = np.array(color_image2)
            grd_3d = data['query'][1]['T_w2cam']*data['query'][0]['points3D']
            grd_2d, valid = data['query'][1]['camera'].world2image(grd_3d) ##camera 3d to 2d
            grd_2d = grd_2d[valid].T
            for j in range(grd_2d.shape[1]):
                cv2.circle(color_image2, (np.int32(grd_2d[0][j]), np.int32(grd_2d[1][j])), 2, (0, 255, 0),
                           -1)
            ax2.imshow(color_image2)

            color_image3 = transforms.functional.to_pil_image(data['query'][2]['image'], mode='RGB')
            color_image3 = np.array(color_image3)
            grd_3d = data['query'][2]['T_w2cam']*data['query'][0]['points3D']
            grd_2d, valid = data['query'][2]['camera'].world2image(grd_3d) ##camera 3d to 2d
            grd_2d = grd_2d[valid].T
            for j in range(grd_2d.shape[1]):
                cv2.circle(color_image3, (np.int32(grd_2d[0][j]), np.int32(grd_2d[1][j])), 2, (0, 255, 0),
                           -1)
            ax3.imshow(color_image3)

            color_image4 = transforms.functional.to_pil_image(data['query'][3]['image'], mode='RGB')
            color_image4 = np.array(color_image4)
            grd_3d = data['query'][3]['T_w2cam']*data['query'][0]['points3D']
            grd_2d, valid = data['query'][3]['camera'].world2image(grd_3d) ##camera 3d to 2d
            grd_2d = grd_2d[valid].T
            for j in range(grd_2d.shape[1]):
                cv2.circle(color_image4, (np.int32(grd_2d[0][j]), np.int32(grd_2d[1][j])), 2, (0, 255, 0),
                           -1)
            ax4.imshow(color_image4)

            plt.show()

        return data

if __name__ == '__main__':
    # test to load 1 data
    conf = {
        'max_num_points3D': 20000,
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


