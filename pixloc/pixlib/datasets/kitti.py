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
from pixloc.pixlib.datasets.transformations import euler_from_matrix, euler_matrix
from pixloc.pixlib.geometry import Camera, Pose

visualise_debug = True

root_dir = '/data/Kitti'
#root_dir = '/students/u6617221/shan/data'
test_csv_file_name = 'test.csv'
ignore_csv_file_name = 'ignore.csv'
satmap_dir = 'satmap_20'
grdimage_dir = 'raw_data'
left_color_camera_dir = 'image_02/data'
right_color_camera_dir = 'image_03/data'
oxts_dir = 'oxts/data'
vel_dir = 'velodyne_points/data'

#grd_ori_size = (375, 1242) # different size in Kitti 375×1242, 370×1224,374×1238, and376×1241
grd_pad_size = (384, 1248)
satellite_ori_size = 1280


ToTensor = transforms.Compose([
    transforms.ToTensor()])

class Kitti(BaseDataset):
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
        #assert split != 'test', 'Not supported'
        return _Dataset(self.conf, split)

def read_calib(calib_file_name):
    with open(calib_file_name, 'r') as f:
        lines = f.readlines()
        for line in lines:
            # left color camera k matrix
            if 'P_rect_02' in line:
                # get 3*3 matrix from P_rect_**:
                items = line.split(':')
                valus = items[1].strip().split(' ')

                camera_P = np.asarray([float(i) for i in valus]).reshape((3, 4))
                fx = float(valus[0])
                cx = float(valus[2])
                fy = float(valus[5])
                cy = float(valus[6])

                camera_k = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
                camera_k = np.asarray(camera_k)
                tx, ty, tz = float(valus[3]), float(valus[7]), float(valus[11])
                camera_t = np.linalg.inv(camera_k) @ np.asarray([tx, ty, tz])

            if 'R_rect_02' in line:
                items = line.split(':')
                valus = items[1].strip().split(' ')
                camera_R = np.asarray([float(i) for i in valus]).reshape((3, 3))

    return camera_k, camera_R, camera_t, camera_P


def read_sensor_rel_pose(file_name):
    with open(file_name, 'r') as f:
        lines = f.readlines()
        for line in lines:
            # left color camera k matrix
            if 'R' in line:
                # get 3*3 matrix from P_rect_**:
                items = line.split(':')
                valus = items[1].strip().split(' ')
                R = np.asarray([float(i) for i in valus]).reshape((3, 3))

            if 'T' in line:
                # get 3*3 matrix from P_rect_**:
                items = line.split(':')
                valus = items[1].strip().split(' ')
                T = np.asarray([float(i) for i in valus]).reshape((3))
    rel_pose_hom = np.eye(4)
    rel_pose_hom[:3, :3] = R
    rel_pose_hom[:3, 3] = T

    return R, T, rel_pose_hom



def project_lidar_to_cam(velodyne, camera_P, camera_k, camera_R, camera_t, lidar2cam_R, lidar2cam_T, img_size):
    # lidar2cam_R2 = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]])
    # translation2 = np.array([ 0.06, 0, -0.27])
    # points3d2 = lidar2cam_R2 @ velodyne.T[:3,:] + translation2[:,None]
    # img_size (H,W)

    points3d = lidar2cam_R @ velodyne.T[:3, :] + lidar2cam_T[:, None]
    points3d = camera_R @ points3d + camera_t[:, None]
    # # ignore out of satellite range 0~64m
    # velo_mask = (points3d[2,:] > 0) & (points3d[2,:] < 31)
    velo_mask = points3d[2, :] > 0
    points3d = points3d[:, velo_mask]
    velodyne_trimed = velodyne.T[:, velo_mask]

    # project the points to the camera
    points2d = camera_k @ points3d
    points2d = points2d[:2, :] / points2d[2, :]

    u_mask = (points2d[0] < img_size[1]) & (points2d[0] > 0)
    v_mask = (points2d[1] < img_size[0]) & (points2d[1] > 0)
    uv_mask = u_mask & v_mask
    return points2d[:, uv_mask], points3d[:, uv_mask], velodyne_trimed[:, uv_mask]

def project_lidar_to_sat(cam_3d, imu_rot, meter_per_pixel, location):
    # cam to sat
    cam2sat = np.eye(4)
    cam2sat[1,1] = -1 # y=-y
    cam2sat[2,2] = -1 # z=-z
    cam2sat = cam2sat@imu_rot
    sat_3d = cam2sat @ cam_3d

    # convert the world to pixels, with respect to the pixel location
    sat_2d = sat_3d[:2, :] / meter_per_pixel
    sat_2d = sat_2d + np.asarray(location)[:, None]
    # visible mask
    u_mask = (sat_2d[0] < satellite_ori_size) & (sat_2d[0] > 0)
    v_mask = (sat_2d[1] < satellite_ori_size) & (sat_2d[1] > 0)
    sat_mask = u_mask & v_mask
    # matching 3D locations
    sat_2d = sat_2d[:, sat_mask].T
    sat_3d = sat_3d[:3, sat_mask].T
    cam_3d = cam_3d[:3, sat_mask].T

    return sat_2d, sat_3d, cam_3d

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
        self.conf = conf

        # satmap NED coords
        file = os.path.join(self.root, grdimage_dir, 'satellite_gps_center.npy')
        self.Geodetic = np.load(file)
        NED_coords_satellite = np.zeros((self.Geodetic.shape[0], 3))
        for i in range(self.Geodetic.shape[0]):
            x, y, z = gps_func.GeodeticToEcef(self.Geodetic[i, 0] * np.pi / 180.0, self.Geodetic[i, 1] * np.pi / 180.0,
                                              self.Geodetic[i, 2])
            xEast, yNorth, zUp = gps_func.EcefToEnu(x, y, z, self.Geodetic[0, 0] * np.pi / 180.0,
                                                    self.Geodetic[0, 1] * np.pi / 180.0, self.Geodetic[0, 2])
            NED_coords_satellite[i, 0] = xEast
            NED_coords_satellite[i, 1] = yNorth
            NED_coords_satellite[i, 2] = zUp
        self.neigh = NearestNeighbors(n_neighbors=1)    
        self.neigh.fit(NED_coords_satellite)

        # read form txt files
        self.file_name = []
        txt_file_name = os.path.join(self.root, grdimage_dir, 'kitti_split', split+'_files.txt')
        with open(txt_file_name, "r") as txt_f:
            lines = txt_f.readlines()
            for line in lines:
                line = line.strip()
                # check grb file exist
                grb_file_name = os.path.join(self.root, grdimage_dir, line[:38], left_color_camera_dir,
                                                  line[38:].lower())
                if not os.path.exists(grb_file_name):
                    # ignore frames with out velodyne
                    print(grb_file_name + ' do not exist!!!')
                    continue

                velodyne_file_name = os.path.join(self.root, grdimage_dir, line[:38], vel_dir,
                                                  line[38:].lower().replace('.png', '.bin'))
                if not os.path.exists(velodyne_file_name):
                    # ignore frames with out velodyne
                    print(velodyne_file_name + ' do not exist!!!')
                    continue

                self.file_name.append(line)

        if 0:  # for debug
            if split == 'train':
                self.file_name = random.sample(self.file_name, len(self.file_name)//6)

            if split == 'val':
                self.file_name = random.sample(self.file_name, len(self.file_name)//6)

    def __len__(self):
        return len(self.file_name)

    def __getitem__(self, idx):
        # read cemera k matrix from camera calibration files, day_dir is first 10 chat of file name
        file_name = self.file_name[idx]
        day_dir = file_name[:10]
        drive_dir = file_name[:38]
        image_no = file_name[38:]

        # get calibration information, do not adjust image size change here
        camera_k, camera_R, camera_t, camera_P = read_calib(
            os.path.join(self.root, grdimage_dir, day_dir, 'calib_cam_to_cam.txt'))
        imu2lidar_R, imu2lidar_T, imu2lidar_H = read_sensor_rel_pose(
            os.path.join(self.root, grdimage_dir, day_dir, 'calib_imu_to_velo.txt'))
        lidar2cam_R, lidar2cam_T, lidar2cam_H = read_sensor_rel_pose(
            os.path.join(self.root, grdimage_dir, day_dir, 'calib_velo_to_cam.txt'))
        # computer the relative pose between imu to camera
        imu2cam_H = lidar2cam_H @ imu2lidar_H
        imu2cam_H[:3, 3] += camera_t
        #imu2cam_eulers = euler_from_matrix(imu2cam_H[:3, :3])
        camera_center_loc = -imu2cam_H[:3, :3].T @ imu2cam_H[:3, 3]

        # get location & rotation
        oxts_file_name = os.path.join(self.root, grdimage_dir, drive_dir, oxts_dir,
                                      image_no.lower().replace('.png', '.txt'))
        with open(oxts_file_name, 'r') as f:
            content = f.readline().split(' ')
        location = [float(content[0]), float(content[1]), float(content[2])]
        roll, pitch, heading = float(content[3]), float(content[4]), float(content[5])
        imu_rot = euler_matrix(roll, pitch, heading)
        cam_rot = euler_matrix(-pitch, -heading, roll)

        # read lidar points
        velodyne_file_name = os.path.join(self.root, grdimage_dir, drive_dir, vel_dir,
                                          image_no.lower().replace('.png', '.bin'))
        velodyne = np.fromfile(velodyne_file_name, dtype=np.float32).reshape(-1, 4)
        velodyne[:, 3] = 1.0

        # ground images, left color camera
        left_img_name = os.path.join(self.root, grdimage_dir, drive_dir, left_color_camera_dir, image_no.lower())
        with Image.open(left_img_name, 'r') as GrdImg:
            grd_left = GrdImg.convert('RGB')
            grd_ori_H = grd_left.size[1]
            grd_ori_W = grd_left.size[0]
            # pad right and bottom
            grd_left = transforms.functional.pad(grd_left, (0, 0, grd_pad_size[1]-grd_ori_W,grd_pad_size[0]-grd_ori_H))
            grd_left = ToTensor(grd_left)

        # project these lidar points to camera coordinate
        # vis_2d: lidar points projected 2D position in the image
        # velodyne_local: lidar points (3D) which are visible to the image, in the velodyne coordinate system
        _, cam_3d, _ = project_lidar_to_cam(velodyne, camera_P, camera_k, camera_R, camera_t,
                                                         lidar2cam_R, lidar2cam_T, (grd_ori_H,grd_ori_W))
        # transform these velodyne_local to the IMU coordinate system
        # velodyne_imu = np.linalg.inv(imu2lidar_H) @ velodyne_local

        # satellite map
        x, y, z = gps_func.GeodeticToEcef(location[0] * np.pi / 180.0, location[1] * np.pi / 180.0, location[2])
        xEast, yNorth, zUp = gps_func.EcefToEnu(x, y, z, self.Geodetic[0, 0] * np.pi / 180.0,
                                                self.Geodetic[0, 1] * np.pi / 180.0, self.Geodetic[0, 2])
        NED_coords_query = np.array([[xEast, yNorth, zUp]])
        distances, indices = self.neigh.kneighbors(NED_coords_query, return_distance=True)
        distances = distances.ravel()[0]
        indices = indices.ravel()[0]
        # find sat image
        sat_gps = self.Geodetic[indices]
        SatMap_name = os.path.join(root_dir, satmap_dir) + "/i" + str(indices) + "_lat_" + str(
            sat_gps[0]) + "_long_" + str(
            sat_gps[1]) + "_zoom_" + str(
            20) + "_size_" + str(640) + "x" + str(640) + "_scale_" + str(2) + ".png"
        with Image.open(SatMap_name, 'r') as SatMap:
            sat_map = SatMap.convert('RGB')
            sat_map = ToTensor(sat_map)

        # get ground-view, satellite image shift
        x_sg, y_sg = gps_func.angular_distance_to_xy_distance_v2(sat_gps[0], sat_gps[1], location[0],
                                                                 location[1])
        meter_per_pixel = Kitti_utils.get_meter_per_pixel(scale=1)
        x_sg = int(x_sg / meter_per_pixel)
        y_sg = int(-y_sg / meter_per_pixel)
        body_location_x = x_sg + satellite_ori_size / 2.0
        body_location_y = y_sg + satellite_ori_size / 2.0

        # add the offset between camera and body to shift the center to query camera
        # dx_cam = -camera_center_loc[1] / meter_per_pixel
        # dy_cam = camera_center_loc[0] / meter_per_pixel
        # convert the offset to ploar coordinate
        tan_theta = -camera_center_loc[1] / camera_center_loc[0]
        length_dxy = np.sqrt(
            camera_center_loc[0] * camera_center_loc[0] + camera_center_loc[1] * camera_center_loc[1])
        offset_cam_theta = np.arctan(tan_theta)
        yaw_FL = heading - offset_cam_theta
        dx_cam_pixel = np.cos(yaw_FL) * length_dxy / meter_per_pixel
        dy_cam_pixel = -np.sin(yaw_FL) * length_dxy / meter_per_pixel
        cam_location_x = x_sg + satellite_ori_size / 2.0 + dx_cam_pixel
        cam_location_y = y_sg + satellite_ori_size / 2.0 + dy_cam_pixel

        # # # # get velodyne_world on the satelite map
        # sat_2d, sat_3d, cam_3d = project_cam_to_sat(cam_3d, imu_rot, meter_per_pixel, [cam_location_x, cam_location_y])
        # velodyne_grd = cam_3d[:,sat_mask].T

        # mask = cam_3d[-1] < satellite_ori_size * meter_per_pixel / 2.0
        # cam_3d = cam_3d[:,mask].T

        # sat
        grd2sat = np.array([[0,0,1,0],[1,0,0,0],[0,1,0,0],[0,0,0,1]]) #grd z->sat x; grd x->sat y, grd y->sat z
        grd2sat = Pose.from_4x4mat(grd2sat).float()

        camera = Camera.from_colmap(dict(
            model='SIMPLE_PINHOLE', params=(1 / meter_per_pixel, cam_location_x, cam_location_y, 0,0,0,0,np.infty),#np.infty for parallel projection
            width=int(satellite_ori_size), height=int(satellite_ori_size)))
        sat_image = {
            'image': sat_map.float(),
            'camera': camera.float(),
            'T_w2cam': Pose.from_4x4mat(np.eye(4)).float() # grd 2 sat in q2r, so just eye(4)
            #'points3D': None # use grd points 3D
        }

        # project to sat and find visible mask
        cam_3d = cam_3d.T
        _, visible = camera.world2image(torch.from_numpy(cam_3d).float())
        cam_3d = cam_3d[visible]

        num_diff = self.conf.max_num_points3D - len(cam_3d)
        if num_diff < 0:
            # select max_num_points
            idx = np.random.choice(range(len(cam_3d)), self.conf.max_num_points3D)
            #idx = farthest_point_sample(torch.from_numpy(cam_3d).float(), self.conf.max_num_points3D)
            cam_3d = cam_3d[idx]
        elif num_diff > 0 and self.conf.force_num_points3D:
            point_add = np.ones((num_diff, 3)) * cam_3d[-1]
            cam_3d = np.vstack((cam_3d, point_add))

        # grd
        camera_para = (camera_k[0,0],camera_k[1,1],camera_k[0,2],camera_k[1,2])
        camera = Camera.from_colmap(dict(
            model='PINHOLE', params=camera_para,
            width=int(grd_pad_size[1]), height=int(grd_pad_size[0])))
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

        # scene
        scene = drive_dir[:4]+drive_dir[5:7]+drive_dir[8:10]+drive_dir[28:32]+image_no[:10]
        data = {
            'ref': sat_image,
            'query': grd_image,
            'overlap': 0.5,
            'T_q2r_init': q2r_init,
            'T_q2r_gt': q2r_gt,
            'scene': scene
        }

        # debug
        if 0:
            image = transforms.functional.to_pil_image(grd_left, mode='RGB')
            image.save('grd.png')
            image = transforms.functional.to_pil_image(sat_map, mode='RGB')
            image.save('sat.png')
        if 0:
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
            ax2.scatter(x=body_location_x, y=body_location_y, c='g', s=20)
            # plot the direction of the body frame
            length_xy = 200.0
            origin = np.array([[body_location_x], [body_location_y]])  # origin point
            dx_east = length_xy * np.cos(heading)
            dy_north = length_xy * np.sin(heading)
            V = np.array([[dx_east, dy_north]])
            ax2.quiver(*origin, V[:, 0], V[:, 1], color=['r'], scale=1000)
            plt.show()

        return data

if __name__ == '__main__':
    # test to load 1 data
    conf = {
        'max_num_points3D': 10000,
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
    dataset = Kitti(conf)
    loader = dataset.get_data_loader('train', shuffle=True)  # or 'train' ‘val’

    for _, data in zip(range(1), loader):
        print(data)


