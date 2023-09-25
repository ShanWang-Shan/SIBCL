# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 11:44:19 2020

@author: loocy
"""

from pixloc.pixlib.datasets.base_dataset import BaseDataset
import numpy as np
import os
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torch
from matplotlib import pyplot as plt
import ford_data_process.gps_coord_func as gps_func
import random
import cv2
from glob import glob
from pixloc.pixlib.datasets.transformations import quaternion_matrix
from pixloc.pixlib.geometry import Camera, Pose
import open3d as o3d
import yaml
ImageFile.LOAD_TRUNCATED_IMAGES = True # add for 'broken data stream'

points_type = 0 # 0: use 3d points from map, 1: use 3d points from lidar blue
gt_from_gps = True #ture: pose gt from gps, False: pose gt from NED pose gt

root_dir = "/data/dataset/Ford_AV" # your Ford_AV dir
sat_dir = 'Satellite_Images_18'
sat_zoom = 18
log_id_train = "2017-08-04-V2-Log4"
log_id_val = "2017-07-24-V2-Log4"
log_id_test = "2017-10-26-V2-Log4"
map_points_dir = 'pcd'
lidar_dir = 'lidar_blue_pointcloud'
calib_dir = 'V2'
info_dir = 'info_files'
satellite_ori_size = 1280
query_size = [432, 816]
query_ori_size = [860, 1656]
LF_height = 1.6

ToTensor = transforms.Compose([
    transforms.ToTensor()])

grd_trans = transforms.Compose([
    transforms.Resize(query_size),
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
    try:
        with open(os.path.join(root_folder, file_name), 'rb') as f:
            cur_file = np.load(f, allow_pickle=True)
    except IOError:
        print("file open IO error, not exist?")
    return cur_file

def convert_body_yaw_to_360(yaw_body):
    yaw_360 = 0

    if (yaw_body >90.0) and (yaw_body <=180.0):
        yaw_360 = 360.0 - yaw_body + 90.0
    else:
        yaw_360 = 90.0 - yaw_body
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
        'two_view': True,
        'seed': 0,
        'max_num_points3D': 15000,
        'force_num_points3D': False,
    }

    def _init(self, conf):
        pass

    def get_dataset(self, split):
        #assert split != 'test', 'Not supported'
        return _Dataset(self.conf, split)


class _Dataset(Dataset):
    def __init__(self, conf, split):
        self.root = root_dir
        self.conf = conf
        if split == 'train':
            self.log_id = log_id_train
        elif split =='val':
            self.log_id = log_id_val
        else:
            self.log_id = log_id_test

        # get calib infor
        calib_folder = os.path.join(self.root, calib_dir)

        # get intrinsic of front left camera and its RT to body
        FL_K_dict = read_calib_yaml(calib_folder, "cameraFrontLeftIntrinsics.yaml")
        self.FL_k = np.asarray(FL_K_dict['K']).reshape(3, 3)
        # turn original img size to process img size
        self.FL_k[0] *= query_size[1] / query_ori_size[1]
        self.FL_k[1] *= query_size[0] / query_ori_size[0]

        FL2body = read_calib_yaml(calib_folder, "cameraFrontLeft_body.yaml")
        self.FL_relPose_body = quaternion_matrix(quat_from_pose(FL2body))
        FL_relTrans_body = trans_from_pose(FL2body)
        self.FL_relPose_body[0, 3] = FL_relTrans_body[0]
        self.FL_relPose_body[1, 3] = FL_relTrans_body[1]
        self.FL_relPose_body[2, 3] = FL_relTrans_body[2]

        if points_type == 1:
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

            # read lidar points
            self.vel_pair = read_numpy(info_folder, 'vel_time_per_images.npy')

        log_folder = os.path.join(self.root, self.log_id )
        info_folder = os.path.join(log_folder, info_dir )

        # get original image & location information
        self.file_name = read_txt(info_folder, self.log_id  + '-FL-names.txt')
        self.file_name.pop(0)

        # get the satellite images
        satellite_folder = os.path.join(log_folder, sat_dir)
        satellite_names = glob(satellite_folder + '/*.png')
        nb_satellite_images = len(satellite_names)
        self.satellite_dict = {}
        for i in range(nb_satellite_images):
            cur_sat = int(os.path.split(satellite_names[i])[-1].split("_")[1])
            self.satellite_dict[cur_sat] = satellite_names[i]

        self.groundview_yaws = read_numpy(info_folder, 'groundview_yaws_pose_gt.npy')  # 'groundview_yaws_pose.npy'
        self.groundview_rolls = read_numpy(info_folder,  'groundview_rolls_pose_gt.npy')  # 'groundview_yaws_pose.npy'
        self.groundview_pitchs = read_numpy(info_folder, 'groundview_pitchs_pose_gt.npy')  # 'groundview_yaws_pose.npy'
        self.groundview_gps = read_numpy(info_folder, 'groundview_gps.npy')
        if not gt_from_gps:
            self.groundview_ned = read_numpy(info_folder, "groundview_NED_pose_gt.npy")
        self.match_pair = read_numpy(info_folder,
                                'groundview_satellite_pair.npy')  # 'groundview_satellite_pair_2.npy'# 'groundview_gps_2.npy'

    def __len__(self):
        return len(self.file_name)

    def __getitem__(self, idx):
        ###############################
        satellite_img = os.path.split(self.satellite_dict[self.match_pair[idx]])[-1].split("_")
        satellite_gps = [float(satellite_img[3]), float(satellite_img[5])]

        # get the current resolution of satellite image
        # a scale at 2 when downloading the dataset
        meter_per_pixel = 156543.03392 * np.cos(satellite_gps[0] * np.pi / 180.0) / np.power(2, sat_zoom) / 2.0

        query_gps = self.groundview_gps[idx, :]
        # using the satellite image as the reference and calculate the offset of the ground-view query
        dx, dy = gps_func.angular_distance_to_xy_distance_v2(satellite_gps[0], satellite_gps[1], query_gps[0],
                                                             query_gps[1])
        # get the pixel offsets of car pose
        dx_pixel_gps = dx / meter_per_pixel # along the east direction
        dy_pixel_gps = -dy / meter_per_pixel # along the north direction

        if not gt_from_gps:
            query_ned = self.groundview_ned[idx, :]
            grdx, grdy = gps_func.angular_distance_to_xy_distance(query_gps[0],
                                                                  query_gps[1])
            dx = query_ned[1]-grdx+dx # long east
            dy = query_ned[0]-grdy+dy # lat north
            # get the pixel offsets of car pose
            dx_pixel_ned = dx / meter_per_pixel # along the east direction
            dy_pixel_ned = -dy / meter_per_pixel # along the north direction

        heading = self.groundview_yaws[idx] * np.pi / 180.0#convert_body_yaw_to_360(self.groundview_yaws[idx]) * np.pi / 180.0 # heading of car
        roll = self.groundview_rolls[idx] * np.pi / 180.0
        pitch = self.groundview_pitchs[idx] * np.pi / 180.0

        if points_type == 0:
            # pints cloud from map
            pcd_file_name = os.path.join(os.path.join(self.root, self.log_id), map_points_dir,
                                         self.file_name[idx][:-5] + '.pcd')
            pcd = o3d.io.read_point_cloud(pcd_file_name)
        elif points_type == 1:
            # velodyne_points
            pcd_file_name = os.path.join(os.path.join(self.root, self.log_id ), lidar_dir, str(int(self.vel_pair[idx,0]))+'.pcd')
            pcd = o3d.io.read_point_cloud(pcd_file_name)
            LidarBlue2FL = inverse_pose(self.FL_relPose_body) @ self.LidarBlue_relPose_body
            pcd.transform(LidarBlue2FL) # move from lidar coordinate to left camera coordinate

        cam_3d = np.asarray(pcd.points)
        if cam_3d.shape[0] <= 0:
            print('empty points', pcd_file_name)

        # sat
        with Image.open(self.satellite_dict[self.match_pair[idx]], 'r') as SatMap:
            sat_map = SatMap.convert('RGB')
            sat_map = ToTensor(sat_map)

        # ned: x: north, y: east, z:down
        ned2sat = np.array([[0,1,0,0],[-1,0,0,0],[0,0,1,0],[0,0,0,1]]) #ned y->sat x; ned -x->sat y, ned z->sat z
        ned2sat = Pose.from_4x4mat(ned2sat).float()
        # sat coordinate to ground(east=z/south=x) coordinate , then, to camera coordinate
        body2ned = Pose.from_aa( np.array([roll, pitch, heading]),np.zeros(3)).float()
        # cam->body->ned->sat without translate
        R_fl2body = self.FL_relPose_body.copy()
        R_fl2body[:,3] = 0.
        cam2sat = ned2sat @ body2ned @ Pose.from_4x4mat(R_fl2body).float()


        # add the offset between camera and body to shift the center to query camera
        cam2sat_ori = ned2sat @ body2ned @ Pose.from_4x4mat(self.FL_relPose_body).float()
        cam_location = cam2sat_ori*torch.tensor([0.,0.,0.])/meter_per_pixel
        cam_location_x = dx_pixel_gps + satellite_ori_size / 2.0 + cam_location[0,0]
        cam_location_y = dy_pixel_gps + satellite_ori_size / 2.0 + cam_location[0,1]
        camera = Camera.from_colmap(dict(
            model='SIMPLE_PINHOLE', params=(1 / meter_per_pixel, cam_location_x, cam_location_y, 0,0,0,0,np.infty),#np.infty for parallel projection
            width=int(satellite_ori_size), height=int(satellite_ori_size)))

        if not gt_from_gps:
            cam_location_x = dx_pixel_ned + satellite_ori_size / 2.0 + cam_location[0,0]
            cam_location_y = dy_pixel_ned + satellite_ori_size / 2.0 + cam_location[0,1]
            camera = Camera.from_colmap(dict(
                model='SIMPLE_PINHOLE', params=(1 / meter_per_pixel, cam_location_x, cam_location_y, 0,0,0,0,np.infty),#np.infty for parallel projection
                width=int(satellite_ori_size), height=int(satellite_ori_size)))

            if 0: #debug gps
                cam_location_x = dx_pixel_gps + satellite_ori_size / 2.0 + cam_location[0, 0]
                cam_location_y = dy_pixel_gps + satellite_ori_size / 2.0 + cam_location[0, 1]
                camera_gps = Camera.from_colmap(dict(
                    model='SIMPLE_PINHOLE',
                    params=(1 / meter_per_pixel, cam_location_x, cam_location_y, 0, 0, 0, 0, np.infty),
                    # np.infty for parallel projection
                    width=int(satellite_ori_size), height=int(satellite_ori_size)))

        sat_image = {
            'image': sat_map.float(),
            'camera': camera.float(),
            'T_w2cam': Pose.from_4x4mat(np.eye(4)).float() # grd 2 sat in q2r, so just eye(4)
        }

        # project to sat and find visible mask
        sat_3d = cam2sat * torch.from_numpy(cam_3d).float()
        _, visible = sat_image['camera'].world2image(sat_3d)
        cam_3d = cam_3d[visible]

        # grd ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        log_folder = os.path.join(self.root, self.log_id)
        key_points = None

        # ground images, front left color camera
        query_image_folder = os.path.join(log_folder, self.log_id + "-FL")
        name = os.path.join(query_image_folder, self.file_name[idx][:-1])
        with Image.open(name, 'r') as GrdImg:
            grd = GrdImg.convert('RGB')
            grd = grd_trans(grd)

        camera_para = (self.FL_k[0,0],self.FL_k[1,1],self.FL_k[0,2],self.FL_k[1,2])
        camera = Camera.from_colmap(dict(
            model='PINHOLE', params=camera_para,
            width=int(query_size[1]), height=int(query_size[0])))

        FL_image = {
            # to array, when have multi query
            'image': grd.float(),
            'camera': camera.float(),
            'T_w2cam': Pose.from_4x4mat(np.eye(4)).float()
        }

        _, FL_visible = camera.world2image(torch.from_numpy(cam_3d).float())
        visible = FL_visible
        cam_3d = cam_3d[visible]
        key_points = torch.from_numpy(cam_3d).float()
        FL_image['points3D_type'] = 'lidar'

        if len(key_points) == 0:
            print(self.file_name[idx][:-1], len(self.FL_kp.item().keys()) )
        num_diff = self.conf.max_num_points3D - len(key_points)
        if num_diff < 0:
            # select max_num_points
            sample_idx = np.random.choice(range(len(key_points)), self.conf.max_num_points3D)
            key_points = key_points[sample_idx]
        elif num_diff > 0 and self.conf.force_num_points3D:
            point_add = torch.ones((num_diff, 3)) * key_points[-1]
            key_points = torch.cat([key_points, point_add], dim=0)
        FL_image['points3D'] = key_points


        # init and gt pose~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # ramdom shift translation and rotation on yaw
        YawShiftRange = 30 * np.pi / 180  # in 10 degree
        yaw = 2 * YawShiftRange * np.random.random() - YawShiftRange
        R_yaw = torch.tensor([[np.cos(yaw),-np.sin(yaw),0],  [np.sin(yaw),np.cos(yaw),0], [0, 0, 1]])
        TShiftRange = 10  # in 5 meter
        T = 2 * TShiftRange * np.random.rand((3)) - TShiftRange
        T[2] = 0  # no shift on height
        #print(f'in dataset: yaw:{yaw/np.pi*180},t:{T}')

        # add random yaw and t to init pose
        init_shift = Pose.from_Rt(R_yaw,T).float()
        cam2sat_init = init_shift@cam2sat

        data = {
            'ref': sat_image,
            'query': FL_image,
            'T_q2r_init': cam2sat_init,
            'T_q2r_gt': cam2sat,
        }

        # debug
        if 0:
            #show sat imge
            color_image0 = transforms.functional.to_pil_image(data['ref']['image'], mode='RGB')
            color_image0 = np.array(color_image0)
            plt.imshow(color_image0)
            plt.show()

            # show grd image
            color_image1 = transforms.functional.to_pil_image(data['query']['image'], mode='RGB')
            color_image1 = np.array(color_image1)
            plt.imshow(color_image1)
            plt.show()
            color_image2 = transforms.functional.to_pil_image(data['query_1']['image'], mode='RGB')
            color_image2 = np.array(color_image2)
            plt.imshow(color_image2)
            plt.show()
            color_image3 = transforms.functional.to_pil_image(data['query_2']['image'], mode='RGB')
            color_image3 = np.array(color_image3)
            plt.imshow(color_image3)
            plt.show()
            color_image4 = transforms.functional.to_pil_image(data['query_3']['image'], mode='RGB')
            color_image4 = np.array(color_image4)
            plt.imshow(color_image4)
            plt.show()

        if 0:
            color_image = transforms.functional.to_pil_image(data['ref']['image'], mode='RGB')
            color_image = np.array(color_image)
            if 1:
                # sat gt green
                sat_3d = data['T_q2r_gt']*data['query']['points3D']
                sat_2d, _ = data['ref']['camera'].world2image(sat_3d)  ##camera 3d to 2d
                sat_2d = sat_2d.T
                for j in range(sat_2d.shape[1]):
                    cv2.circle(color_image, (np.int32(sat_2d[0][j]), np.int32(sat_2d[1][j])), 2, (0, 255, 0),
                               -1)

                # sat init red
                sat_3d = data['T_q2r_init']* data['query']['points3D']
                sat_2d, _ = data['ref']['camera'].world2image(sat_3d)  ##camera 3d to 2d
                sat_2d = sat_2d.T
                for j in range(sat_2d.shape[1]):
                    cv2.circle(color_image, (np.int32(sat_2d[0][j]), np.int32(sat_2d[1][j])), 2, (255, 0, 0),
                               -1)

            plt.imshow(color_image)

            # camera position
            # camera gt position
            origin = torch.zeros(3)
            origin_2d_gt, _ = data['ref']['camera'].world2image(data['T_q2r_gt'] * origin)
            origin_2d_init, _ = data['ref']['camera'].world2image(data['T_q2r_init'] * origin)
            direct = torch.tensor([0,0,6.])
            direct_2d_gt, _ = data['ref']['camera'].world2image(data['T_q2r_gt'] * direct)
            direct_2d_init, _ = data['ref']['camera'].world2image(data['T_q2r_init'] * direct)
            origin_2d_gt = origin_2d_gt.squeeze(0)
            origin_2d_init = origin_2d_init.squeeze(0)
            direct_2d_gt = direct_2d_gt.squeeze(0)
            direct_2d_init = direct_2d_init.squeeze(0)

            # plot the init direction of the body frame
            plt.scatter(x=origin_2d_init[0], y=origin_2d_init[1], c='r', s=5)
            plt.quiver(origin_2d_init[0], origin_2d_init[1], direct_2d_init[0] - origin_2d_init[0],
                       origin_2d_init[1] - direct_2d_init[1], color=['r'], scale=None)
            # plot the gt direction of the body frame
            plt.scatter(x=origin_2d_gt[0], y=origin_2d_gt[1], c='g', s=5)
            plt.quiver(origin_2d_gt[0], origin_2d_gt[1], direct_2d_gt[0] - origin_2d_gt[0],
                       origin_2d_gt[1] - direct_2d_gt[1], color=['g'], scale=None)

            plt.show()

            # grd images
            fig = plt.figure(figsize=plt.figaspect(0.5))
            ax1 = fig.add_subplot(1, 1, 1)

            color_image1 = transforms.functional.to_pil_image(data['query']['image'], mode='RGB')
            color_image1 = np.array(color_image1)
            if 1:
                grd_3d = data['query']['T_w2cam']*data['query']['points3D']
                grd_2d, valid = data['query']['camera'].world2image(grd_3d) ##camera 3d to 2d
                grd_2d = grd_2d[valid].T
                for j in range(grd_2d.shape[1]):
                    cv2.circle(color_image1, (np.int32(grd_2d[0][j]), np.int32(grd_2d[1][j])), 5, (0, 255, 0),
                               -1)
            ax1.imshow(color_image1)

            plt.show()
            print(self.file_name[idx][:-1])

        return data

if __name__ == '__main__':
    # test to load 1 data
    conf = {
        'max_num_points3D': 1024,
        'force_num_points3D': True,
        'batch_size': 1,
        'seed': 1,
        'num_workers': 0,
    }
    dataset = FordAV(conf)
    loader = dataset.get_data_loader('train', shuffle=True)  # or 'train' ‘val’

    for i, data in zip(range(1000), loader):
        print(i)


