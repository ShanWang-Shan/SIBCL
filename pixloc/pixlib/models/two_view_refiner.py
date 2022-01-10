"""
The top-level model of training-time PixLoc.
Encapsulates the feature extraction, pose optimization, loss and metrics.
"""
import torch
from torch.nn import functional as nnF
import logging
from copy import deepcopy
import omegaconf
import numpy as np

from pixloc.pixlib.models.base_model import BaseModel
from pixloc.pixlib.models import get_model
from pixloc.pixlib.models.utils import masked_mean
from pixloc.pixlib.geometry.losses import scaled_barron

from matplotlib import pyplot as plt
from torchvision import transforms
import cv2


logger = logging.getLogger(__name__)

# add by shan
share_weight = False #
cal_confidence = 3 # 0: no confidence, 1:only ref 2: only query, 3:both query and ref
no_opt = False 
l1_loss = True

class TwoViewRefiner(BaseModel):
    default_conf = {
        'extractor': {
            'name': 's2dnet',
        },
        'optimizer': {
            'name': 'basic_optimizer',
        },
        'duplicate_optimizer_per_scale': False,
        'success_thresh': 2,
        'clamp_error': 50,
        'normalize_features': True,
        'normalize_dt': True,

        # deprecated entries
        'init_target_offset': None,
    }
    required_data_keys = {
        'ref': ['image', 'camera', 'T_w2cam'],
        'query': ['image', 'camera', 'T_w2cam'],
    }
    strict_conf = False  # need to pass new confs to children models

    def _init(self, conf):
        self.extractor = get_model(conf.extractor.name)(conf.extractor)
        assert hasattr(self.extractor, 'scales')
        # if not share_weight:
        #     self.extractor_sat = deepcopy(self.extractor) # add by shan

        Opt = get_model(conf.optimizer.name)
        if conf.duplicate_optimizer_per_scale:
            oconfs = [deepcopy(conf.optimizer) for _ in self.extractor.scales]
            feature_dim = self.extractor.conf.output_dim
            if not isinstance(feature_dim, int):
                for d, oconf in zip(feature_dim, oconfs):
                    with omegaconf.read_write(oconf):
                        with omegaconf.open_dict(oconf):
                            oconf.feature_dim = d
            self.optimizer = torch.nn.ModuleList([Opt(c) for c in oconfs])
        else:
            self.optimizer = Opt(conf.optimizer)

        if conf.init_target_offset is not None:
            raise ValueError('This entry has been deprecated. Please instead '
                             'use the `init_pose` config of the dataloader.')

    def _forward(self, data):
        def process_siamese(data_i, data_type):
            if data_type == 'ref':
                data_i['type'] = 'sat'
            pred_i = self.extractor(data_i)
            pred_i['camera_pyr'] = [data_i['camera'].scale(1/s)
                                    for s in self.extractor.scales]
            return pred_i

        pred = {i: process_siamese(data[i], i) for i in ['ref', 'query']}

        # # change by shan
        # if share_weight:
        #     pred = {i: process_siamese(data[i]) for i in ['ref', 'query']}
        # else:
        #
        #     # add by shan for satellite image extractor
        #     def process_sat(data_i):
        #         pred_i = self.extractor_sat(data_i, sat_flag=True)
        #         pred_i['camera_pyr'] = [data_i['camera'].scale(1/s)
        #                             for s in self.extractor_sat.scales]
        #         return pred_i
        #     pred = {i: process_siamese(data[i]) for i in ['query']}
        #     pred.update({i: process_sat(data[i]) for i in ['ref']})

        p3D_query = data['query']['points3D']
        T_init = data['T_q2r_init']

        pred['T_q2r_init'] = []
        pred['T_q2r_opt'] = []
        pred['valid_masks'] = []
        pred['L1_loss'] = []
        for i in reversed(range(len(self.extractor.scales))):
            F_ref = pred['ref']['feature_maps'][i]
            F_q = pred['query']['feature_maps'][i]
            cam_ref = pred['ref']['camera_pyr'][i]
            cam_q = pred['query']['camera_pyr'][i]
            if self.conf.duplicate_optimizer_per_scale:
                opt = self.optimizer[i]
            else:
                opt = self.optimizer

            # debug original image
            if 0:
                # _,_,H,W = pred['ref']['feature_maps'][i].size()
                # F_ref = nnF.interpolate(data['ref']['image'], size=(H,W), mode='bilinear')
                # _, _, H, W = pred['query']['feature_maps'][i].size()
                # F_q = nnF.interpolate(data['query']['image'], size=(H,W), mode='bilinear')

                fig = plt.figure(figsize=plt.figaspect(0.5))
                ax1 = fig.add_subplot(1, 2, 1)
                ax2 = fig.add_subplot(1, 2, 2)
                # color_image0 = transforms.functional.to_pil_image(F_q[0], mode='RGB')  # grd
                # color_image0 = np.array(color_image0)
                # color_image1 = transforms.functional.to_pil_image(F_ref[0], mode='RGB')  # sat
                # color_image1 = np.array(color_image1)
                color_image1, color_image0 = features_to_RGB(F_ref[0].cpu().numpy(), F_q[0].cpu().numpy(), skip=1)

                # sat
                p3D_ref = data['T_q2r_gt'] * data['query']['points3D']
                p2D_ref, visible = cam_ref.world2image(p3D_ref)
                p2D_ref = p2D_ref.cpu().detach()
                for j in range(p2D_ref.shape[1]):
                    cv2.circle(color_image1, (np.int32(p2D_ref[0][j][0]), np.int32(p2D_ref[0][j][1])), 2, (255, 0, 0),
                               -1)

                p3D_q = data['query']['T_w2cam'] * data['query']['points3D']
                p2D, visible = cam_q.world2image(p3D_q)
                p2D = p2D.cpu().detach()
                # valid = valid & visible
                for j in range(p2D.shape[1]):
                    cv2.circle(color_image0, (np.int32(p2D[0][j][0]), np.int32(p2D[0][j][1])), 2, (255, 0, 0),
                               -1)

                ax1.imshow(color_image0)
                ax2.imshow(color_image1)
                plt.show()

            # p2D_ref, visible = cam_ref.world2image(p3D_ref)
            # F_ref, mask, _ = opt.interpolator(F_ref, p2D_ref)
            # mask &= visible
            p2D_query, visible = cam_q.world2image(data['query']['T_w2cam']*p3D_query)
            F_q, mask, _ = opt.interpolator(F_q, p2D_query)
            mask &= visible

            W_ref_q = None
            if cal_confidence != 0 and self.extractor.conf.get('compute_uncertainty', False):
            #if self.extractor.conf.get('compute_uncertainty', False):
                W_q = pred['query']['confidences'][i]
                W_q, _, _ = opt.interpolator(W_q, p2D_query)
                W_ref = pred['ref']['confidences'][i]
                # W_ref, _, _ = opt.interpolator(W_ref, p2D_ref)

                if cal_confidence == 1:
                    # only use confidence of ref
                    W_ref_q = (W_ref, None)
                elif cal_confidence == 2:
                    # only use confidence of query
                    W_ref_q = (None, W_q)
                else:
                    W_ref_q = (W_ref, W_q)


            if self.conf.normalize_features:
                # F_ref = nnF.normalize(F_ref, dim=2)  # B x N x C
                # F_q = nnF.normalize(F_q, dim=1)  # B x C x W x H
                F_q = nnF.normalize(F_q, dim=2)  # B x N x C
                F_ref = nnF.normalize(F_ref, dim=1)  # B x C x W x H


            if no_opt:
                T_opt = T_init.detach()
            else:
                # T_opt, failed = opt(dict(
                #     p3D=p3D_ref, F_ref=F_ref, F_q=F_q, T_init=T_init, cam_q=cam_q,
                #     mask=mask, W_ref_q=W_ref_q))
                T_opt, failed = opt(dict(
                    p3D=p3D_query, F_ref=F_ref, F_q=F_q, T_init=T_init, camera=cam_ref,
                    mask=mask, W_ref_q=W_ref_q))

            pred['T_q2r_init'].append(T_init)
            pred['T_q2r_opt'].append(T_opt)
            T_init = T_opt.detach()

            # add by shan, query & reprojection GT error, for query unet back propogate
            if l1_loss and not share_weight:
                loss = self.preject_l1loss(opt, p3D_query, F_ref, F_q, data['T_q2r_gt'], cam_ref, mask=mask, W_ref_query=W_ref_q)
                pred['L1_loss'].append(loss)

        return pred

    def preject_l1loss(self, opt, p3D, F_ref, F_query, T_gt, camera, mask=None, W_ref_query= None):
        args = (camera, p3D, F_ref, F_query, W_ref_query)
        res, valid, w_unc, _, _ = opt.cost_fn.residuals(T_gt, *args)
        if mask is not None:
            valid &= mask

        # compute the cost and aggregate the weights
        cost = (res ** 2).sum(-1)
        cost, w_loss, _ = opt.loss_fn(cost)
        loss = cost * valid.float()
        if w_unc is not None:
            # do not gradient back to w_unc
            weight = w_unc.detach()
            loss = loss * weight

        return torch.sum(loss, dim=-1)

    # add by shan for satellite image extractor
    def add_sat_extractor(self):
        self.extractor.add_sat_unet()
        # self.extractor_sat = deepcopy(self.extractor)

    def loss(self, pred, data):
        cam_ref = data['ref']['camera']

        def project(T_q2r):
            return cam_ref.world2image(T_q2r * data['query']['points3D'])

        p2D_r_gt, mask = project(data['T_q2r_gt'])
        p2D_r_i, mask_i = project(data['T_q2r_init'])
        mask = (mask & mask_i).float()

        too_few = torch.sum(mask, -1) < 10
        if torch.any(too_few):
            logger.warning('Few points in batch '+str(data['scene']))
            # logger.warning(
            #     'Few points in batch '+str([
            #         (data['scene'][i], data['ref']['index'][i].item(),
            #          data['query']['index'][i].item())
            #         for i in torch.where(too_few)[0]]))

        def reprojection_error(T_q2r):
            p2D_r, _ = project(T_q2r)
            err = torch.sum((p2D_r_gt - p2D_r)**2, dim=-1)
            err = scaled_barron(1., 2.)(err)[0]/4
            err = masked_mean(err, mask, -1)
            return err

        num_scales = len(self.extractor.scales)
        success = None
        losses = {'total': 0.}
        for i, T_opt in enumerate(pred['T_q2r_opt']):
            err = reprojection_error(T_opt).clamp(max=self.conf.clamp_error)
            loss = err / num_scales
            if i > 0:
                loss = loss * success.float()
            thresh = self.conf.success_thresh * self.extractor.scales[-1-i]
            success = err < thresh
            losses[f'reprojection_error/{i}'] = err
            losses['total'] += loss
        losses['reprojection_error'] = err
        losses['total'] *= (~too_few).float()

        err_init = reprojection_error(pred['T_q2r_init'][0])
        losses['reprojection_error/init'] = err_init

        # add by shan, query & reprojection GT error, for query unet back propogate
        if l1_loss and not share_weight:
            losses['L1_loss'] = sum(pred['L1_loss'])/num_scales
        else:
            losses['L1_loss'] = torch.tensor(0.)

        return losses

    def metrics(self, pred, data):
        T_r2q_gt = data['T_q2r_gt'].inv()

        @torch.no_grad()
        def scaled_pose_error(T_q2r):
            err_R, err_t = (T_q2r @ T_r2q_gt).magnitude()
            err_x = (T_q2r @ T_r2q_gt).magnitude_lateral()
            if self.conf.normalize_dt:
                err_t /= torch.norm(T_r2q_gt.t, dim=-1)
            # change for validate lateral error only, change by shan
            # return err_R, err_t
                err_x /= T_r2q_gt.magnitude_lateral()
            return err_R, err_x

        metrics = {}
        for i, T_opt in enumerate(pred['T_q2r_opt']):
            err = scaled_pose_error(T_opt)
            metrics[f'R_error/{i}'], metrics[f't_error/{i}'] = err
        metrics['R_error'], metrics['t_error'] = err

        err_init = scaled_pose_error(pred['T_q2r_init'][0])
        metrics['R_error/init'], metrics['t_error/init'] = err_init

        return metrics
