
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from omegaconf import OmegaConf
import os

#from pixloc import run_Kitti
# from pixloc.pixlib.datasets.cmu import CMU
from pixloc.pixlib.datasets.kitti import Kitti
from pixloc.pixlib.utils.tensor import batch_to_device, map_tensor
from pixloc.pixlib.utils.tools import set_seed
from pixloc.pixlib.utils.experiments import load_experiment
from pixloc.visualization.viz_2d import (
    plot_images, plot_keypoints, plot_matches, cm_RdGn,
    features_to_RGB, add_text)

torch.set_grad_enabled(False);
mpl.rcParams['image.interpolation'] = 'bilinear'

conf = {
    'min_overlap': 0.3, #?
    'max_overlap': 1.0, #?
    'max_num_points3D': 18000,#512
    'force_num_points3D': True,
    'batch_size': 1, # only one, because 3D points not the same
    'min_baseline': 1.,
    'max_baseline': 7.,

    'resize': 720, #?
    'resize_by': 'min',
    'crop': 720, #?
    'optimal_crop': False,
    'seed': 1,
    'num_workers': 0,
}
# dataset = CMU(conf)
# loader = dataset.get_data_loader('train', shuffle=True)  # or 'train' ‘val’
dataset = Kitti(conf)
loader = dataset.get_data_loader('train', shuffle=True)  # or 'train' ‘val’
val_loader = dataset.get_data_loader('val', shuffle=True)  # or 'train' ‘val’

# Name of the example experiment. Replace with your own training experiment.
exp = 'pixloc_kitti' #run_CMU.experiment
device = 'cuda'
conf = {
    'normalize_dt': False,
    'optimizer': {'num_iters': 20,},
}
refiner = load_experiment(exp, conf,get_last=True).to(device)
#refiner.load_state_dict(torch.load('/home/shan/projects/localization/pixloc/pixloc/parameterModel_best.pth'))
print(OmegaConf.to_yaml(refiner.conf))


class Logger:
    def __init__(self, optimizers=None):
        self.costs = []
        self.dt = []
        self.p2D_trajectory = []

        if optimizers is not None:
            for opt in optimizers:
                opt.logging_fn = self.log

    def log(self, **args):
        if args['i'] == 0:
            self.costs.append([])
        self.costs[-1].append(args['cost'].mean(-1).cpu().numpy())
        self.dt.append(args['T_delta'].magnitude()[1].cpu().numpy())
        p2D, valid = self.data['ref']['camera'].world2image(args['T'] * self.data['query']['points3D'])
        self.p2D_trajectory.append((p2D[0].cpu().numpy(), valid[0].cpu().numpy()))

    def set(self, data):
        self.data = data


logger = Logger(refiner.optimizer)
# trainning
set_seed(20)
def Train(refiner, train_loader, val_loader, epochs, save_path):
    bestResult = 0.
    for epoch in range(epochs):
        refiner.train()
        for _, data in zip(range(34556), train_loader):
            data_ = batch_to_device(data, device)
            logger.set(data_)
            pred_ = refiner(data_)
            pred = map_tensor(pred_, lambda x: x[0].cpu())
            data = map_tensor(data, lambda x: x[0].cpu())
            cam_f = data['ref']['camera']
            p3D_q = data['query']['points3D']

            p2D_q, valid_q = data['query']['camera'].world2image(data['query']['T_w2cam']*p3D_q)
            p2D_r_gt, valid_r = cam_f.world2image(data['T_q2r_gt'] * p3D_q)
            p2D_q_init, _ = cam_f.world2image(data['T_q2r_init'] * p3D_q)
            p2D_q_opt, _ = cam_f.world2image(pred['T_q2r_opt'][-1] * p3D_q)
            valid = valid_q & valid_r

            losses = refiner.loss(pred_, data_)
            mets = refiner.metrics(pred_, data_)
            errP = f"ΔP {losses['reprojection_error/init'].item():.2f} -> {losses['reprojection_error'].item():.3f} px; "
            errR = f"ΔR {mets['R_error/init'].item():.2f} -> {mets['R_error'].item():.3f} deg; "
            errt = f"Δt {mets['t_error/init'].item():.2f} -> {mets['t_error'].item():.3f} m"
            print(errP, errR, errt)

        ## save modelget_similarity_fn
        compNum = epoch % 100
        print('taking snapshot ...')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save(refiner.state_dict(), save_path + 'model_' + str(compNum + 1) + '.pth')

        ## test
        current_ac = Val(refiner, val_loader, save_path, bestResult)
        if current_ac > bestResult:
            bestResult = current_ac

    print('Finished Training')

#val
def Val(refiner, val_loader, save_path, best_result):
    refiner.eval()
    acc = 0
    cnt = 0
    for idx, data in zip(range(2959), val_loader):
        data_ = batch_to_device(data, device)
        logger.set(data_)
        pred_ = refiner(data_)
        pred = map_tensor(pred_, lambda x: x[0].cpu())
        data = map_tensor(data, lambda x: x[0].cpu())
        cam_r = data['ref']['camera']
        p3D_q = data['query']['points3D']

        p2D_q, valid_q = data['query']['camera'].world2image(data['query']['T_w2cam']*p3D_q)
        p2D_r_gt, valid_r = cam_r.world2image(data['T_q2r_gt'] * p3D_q)
        p2D_q_init, _ = cam_r.world2image(data['T_q2r_init'] * p3D_q)
        p2D_q_opt, _ = cam_r.world2image(pred['T_q2r_opt'][-1] * p3D_q)
        valid = valid_q & valid_r

        losses = refiner.loss(pred_, data_)
        mets = refiner.metrics(pred_, data_)
        errP = f"ΔP {losses['reprojection_error/init'].item():.2f} -> {losses['reprojection_error'].item():.3f} px; "
        errR = f"ΔR {mets['R_error/init'].item():.2f} -> {mets['R_error'].item():.3f} deg; "
        errt = f"Δt {mets['t_error/init'].item():.2f} -> {mets['t_error'].item():.3f} m"
        print(errP, errR, errt)

        if mets['t_error'].item() < 1 and mets['R_error'].item() < 2:
            acc += 1
        cnt += 1

        # for debug
        if 1:
            imr, imq = data['ref']['image'].permute(1, 2, 0), data['query']['image'].permute(1, 2, 0)
            plot_images([imr, imq],
                        dpi=50,  # set to 100-200 for higher res
                        titles=[(data['scene'], valid_r.sum().item(), valid_q.sum().item()), errP + errt])
            plot_keypoints([p2D_r_gt[valid], p2D_q[valid_q]], colors='lime') #[cm_RdGn(valid[valid_q]), 'lime'])
            plot_keypoints([p2D_q_init[valid], np.empty((0, 2))], colors='red')
            plot_keypoints([p2D_q_opt[valid], np.empty((0, 2))], colors='blue')
            add_text(0, 'reference')
            add_text(1, 'query')
            plt.show()

            #continue
            for i, (F0, F1) in enumerate(zip(pred['ref']['feature_maps'], pred['query']['feature_maps'])):
                C_r, C_q = pred['ref']['confidences'][i][0], pred['query']['confidences'][i][0]
                plot_images([C_r, C_q], cmaps=mpl.cm.turbo, dpi=50)
                add_text(0, f'Level {i}')

                axes = plt.gcf().axes
                axes[0].imshow(imr, alpha=0.2, extent=axes[0].images[0]._extent)
                axes[1].imshow(imq, alpha=0.2, extent=axes[1].images[0]._extent)
                print(F0.dtype, torch.min(F0), torch.max(F0))
                print(F1.dtype, torch.min(F1), torch.max(F1))
                plot_images(features_to_RGB(F0.numpy(), F1.numpy(), skip=1), dpi=50)
                plt.show()
                plt.close()

            costs = logger.costs
            fig, axes = plt.subplots(1, len(costs), figsize=(len(costs)*4.5, 4.5))
            for i, (ax, cost) in enumerate(zip(axes, costs)):
                ax.plot(cost) if len(cost)>1 else ax.scatter(0., cost)
                ax.set_title(f'({i}) Scale {i//3} Level {i%3}')
                ax.grid()
            plt.show()
            plt.close()

            idxs = np.random.RandomState(0).choice(np.where(valid)[0], 15, replace=False)
            colors = mpl.cm.jet(1 - np.linspace(0, 1, len(logger.p2D_trajectory)))[:, :3]
            plot_images([imr])
            #for (p2D, valid), c in zip(logger.p2D_trajectory, colors):
            for (p2D, _), c in zip(logger.p2D_trajectory, colors):
                plot_keypoints([p2D[idxs]], colors=c[None])
            plt.show()
            plt.close()

    acc = acc/cnt
    print('acc of a epoch:#####',acc)
    if acc > best_result:
        print('best acc:@@@@@', acc)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save(refiner.state_dict(), save_path + 'Model_best.pth')
    return acc



if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    save_path = 'parameter'

    if 1: # test val
        Val(refiner, val_loader, save_path, 0)
    Train(refiner, loader, val_loader, 5, save_path)
