import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import shutil
import argparse
import json
import random
import math
import numpy as np
import yaml
from functools import partial
from more_itertools import pairwise
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
#from torchmetrics.image import StructuralSimilarityIndexMeasure
#from pytorch_msssim import SSIM
#from skimage.metrics import structural_similarity
from basicsr.metrics.psnr_ssim import calculate_ssim
from src.schedulers import get_sampler
from src.script_util import create_sigma_eps_model,create_simple_sigma_eps_model
from src.experiments import ImageExperiment
from src.utils import get_model_size, vector_norm
from src import logger
from src.constraint_functions import svd_constraint, simple_constraint, svd_constraint_ddrm
from time import time

from datasets import get_dataset
import joblib


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default='cifar10',choices=['cifar10','imagenet','celeba','celeba_hq'])
    parser.add_argument("--config_path", type=str, default='cifar10_adm') #None cifar10_adm celeba_ddim
    parser.add_argument("--constraint", type=str, default='none',
                        choices=['none','sr_bicubic','sr_averagepooling','deblur_gauss','colorization','cs_walshhadamard','inpainting','inpainting_half'])
    parser.add_argument("--constraint_proj", type=str, default='svd',
                        choices=['none','simple','svd','simple_gd','svd_gd','ddrm'])
    parser.add_argument("--constraint_scale", type=float, default=4.0)
    parser.add_argument("--constraint_lr", type=float, default=10)
    parser.add_argument("--constraint_iter", type=int, default=10) # for proj GD
    parser.add_argument("--constraint_loss", type=str, default='l1', choices=['l1','l2']) # prior_xt
    parser.add_argument("--prior_xt", type=int, default=0)

    parser.add_argument("--norm_eps", type=int, default=0)
    parser.add_argument("--sigma_type", type=str, default='pred',choices=['base', 'pred','pred_partial'])
    parser.add_argument("--sampling", type=str, default='project',choices=['denoise', 'project',])
    parser.add_argument("--norm_init_noise", type=int, default=0)
    parser.add_argument("--redesign_sigma", type=int, default=1)
    parser.add_argument("--min_sigma", type=float, default=0.003) # 0.1, 0.01 is a better for ts=50 for max_T=100
    parser.add_argument("--max_sigma", type=float, default=0.02) # sigma_gamm cycle_size
    parser.add_argument("--sigma_gamma", type=float, default=1.0)
    parser.add_argument("--cycle_size", type=int, default=10)
    parser.add_argument("--max_T", type=int, default=10)
    parser.add_argument("--sampler", type=str, default='ddim_simple_orig',choices=['ddpm', 'ddim', 'ge','ddim_simple','ddim_orig','ddpm_orig','ddim_simple_orig','ddim_simple_drag']) # ddim_simple
    parser.add_argument("--num_timesteps", type=int, default=100)
    parser.add_argument("--start_sigma", type=float, default=100) # 143.0 150 # 100 0
    parser.add_argument("--end_sigma", type=float, default=0) # 0.01
    parser.add_argument("--start_t", type=int, default=-1)
    parser.add_argument("--end_t", type=int, default=-1)
    parser.add_argument("--sigma_style", type=str, default='DDIM',choices=['Linear','DDIM','Scaled'])
    parser.add_argument("--linear_scale", type=float, default=1.0)
    parser.add_argument("--sampler_var", type=str, default='learned', choices=['learned','fixedsmall','fixedlarge','none']) #fixedlarge
    parser.add_argument("--eta", type=float, default=0.85) # 0.85
    parser.add_argument("--new_eta", type=float, default=None) # 0.85
    parser.add_argument("--refine_sigma", type=int, default=1)
    parser.add_argument("--continuous_t", type=int, default=1)
    parser.add_argument("--final_alpha_one", type=int, default=1)
    parser.add_argument("--time_shift", type=int, default=0)
    parser.add_argument("--sigma_estimate", type=str, default='1000')
    parser.add_argument("--sigma_pred_threshold", type=int, default=960) #clip_denoised
    parser.add_argument("--clip_fn", type=str, default='none',choices=['none','clamp','dynamic']) #recal_sigma_prev
    parser.add_argument("--recal_sigma_prev", type=int, default=1)
    # cifar-10  pred_proj max_T=100 num_timesteps=80 start_sigma=150 eta=1 max_sigma=0.03 min_sigma=0.01 cycle_size=10 sigma_gamma=0.7 new_eta=0.0
    # imagenet pred noise ts=100 {'mse': 0.000832389914483594, 'psner': 33.32718259638006, 'ssim': 0.9518115043640136,  'fid': 215.28264841113526} ts=80 {'mse': 0.0008208760465177792, 'psner': 33.38257123773748, 'ssim': 0.9536194617098028,  'fid': 214.80108022645686}
    parser.add_argument("--batch_size", type=int, default=10) # cifar 10:500, imagenet: 10
    parser.add_argument("--device", type=str, default='cuda:1') # 'cpu, cuda:0
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--result_dir", type=str, default='results',) #'results'
    parser.add_argument("--test_dir", type=str, default='temp2',) #'test_result' 'temp'
    parser.add_argument("--sample_size", type=int, default=1000) # 50000, 100

    parser.add_argument("--save_folder", type=str, default=None) # 50000, 100
    parser.add_argument("--save_flag", type=str, default='0')

    parser.add_argument("--sample_overwrite", type=int, default=0) # 0
    parser.add_argument("--load_folder", type=str, default='7') # None, cifar10:'7', imagenet: '13', celeba '0'
    parser.add_argument("--load_eps", type=str, default=None) # None
    parser.add_argument("--load_sigma", type=str, default='results/cifar10/7/ema_sigma_ckpt_299.pt') # 'results/imagenet/13/ema_sigma_ckpt_20.pt' # 'results/imagenet/2/ema_sigma_ckpt_20.pt' # results/celeba/0/ema_sigma_ckpt_299.pt
    parser.add_argument("--fid_target", type=str, default=None)
    # 36.569
    parser.add_argument("--method", type=str, default='pred_denoise_base',choices=['default','base','pred_denoise_base','pred_denoise_proj','pred_denoise_proj_arbit','pred_proj',
                                                                                   'pred_denoise_base_nonorm','pred_denoise_base_norefine','pred_partial_denoise_base'])

    args = parser.parse_args()
    if args.config_path is None:
        args.config_path = args.config

    args.result_dir = os.path.join(args.result_dir,args.config_path)
    args.root_dir = args.result_dir
    args.result_dir = os.path.join(args.root_dir, args.load_folder)

    args.test_dir = os.path.join(args.test_dir, args.config,args.constraint)
    sigma_estimate_rate = [float(x) for x in args.sigma_estimate]
    s = sum(sigma_estimate_rate)
    sigma_estimate_rate = [round(x/s,2) for x in sigma_estimate_rate]
    d = 1 - sum(sigma_estimate_rate)
    sigma_estimate_rate[0] += d
    args.sigma_estimate_rate = sigma_estimate_rate

    with open(os.path.join(args.result_dir,'args.json'), "r") as f:
        saved_args = json.load(f)
    args.load_eps = saved_args['load_eps']
    args.fid_target = saved_args['fid_target']
    if args.config=='imagenet':
        args.sigma_block = 2# 2 #saved_args['sigma_block']
    else:
        args.sigma_block =  saved_args['sigma_block']
    args.sigma_dropout = saved_args['sigma_dropout']
    args.use_sigma_fp16 = saved_args['use_sigma_fp16']

    with open(os.path.join("store","config", args.config_path+'.yml'), "r") as f:
        config = yaml.safe_load(f)

    def dict2namespace(config):
        namespace = argparse.Namespace()
        for key, value in config.items():
            if isinstance(value, dict):
                new_value = dict2namespace(value)
            else:
                new_value = value
            setattr(namespace, key, new_value)
        return namespace

    new_config = dict2namespace(config)
    new_config.model.use_sigma_fp16 = args.use_sigma_fp16
    new_config.model.sigma_block = args.sigma_block
    new_config.model.sigma_dropout = args.sigma_dropout
    if 'feat_layer' in saved_args:
        new_config.model.feat_layer = saved_args['feat_layer']

    def get_default(args):
        if args.config == 'cifar10':
            args.norm_max = 54.63 # 54.63  # fid 15.961
            args.norm_min = 0 # 0
            # args.load_folder = '7'  # 7 is ADM others are EDM
            # args.load_sigma = 'results/cifar10/7/ema_sigma_ckpt_299.pt' # or 'results/cifar10/8/ema_sigma_ckpt_299.pt'
            args.clip_fn = 'clamp'
            args.sampler_var = 'learned'
            #args.sampler = 'ddim'
            #args.eta = 1.0
        elif args.config == 'imagenet':
            args.norm_max = 440.0 # 440.0
            args.norm_min = 0 # 0
            # args.load_folder ='13'
            # args.load_sigma ='results/imagenet/13/ema_sigma_ckpt_20.pt'   # 'results/imagenet/0/ema_sigma_ckpt_199.pt'
            #args.sampler = 'ddim_simple'
            # args.eta = 0.85
            args.clip_fn='dynamic'
            args.sampler_var = 'learned'
        elif args.config == 'celeba':
            args.norm_max = 110  # 102.0 110
            args.norm_min = -2  # 0 -5
            # args.load_folder = '3'
            # args.load_sigma = 'results/celeba/3/ema_sigma_ckpt_20.pt'  # 'results/imagenet/0/ema_sigma_ckpt_299.pt'
            args.clip_fn = 'clamp'
            args.sampler_var = 'learned'
            #args.sampler = 'ddim'
            #args.eta = 1.0
        elif args.config == 'celeba_hq':
            args.norm_max =397.0 #397.0 # 397.0
            args.norm_min = 0.0
            args.sampler_var='fixedsmall'
            #args.sampler = 'ddim_simple'
            # args.eta = 0.85
            # args.load_folder ='3'
            # args.load_sigma = 'results/celeba_hq/3/ema_sigma_ckpt_140.pt'
        else:
            args.norm_max = None
            args.norm_min = None

        if args.method=='base':
            args.sampling = 'denoise'
            args.sigma_type = 'base'
            args.sigma_style = 'DDIM'
            args.norm_eps = False
            args.redesign_sigma = 0
            args.continuous_t = 0
            args.refine_sigma = 0
            args.num_timesteps = args.max_T
        elif args.method=='pred_denoise_base':
            args.sampling = 'denoise'
            args.sigma_type =  'pred'   #'pred' 'pred_partial'
            args.sigma_style = 'DDIM'
            args.norm_eps = True
            args.redesign_sigma = 0
            args.continuous_t = 0
            args.refine_sigma = 1
            args.num_timesteps = args.max_T
        elif args.method=='pred_partial_denoise_base':
            args.sampling = 'denoise'
            args.sigma_type =  'pred_partial'   #'pred' 'pred_partial'
            args.sigma_style = 'DDIM'
            args.norm_eps = True
            args.redesign_sigma = 0
            args.continuous_t = 0
            args.refine_sigma = 1
            args.num_timesteps = args.max_T
        elif args.method=='pred_denoise_base_nonorm':
            args.sampling = 'denoise'
            args.sigma_type =  'pred'   #'pred' 'pred_partial'
            args.sigma_style = 'DDIM'
            args.norm_eps = False
            args.redesign_sigma = 0
            args.continuous_t = 0
            args.refine_sigma = 1
            args.num_timesteps = args.max_T
        elif args.method=='pred_denoise_base_norefine':
            args.sampling = 'denoise'
            args.sigma_type =  'pred'   #'pred' 'pred_partial'
            args.sigma_style = 'DDIM'
            args.norm_eps = True
            args.redesign_sigma = 0
            args.continuous_t = 0
            args.refine_sigma = 0
            args.num_timesteps = args.max_T
        elif args.method=='pred_denoise_proj':
            args.sampling = 'denoise'
            args.sigma_type =  'pred'   #'pred' 'pred_partial'
            args.sigma_style = 'Linear'
            args.norm_eps = True
            args.redesign_sigma = 0
            args.continuous_t = 1
            #args.refine_sigma = 1
            args.num_timesteps = args.max_T
        elif args.method=='pred_denoise_proj_arbit':
            args.sampling = 'denoise'
            args.sigma_type =  'pred'   #'pred' 'pred_partial'
            args.sigma_style = 'Linear'
            args.norm_eps = True
            args.redesign_sigma = 1
            args.continuous_t = 1
            #args.refine_sigma = 1
            if args.max_T>=50:
                args.num_timesteps = int(0.8 * args.max_T)
                args.cycle_size =  int(0.1 * args.max_T)
            else:
                args.num_timesteps = args.max_T
        elif 'pred_proj' in args.method: # pred_proj sigma_type=pred sigma_estimate='0100' 16.72
            args.sampling = 'project'
            args.sigma_type = 'pred'  #'pred'
            args.sigma_style = 'Linear'
            args.norm_eps = True
            args.redesign_sigma = 1
            args.continuous_t = 1
            #args.refine_sigma = 1

        if args.sigma_type == 'base':
            args.norm_eps = False
            args.sampling = 'denoise'
            args.redesign_sigma = 0
            args.continuous_t = 0
            args.refine_sigma = 0
        else:
            args.norm_eps = True

        return args
    args = get_default(args)

    if new_config.data.dataset=='ImageNet':
        if new_config.data.subset_1k and args.constraint !='none':
            args.fid_target = 'store/fid/fid_imagenet_1k_orig.npz'
    if new_config.data.dataset=='CelebA_HQ':
        if new_config.data.subset_1k and args.constraint !='none':
            args.fid_target = 'store/fid/fid_celebahq_1k_orig.npz'
    if new_config.data.dataset == 'CELEBA':
        args.fid_target = 'store/fid/celeba_stats.npz'
    return args,new_config


class Constraint_Function:
    def __init__(self, deg,  A, Ap, constraint_fn, proj='simple',channels=3,image_size=256, lr=1.0):
        self.deg = deg
        self.A = A
        self.Ap = Ap
        self.constraint_fn = constraint_fn
        self.proj = proj
        self.channels = channels
        self.image_size = image_size
        self.lr = lr

    def transform(self,x):
        if 'simple' in self.proj:
            y = self.A(x)
        else:
            y = self.A(x)
            b, hwc = y.size()
            if 'color' in self.deg:
                hw = hwc / 1
                h = w = int(hw ** 0.5)
                y = y.reshape((b, 1, h, w))
            elif 'inp' in self.deg or 'cs' in self.deg:
                pass
            else:
                hw = hwc / 3
                h = w = int(hw ** 0.5)
                y = y.reshape((b, 3, h, w))
            y = y.reshape((b, hwc))
        return y

    def inv_transform(self,y):
        if 'simple' in self.proj:
            Apy = self.Ap(y)
        else:
            Apy = self.Ap(y).view(y.shape[0], self.channels, self.image_size,self.image_size)
            if self.deg[:6] == 'deblur':
                Apy = y.view(y.shape[0], self.channels,  self.image_size,self.image_size)
            elif self.deg == 'colorization':
                Apy = y.view(y.shape[0], 1, self.image_size, self.image_size).repeat(1, 3, 1, 1)
            elif self.deg == 'inpainting':
                Apy += self.Ap(self.A(torch.ones_like(Apy))).reshape(*Apy.shape) - 1
        return Apy

    def loss(self,x,y):
        y_hat = self.transform(x)
        #y_hat = self.A(x)
        x_hat = self.inv_transform(y)
        dim = tuple(range(1, len(y.shape)))
        forward_loss = torch.linalg.vector_norm(y_hat-y, ord=1, dim=dim).cpu()
        dim = tuple(range(1, len(x.shape)))
        backward_loss = torch.linalg.vector_norm(x_hat-x, ord=1, dim=dim).cpu()
        return forward_loss, backward_loss

    def const_loss(self,y_hat,y, ord=1, reduce='none'):
        dim = tuple(range(1, len(y.shape)))
        forward_loss = torch.linalg.vector_norm(y_hat - y, ord=ord, dim=dim)
        if reduce=='mean':
            forward_loss = torch.mean(forward_loss)
        elif reduce=='sum':
            forward_loss = torch.sum(forward_loss)
        return forward_loss

def affine_proj_GD(x0_t, y, lambda_t, infer_fn, loss_fn, n_iter=1):
    prev = torch.is_grad_enabled()
    torch.set_grad_enabled(True)
    x0_t_hat = x0_t
    for _ in range(n_iter):
        # Ax == b
        X = x0_t_hat.clone().detach().requires_grad_(True)
        y_hat = infer_fn(X)
        loss = loss_fn(y_hat,y)
        grad = torch.autograd.grad(loss, X)[0]
        new_X = X - lambda_t * grad
        x0_t_hat = new_X.detach()
    torch.set_grad_enabled(prev)
    return x0_t_hat

def get_constraint_function(args):
    if  args.constraint_proj == 'ddrm':
        svd_constraint = svd_constraint_ddrm
        args.constraint_proj = 'svd'
    if args.constraint_proj=='simple':
        A, Ap = simple_constraint(args.constraint,fn_scale=args.constraint_scale,device=args.device, base_mask_dir='store/inp_masks')
        def affine_simple(x0_t,y,lambda_t, A, Ap):
            x0_t_hat = x0_t - lambda_t * Ap(A(x0_t) - y)
            return x0_t_hat
        constraint_fn = partial(affine_simple, A=A, Ap=Ap)
        Constraint = Constraint_Function(args.constraint, A, Ap, constraint_fn, proj=args.constraint_proj,
                                         lr=args.constraint_lr)
    elif args.constraint_proj=='svd':
        A_funcs = svd_constraint(args.constraint,fn_scale=args.constraint_scale,device=args.device, base_mask_dir='store/inp_masks',
                                 image_size=config.data.image_size,channels=config.data.channels)
        A = A_funcs.A
        Ap = A_funcs.A_pinv
        def affine_svd(x0_t,y,lambda_t, A, Ap):
            x0_t_hat = x0_t - Ap(
                A(x0_t.reshape(x0_t.size(0), -1)) - y.reshape(y.size(0), -1)).reshape(*x0_t.size())
            return x0_t_hat

        constraint_fn = partial(affine_svd, A=A, Ap=Ap)
        Constraint = Constraint_Function(args.constraint, A, Ap, constraint_fn, proj=args.constraint_proj,
                                         lr=args.constraint_lr)
    elif 'gd' in args.constraint_proj:
        if args.constraint_proj == 'simple_gd':
            A, Ap = simple_constraint(args.constraint, fn_scale=args.constraint_scale, device=args.device,
                                      base_mask_dir='store/inp_masks')
        else:
            A_funcs = svd_constraint(args.constraint, fn_scale=args.constraint_scale, device=args.device,
                                     base_mask_dir='store/inp_masks',
                                     image_size=config.data.image_size, channels=config.data.channels)
            A = A_funcs.A
            Ap = A_funcs.A_pinv
        Constraint = Constraint_Function(args.constraint, A, Ap, constraint_fn=None, proj=args.constraint_proj,
                                         lr=args.constraint_lr)
        infer_fn = Constraint.transform
        loss_fn =  partial(Constraint.const_loss, ord=1 if 'l1' in args.constraint_loss else 2, reduce='sum')
        constraint_fn = partial(affine_proj_GD, infer_fn=infer_fn, loss_fn= loss_fn, n_iter=args.constraint_iter)
        Constraint.constraint_fn = constraint_fn
    else:
        A= lambda x: x
        Ap =lambda x: x
        Constraint = Constraint_Function(args.constraint, A, Ap, constraint_fn=None, proj=args.constraint_proj,
                                         lr=args.constraint_lr)
    return Constraint

def get_val_loader(args):
    def seed_worker(worker_id):
        worker_seed = args.seed % 2 ** 32
        np.random.seed(worker_seed)
        random.seed(worker_seed)
    g = torch.Generator()
    g.manual_seed(args.seed)
    if args.constraint != 'none':
        dataset, test_dataset = get_dataset(args, config)
        val_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=True, # True
            num_workers=config.data.num_workers,
            worker_init_fn=seed_worker,
            generator=g,
        )
    else:
        test_dataset = None
        val_loader = None
    print(f'Dataset has size {len(test_dataset)}')
    return test_dataset, val_loader

@torch.no_grad()
def projection_loop(self, shape, gen=None, norm_init_noise=False, style='base', constrain_fn=None,
                    norm_eps=False,  refine_prior_sigma=False, xT=None, return_log=False, chunk_size=2,
                    sigma_estimate_rate=[1,0,0], constrain_loss=None, stop_condition=0.0,max_T=None,sigma_pred_threshold=1000,
                    new_eta=None,recal_sigma_prev=False, ):
    if max_T is None:
        max_T = len(self.scheduler.timesteps)-1
    self.scheduler.reset_state()
    if xT is None:
        xt, zt = self.get_noise_xt(shape=shape, gen=gen, norm_noise=norm_init_noise, sigma=self.scheduler.sampling_sigmas[0])
    else:
        xt = xT
        zt = self.convert_coordinate(xt, sigma=self.scheduler.sampling_sigmas[0])
    x0 = xt
    eps_list, z_list, sigma_list=[], [],[]
    x0_prec_list,x0_postc_list = [], []
    const_loss_list=[]
    sigma_t = self.scheduler.sampling_sigmas[0]
    t =  self.scheduler.timesteps[0]
    costheta=0.99
    last_norm= vector_norm(xt) / math.sqrt(self.dim)
    if return_log:
        z_list = [zt.cpu()]
        sigma_list = [sigma_t.cpu()]
    sampling_sigmas = self.scheduler.sampling_sigmas
    T = len(sampling_sigmas)
    best_val, best_x0 = 10000, x0
    const_val =  None
    sample_time_step=len(self.scheduler.timesteps)
    for ind in range(max_T):
        if ind == sample_time_step-1 and new_eta is not None:
            self.scheduler.eta = new_eta
        sigma_prev_orig = sampling_sigmas[-1] if ind >= T-1 else sampling_sigmas[ind+1]
        if recal_sigma_prev:
            sigma_prev = sigma_t * (sampling_sigmas[ind+1]/sampling_sigmas[ind])
        else:
            sigma_prev = sigma_prev_orig

        #print(ind, t.mean().item(), sigma_prev.mean().item())
        cur_style = style
        cur_refine_prior = refine_prior_sigma
        if t.max() > sigma_pred_threshold:
            cur_style = 'base'
            cur_refine_prior = False
        eps, eps_logvar, sigma_t, sigma_prev = self.get_denoise_vector(xt, t, sigma_t, sigma_prev, cur_style, norm_eps, refine_prior_sigma=cur_refine_prior, chunk_size=chunk_size)
        # if style=='pred_partial' or style=='base':
        #     sigma_prev = sampling_sigmas[-1] if ind >= T-1 else sampling_sigmas[ind+1]
        x0_hat = self.scheduler.pred_xstart(xt, eps, sigma_t)
        x0_hat = self.clip_denoise_fn(x0_hat)
        if constrain_fn is not None:
            x0 = constrain_fn(x0_hat)
        else:
            x0 = x0_hat
        #xt = self.scheduler.pred_xprev(x0, eps, sigma_t, sigma_prev, eps_logvar)
        xt = self.scheduler.pred_xprev(x0=x0, eps=eps, sigma_t=sigma_t, sigma_prev=sigma_prev, xt=xt,
                                       log_variance=eps_logvar)
        #xt = self.scheduler.pred_xprev_with_sigma_noise(x0, eps, signal_sigma, noise_sigma)
        cur_norm = vector_norm(xt) / math.sqrt(self.dim)
        cur_dist = torch.sqrt(cur_norm ** 2 + self.norm_max ** 2 - 2 * cur_norm *  self.norm_max * costheta + 1e-8)
        norm_ratio = cur_norm / last_norm
        sigma_t0 = sigma_prev_orig
        sigma_t1 = sigma_prev
        sigma_t2 = sigma_t * norm_ratio
        sigma_t3 = cur_dist
        sigma_t = sigma_estimate_rate[0]*sigma_t0+sigma_estimate_rate[1]*sigma_t1+ sigma_estimate_rate[2]*sigma_t2+ sigma_estimate_rate[3]*sigma_t3
        t = self.scheduler.get_t_from_sigma(sigma_t)
        last_norm = cur_norm
        if constrain_loss is not None:
            const, _ = constrain_loss(x0.clamp(-1, 1))
            const_val = torch.mean(const)
            if const_val < best_val:
                best_x0 = x0
                best_val = const_val
            if return_log:
                const_loss_list.append(const.cpu())
        else:
            best_x0 = x0
        if return_log:
            zt = self.convert_coordinate(xt, sigma=sigma_prev)
            z_list.append(zt.cpu())
            eps_list.append(eps.cpu())
            x0_prec_list.append(x0_hat.cpu())
            x0_postc_list.append(x0.cpu())
            sigma_list.append(sigma_t.cpu())
        if torch.isnan(xt).any() or ( const_val is not None and const_val <= stop_condition):
            break
        #print('constraint loss: iter',ind, ', loss:', const_val)
    xt = best_x0.cpu()
    return_list = [z_list, eps_list, x0_prec_list, x0_postc_list, sigma_list,const_loss_list]
    return xt, return_list

@torch.no_grad()
def evaluate_unconstraint(experiment, n_samples, images_dir, norm_init_noise=False, style='base', sampling='denoise',norm_eps=False,refine_prior_sigma=False,
                          sigma_estimate_rate=[1,0,0],max_T=None,sigma_pred_threshold=1000,new_eta=None,recal_sigma_prev=False,
                          return_log=False, res_pkl_path=''):
    log_dict = {}
    batch_size = experiment.batch_size
    n_batches =math.ceil(n_samples / batch_size)
    shape = (batch_size,) + experiment.data_shape
    gen = experiment.new_gen()
    rank=0
    return_lists = []
    for i in range(n_batches):
        skip_flag = True
        for j in range(batch_size):
            path = os.path.join(images_dir, f'{rank:02}-{i:05}-{j:03}.png')
            skip_flag = skip_flag and os.path.exists(path)
            if not skip_flag:
                break
        if skip_flag:
            print('skip images for:', f'{rank:02}-{i:05}-({0:03}~{batch_size-1:03}).png')
            continue
        t1 = time()
        if sampling=='project':
            sample, return_list = projection_loop(self=experiment,shape=shape, gen=gen, norm_init_noise=norm_init_noise,
                                                          style=style, constrain_fn=None,
                                                          norm_eps=norm_eps, refine_prior_sigma=refine_prior_sigma,
                                                          xT=None, return_log=return_log, chunk_size=1, sigma_estimate_rate=sigma_estimate_rate,
                                                         constrain_loss = None, max_T=max_T,stop_condition=0.0,
                                                  sigma_pred_threshold=sigma_pred_threshold,new_eta=new_eta,recal_sigma_prev=recal_sigma_prev)
        else:
            sample, return_list = experiment.denoise_loop(shape=shape, gen=gen, norm_init_noise=norm_init_noise,
                                                          style=style, constrain_fn=None,
                                                          norm_eps=norm_eps, refine_prior_sigma=refine_prior_sigma,
                                                          return_log=return_log, chunk_size=1,
                                                          sigma_pred_threshold=sigma_pred_threshold,new_eta=new_eta)
        print('time:', time()-t1)
        return_lists.append(return_list)
        if return_log:
            joblib.dump(return_lists, res_pkl_path)
            print('res pkl save done', i, res_pkl_path)
        sample = sample.add(1).div(2).clamp(0, 1)
        for j, img in enumerate(sample):
            path = os.path.join(images_dir, f'{rank:02}-{i:05}-{j:03}.png')
            save_image(img, path)
        print(f'done batches:{i}/{n_batches}')
    fid = experiment.fid_fn(images_dir)
    #torch.cuda.empty_cache()
    log_dict['fid'] = fid
    return log_dict,return_lists

def ssim_fn(sample, orig):
    sample = torch.round(sample*255).to(torch.uint8)  #sample.cpu().numpy()
    orig =torch.round(orig*255).to(torch.uint8)  #s #orig.cpu().numpy()
    ssims=[]
    for i in range(len(sample)):
        sr=sample[i]
        gt= orig[i]
        val =calculate_ssim(sr, gt, crop_border=0, test_y_channel=False)
        #val = structural_similarity(a, b, data_range=1.0, channel_axis=0, win_size=11)
        ssims.append(val)
    #ssims = np.array(ssims)
    return ssims

def analyze_log(return_list,x_orig,y, constrain_loss):
    [z_list, eps_list, x0_prec_list, x0_postc_list] = return_list[:4]
    res_dict={}
    key_list =['zt','z0_prec','z0_postc']
    for key in key_list:
        res_dict[key]={'psnr':[],'ssim':[],'const':[]}
    #ssim_fn = StructuralSimilarityIndexMeasure(data_range=1.0)
    for kk in range(len(eps_list)):
        zt = z_list[kk]
        z0_prec =  x0_prec_list[kk]
        z0_postc = x0_postc_list[kk]
        for key, sample in zip(key_list,[zt,z0_prec,z0_postc]):
            sample = sample.add(1).div(2).clamp(0, 1)
            mse = torch.mean((sample - x_orig) ** 2)
            psnr = 10 * torch.log10(1 / mse)
            ssim = ssim_fn(sample, x_orig)
            x_hat = 2 * sample - 1.0
            const, _ = constrain_loss(x_hat.to(y.device), y)
            const = torch.mean(const)
            res_dict[key]['psnr'].append(psnr.item())
            res_dict[key]['ssim'].append(np.mean(ssim))
            res_dict[key]['const'].append(const.item())
    return res_dict
@torch.no_grad()
def evaluate_constraint(experiment, data_loader, Constraint, images_dir,n_samples=-1,
                        transform_dir=None,  norm_init_noise=False, style='base',sampling='denoise', norm_eps=False,
                        refine_prior_sigma=False,prior_xt=False,sigma_estimate_rate=[1,0,0],return_log=False,
                        max_T=None,sigma_pred_threshold=1000, new_eta=None,recal_sigma_prev=False):
    log_dict = {}
    n_batches = len(data_loader)
    device = experiment.device
    gen = experiment.new_gen()
    #ssim_fn = StructuralSimilarityIndexMeasure(data_range=1.0,reduction='none')
    #ssim_fn = SSIM(win_size=11,data_range=1.0,win_sigma=1.5,K=(0.01,0.1),size_average=False)
    rank=0
    psnr_list=[]
    mse_list=[]
    ssim_list=[]
    const_f_loss, const_b_loss, const_orig_loss=[],[],[]
    full_results=[]
    for i, (x_orig, classes) in enumerate(data_loader):
        batch_size = x_orig.shape[0]
        batch_x = x_orig.to(device)
        batch_x = 2 * batch_x - 1.0
        skip_flag = True
        for j in range(batch_size):
            path = os.path.join(images_dir, f'{rank:02}-{i:05}-{j:03}.png')
            skip_flag = skip_flag and os.path.exists(path)
            if not skip_flag:
                break
        if skip_flag:
            print('skip images for:', f'{rank:02}-{i:05}-({0:03}~{batch_size-1:03}).png')
            continue

        y = Constraint.transform(batch_x)
        Apy = Constraint.inv_transform(y)
        if transform_dir is not None:
            sample_apy = Apy.add(1).div(2).clamp(0, 1)
            for j in range(len(Apy)):
                save_image(sample_apy[j],os.path.join(transform_dir, f'Apy_{rank:02}-{i:05}-{j:03}.png') )
                save_image(x_orig[j], os.path.join(transform_dir,f'orig_{rank:02}-{i:05}-{j:03}.png'))

        shape = (batch_size,) + experiment.data_shape
        constraint_fn = partial(Constraint.constraint_fn,y=y, lambda_t=Constraint.lr)
        constrain_loss = partial(Constraint.loss,y=y)
        if prior_xt:
           xT = Apy + experiment.scheduler.sampling_sigmas[0] * torch.randn_like(Apy)
        else:
            xT = None
        t1 = time()
        if sampling=='project':
            sample, return_list = projection_loop(self=experiment,shape=shape, gen=gen, norm_init_noise=norm_init_noise,
                                                          style=style, constrain_fn=constraint_fn,
                                                          norm_eps=norm_eps, refine_prior_sigma=refine_prior_sigma,
                                                          xT=xT, return_log=return_log, chunk_size=1, sigma_estimate_rate=sigma_estimate_rate,
                                                         constrain_loss = constrain_loss, max_T=max_T,stop_condition=0.0,
                                                  sigma_pred_threshold=sigma_pred_threshold,new_eta=new_eta,recal_sigma_prev=recal_sigma_prev)
        else:
            sample, return_list = experiment.denoise_loop(shape=shape, gen=gen, norm_init_noise=norm_init_noise,
                                                          style=style, constrain_fn=constraint_fn,
                                                          norm_eps=norm_eps, refine_prior_sigma=refine_prior_sigma,
                                                          xT=xT, return_log=return_log, chunk_size=1,constrain_loss = constrain_loss,
                                                          sigma_pred_threshold=sigma_pred_threshold,new_eta=new_eta)
        print('time:', time() - t1)

        sample = sample.add(1).div(2).clamp(0, 1)
        for j, img in enumerate(sample):
            path = os.path.join(images_dir, f'{rank:02}-{i:05}-{j:03}.png')
            save_image(img, path)
        mse =  torch.mean((sample - x_orig) ** 2, dim=(1,2,3))
        psnr = 10 * torch.log10(1 / mse)
        ssim = ssim_fn(sample, x_orig)
        x_hat = 2 * sample - 1.0
        x_hat = x_hat.to(device)
        const_f, const_b = Constraint.loss(x_hat, y)
        cons_orig = torch.linalg.vector_norm(x_hat - batch_x, ord=1, dim=(1,2,3)).cpu()
        mse_list += mse.numpy().tolist()
        psnr_list += psnr.numpy().tolist()
        ssim_list += ssim
        const_f_loss += const_f.numpy().tolist()
        const_b_loss += const_b.numpy().tolist()
        const_orig_loss += cons_orig.numpy().tolist()

        print(f'done batches:{i}/{n_batches},  psnr:{np.mean(psnr_list)}, ssim:{np.mean(ssim_list)}, cost:{np.mean(const_f_loss)}')

        if return_log:
            result = analyze_log(return_list,x_orig,y,Constraint.loss)
            full_results.append(result)

        if n_samples>0 and (i+1)*batch_size>n_samples:
            break

    log_dict['mse'] = np.mean(mse_list)
    log_dict['psner'] = np.mean(psnr_list)
    log_dict['ssim'] = np.mean(ssim_list)
    log_dict['const_f_loss'] = np.mean(const_f_loss)
    log_dict['const_b_loss'] = np.mean(const_b_loss)
    log_dict['const_orig_loss'] = np.mean(const_orig_loss)

    fid = experiment.fid_fn(images_dir)
    #torch.cuda.empty_cache()
    log_dict['fid'] = fid
    log_dict['full_log']={'psnr':psnr_list,'mse':mse_list,'ssim':ssim_list,'const_forward':const_f_loss,'const_backward':const_b_loss,'const_orig_loss':const_orig_loss }
    log_dict['full_results'] = full_results
    #return_list.append([sample, x_orig])
    return log_dict, return_list


def main(args, config):
    logger.configure(dir='./logs/')

    if args.save_folder is not None :
        args.test_dir = args.save_folder
        print('save folder',  args.save_folder)
        if not os.path.exists(args.test_dir):
            os.makedirs(args.test_dir)
            with open(os.path.join(args.test_dir, 'args.json'), 'w') as f:
                arg_dict = args.__dict__
                arg_dict['device'] = str(arg_dict['device'])
                json.dump(arg_dict, f)
    else:
        i=0
        root_dir = args.test_dir
        save_dir = os.path.join(root_dir, str(i))
        while os.path.exists(save_dir):
            i += 1
            save_dir = os.path.join(root_dir, str(i))
        args.test_dir = save_dir
        os.makedirs(args.test_dir, exist_ok=True)
        with open(os.path.join(args.test_dir, 'args.json'), 'w') as f:
            arg_dict = args.__dict__
            arg_dict['device'] = str(arg_dict['device'])
            json.dump(arg_dict, f)

    print('args:',args)
    print('config:', config)

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True

    # model
    model_config = config.model
    if model_config.type=='openai':
        model, sigma_model,feat_shape = create_sigma_eps_model(**vars(model_config))
    else:
        model, sigma_model, feat_shape = create_simple_sigma_eps_model(config)
    print('eps model size:', get_model_size(model))
    print('sigma model size:', get_model_size(sigma_model))

    ckpt = torch.load(args.load_eps, map_location='cpu')
    model.load_state_dict(ckpt)
    print('load eps model from', args.load_eps)
    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    model.to(args.device)

    ckpt = torch.load(args.load_sigma, map_location='cpu')
    sigma_model.load_state_dict(ckpt)
    print('load sigma model from', args.load_sigma)
    for param in sigma_model.parameters():
        param.requires_grad = False
    sigma_model.eval()
    sigma_model.to(args.device)

    if model_config.use_fp16:
        model.convert_to_fp16()
    if model_config.use_sigma_fp16:
        sigma_model.convert_to_fp16()
    #model = torch.nn.DataParallel(model)
    #sigma_model = torch.nn.DataParallel(sigma_model)

    # sampler
    samp_config = config.diffusion
    sampler = get_sampler(args.sampler, samp_config.num_diffusion_timesteps, args.num_timesteps,
                          beta_schedule= samp_config.beta_schedule, sigma_style=args.sigma_style,
                          set_alpha_to_one=args.final_alpha_one, start_sigma=args.start_sigma,end_sigma=args.end_sigma,
                         sampler_var=args.sampler_var, continuous_t=args.continuous_t,linear_scale=args.linear_scale,
                          eta=args.eta, norm_eps=args.norm_eps,start_t=args.start_t, end_t=args.end_t)

    if args.redesign_sigma and args.max_T>args.num_timesteps:
        print('redesign sigma', args.num_timesteps, args.max_T)
        sampler.continuous_t = True
        cycle_size =  args.cycle_size
        iterations = np.arange(args.max_T - args.num_timesteps)
        cycle = np.floor(1 + iterations / cycle_size)
        x = np.abs(iterations / cycle_size - cycle + 1)
        sigma_res = np.log(args.min_sigma) + (np.log(args.max_sigma) - np.log(args.min_sigma)) * np.maximum(0, (1 - x)) * args.sigma_gamma ** (cycle-1)
        sigma_res = torch.tensor(np.exp(sigma_res))
        sampler.sampling_sigmas = torch.cat([torch.clamp(sampler.sampling_sigmas[:-1], min = args.min_sigma), sigma_res])
        sampler.timesteps =  sampler.get_t_from_sigma(sampler.sampling_sigmas)
        sampler.timesteps = torch.cat([sampler.timesteps, torch.tensor([-1])])
        sampler.sampling_sigmas = torch.cat([sampler.sampling_sigmas, torch.tensor([sampler.final_sigma])])


    sampler.to(args.device)

    # experiment
    data_config=config.data
    experiment = ImageExperiment(model, sampler, batch_size=args.batch_size, data_shape=(data_config.channels,data_config.image_size,data_config.image_size),seed=args.seed,  device=args.device,
                                 save_folder=args.test_dir,dist_train=False, time_shift=args.time_shift)
    experiment.set_model(model, sigma_model,learn_epsvar = True if model_config.type=='openai' else False)
    experiment.fid_helper(args.fid_target)
    experiment.set_norm_maxmin(args.norm_min, args.norm_max)
    experiment.set_clip_fn(args.clip_fn)

    # evaluate
    #torch.cuda.empty_cache()
    images_dir = os.path.join(args.test_dir,args.save_flag,'images')
    if os.path.exists(images_dir):
        if args.sample_overwrite:
            shutil.rmtree(images_dir)
    os.makedirs(images_dir, exist_ok=True)
    #print('sampling_sigmas:',experiment.scheduler.sampling_sigmas)
    return_log=True
    if args.constraint == 'none':
        # evaluate
        log_dict, return_lists = evaluate_unconstraint(experiment, args.sample_size, images_dir, norm_init_noise=args.norm_init_noise, style=args.sigma_type, sampling=args.sampling, norm_eps=args.norm_eps,refine_prior_sigma=args.refine_sigma,
                                         sigma_estimate_rate=args.sigma_estimate_rate, max_T=args.max_T,sigma_pred_threshold=args.sigma_pred_threshold,
                                         new_eta=args.new_eta,recal_sigma_prev=args.recal_sigma_prev,return_log=return_log, res_pkl_path = os.path.join(args.test_dir, args.save_flag, 'results_dump.pkl'))
        if return_log:
            res_pkl_path = os.path.join(args.test_dir, args.save_flag, 'results_dump.pkl')
            joblib.dump(return_lists, res_pkl_path)
            print('res pkl save done', res_pkl_path)
    else:
        # data
        test_dataset, val_loader = get_val_loader(args)
        # constraint
        Constraint = get_constraint_function(args)
        Constraint.channels = data_config.channels
        Constraint.image_size = data_config.image_size
        # evaluate
        transform_dir = os.path.join(args.test_dir,args.save_flag, 'transform')
        if os.path.exists(transform_dir):
            if args.sample_overwrite:
                shutil.rmtree(transform_dir)
        os.makedirs(transform_dir, exist_ok=True)
        #transform_dir = None
        log_dict, return_list = evaluate_constraint(experiment, val_loader, Constraint,images_dir,args.sample_size, transform_dir=transform_dir,
                                       norm_init_noise=args.norm_init_noise, style=args.sigma_type, sampling=args.sampling, norm_eps=args.norm_eps,
                                       refine_prior_sigma=args.refine_sigma,prior_xt=args.prior_xt,sigma_estimate_rate=args.sigma_estimate_rate,
                                                    return_log=False,max_T=args.max_T,sigma_pred_threshold=args.sigma_pred_threshold,new_eta=args.new_eta,
                                                    recal_sigma_prev=args.recal_sigma_prev)
        #torch.save(return_list, os.path.join(args.test_dir, 'return_log.pt'))
    with open(os.path.join(args.test_dir,args.save_flag, 'results.json'), 'w') as f:
        json.dump(log_dict, f)
    if 'full_log' in log_dict:
        del log_dict['full_log']
    if 'full_results' in log_dict:
        del log_dict['full_results']
    print(log_dict)
    print('evaluate done')
    #torch.cuda.empty_cache()

if __name__=='__main__':
    args, config=get_args()
    main(args, config)