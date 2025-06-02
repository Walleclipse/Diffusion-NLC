import math
import os
import copy
from more_itertools import pairwise
from functools import partial
import shutil
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import OrderedDict
import torch
import torch.nn.functional as F
import torch.distributed as dist
from pytorch_fid.fid_score import calculate_fid_given_paths, compute_statistics_of_path,calculate_frechet_distance
from pytorch_fid.inception import InceptionV3
from torchvision.utils import save_image
from torch.nn.parallel.distributed import DistributedDataParallel as DDP

from .utils import normalize, cosine_distance, EMA_smooth, normalized_MSE, vector_norm, normalized_huber,MSE_normalized_MSE
from .fp16_util import MixedPrecisionTrainer

def get_loss(loss, reduction='mean'):
    if loss == 'mse' or loss == 'l2':
        loss_fn = torch.nn.MSELoss(reduction=reduction ) # F.mse_loss
    elif loss == 'mae' or loss == 'l1':
        loss_fn =torch.nn.L1Loss(reduction=reduction ) # F.l1_loss
    elif loss == 'huber':
        loss_fn = torch.nn.SmoothL1Loss(reduction=reduction ) #F.smooth_l1_loss
    elif loss == 'cosine':
        loss_fn = cosine_distance
    elif loss == 'norm_mse':
        loss_fn = normalized_MSE
    elif loss == 'norm_huber':
        loss_fn = normalized_huber
    elif loss == 'mse_norm_mse':
        loss_fn = MSE_normalized_MSE
    else:
        raise NotImplementedError
    return loss_fn

def get_affine_proj(z, A, b, include_negative=False):
    # Ax <= b
    residual = A@z.T - b
    inv = torch.linalg.inv(A @ A.T)
    delta =  A.T @ inv @ residual
    if include_negative:
        mask = (residual > 0).to(torch.float32)
    else:
        mask =1.0
    xT = z.T - delta * mask
    return xT.T

def get_clamp_proj(z, xmin=-0.1, xmax=1):
    xt = torch.clamp(z, min=xmin, max=xmax)
    return xt

def affine_proj_GD(z, A, b, lr=1.0):
    prev = torch.is_grad_enabled()
    torch.set_grad_enabled(True)
    # Ax <= b
    X = z.clone().detach().requires_grad_(True)
    loss = ((A@X.T - b).T**2).mean()
    grad = torch.autograd.grad(loss, X)[0]
    new_X = X - lr*grad
    x = new_X.detach()
    torch.set_grad_enabled(prev)
    return x


class StackedRandomGenerator:
    def __init__(self, device, seeds):
        super().__init__()
        self.generators = [torch.Generator(device).manual_seed(int(seed) % (1 << 32)) for seed in seeds]

    def randn(self, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randn(size[1:], generator=gen, **kwargs) for gen in self.generators])

    def randn_like(self, input):
        return self.randn(input.shape, dtype=input.dtype, layout=input.layout, device=input.device)

    def randint(self, *args, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randint(*args, size=size[1:], generator=gen, **kwargs) for gen in self.generators])

class ExperimentDiffusion:
    def __init__(self, model, scheduler, batch_size, data_shape, save_folder, seed=0, device='cpu',dist_train=0,time_shift=0):
        self.model = model
        self.scheduler = scheduler
        self.device = device
        self.seed = seed
        self.batch_size = batch_size
        self.data_shape = data_shape
        self.shape = (batch_size,) + data_shape
        self.dim = np.prod(data_shape)
        self.dim_coord = len(data_shape)
        self.save_folder = save_folder
        self.dist_train = dist_train
        self.time_shift = time_shift
        self.clip_denoise_fn = lambda x : x
        self.gen = self.new_gen()

    def set_model(self, model=None, sigma_model=None, learn_epsvar=True):
        if model is not None:
            self.model = model
            self.learn_epsvar = learn_epsvar
        else:
            self.learn_epsvar = False
        if sigma_model is not None:
            self.sigma_model = sigma_model
            if self.dist_train:
                from src import dist_util
                dist_util.sync_params(self.sigma_model.parameters())

    def set_optimizers(self, lr, loss_sigma=None, weight_decay=0.0,
                       use_fp16=False, fp16_scale_growth=1e-3, ema_rate=0.999,
                       resume_ema_model=None, resume_optim=None, reduction='mean'):

        self.use_fp16 = use_fp16
        self.ema_rate = ema_rate
        self.sigma_loss_fn = get_loss(loss_sigma,reduction=reduction)
        self.mp_sigma_trainer = MixedPrecisionTrainer(
            model=self.sigma_model,
            use_fp16=self.use_fp16,
            fp16_scale_growth=fp16_scale_growth,
        )

        self.sigma_ema_params = copy.deepcopy(self.mp_sigma_trainer.master_params)
        if resume_ema_model and os.path.exists(resume_ema_model):
            state_dict = dist_util.load_state_dict(
                resume_ema_model, map_location='cpu'
            )
            master_params = [state_dict[name].to(self.device) for name, _ in self.sigma_model.named_parameters()]
            self.sigma_ema_params = master_params
            #state_dict = torch.load(resume_ema_model, map_location='cpu')
            #self.sigma_ema_params = self.mp_sigma_trainer.state_dict_to_master_params(state_dict)
            #master_params = [state_dict[name].to(self.device) for name, _ in self.sigma_model.named_parameters()]
            #self.sigma_ema_params = master_params
            print('resume sigma ema model from', resume_ema_model)
            if self.dist_train:
                dist_util.sync_params(self.sigma_ema_params)

        self.optim = torch.optim.AdamW(self.mp_sigma_trainer.master_params, lr=lr, weight_decay=weight_decay)
        if resume_optim and os.path.exists(resume_optim):
            ckpt = dist_util.load_state_dict(
                resume_optim, map_location='cpu'
            )
            self.optim.load_state_dict(ckpt)
            def optimizer_to(optim, device):
                for param in optim.state.values():
                    # Not sure there are any global tensors in the state dict
                    if isinstance(param, torch.Tensor):
                        param.data = param.data.to(device)
                        if param._grad is not None:
                            param._grad.data = param._grad.data.to(device)
                    elif isinstance(param, dict):
                        for subparam in param.values():
                            if isinstance(subparam, torch.Tensor):
                                subparam.data = subparam.data.to(device)
                                if subparam._grad is not None:
                                    subparam._grad.data = subparam._grad.data.to(device)

            optimizer_to(self.optim, self.device)
            #ckpt = torch.load(resume_optim, map_location='cuda:0')
            #self.optim.load_state_dict(ckpt)
            print('resume optim from', resume_optim)


    def set_dataset(self, dataloader, fid_target=None, norm_min=None, norm_max=None):
        self.dataloader = dataloader
        self.fid_helper(fid_target)
        self.set_norm_maxmin(norm_min,norm_max)


    def set_norm_maxmin(self, norm_min=None, norm_max=None):
        if norm_min is not None:
            self.norm_min = norm_min /math.sqrt(self.dim)
        else:
            self.norm_min = 0.0
        if norm_max is not None:
            self.norm_max = norm_max / math.sqrt(self.dim)
        else:
            self.norm_max = 1.0

    def set_clip_fn(self, clip_fn='none'):
        if clip_fn=='clamp':
            self.clip_denoise_fn = lambda x : x.clamp(-1, 1)
        elif clip_fn=='dynamic':
            def _threshold_sample(sample,dynamic_thresholding_ratio=0.995,sample_max_value=100):
                batch_size, channels, height, width = sample.shape
                # Flatten sample for doing quantile calculation along each image
                sample = sample.reshape(batch_size, channels * height * width)
                abs_sample = sample.abs()  # "a certain percentile absolute pixel value"
                s = torch.quantile(abs_sample, dynamic_thresholding_ratio, dim=1)
                s = torch.clamp(
                    s, min=1, max=sample_max_value
                )  # When clamped to min=1, equivalent to standard clipping to [-1, 1]
                s = s.unsqueeze(1)  # (batch_size, 1) because clamp will broadcast along dim=0
                sample = torch.clamp(sample, -s, s) / s  # "we threshold xt0 to the range [-s, s] and then divide by s"
                sample = sample.reshape(batch_size, channels, height, width)
                return sample

            self.clip_denoise_fn = partial(_threshold_sample, dynamic_thresholding_ratio=0.99,sample_max_value=100)

        else:
            self.clip_denoise_fn = lambda x : x


    def fid_helper(self,fid_target, dims=2048):
        dims=2048
        batch_size=128
        device = self.device
        num_workers=1
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
        model = InceptionV3([block_idx]).to(device)
        with np.load(fid_target) as f:
            m1, s1 = f['mu'][:], f['sigma'][:]

        def calc_fid(img_path):
            m2, s2 = compute_statistics_of_path(img_path, model, batch_size,
                                                dims, device, num_workers)
            fid_value = calculate_frechet_distance(m1, s1, m2, s2)
            return fid_value

        self.fid_fn = calc_fid

    def set_perturb_coefficient(self, eta1_min=0, eta1_scale=0, eta2_min=0, eta2_scale=0):
        noise_size = (1,) * self.dim_coord
        self.eta1_fn = lambda bsz: eta1_min + torch.rand(size=(bsz,) + noise_size).to(self.device) * eta1_scale
        self.eta2_fn = lambda bsz: eta2_min + torch.rand(size=(bsz,) + noise_size).to(self.device) * eta2_scale

    def update_ema(self):
        rate = self.ema_rate
        for targ, src in zip(self.sigma_ema_params, self.mp_sigma_trainer.master_params):
            targ.detach().mul_(rate).add_(src, alpha=1 - rate)

    def save_checkpoint(self, epoch):
        if not self.dist_train or dist.get_rank() == 0:
            params = self.mp_sigma_trainer.master_params
            state_dict =  self.mp_sigma_trainer.master_params_to_state_dict(params)
            filepath =  os.path.join(self.save_folder, f"sigma_ckpt_{epoch}.pt")
            torch.save(state_dict, filepath)
            params = self.sigma_ema_params
            state_dict = self.mp_sigma_trainer.master_params_to_state_dict(params)
            filepath = os.path.join(self.save_folder, f"ema_sigma_ckpt_{epoch}.pt")
            torch.save(state_dict, filepath)

            torch.save(self.optim.state_dict(), os.path.join(self.save_folder, f"optim_state.pt"))

        if self.dist_train:
            dist.barrier()


    def batched_t(self, t, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        return torch.ones(batch_size, dtype=int, device=self.device) * t

    def new_gen(self, seed=None):
        return torch.manual_seed(self.seed if seed is None else seed)

    def get_noise(self, shape=None, gen=None, norm_noise=False):
        if shape is None:
            shape = self.shape
        if gen is None:
            gen = self.gen
        noise = torch.randn(shape, generator=gen).to(self.device)
        if norm_noise:
            noise = normalize(noise, self.dim)
        return noise

    def convert_coordinate(self, xt, sigma=None, t=None):
        """
        x_t to z_t
        """
        if sigma is not None:
            alpha_bar = 1 / (sigma ** 2 + 1)
        else:
            alpha_bar = self.scheduler.get_alpha_bar(t)
        zt = xt * alpha_bar.sqrt()
        return zt

    def inv_convert_coordinate(self, zt, sigma=None, t=None):
        """
        z_t to x_t
        """
        if sigma is not None:
            alpha_bar = 1 / (sigma ** 2 + 1)
        else:
            alpha_bar = self.scheduler.get_alpha_bar(t)
        xt = zt / alpha_bar.sqrt()
        return xt

    def pred_xt(self, xt, t, sigma=None, batch_t=True):
        zt = self.convert_coordinate(xt, sigma, t)
        if batch_t:
            t = self.batched_t(t, len(xt))
        while len(t.size()) > 1:
            t = torch.squeeze(t,dim=-1)
        t = t.to(xt.device)
        return self.model(zt, t)

    def encode_xt(self, xt, t, sigma=None, batch_t=True):
        zt = self.convert_coordinate(xt, sigma, t)
        if batch_t:
            t = self.batched_t(t, len(xt))
        while len(t.size()) > 1:
            t = torch.squeeze(t,dim=-1)
        t = t.to(xt.device)
        return self.model.encode(zt, t)

    def forward_and_encode_xt(self, xt, t, sigma=None, batch_t=True):
        zt = self.convert_coordinate(xt, sigma, t)
        if batch_t:
            t = self.batched_t(t, len(xt))
        while len(t.size()) > 1:
            t = torch.squeeze(t,dim=-1)
        t = t.to(xt.device)
        return self.model.forward_and_encode(zt, t)

    def get_noise_xt(self, shape=None, gen=None, norm_noise=False, t=None, sigma=None, ):
        zt = self.get_noise(shape=shape, gen=gen, norm_noise=norm_noise)
        xt = self.inv_convert_coordinate(zt, sigma, t)
        return xt, zt


    @torch.no_grad()
    def denoise_loop(self, shape, gen=None, norm_init_noise=False, style='base', constrain_fn=None,
                           norm_eps=False, refine_prior_sigma=False, xT=None, return_log=True, chunk_size=2,
                     sigma_pred_threshold=1000,new_eta=None,constrain_loss=None,return_best=True,
                     free_const_steps=-1):  # 'base' 'pred'
        self.scheduler.reset_state()
        if xT is None:
            xt, zt = self.get_noise_xt(shape=shape, gen=gen, norm_noise=norm_init_noise, sigma=self.scheduler.sampling_sigmas[0])
        else:
            xt = xT
            zt = self.convert_coordinate(xt, sigma=self.scheduler.sampling_sigmas[0])
        eps_list, z_list=[], []
        x0_prec_list,x0_postc_list = [], []
        const_loss_list=[]
        if return_log:
            z_list = [zt.cpu()]
        sample_time_step = self.scheduler.num_inference_steps
        best_val, best_x0 = 10000, xt
        for ind, (t, t_prev) in enumerate(pairwise(self.scheduler.timesteps)):
            if ind == sample_time_step - 1 and new_eta is not None:
                self.scheduler.eta = new_eta
            sigma_t = self.scheduler.sampling_sigmas[ind]
            sigma_prev = self.scheduler.sampling_sigmas[ind + 1]
            #print(ind, t.mean().item(), sigma_prev.mean().item())
            cur_style=style
            cur_refine_prior = refine_prior_sigma
            if t>sigma_pred_threshold:
                cur_style='base'
                cur_refine_prior = False
            eps, eps_logvar, sigma_t, sigma_prev = self.get_denoise_vector(xt, t, sigma_t, sigma_prev, cur_style, norm_eps, refine_prior_sigma=cur_refine_prior, chunk_size=chunk_size)
            # if style=='pred_partial' or style=='base':
            #     sigma_prev =  self.scheduler.sampling_sigmas[ind + 1]
            x0_hat = self.scheduler.pred_xstart(xt, eps, sigma_t)
            x0_hat = self.clip_denoise_fn(x0_hat)
            if constrain_fn is not None:
                if free_const_steps<=0 or ind<=free_const_steps:
                    x0 = constrain_fn(x0_hat)
                else:
                    x0 = x0_hat
            else:
                x0 = x0_hat
            # x0, eps, sigma_t, sigma_prev,xt=None, log_variance=None
            xt = self.scheduler.pred_xprev(x0=x0, eps=eps, sigma_t=sigma_t, sigma_prev=sigma_prev,xt=xt, log_variance=eps_logvar)
            if constrain_loss is not None:
                const, _ = constrain_loss(x0.clamp(-1, 1))
                const_val = torch.mean(const)
                if const_val < best_val:
                    best_x0 = x0
                    best_val = const_val
                #print('constraint loss: iter', ind, ', loss:', const_val.item())
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
            if torch.isnan(xt).any():
                break
        #xt = xt.cpu()
        if return_best:
            xt = best_x0.cpu()
        else:
            xt = x0.cpu()
        return_list =  [z_list,eps_list, x0_prec_list, x0_postc_list,const_loss_list]
        return xt, return_list

    @torch.no_grad()
    def get_denoise_vector(self, xt, t, sigma_t,sigma_prev,  style='base',norm_eps=False, refine_prior_sigma=False, chunk_size = 2):
        if refine_prior_sigma:
            norm_x = vector_norm(xt, keepdim=True)/math.sqrt(self.dim)
            min_dist = torch.clamp(norm_x - self.norm_max, min=0)
            max_dist =  norm_x + self.norm_min
            if len(sigma_t.unsqueeze(-1)) ==1:
                raw_sigma = torch.ones_like(norm_x)*sigma_t
            else:
                raw_sigma = sigma_t
            sigma_t= torch.clamp(raw_sigma, min=min_dist, max=max_dist)
            t = self.scheduler.get_t_from_sigma(sigma_t)
            if t.min()>0:
                t = t -self.time_shift
            if len(sigma_prev.unsqueeze(-1)) ==1:
                sigma_prev = torch.ones_like(norm_x)*sigma_prev
            else:
                sigma_prev = sigma_prev
            #sigma_prev = torch.clamp(raw_sigma_prev,  max=sigma_t)

        t = torch.clamp(t, min=0.0, max=1000.0)
        if 'pred' in style:
            #is_short =( len(t.unsqueeze(-1)) == 1
            is_short = len(t.size()) <=1
            feat = self.encode_xt(xt, t, sigma=sigma_t, batch_t=True if is_short else False)
            sigma_residual = self.sigma_model(feat)
            dist_hat = sigma_t * (1 + sigma_residual)
            dist_prev_hat = dist_hat * (sigma_prev / sigma_t)
            t = self.scheduler.get_t_from_sigma(dist_hat)
            t = torch.clamp(t,min=0.0,max=1000.0)
            sigma_t = dist_hat
            if style=='pred':
                sigma_prev =  dist_prev_hat
        else:
            if len(t.unsqueeze(-1)) == 1:
                t = self.batched_t(t, len(xt))

        microbatch = max(len(xt)//chunk_size,1)
        eps_out=None
        is_short = len(sigma_t.unsqueeze(-1)) ==1
        for i_micro in range(0, len(xt), microbatch):
            micro_x = xt[i_micro: i_micro + microbatch]
            micro_t = t[i_micro: i_micro + microbatch]
            if not is_short:
                micro_sigma = sigma_t[i_micro: i_micro + microbatch]
            else:
                micro_sigma = sigma_t
            micro_eps = self.pred_xt(micro_x, micro_t, sigma=micro_sigma, batch_t=False)
            if eps_out is None:
                eps_out = micro_eps
            else:
                eps_out = torch.cat([eps_out, micro_eps])
        if self.learn_epsvar:
            C = eps_out.size(1)//2
            eps_mean, eps_logvar = torch.split(eps_out,C,dim=1)
        else:
            eps_mean = eps_out
            eps_logvar = None
        if norm_eps:
            eps_mean = normalize(eps_mean, self.dim)
        eps_logvar = self.scheduler.get_eps_logvar(sigma_t=sigma_t, sigma_prev=sigma_prev, learned_logvar=eps_logvar)
        return eps_mean, eps_logvar, sigma_t,sigma_prev

    @torch.no_grad()
    def projection_loop(self, shape, gen=None, norm_init_noise=False, style='base', constrain_fn=None,
                        norm_eps=False,  refine_prior_sigma=False, xT=None, return_log=True, chunk_size=2,
                        sigma_estimate_rate=[1,0,0]):
        max_steps = len(self.scheduler.timesteps)-1
        self.scheduler.reset_state()
        if xT is None:
            xt, zt = self.get_noise_xt(shape=shape, gen=gen, norm_noise=norm_init_noise, sigma=self.scheduler.sampling_sigmas[0])
        else:
            xt = xT
            zt = self.convert_coordinate(xt, sigma=self.scheduler.sampling_sigmas[0])
        x0 = xt
        eps_list, z_list, sigma_list=[], [],[]
        x0_prec_list,x0_postc_list = [], []
        sigma_t = self.scheduler.sampling_sigmas[0]
        t =  self.scheduler.timesteps[0]
        costheta=0.99
        last_norm= vector_norm(xt) / math.sqrt(self.dim)
        if return_log:
            z_list = [zt.cpu()]
            sigma_list = [sigma_t.cpu()]
        for ind in range(max_steps):
            sigma_orig = sigma_t
            sigma_decay = self.scheduler.sigma_decay[ind]
            sigma_prev = sigma_t * sigma_decay
            eps, eps_logvar, sigma_t, sigma_prev = self.get_denoise_vector(xt, t, sigma_t, sigma_prev, style, norm_eps, refine_prior_sigma=refine_prior_sigma, chunk_size=chunk_size)
            # if style=='pred_partial' or style=='base':
            #     sigma_prev = sigma_orig * self.scheduler.sigma_decay[ind]
            x0_hat = self.scheduler.pred_xstart(xt, eps, sigma_t)
            if constrain_fn is not None:
                x0 = constrain_fn(x0_hat)
            else:
                x0 = x0_hat
            #xt = self.scheduler.pred_xprev(x0, eps, sigma_t, sigma_prev, eps_logvar)
            xt = self.scheduler.pred_xprev_with_sigma_decay(x0, eps, sigma_t, signal_decay= self.scheduler.signal_decay[ind] , noise_decay=self.scheduler.noise_decay[ind])
            cur_norm = vector_norm(xt) / math.sqrt(self.dim)
            cur_dist = torch.sqrt(cur_norm ** 2 + self.norm_max ** 2 - 2 * cur_norm *  self.norm_max * costheta + 1e-8)
            norm_ratio = torch.clamp(cur_norm / last_norm, max=sigma_decay)
            sigma_t1 = sigma_prev
            sigma_t2 = sigma_t * norm_ratio
            sigma_t3 = cur_dist
            sigma_t = sigma_estimate_rate[0]*sigma_t1+ sigma_estimate_rate[1]*sigma_t2+ sigma_estimate_rate[2]*sigma_t3
            t = self.scheduler.get_t_from_sigma(sigma_t)
            last_norm = cur_norm
            if return_log:
                zt = self.convert_coordinate(xt, sigma=sigma_prev)
                z_list.append(zt.cpu())
                eps_list.append(eps.cpu())
                x0_prec_list.append(x0_hat.cpu())
                x0_postc_list.append(x0.cpu())
                sigma_list.append(sigma_t.cpu())
            if torch.isnan(xt).any():
                break
        #xt = xt.cpu()
        xt = x0.cpu()
        return_list = [z_list, eps_list, x0_prec_list, x0_postc_list, sigma_list]
        return xt, return_list

    @torch.no_grad()
    def generate_projection_one_batch(self, shape, gen=None, norm_init_noise=False, style='base',
                           norm_eps=False, sigma_decay=0.9, constrain_fn = None, max_steps = None):  # 'base' 'pred' 'dist'
        if max_steps is None:
            max_steps = len(self.scheduler.timesteps) - 1

        self.scheduler.reset_state()
        sigma_t = self.scheduler.sampling_sigmas[0]
        t =  self.scheduler.timesteps[0]
        xt, zt = self.get_noise_xt(shape=shape, gen=gen, norm_noise=norm_init_noise,
                                   sigma=sigma_t)
        x_norm = vector_norm(xt) / np.sqrt(self.dim)
        z_list = [zt.cpu()]
        x_prev_list=[xt.cpu()]
        eps_list=[]
        for ind in range(max_steps):
            eps, dist_t,dist_prev = self.get_denoise_vector(xt, t, sigma_t,sigma_prev=sigma_t*0,  style=style,norm_eps=norm_eps)
            xt = self.scheduler.step_with_sigma(eps, dist_t, dist_prev, xt)
            x_prev_list.append(xt.cpu())
            if constrain_fn is not None:
                xt = constrain_fn(xt)
            prev_norm = vector_norm(xt) / np.sqrt(self.dim)
            sigma_t = sigma_t * torch.clamp(prev_norm/x_norm, max=sigma_decay)
            t = self.scheduler.get_t_from_sigma(sigma_t).squeeze()
            x_norm = prev_norm
            z_list.append(xt.cpu())
            eps_list.append(eps.cpu())
            if torch.isnan(xt).any():
                break
        xt = xt.cpu()
        return_list = [z_list,eps_list,x_prev_list]
        return xt, return_list

class ImageExperiment(ExperimentDiffusion):
    def __init__(self, model, scheduler, batch_size=64, data_shape=(3, 32, 32), seed=0, device='cuda:0',
                 save_folder='./',dist_train=False,time_shift=0):
        ExperimentDiffusion.__init__(self, model=model, scheduler=scheduler, batch_size=batch_size,
                                     data_shape=data_shape, save_folder=save_folder, seed=seed,
                                     device=device, dist_train=dist_train,time_shift=time_shift)


    def plot_samples(self, x_hat_list, sample_path, n_per_row = 5):
        fig = plt.figure(figsize=(20, 20))
        ims = []
        c,h,w = self.data_shape
        for t in range(len(x_hat_list)):
            big_img = np.zeros(
                shape=(h * n_per_row, w * n_per_row, c))
            for i_row in range(n_per_row):
                for i_col in range(n_per_row):
                    img = x_hat_list[t][i_row * n_per_row + i_col].permute(1, 2,0).reshape(h, w,c)
                    big_img[i_row * h:(i_row + 1) * h, i_col * w:(i_col + 1) * w] = img
            im = plt.imshow(big_img, cmap="gray" if c == 1 else None, animated=True)
            ims.append([im])
        animate = animation.ArtistAnimation(fig, ims, interval=500, blit=True, repeat_delay=2000)
        animate.save(sample_path)
        plt.close()

    @torch.no_grad()
    def evaluate(self, n_samples, images_dir, gen=None, norm_init_noise=False, style='base', norm_eps=False, microbatch=-1,
                 return_log=True,chunk_size=2):
        if os.path.exists(images_dir):
            shutil.rmtree(images_dir)
        os.makedirs(images_dir, exist_ok=True)
        log_dict = {}
        batch_size = microbatch if microbatch>0 else self.batch_size
        n_batches = max(n_samples // batch_size, 1)
        shape = (batch_size,) + self.data_shape
        rank = 0 if not self.dist_train else dist.get_rank()
        for i in range(n_batches):
            sample, return_list = self.denoise_loop(shape=shape, gen=gen, norm_init_noise=norm_init_noise,style=style, norm_eps=norm_eps,return_log=return_log,chunk_size=chunk_size)
            sample = sample.add(1).div(2).clamp(0, 1)
            for j, img in enumerate(sample):
                path = os.path.join(images_dir, f'{rank:02}-{i:05}-{j:03}.png')
                save_image(img, path)
        z_list = return_list[0]
        eps_list = return_list[1]
        if self.dist_train:
            dist.barrier()
        if (not self.dist_train or dist.get_rank() == 0):
            fid = self.fid_fn(images_dir)
            #torch.cuda.empty_cache()
        else:
            fid=0.0
        #fid = self.fid_fn(images_dir)
        log_dict['fid'] = fid
        return log_dict, z_list, eps_list

    @torch.no_grad()
    def evaluate_projection(self, n_samples, images_dir, gen=None, norm_init_noise=False, style='base', norm_eps=False,
                            sigma_decay=0.9,constrain_fn=None):
        if os.path.exists(images_dir):
            shutil.rmtree(images_dir)
        os.makedirs(images_dir, exist_ok=True)
        log_dict = {}
        batch_size = self.batch_size
        n_batches = max(n_samples // batch_size, 1)
        shape = (batch_size,) + self.data_shape
        for i in range(n_batches):
            sample, return_list = self.generate_projection_one_batch(shape=shape, gen=gen, norm_init_noise=norm_init_noise,
                                                          style=style, norm_eps=norm_eps,sigma_decay=sigma_decay,constrain_fn=constrain_fn)
            sample = sample.add(1).div(2).clamp(0, 1)
            for j, img in enumerate(sample):
                path = os.path.join(images_dir, f'{i:05}-{j:03}.png')
                save_image(img, path)
        z_list = return_list[0]
        eps_list =  return_list[1]
        #fid = calculate_fid_given_paths((images_dir, self.fid_target), batch_size=64, device=self.device, dims=2048)
        fid = self.fid_fn(images_dir)
        log_dict['fid'] = fid
        return log_dict, z_list, eps_list

    def train(self, epoch=101,iter_per_epoch=1000, eval_per_epoch=100,save_per_epoch=100, max_T=-1, microbatch=16):
        dataloader = self.dataloader
        num_train_timesteps = self.scheduler.num_train_timesteps
        if max_T>0 :
            num_train_timesteps = max_T
        figs_dir = os.path.join(self.save_folder, 'figs')
        os.makedirs(figs_dir, exist_ok=True)
        images_dir = os.path.join(self.save_folder, 'images')
        os.makedirs(images_dir, exist_ok=True)
        losses = []
        logs = {'epoch': [], 'loss': [], 'fid': [],'fid_sigma_ped':[]}
        self.sigma_model.train()
        if self.dist_train:
            self.ddp_sigma_model = DDP(
                self.sigma_model,
                device_ids=[self.device],
                output_device=self.device,
                broadcast_buffers=False,
                bucket_cap_mb=128,
                find_unused_parameters=False,
            )

        for ep in range(epoch):
            loss_list = []
            for i in range(iter_per_epoch):
                img, y = next(dataloader)
                batch_size = img.shape[0]
                batch_x = img.to(self.device)
                # add noise
                t = torch.randint(0, num_train_timesteps, size=(batch_size // 2 + 1,))
                t = torch.cat([t, num_train_timesteps - t - 1], dim=0)[:batch_size].long().to(self.device)
                noise = self.get_noise(shape=(batch_size,) + self.data_shape)
                eta1 =  self.eta1_fn(batch_size)
                eta2= self.eta2_fn(batch_size)
                noise_delta = eta1 * noise  + eta1* eta2 * torch.randn((batch_size,) + self.data_shape).to(self.device)
                new_noise = noise + noise_delta
                dist_real = vector_norm(new_noise) / np.sqrt(self.dim)
                noisy_x, _ = self.scheduler.diffusion(batch_x, t, new_noise)
                # pred and Compute the loss.
                self.mp_sigma_trainer.zero_grad()
                feat=None
                with torch.no_grad():
                    for i_micro in range(0, batch_size, microbatch):
                        micro_x = noisy_x[i_micro: i_micro + microbatch]
                        micro_t =  t[i_micro: i_micro + microbatch]
                        micro_feat = self.model.encode(micro_x, micro_t)
                        if feat is None:
                            feat = micro_feat
                        else:
                            feat = torch.cat([feat,micro_feat])
                if self.dist_train:
                    with self.ddp_sigma_model.no_sync():
                        dist_res = self.ddp_sigma_model(feat)
                        dist_hat = dist_res + 1
                        loss = self.sigma_loss_fn(dist_real, dist_hat)
                else:
                    dist_res = self.sigma_model(feat)
                    dist_hat = dist_res + 1
                    loss = self.sigma_loss_fn(dist_real, dist_hat)
                self.mp_sigma_trainer.backward(loss)
                took_step = self.mp_sigma_trainer.optimize(self.optim)
                if took_step:
                    self.update_ema()
                loss_list.append([loss.item()])
                if i%100==0:
                    print(f"[rank={dist.get_rank()}] epoch={ep}, iteration={i}, loss={np.mean(loss_list[-10:])}")
            losses += loss_list
            loss_list = np.array(loss_list)
            print('----------------------------')
            print(f"[rank={dist.get_rank()}] epoch={ep}, loss={np.mean(loss_list[:, 0])}")
            # Print loss
            if (ep % eval_per_epoch == 0) or (ep == epoch - 1):
                sample_size =self.batch_size*8 if ep != epoch - 1 else 2560 #self.batch_size*16 if ep != epoch - 1 else 5000
                self.sigma_model.eval()
                logs['epoch'].append(ep)
                logs['loss'].append(np.mean(loss_list[:, 0]))
                log_dict, _ ,_ = self.evaluate(sample_size, os.path.join(images_dir, 'base_samples'),gen=self.new_gen(),
                                                 norm_init_noise=False, style='base', norm_eps=False, microbatch=microbatch,
                                               return_log=False)
                logs['fid'].append(log_dict['fid'])
                #self.plot_samples(z_list, sample_path=os.path.join(figs_dir, f"base_generation_epoch{ep}.gif"))
                log_dict, _, _ = self.evaluate(sample_size, os.path.join(images_dir, 'pred_samples'),gen=self.new_gen(),
                                                 norm_init_noise=False, style='pred', norm_eps=True, microbatch=microbatch,
                                               return_log=False)
                logs['fid_sigma_ped'].append(log_dict['fid'])
                #self.plot_samples(z_list, sample_path=os.path.join(figs_dir, f"pred_generation_epoch{ep}.gif"))

                print(f" ---[rank={dist.get_rank()}] Train on  Epoch={ep}, Loss={logs['loss'][-1]}, fid={logs['fid'][-1]}" + f", fid pred sigma={logs['fid_sigma_ped'][-1]}")
                self.sigma_model.train()
            if (ep % save_per_epoch == 0) or (ep == epoch - 1):
                self.save_checkpoint(ep)
                print(f'[rank={dist.get_rank()}] save model on epoch', ep)

        if (not self.dist_train or dist.get_rank() == 0):
            logs = pd.DataFrame(logs)
            logs.to_csv(os.path.join(self.save_folder, f'train_logs.tsv'), sep='\t',
                        index=False)
            losses = np.array(losses)
            np.savetxt(os.path.join(self.save_folder, f'train_losses.txt'), losses)

            fig = plt.figure(figsize=(8, 6))
            plt.plot(np.arange(len(losses)), EMA_smooth(losses[:, 0], alpha=0.9), label='loss', color='red', )
            plt.xlabel('iteration')
            plt.ylabel('loss')
            plt.tight_layout()
            plt.legend()
            plt.yscale('log')
            plt.savefig(os.path.join(self.save_folder, f'train_loss.png'))
            plt.close()

            fig = plt.figure(figsize=(8, 6))
            plt.plot(logs['epoch'].values, EMA_smooth(logs['fid'].values, alpha=0.3), label='baseline sampling',
                     color='red', )
            plt.plot(logs['epoch'].values, EMA_smooth(logs['fid_sigma_ped'].values, alpha=0.3),
                         label='sigma-pred sampling')
            plt.xlabel('epoch')
            plt.ylabel('fid score')
            plt.legend()
            plt.tight_layout()
            plt.yscale('log')
            plt.savefig(os.path.join(self.save_folder, f'fid_score.png'))
            plt.close()


class EDMImageExperiment(ImageExperiment):
    def __init__(self, model, scheduler, batch_size=64, data_shape=(3, 32, 32), seed=0, device='cuda:0',
                 save_folder='./',dist_train=False,time_shift=0,
                 sigma_min=0.002, sigma_max=80, rho=7, S_churn=0, S_min=0, S_max=float('inf'), S_noise=1,
                 sigma_data=0.5,  P_mean=-1.2, P_std=1.2, num_timesteps=18):
        ImageExperiment.__init__(self, model=model, scheduler=scheduler, batch_size=batch_size,
                                     data_shape=data_shape, save_folder=save_folder, seed=seed,
                                     device=device, dist_train=dist_train,time_shift=time_shift)
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho
        self.S_churn = S_churn
        self.S_min = S_min
        self.S_max = S_max
        self.S_noise = S_noise
        self.sigma_data = sigma_data
        self.P_mean = P_mean
        self.P_std = P_std
        self.num_timesteps = num_timesteps


    def encode_edm(self, xt, sigma):
        xt = xt.to(torch.float32)
        sigma = sigma.to(torch.float32).reshape(-1, 1, 1, 1)
        class_labels = None
        dtype = torch.float32
        c_in = 1 / (self.sigma_data ** 2 + sigma ** 2).sqrt()
        c_noise = sigma.log() / 4

        feat = self.model.encode((c_in * xt).to(dtype), c_noise.flatten(), class_labels=class_labels)
        return feat

    def pred_edm(self, xt, sigma):
        xt = xt.to(torch.float32)
        sigma = sigma.to(torch.float32).reshape(-1, 1, 1, 1)
        class_labels = None
        dtype = torch.float32

        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()
        c_in = 1 / (self.sigma_data ** 2 + sigma ** 2).sqrt()
        c_noise = sigma.log() / 4

        F_x = self.model((c_in * xt).to(dtype), c_noise.flatten(), class_labels=class_labels)
        assert F_x.dtype == dtype
        D_x = c_skip * xt + c_out * F_x.to(torch.float32)
        return D_x

    @torch.no_grad()
    def get_denoise_vector(self, xt,  sigma_t, sigma_prev, style='base',norm_eps=False, refine_prior_sigma=False):
        sigma_t_orig = sigma_t
        if refine_prior_sigma:
            norm_x = vector_norm(xt, keepdim=True)/math.sqrt(self.dim)
            min_dist = torch.clamp(norm_x - self.norm_max, min=0)
            max_dist =  norm_x + self.norm_min
            if len(sigma_t.unsqueeze(-1)) ==1:
                raw_sigma = torch.ones_like(norm_x)*sigma_t
            else:
                raw_sigma = sigma_t
            sigma_t= torch.clamp(raw_sigma, min=min_dist, max=max_dist)
            if len(sigma_prev.unsqueeze(-1)) ==1:
                sigma_prev = torch.ones_like(norm_x)*sigma_prev
            else:
                sigma_prev = sigma_prev

        if 'pred' in style:
            feat = self.encode_edm(xt, sigma=sigma_t)
            sigma_residual = self.sigma_model(feat)
            dist_hat = sigma_t * (1 + sigma_residual)
            dist_prev_hat = dist_hat * (sigma_prev / sigma_t)
            sigma_t = dist_hat
            if style=='pred':
                sigma_prev = dist_prev_hat
        if len(sigma_t_orig.unsqueeze(-1)) ==1:
            sigma_t_orig = sigma_t_orig.reshape(-1, 1, 1, 1)
        if len(sigma_t.unsqueeze(-1)) ==1:
            sigma_t = sigma_t.reshape(-1, 1, 1, 1)
        if len(sigma_prev.unsqueeze(-1)) ==1:
            sigma_prev = sigma_prev.reshape(-1, 1, 1, 1)
        if style=='pred_sigma':
            denoised = self.pred_edm(xt, sigma_t_orig).to(torch.float64)
            eps = (xt - denoised) / sigma_t_orig
        else:
            denoised = self.pred_edm(xt, sigma_t).to(torch.float64)
            eps = (xt - denoised) / sigma_t
        if norm_eps:
            eps = normalize(eps, self.dim)
        return eps, denoised, sigma_t,sigma_prev


    @torch.no_grad()
    def edm_sampler(self, shape, gen=None, style='base,base', norm_eps='000',refine_prior_sigma=False,num_steps=None, sigma_scheduler='EDM',
                    eps_ratio=0.5,eps_scale=1.0, use_second_order=True):

        norm_eps, norm_eps_combine = bool(int(norm_eps[0])), bool(int(norm_eps[1]))
        style_t, style_next = style.split(',')
        if num_steps is None:
            num_steps = self.num_timesteps
        # Adjust noise levels based on what's supported by the network.
        #latents = self.get_noise(shape=shape, gen=gen, norm_noise=False)
        latents = gen.randn(shape, device=self.device)
        sigma_min = self.sigma_min
        sigma_max = self.sigma_max
        # Time step discretization.
        step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
        if sigma_scheduler=='EDM':
            sigma_steps = (sigma_max ** (1 / self.rho) + step_indices / (num_steps - 1) * (
                        sigma_min ** (1 / self.rho) - sigma_max ** (1 / self.rho))) ** self.rho
        elif sigma_scheduler=='Linear':
            sigma_steps = torch.tensor(np.exp(np.linspace(np.log(sigma_max), np.log(sigma_min), num_steps)))
        else:
            raise  NotImplementedError
        sigma_steps = torch.cat([torch.as_tensor(sigma_steps), torch.zeros_like(sigma_steps[:1])])  # t_N = 0

        sim_fn = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        # Main sampling loop.
        x_next = latents.to(torch.float64) * sigma_steps[0]
        for i, (sigma_cur, sigma_next) in enumerate(zip(sigma_steps[:-1], sigma_steps[1:])):  # 0, ..., N-1
            x_cur = x_next
            sigma_next0 = sigma_next
            # Increase noise temporarily.
            gamma = min(self.S_churn / num_steps, np.sqrt(2) - 1) if self.S_min <= sigma_cur <= self.S_max else 0
            sigma_hat = torch.as_tensor(sigma_cur + gamma * sigma_cur)
            sigma_hat0 = sigma_hat
            x_hat = x_cur + (sigma_hat ** 2 - sigma_cur ** 2).sqrt() * self.S_noise * torch.randn_like(x_cur)
            # Euler step.
            eps, denoised, sigma_hat, sigma_next = self.get_denoise_vector(x_hat,  sigma_hat, sigma_next, style=style_t,norm_eps=norm_eps, refine_prior_sigma=refine_prior_sigma)
            #eps = (x_hat - denoised) / sigma_hat
            eps = eps * (sigma_hat/sigma_hat0)
            if 'pred_partial' in style_t:
                sigma_next = sigma_next0
            if style_t=='pred_partial':
                x_next = x_hat + (sigma_next - sigma_hat0) * eps
            else:
                x_next = x_hat + (sigma_next - sigma_hat) * eps
            if style_t == 'pred_partial3':
                sigma_hat = sigma_hat0
            #x_next = x_hat + (sigma_next - sigma_hat0) * eps
            #x0 = x_hat - sigma_hat * eps
            #x0 = x_hat -sigma_hat0 * pred_eps
            #x_next = x0 + sigma_next*eps

            # Apply 2nd order correction.
            if i < num_steps - 1 and  use_second_order:
                eps_next, denoised, sigma_next, _ = self.get_denoise_vector(x_next, sigma_next, sigma_next*0, style=style_next,
                                                         norm_eps=norm_eps, refine_prior_sigma=refine_prior_sigma)

                #eps_next = (x_next - denoised) / sigma_next
                eps_next = eps_next * (sigma_next / sigma_next0)
                if 'pred_partial' in style_next:
                    sigma_next = sigma_next0
                new_eps = eps_ratio * eps + (1-eps_ratio) * eps_next
                if norm_eps_combine:
                    new_eps = normalize(new_eps, self.dim)
                if eps_scale is not None:
                    new_eps = new_eps/eps_scale
                else:
                    bsz = len(new_eps)
                    cur_eps_scale = sim_fn(new_eps.reshape(bsz, -1), eps.reshape(bsz, -1))
                    cur_eps_scale = cur_eps_scale.reshape(bsz, 1,1,1)
                    new_eps = new_eps * cur_eps_scale
                x_next = x_hat + (sigma_next - sigma_hat) * new_eps
        return x_next



    @torch.no_grad()
    def evaluate_edm(self, n_samples, images_dir, gen=None,   style='base,base', norm_eps='000',refine_prior_sigma=False, microbatch=-1,sigma_scheduler='EDM',eps_ratio=0.5,eps_scale=1.0,use_second_order=True):
        log_dict = {}
        batch_size = microbatch if microbatch>0 else self.batch_size
        n_batches =math.ceil(n_samples / batch_size)
        shape = (batch_size,) + self.data_shape
        rank = 0 if not self.dist_train else dist.get_rank()
        seeds = np.arange(n_samples)
        num_batches = ((len(seeds) - 1) // (batch_size) + 1)
        all_batches = torch.as_tensor(seeds).tensor_split(num_batches)
        for i in range(n_batches):
            gen = StackedRandomGenerator(self.device, all_batches[i])
            #gen = self.new_gen(seed=i)
            skip_flag = True
            for j in range(batch_size):
                path = os.path.join(images_dir, f'{rank:02}-{i:05}-{j:03}.png')
                skip_flag = skip_flag and os.path.exists(path)
                if not skip_flag:
                    break
            if skip_flag:
                print('skip images for:', f'{rank:02}-{i:05}-({0:03}~{batch_size - 1:03}).png')
                continue


            sample = self.edm_sampler(shape=shape, gen=gen, style=style, norm_eps=norm_eps,refine_prior_sigma=refine_prior_sigma,sigma_scheduler=sigma_scheduler,eps_ratio=eps_ratio,eps_scale=eps_scale,use_second_order=use_second_order)
            sample = sample.add(1).div(2).clamp(0, 1)
            for j, img in enumerate(sample):
                path = os.path.join(images_dir, f'{rank:02}-{i:05}-{j:03}.png')
                save_image(img, path)
            print(f'done batches:{i}/{n_batches}')
        if self.dist_train:
            dist.barrier()
        if (not self.dist_train or dist.get_rank() == 0):
            fid = self.fid_fn(images_dir)
            #torch.cuda.empty_cache()
        else:
            fid=0.0
        #fid = self.fid_fn(images_dir)
        log_dict['fid'] = fid
        return log_dict

    def train_edm(self, epoch=101,iter_per_epoch=1000, eval_per_epoch=100,save_per_epoch=100, sigma_sampler ='edm', loss_weighted=False):
        dataloader = self.dataloader
        figs_dir = os.path.join(self.save_folder, 'figs')
        os.makedirs(figs_dir, exist_ok=True)
        images_dir = os.path.join(self.save_folder, 'images')
        os.makedirs(images_dir, exist_ok=True)
        losses = []
        logs = {'epoch': [], 'loss': [], 'fid': [],'fid_sigma_ped':[]}
        self.sigma_model.train()
        if self.dist_train:
            self.ddp_sigma_model = DDP(
                self.sigma_model,
                device_ids=[self.device],
                output_device=self.device,
                broadcast_buffers=False,
                bucket_cap_mb=128,
                find_unused_parameters=False,
            )

        for ep in range(epoch):
            loss_list = []
            for i in range(iter_per_epoch):
                img, y = next(dataloader)
                batch_size = img.shape[0]
                batch_x = img.to(self.device)
                # add noise
                if sigma_sampler=='edm':
                    rnd_normal = torch.randn([batch_size, 1, 1, 1], device=batch_x.device)
                    sigma = (rnd_normal * self.P_std + self.P_mean).exp()
                else:
                    sigma = self.sigma_min*0.95 + (self.sigma_max*1.05-self.sigma_min*0.95) * torch.rand([batch_size, 1, 1, 1], device=batch_x.device)
                weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
                noise = self.get_noise(shape=(batch_size,) + self.data_shape)
                eta1 =  self.eta1_fn(batch_size)
                eta2= self.eta2_fn(batch_size)
                noise_delta = eta1 * ( noise + eta2 * torch.randn((batch_size,) + self.data_shape).to(self.device) )
                new_noise = noise + noise_delta
                dist_real = vector_norm(new_noise) / np.sqrt(self.dim)
                noisy_img =  batch_x + sigma*new_noise

                self.mp_sigma_trainer.zero_grad()
                with torch.no_grad():
                    feat = self.encode_edm(noisy_img, sigma)
                # pred and Compute the loss.

                if self.dist_train:
                    with self.ddp_sigma_model.no_sync():
                        dist_res = self.ddp_sigma_model(feat)
                        dist_hat = dist_res + 1
                        loss = self.sigma_loss_fn(dist_real, dist_hat)
                else:
                    dist_res = self.sigma_model(feat)
                    dist_hat = dist_res + 1
                    loss = self.sigma_loss_fn(dist_real, dist_hat)
                if loss_weighted:
                    weight = weight/weight.sum()
                    loss = (loss*weight).sum()
                else:
                    loss = loss.mean()
                self.mp_sigma_trainer.backward(loss)
                took_step = self.mp_sigma_trainer.optimize(self.optim)
                if took_step:
                    self.update_ema()
                loss_list.append([loss.item()])
                if i%100==0:
                    print(f"[rank={dist.get_rank()}] epoch={ep}, iteration={i}, loss={np.mean(loss_list[-10:])}")
            losses += loss_list
            loss_list = np.array(loss_list)
            print('----------------------------')
            print(f"[rank={dist.get_rank()}] epoch={ep}, loss={np.mean(loss_list[:, 0])}")
            # Print loss
            if (ep % eval_per_epoch == 0) or (ep == epoch - 1):
                sample_size =self.batch_size*8 if ep != epoch - 1 else 2560 #self.batch_size*16 if ep != epoch - 1 else 5000
                self.sigma_model.eval()
                logs['epoch'].append(ep)
                logs['loss'].append(np.mean(loss_list[:, 0])) #
                log_dict= self.evaluate_edm(sample_size, os.path.join(images_dir, 'base_samples'),gen=self.new_gen(),
                                                  style='base,base', norm_eps='000', refine_prior_sigma=False)
                logs['fid'].append(log_dict['fid'])
                #self.plot_samples(z_list, sample_path=os.path.join(figs_dir, f"base_generation_epoch{ep}.gif"))
                log_dict = self.evaluate_edm(sample_size, os.path.join(images_dir, 'pred_samples'),gen=self.new_gen(),
                                                 style='pred_partial,pred_partial', norm_eps='110', refine_prior_sigma=True )
                logs['fid_sigma_ped'].append(log_dict['fid'])
                #self.plot_samples(z_list, sample_path=os.path.join(figs_dir, f"pred_generation_epoch{ep}.gif"))

                print(f" ---[rank={dist.get_rank()}] Train on  Epoch={ep}, Loss={logs['loss'][-1]}, fid={logs['fid'][-1]}" + f", fid pred sigma={logs['fid_sigma_ped'][-1]}")
                self.sigma_model.train()
            if (ep % save_per_epoch == 0) or (ep == epoch - 1):
                self.save_checkpoint(ep)
                print(f'[rank={dist.get_rank()}] save model on epoch', ep)

        if (not self.dist_train or dist.get_rank() == 0):
            logs = pd.DataFrame(logs)
            logs.to_csv(os.path.join(self.save_folder, f'train_logs.tsv'), sep='\t',
                        index=False)
            losses = np.array(losses)
            np.savetxt(os.path.join(self.save_folder, f'train_losses.txt'), losses)

            fig = plt.figure(figsize=(8, 6))
            plt.plot(np.arange(len(losses)), EMA_smooth(losses[:, 0], alpha=0.9), label='loss', color='red', )
            plt.xlabel('iteration')
            plt.ylabel('loss')
            plt.tight_layout()
            plt.legend()
            plt.yscale('log')
            plt.savefig(os.path.join(self.save_folder, f'train_loss.png'))
            plt.close()

            fig = plt.figure(figsize=(8, 6))
            plt.plot(logs['epoch'].values, EMA_smooth(logs['fid'].values, alpha=0.3), label='baseline sampling',
                     color='red', )
            plt.plot(logs['epoch'].values, EMA_smooth(logs['fid_sigma_ped'].values, alpha=0.3),
                         label='sigma-pred sampling')
            plt.xlabel('epoch')
            plt.ylabel('fid score')
            plt.legend()
            plt.tight_layout()
            plt.yscale('log')
            plt.savefig(os.path.join(self.save_folder, f'fid_score.png'))
            plt.close()
