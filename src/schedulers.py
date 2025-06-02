import math
import numpy as np
import torch
from .torchinterp1d import interp1d
#from torchinterp1d import interp1d

def normalize(x, inp_dim=None, eps=1e-12):
    if inp_dim is None:
        inp_dim = torch.numel(x[0])
    denom = torch.clamp(torch.linalg.vector_norm(x, dim=1,keepdim=True), min=eps)
    x = math.sqrt(inp_dim) * x / denom
    return x


def replace_duplicate_t(ts, max_step=999):
    new_ts = torch.zeros_like(ts)
    new_ts[-2:] = ts[-2:]
    for i in range(len(ts) - 1, 0, -1):
        if ts[i - 1] > new_ts[i]:
            new_ts[i - 1] = ts[i - 1]
        else:
            new_ts[i - 1] = new_ts[i] + 1
    new_ts2 = torch.zeros_like(new_ts)
    cur_big_t = max_step
    for i in range(len(new_ts) - 1):
        if new_ts[i]>cur_big_t:
            new_ts2[i] = cur_big_t
        else:
            new_ts2[i] = new_ts[i]
        cur_big_t = new_ts2[i] -1
    return new_ts2

def expand_shape(res,broadcast_shape):
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)

def space_timesteps(num_timesteps, section_counts):
    """
    Create a list of timesteps to use from an original diffusion process,
    given the number of timesteps we want to take from equally-sized portions
    of the original process.

    For example, if there's 300 timesteps and the section counts are [10,15,20]
    then the first 100 timesteps are strided to be 10 timesteps, the second 100
    are strided to be 15 timesteps, and the final 100 are strided to be 20.

    If the stride is a string starting with "ddim", then the fixed striding
    from the DDIM paper is used, and only one section is allowed.

    :param num_timesteps: the number of diffusion steps in the original
                          process to divide up.
    :param section_counts: either a list of numbers, or a string containing
                           comma-separated numbers, indicating the step count
                           per section. As a special case, use "ddimN" where N
                           is a number of steps to use the striding from the
                           DDIM paper.
    :return: a set of diffusion steps from the original process to use.
    """
    if isinstance(section_counts, str):
        if section_counts.startswith("ddim"):
            desired_count = int(section_counts[len("ddim") :])
            for i in range(1, num_timesteps):
                if len(range(0, num_timesteps, i)) == desired_count:
                    return set(range(0, num_timesteps, i))
            raise ValueError(
                f"cannot create exactly {num_timesteps} steps with an integer stride"
            )
        section_counts = [int(x) for x in section_counts.split(",")]
    size_per = num_timesteps // len(section_counts)
    extra = num_timesteps % len(section_counts)
    start_idx = 0
    all_steps = []
    for i, section_count in enumerate(section_counts):
        size = size_per + (1 if i < extra else 0)
        if size < section_count:
            raise ValueError(
                f"cannot divide section of {size} steps into {section_count}"
            )
        if section_count <= 1:
            frac_stride = 1
        else:
            frac_stride = (size - 1) / (section_count - 1)
        cur_idx = 0.0
        taken_steps = []
        for _ in range(section_count):
            taken_steps.append(start_idx + round(cur_idx))
            cur_idx += frac_stride
        all_steps += taken_steps
        start_idx += size
    return set(all_steps)


# Implementation Based on https://huggingface.co/docs/diffusers/using-diffusers/schedulers
class Scheduler:
    def __init__(
            self,
            num_train_timesteps: int = 1000,
            beta_start: float = 0.0001,
            beta_end: float = 0.02,
            beta_schedule: str = 'linear',
            set_alpha_to_one: bool = True,
            sampler_var: str = 'none',
            eta: float = 0.0,
    ):
        if beta_schedule == 'linear':
            self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps, dtype=torch.float32)
        elif beta_schedule == 'quadratic':
            # this schedule is very specific to the latent diffusion model.
            self.betas = (
                    torch.linspace(beta_start ** 0.5, beta_end ** 0.5, num_train_timesteps, dtype=torch.float32) ** 2
            )
        elif beta_schedule == 'cosine':
            s = 0.008
            min_beta=1e-6
            max_beta=0.999
            steps = num_train_timesteps + 1
            x = torch.linspace(0, num_train_timesteps, steps)
            alphas_cumprod = torch.cos(((x / num_train_timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            self.betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            self.betas = torch.clip(self.betas, min_beta, max_beta)
        elif beta_schedule == 'sigmoid':
            betas = torch.linspace(-6, 6, num_train_timesteps)
            self.betas = torch.sigmoid(betas) * (beta_end - beta_start) + beta_start
        else:
            raise NotImplementedError(f'{beta_schedule} does is not implemented for {self.__class__}')

        self.set_alpha_to_one =set_alpha_to_one
        self.num_train_timesteps = num_train_timesteps
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.final_alpha_cumprod = torch.tensor(1.0) # if set_alpha_to_one else self.alphas_cumprod[0]
        self.sigmas = (1 / self.alphas_cumprod - 1).sqrt()
        self.final_sigma = (1 / self.final_alpha_cumprod - 1).sqrt()
        self.train_timesteps = torch.tensor(np.arange(0, num_train_timesteps).copy().astype(np.int64))
        self.timesteps = self.train_timesteps
        self.sampling_sigmas = self.sigmas
        self.continuous_t = False
        self.sampler_var = sampler_var
        self.eta =eta

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        alphas_cumprod_prev =torch.cat([self.final_alpha_cumprod.view(1), self.alphas_cumprod[:-1]])
        self.posterior_variance = (
            self.betas * (1.0 - alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.min_var_coef = self.posterior_variance[1]
        # log calculation clipped because the posterior variance is 0 at the
        # beginning of the diffusion chain.
        self.posterior_log_variance_clipped = torch.log(
            torch.cat([self.posterior_variance[1:2], self.posterior_variance[1:]])
        )
        # self.posterior_mean_coef1 = (
        #     self.betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        # )
        # self.posterior_mean_coef2 = (
        #     (1.0 - alphas_cumprod_prev) * torch.sqrt(self.alphas)/ (1.0 - self.alphas_cumprod)
        # )

        self.logvar_min = self.posterior_log_variance_clipped #self.posterior_variance.clamp(min=1e-20).log()
        self.logvar_max = self.betas.log()

        self.reset_state()

    def to(self, device):
        self.betas = self.betas.to(device)
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        self.final_alpha_cumprod = self.final_alpha_cumprod.to(device)
        self.sigmas = self.sigmas.to(device)
        self.final_sigma = self.final_sigma.to(device)
        self.train_timesteps = self.train_timesteps.to(device)
        self.timesteps = self.timesteps.to(device)
        self.sampling_sigmas = self.sampling_sigmas.to(device)
        self.posterior_variance = self.posterior_variance.to(device)
        self.posterior_log_variance_clipped = self.posterior_log_variance_clipped.to(device)
        self.logvar_min = self.logvar_min.to(device)
        self.logvar_max = self.logvar_max.to(device)
        self.min_var_coef = self.min_var_coef.to(device)

    def reset_state(self):
        self.state = {}
        self.i = 0

    def sigma_to_t(self, sigma):
        '''Returns t so that self.nsr(t-1) <= sigma < self.nsr(t)'''
        #alpha_bar_t = 1 / (sigma ** 2 + 1)
        #ts = self.num_train_timesteps - torch.searchsorted(reversed(self.alphas_cumprod), alpha_bar_t)
        ts =  torch.searchsorted(self.sigmas, sigma)
        return ts

    def t_to_sigma_interp(self, t):
        x = self.train_timesteps.float()
        y = self.alphas_cumprod
        xnew=t.to(x.device).squeeze()
        if len(xnew.size())==0:
            xnew = xnew.unsqueeze(0)
        y_new = interp1d(x, y, xnew)
        y_new = y_new.squeeze(0)
        sigma = (1 / y_new - 1).sqrt()
        sigma = torch.where(t >= 0, sigma, self.final_sigma)
        sigma =sigma.float()
        return sigma

    def t_to_alphabar_interp(self, t):
        sigma = self.t_to_sigma_interp(t)
        alpha_bar_t = 1 / (sigma ** 2 + 1)
        return alpha_bar_t

    def sigma_to_t_interp(self, sigma):
        x = self.sigmas
        y = self.train_timesteps.float()
        xnew=sigma.to(x.device).squeeze()
        if len(xnew.size())==0:
            xnew = xnew.unsqueeze(0)
        y_new = interp1d(x, y, xnew)
        t = y_new.squeeze(0)
        #t=torch.clamp(y_new, min=0, max=self.num_train_timesteps,)
        t =t.float()
        return t

    def get_end_sigma(self, factor=2):
        sigma_1 = self.get_sigma(self.timesteps[-2])
        sigma_2 = self.get_sigma(self.timesteps[-1])
        return ((sigma_1.log() + sigma_2.log()) / factor).exp()

    def set_timesteps_sigma(self,
                            start: float,
                            end: float,
                            num_inference_steps: int,
                            style: str = 'DDIM',scale: float =1, continuous_t: bool = False):
        self.continuous_t = continuous_t
        self.num_inference_steps = num_inference_steps
        dtype = torch.long if not continuous_t else torch.float32
        if not self.set_alpha_to_one:
            num_inference_steps = num_inference_steps+1
        if style == 'DDIM':
            start_t, end_t = map(self.get_t_from_sigma, (start, end))
            start_t = start_t.item()
            end_t = end_t.item()
            ts = space_timesteps(num_timesteps=start_t+1-end_t, section_counts=str(num_inference_steps))
            ts =   end_t + np.array(sorted(ts, reverse=True))
            self.timesteps =  torch.tensor(ts,dtype=dtype)
            #self.timesteps = torch.tensor(np.linspace(start_t, end_t, num_inference_steps),dtype=dtype)
            sigmas =  self.get_sigma(self.timesteps)
        elif style == 'EDM':
            rho = 7
            N = num_inference_steps
            sigmas = torch.tensor([
                (start ** (1 / rho) + i / (N - 1) * (end ** (1 / rho) - start ** (1 / rho))) ** rho
             for i in range(N)])
            self.timesteps = self.get_t_from_sigma(sigmas)
        elif style == 'Linear':
            sigmas = torch.tensor(np.exp(np.linspace(np.log(start), np.log(end), num_inference_steps)))
            self.timesteps = self.get_t_from_sigma(sigmas)
        elif style == 'Scaled':
            diff = np.log(end) - np.log(start)
            a_t = scale**np.arange(num_inference_steps-1)
            a_t_cumsum = np.cumsum(a_t)
            scaler_factor = diff/a_t_cumsum[-1]
            sigma_logs = np.log(start) + scaler_factor*a_t_cumsum
            sigma_logs = np.insert(sigma_logs, 0, np.log(start))
            sigmas = torch.tensor(np.exp(sigma_logs))
            self.timesteps = self.get_t_from_sigma(sigmas)
        else:
            raise ValueError('Invalid style!')
        self.timesteps = self.timesteps.squeeze()
        sigmas = sigmas.squeeze()
        if not continuous_t:
            self.timesteps = replace_duplicate_t(self.timesteps)
            self.sampling_sigmas = self.get_sigma(self.timesteps)
        else:
            self.sampling_sigmas =sigmas

        if self.set_alpha_to_one:
            self.timesteps = torch.cat([self.timesteps, torch.tensor([-1])])
            self.sampling_sigmas = torch.cat([self.sampling_sigmas, torch.tensor([self.final_sigma])])

        sigma_t = self.sampling_sigmas[-3]
        sigma_prev = self.sampling_sigmas[-2]
        beta_t = (sigma_t ** 2 - sigma_prev ** 2) / (sigma_t ** 2 + 1)
        alpha_t = 1 / (sigma_t ** 2 + 1)
        alpha_prev = 1 / (sigma_prev ** 2 + 1)
        self.min_var_coef = beta_t*(1 - alpha_prev) / (1 - alpha_t)


    def set_timesteps(self, num_inference_steps: int, steps_offset: int):
        if num_inference_steps > self.num_train_timesteps:
            raise ValueError(
                f'`num_inference_steps` ({num_inference_steps}) must be smaller than'
                f'`self.num_train_timesteps` ({self.num_train_timesteps})'
            )
        if steps_offset >= self.num_train_timesteps:
            raise ValueError(
                f'`steps_offset` ({steps_offset}) must be smaller than '
                f'`self.num_train_timesteps` ({self.num_train_timesteps})'
            )
        start = self.num_train_timesteps - steps_offset
        if not self.set_alpha_to_one:
            num_inference_steps = num_inference_steps+1
        self.timesteps = torch.tensor(np.linspace(start, 0, num_inference_steps), dtype=torch.long)
        if self.set_alpha_to_one:
            self.timesteps = torch.cat([self.timesteps, torch.tensor([-1])])
        self.sampling_sigmas = self.get_sigma(self.timesteps)

    def alpha_bar(self, timestep):  # ap
        #abar= self.alphas_cumprod[timestep] if timestep >= 0 else self.final_alpha_cumprod
        abar=self.alphas_cumprod[timestep]
        abar= torch.where(timestep >= 0, abar, self.final_alpha_cumprod)
        return abar

    def sigma(self, timestep):  # nsr
        #sig =  self.sigmas[timestep] if timestep >= 0 else self.final_sigma
        sig = self.sigmas[timestep]
        sig = torch.where(timestep >= 0, sig, self.final_sigma)
        return sig

    def eps_logvar(self, logvar_list, timestep):
        logvar = logvar_list[timestep]
        logvar = torch.where(timestep >= 0, logvar, logvar[0])
        return logvar

    def diffusion(self, x_0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_0)
        shape = (-1,) + (1,) * (len(x_0.size()) - 1)
        alpha = self.alphas_cumprod[t].view(*shape)
        x_n = x_0 * alpha.sqrt() + noise * (1 - alpha).sqrt()
        return x_n, noise

    def get_sigma(self, timestep):
        if self.continuous_t:
            return self.t_to_sigma_interp(timestep)
        else:
            return self.sigma(timestep)

    def get_alpha_bar(self, timestep):
        if self.continuous_t:
            return self.t_to_alphabar_interp(timestep)
        else:
            return self.alpha_bar(timestep)

    def get_t_from_sigma(self, sigma):
        if self.continuous_t:
            return self.sigma_to_t_interp(sigma)
        else:
            times = self.sigma_to_t(sigma)
            return times

    # def get_eps_logvar(self, t,x_shape, learned_logvar=None):
    #     t = t.long()
    #     min_logvar = self.eps_logvar(self.logvar_min, t)
    #     min_logvar = expand_shape(min_logvar, x_shape)
    #     max_logvar = self.eps_logvar(self.logvar_max, t)
    #     max_logvar = expand_shape(max_logvar, x_shape)
    #     if self.sampler_var =='learned':
    #         frac = (learned_logvar + 1) / 2
    #         log_variance = frac * max_logvar + (1 - frac) * min_logvar
    #     elif self.sampler_var =='fixedsmall':
    #         log_variance = min_logvar
    #     elif self.sampler_var =='fixedlarge':
    #         log_variance = max_logvar
    #     else:
    #         log_variance = None
    #     return log_variance

    def get_eps_logvar(self, sigma_t,sigma_prev, learned_logvar=None):
        beta_t = (sigma_t**2 - sigma_prev**2)/(sigma_t**2+1)
        beta_t = beta_t.abs().clamp(min=1e-20)
        alpha_t = 1 / (sigma_t ** 2 + 1)
        alpha_prev = 1 / (sigma_prev ** 2 + 1)
        coef = (1 - alpha_prev) / (1 - alpha_t)
        coef = coef.clamp(min=0, max=1)
        #coef = 1 - (1 - coef).abs()
        post_var = beta_t * coef
        max_logvar = beta_t.log()
        #min_logvar = post_var.clamp(min=1e-20).log()
        min_logvar = post_var.clamp(min=self.min_var_coef).log()
        max_var = torch.exp(max_logvar)
        min_var = torch.exp(min_logvar)
        if self.sampler_var =='learned':
            frac = (learned_logvar + 1) / 2
            log_variance = frac * max_logvar + (1 - frac) * min_logvar
        elif self.sampler_var =='fixedsmall':
            log_variance = min_logvar
        elif self.sampler_var =='fixedlarge':
            log_variance = max_logvar
        else:
            log_variance = None
        return log_variance

    def step(self, eps, t, t_prev, xt,learned_logvar=None):
        sigma_prev = self.get_sigma(t_prev)
        sigma_t = self.get_sigma(t)
        log_variance = self.get_eps_logvar(t, xt.shape, learned_logvar)
        x_prev = self.step_with_sigma(eps, sigma_t, sigma_prev, xt,log_variance)
        return x_prev

    def step_with_iter(self, eps, iter, xt,learned_logvar=None):
        sigma_t = self.sampling_sigmas[iter]
        sigma_prev = self.sampling_sigmas[iter + 1]
        t = self.timesteps[iter]
        log_variance = self.get_eps_logvar(t, xt.shape, learned_logvar)
        x_prev = self.step_with_sigma(eps, sigma_t, sigma_prev, xt,log_variance)
        return x_prev

    def pred_xstart(self, xt, eps, sigma_t):
        x_start = xt - sigma_t * eps
        return x_start

    def step_with_sigma(self, eps, sigma_t, sigma_prev, xt, log_variance=None):
        x0 = self.pred_xstart(xt=xt,eps=eps,sigma_t=sigma_t)
        x_prev = self.pred_xprev(x0, eps, sigma_t, sigma_prev, log_variance)
        return x_prev

    def pred_xprev(self, x0, eps, sigma_t, sigma_prev,xt=None, log_variance=None):
        raise NotImplementedError

    def pred_xprev_with_sigma_noise(self, x0, eps, signal_sigma, noise_sigma):
        noise = torch.randn_like(x0)
        x_prev = x0 + signal_sigma * eps + noise_sigma * noise
        return x_prev


class DDIM_Scheduler(Scheduler):
    def __init__(self, num_train_timesteps: int = 1000, beta_start: float = 0.0001, beta_end: float = 0.02,
                 beta_schedule: str = 'linear', set_alpha_to_one: bool = True,
                sampler_var: str = 'none',  eta : float = 0.0):
        super().__init__(num_train_timesteps=num_train_timesteps, beta_start=beta_start, beta_end=beta_end,
                         beta_schedule=beta_schedule, set_alpha_to_one=set_alpha_to_one,sampler_var=sampler_var,eta=eta)

    def pred_xprev(self, x0, eps, sigma_t, sigma_prev,xt=None, log_variance=None):
        if self.eta > 0:
            ddpm_noise_sigma = torch.exp(0.5 * log_variance)
            alpha_bar_prev = 1 / (sigma_prev ** 2 + 1)
            noise_sigma = self.eta * ddpm_noise_sigma / torch.sqrt(alpha_bar_prev)
            # noise_sigma = self.eta * sigma_prev * torch.sqrt(1 - (sigma_prev/sigma_t)**2)
            noise = torch.randn_like(x0)
            mask = sigma_prev > 0
            noise = mask * noise
        else:
            noise_sigma = 0
            noise = 0

        signal_sigma = torch.sqrt((sigma_prev**2 - noise_sigma**2).clamp(min=0))
        noise_sigma =  torch.sqrt(sigma_prev**2 - signal_sigma**2)
        x_prev = x0 + signal_sigma * eps + noise_sigma * noise
        self.i += 1
        return x_prev

    def get_noise_signal_sigma(self, sigma_prev, log_variance=None):
        ddpm_noise_sigma = torch.exp(0.5 * log_variance)
        alpha_bar_prev = 1 / (sigma_prev ** 2 + 1)
        noise_sigma = self.eta * ddpm_noise_sigma / torch.sqrt(alpha_bar_prev)
        signal_sigma = torch.sqrt((sigma_prev ** 2 - noise_sigma ** 2).clamp(min=0))
        return noise_sigma, signal_sigma

class DDIM_simple_Scheduler(Scheduler):
    def __init__(self, num_train_timesteps: int = 1000, beta_start: float = 0.0001, beta_end: float = 0.02,
                 beta_schedule: str = 'linear', set_alpha_to_one: bool = True,
                sampler_var: str = 'none', eta : float = 0.0):
        super().__init__(num_train_timesteps=num_train_timesteps, beta_start=beta_start, beta_end=beta_end,
                         beta_schedule=beta_schedule, set_alpha_to_one=set_alpha_to_one,sampler_var=sampler_var,eta=eta)

    def pred_xprev(self, x0, eps, sigma_t, sigma_prev,xt=None, log_variance=None):
        signal_sigma = math.sqrt(1-self.eta**2) * sigma_prev
        x_prev = x0 + signal_sigma * eps
        if self.eta>0:
            noise = torch.randn_like(x0)
            noise_sigma = self.eta * sigma_prev
            x_prev = x_prev + noise_sigma * noise
        self.i += 1
        return x_prev

    def get_noise_signal_sigma(self, sigma_prev, log_variance=None):
        noise_sigma = self.eta * sigma_prev
        signal_sigma = math.sqrt(1 - self.eta ** 2) * sigma_prev
        return noise_sigma, signal_sigma

class DDIM_simple_orig_Scheduler(Scheduler):
    def __init__(self, num_train_timesteps: int = 1000, beta_start: float = 0.0001, beta_end: float = 0.02,
                 beta_schedule: str = 'linear', set_alpha_to_one: bool = True,
                sampler_var: str = 'none', eta : float = 0.0):
        super().__init__(num_train_timesteps=num_train_timesteps, beta_start=beta_start, beta_end=beta_end,
                         beta_schedule=beta_schedule, set_alpha_to_one=set_alpha_to_one,sampler_var=sampler_var,eta=eta)

    def pred_xprev(self, x0, eps, sigma_t, sigma_prev,xt=None, log_variance=None):
        eps = (xt - x0) / sigma_t
        signal_sigma = math.sqrt(1-self.eta**2) * sigma_prev
        x_prev = x0 + signal_sigma * eps
        if self.eta>0:
            noise = torch.randn_like(x0)
            noise_sigma = self.eta * sigma_prev
            x_prev = x_prev + noise_sigma * noise
        self.i += 1
        return x_prev

class DDIM_simple_drag_Scheduler(Scheduler):
    def __init__(self, num_train_timesteps: int = 1000, beta_start: float = 0.0001, beta_end: float = 0.02,
                 beta_schedule: str = 'linear', set_alpha_to_one: bool = True,
                sampler_var: str = 'none', eta : float = 0.0):
        super().__init__(num_train_timesteps=num_train_timesteps, beta_start=beta_start, beta_end=beta_end,
                         beta_schedule=beta_schedule, set_alpha_to_one=set_alpha_to_one,sampler_var=sampler_var,eta=eta)

    def pred_xprev(self, x0, eps, sigma_t, sigma_prev,xt=None, log_variance=None):
        eps = (xt - x0) / sigma_t
        signal_sigma = sigma_prev
        x_prev = x0 + signal_sigma * eps
        if self.eta>0:
            noise = torch.randn_like(x0)
            noise_sigma = self.eta * sigma_prev
            x_prev = x_prev + noise_sigma * noise
        self.i += 1
        return x_prev

# class DDPM_Scheduler(Scheduler):
#     def __init__(self, num_train_timesteps: int = 1000, beta_start: float = 0.0001, beta_end: float = 0.02,
#                  beta_schedule: str = 'linear', set_alpha_to_one: bool = True,
#                   sampler_var: str = 'none', eta : float = 1.0):
#         super().__init__(num_train_timesteps=num_train_timesteps, beta_start=beta_start, beta_end=beta_end,
#                          beta_schedule=beta_schedule, set_alpha_to_one=set_alpha_to_one,sampler_var=sampler_var,eta=eta)
#
#     def pred_xprev(self, x0, eps, sigma, sigma_prev, log_variance=None):
#         noise_sigma = self.eta * sigma_prev * torch.sqrt(1 - (sigma_prev/sigma)**2)
#         signal_sigma = torch.sqrt(sigma_prev**2 - noise_sigma**2)
#         x_prev = x0 + signal_sigma * eps
#
#         noise = torch.randn_like(x0)
#         mask = sigma > self.sigmas[0]
#         noise = mask * noise
#         ddpm_noise_sigma = torch.exp(0.5*log_variance)
#         alpha_bar_prev = 1 / (sigma_prev ** 2 + 1)
#         ddpm_noise_sigma = ddpm_noise_sigma / torch.sqrt(alpha_bar_prev)
#         noise_sigma = self.eta * ddpm_noise_sigma + (1-self.eta)*noise_sigma
#         x_prev = x_prev + noise_sigma * noise
#
#         self.i += 1
#         return x_prev


class DDPM_Scheduler(Scheduler):
    def __init__(self, num_train_timesteps: int = 1000, beta_start: float = 0.0001, beta_end: float = 0.02,
                 beta_schedule: str = 'linear', set_alpha_to_one: bool = True,
                sampler_var: str = 'none', eta : float = 1.0):
        super().__init__(num_train_timesteps=num_train_timesteps, beta_start=beta_start, beta_end=beta_end,
                         beta_schedule=beta_schedule, set_alpha_to_one=set_alpha_to_one,sampler_var=sampler_var,eta=eta)

    def pred_xprev(self, x0, eps, sigma_t, sigma_prev, xt=None, log_variance=None):
        ddpm_noise_sigma = torch.exp(0.5 * log_variance)
        alpha_bar_prev = 1 / (sigma_prev ** 2 + 1)
        noise_sigma = ddpm_noise_sigma / torch.sqrt(alpha_bar_prev)

        signal_sigma = torch.sqrt( (sigma_prev**2 - noise_sigma**2).clamp(min=0))
        x_prev = x0 + signal_sigma * eps

        noise = torch.randn_like(x0)
        mask = sigma_prev > 0
        noise = mask * noise
        x_prev = x_prev + noise_sigma * noise

        self.i += 1
        return x_prev

    def get_noise_signal_sigma(self, sigma_prev, log_variance=None):
        ddpm_noise_sigma = torch.exp(0.5 * log_variance)
        alpha_bar_prev = 1 / (sigma_prev ** 2 + 1)
        noise_sigma =  ddpm_noise_sigma / torch.sqrt(alpha_bar_prev)
        signal_sigma = torch.sqrt((sigma_prev ** 2 - noise_sigma ** 2).clamp(min=0))
        return noise_sigma, signal_sigma



class DDPM_orig_Scheduler(Scheduler):
    def __init__(self, num_train_timesteps: int = 1000, beta_start: float = 0.0001, beta_end: float = 0.02,
                 beta_schedule: str = 'linear', set_alpha_to_one: bool = True,
                sampler_var: str = 'none', eta : float = 1.0):
        eta = 1.0
        super().__init__(num_train_timesteps=num_train_timesteps, beta_start=beta_start, beta_end=beta_end,
                         beta_schedule=beta_schedule, set_alpha_to_one=set_alpha_to_one,sampler_var=sampler_var,eta=eta)

    def pred_xprev(self, x0, eps, sigma_t, sigma_prev, xt=None, log_variance=None):
        alpha_bar = 1 / (sigma_t ** 2 + 1)
        alpha_bar_prev = 1 / (sigma_prev ** 2 + 1)
        alpha_t = alpha_bar/alpha_bar_prev
        beta_t = 1 - alpha_t
        zt = xt * alpha_bar.sqrt()
        z0 = x0

        posterior_mean_coef1 = beta_t * alpha_bar_prev.sqrt() / (1.0 - alpha_bar)
        posterior_mean_coef2 = (1.0 - alpha_bar_prev) * alpha_t.sqrt()/ (1.0 - alpha_bar)
        posterior_mean = posterior_mean_coef1 *z0 + posterior_mean_coef2*zt

        noise = torch.randn_like(x0)
        mask = (sigma_prev > 0).float()
        z_prev = posterior_mean + mask * torch.exp(0.5 * log_variance) * noise

        x_prev = z_prev / alpha_bar_prev.sqrt()
        self.i += 1
        return x_prev


class DDIM_orig_Scheduler(Scheduler):
    def __init__(self, num_train_timesteps: int = 1000, beta_start: float = 0.0001, beta_end: float = 0.02,
                 beta_schedule: str = 'linear', set_alpha_to_one: bool = True,
                sampler_var: str = 'none',  eta : float = 0.0):
        super().__init__(num_train_timesteps=num_train_timesteps, beta_start=beta_start, beta_end=beta_end,
                         beta_schedule=beta_schedule, set_alpha_to_one=set_alpha_to_one,sampler_var=sampler_var,eta=eta)

    def pred_xprev(self, x0, eps, sigma_t, sigma_prev,xt=None, log_variance=None):
        eps = (xt - x0) / sigma_t
        if self.eta > 0:
            ddpm_noise_sigma = torch.exp(0.5 * log_variance)
            alpha_bar_prev = 1 / (sigma_prev ** 2 + 1)
            noise_sigma = self.eta * ddpm_noise_sigma / torch.sqrt(alpha_bar_prev)
            # noise_sigma = self.eta * sigma_prev * torch.sqrt(1 - (sigma_prev/sigma_t)**2)
            noise = torch.randn_like(x0)
            mask = (sigma_prev > 0).float()
            noise = mask * noise
        else:
            noise_sigma = 0.0
            noise = 0.0

        signal_sigma = torch.sqrt((sigma_prev**2 - noise_sigma**2).clamp(min=0))
        #noise_sigma =  torch.sqrt(sigma_prev**2 - signal_sigma**2)
        x_prev = x0 + signal_sigma * eps + noise_sigma * noise
        self.i += 1
        return x_prev


class GE_Scheduler(Scheduler):
    '''Gradient Estimation Second Order Scheduler'''

    def __init__(self, gamma: float = 2.0, num_train_timesteps: int = 1000, beta_start: float = 0.0001,
                 beta_end: float = 0.02, beta_schedule: str = 'linear', set_alpha_to_one: bool = True,
                  sampler_var: str = 'none', eta : float = 1.0,norm_eps=False):
        super().__init__(num_train_timesteps=num_train_timesteps, beta_start=beta_start, beta_end=beta_end,
                         beta_schedule=beta_schedule, set_alpha_to_one=set_alpha_to_one,sampler_var=sampler_var,eta=eta)
        self.gamma = gamma
        self.norm_eps =norm_eps

    def reset_state(self):
        self.state = {}
        self.state['eps_prev'] = None
        self.state['eps_avg'] = None
        self.i = 0

    def pred_xstart(self, xt, eps, sigma_t):
        eps_prev = self.state['eps_prev']
        eps_av = eps * self.gamma + eps_prev * (1 - self.gamma) if self.i > 0 else eps
        if self.norm_eps:
            eps_av = normalize(eps_av)
        x_start = xt - sigma_t * eps_av
        return x_start

    def pred_xprev(self, x0, eps, sigma_t, sigma_prev, log_variance=None):
        eps_prev = self.state['eps_prev']
        eps_av = eps * self.gamma + eps_prev * (1 - self.gamma) if self.i > 0 else eps
        if self.norm_eps:
            eps_av = normalize(eps_av)

        noise_sigma = self.eta * sigma_prev * torch.sqrt(1 - (sigma_prev/sigma_t)**2)
        signal_sigma = torch.sqrt(sigma_prev**2 - noise_sigma**2)
        x_prev = x0 + signal_sigma * eps_av
        if self.eta>0:
            noise = torch.randn_like(x0)
            mask = sigma_t > self.sigmas[0]
            noise = mask * noise
            x_prev = x_prev + noise_sigma * noise

        self.i += 1
        self.state['eps_prev'] = eps.detach()
        self.state['eps_avg'] = eps_av.detach()
        return x_prev


def get_sampler(sampler_name, train_timesteps, inference_timesteps,  beta_start = 0.0001,
            beta_end = 0.02, beta_schedule='linear', sigma_style='DDIM', set_alpha_to_one=True, start_sigma=None,
            end_sigma=None, sampler_var='none',  continuous_t=False,  linear_scale=1.0, eta=0.0, ge_gamma=2,norm_eps=False,
                start_t=None, end_t=None):
    if sampler_name == 'ddpm':
        sampler = DDPM_Scheduler(num_train_timesteps=train_timesteps,beta_start=beta_start,beta_end=beta_end, beta_schedule=beta_schedule, set_alpha_to_one=set_alpha_to_one,
                                 sampler_var=sampler_var,eta=eta)
    elif sampler_name== 'ddim':
        sampler = DDIM_Scheduler(num_train_timesteps=train_timesteps,beta_start=beta_start,beta_end=beta_end, beta_schedule=beta_schedule,set_alpha_to_one=set_alpha_to_one,
                                 sampler_var=sampler_var,eta=eta)
    elif sampler_name == 'ddim_simple':
        sampler = DDIM_simple_Scheduler(num_train_timesteps=train_timesteps, beta_start=beta_start, beta_end=beta_end,
                                 beta_schedule=beta_schedule, set_alpha_to_one=set_alpha_to_one,
                                 sampler_var=sampler_var, eta=eta)
    elif sampler_name == 'ddim_orig':
        sampler = DDIM_orig_Scheduler(num_train_timesteps=train_timesteps, beta_start=beta_start, beta_end=beta_end,
                                 beta_schedule=beta_schedule, set_alpha_to_one=set_alpha_to_one,
                                 sampler_var=sampler_var, eta=eta)
    elif sampler_name == 'ddim_simple_orig':
        sampler  = DDIM_simple_orig_Scheduler(num_train_timesteps=train_timesteps, beta_start=beta_start, beta_end=beta_end,
                                 beta_schedule=beta_schedule, set_alpha_to_one=set_alpha_to_one,
                                 sampler_var=sampler_var, eta=eta)
    elif sampler_name == 'ddim_simple_drag':
        sampler  = DDIM_simple_drag_Scheduler(num_train_timesteps=train_timesteps, beta_start=beta_start, beta_end=beta_end,
                                 beta_schedule=beta_schedule, set_alpha_to_one=set_alpha_to_one,
                                 sampler_var=sampler_var, eta=eta)
    elif sampler_name == 'ddpm_orig':
        sampler = DDPM_orig_Scheduler(num_train_timesteps=train_timesteps, beta_start=beta_start, beta_end=beta_end,
                                 beta_schedule=beta_schedule, set_alpha_to_one=set_alpha_to_one,
                                 sampler_var=sampler_var, eta=eta)
    elif sampler_name =='ge':
        sampler = GE_Scheduler(num_train_timesteps=train_timesteps,beta_start=beta_start,beta_end=beta_end,beta_schedule=beta_schedule, set_alpha_to_one=set_alpha_to_one,
                               sampler_var=sampler_var,eta=eta, gamma=ge_gamma,norm_eps=norm_eps)
    else:
        raise NotImplementedError
    if start_sigma is None or start_sigma<=0:
        if start_t is None or start_t<0:
            start_sigma = sampler.sigmas[-1]
        else:
            start_sigma = min(sampler.sigmas[start_t], sampler.sigmas[-1])
    else:
        start_sigma = min(start_sigma, sampler.sigmas[-1])
        start_sigma = torch.tensor(start_sigma, device=sampler.sigmas.device)
    if end_sigma is None or end_sigma<=0:
        if end_t is None or end_t<0:
            end_sigma = sampler.sigmas[0]
        else:
            end_sigma = sampler.sigmas[end_t]
    sampler.set_timesteps_sigma(start=start_sigma, end=end_sigma, num_inference_steps=inference_timesteps, style=sigma_style,
                                scale=linear_scale,continuous_t=continuous_t)
    return sampler

# if __name__=='__main__':
#     start_sigma = 0
#     end_sigma = 0
#     sampler  =get_sampler('ddim', 1000, 100,  beta_start = 0.0001,
#             beta_end = 0.02, beta_schedule='linear', sigma_style='DDIM', set_alpha_to_one=True, start_sigma=None,
#                 end_sigma=None, sampler_var='none',  continuous_t=False,linear_scale=1.0, eta=0.0, ge_gamma=2,norm_eps=False )
#     print(sampler.sampling_sigmas)
#     print(sampler.sigmas[::10])