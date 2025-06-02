import os
import numpy as np
import torch


class LinearProcessing:
    def __init__(self, function='inpaint_center', data_shape=(1,28,28), device='cpu',
                 drop_ratio=0.3,total_box = 5):
        self.fun_name = function
        self.data_shape = data_shape
        self.device = device
        if function =='inpaint_center':
            box_size = int(drop_ratio * data_shape[1])
            self.A_fun, self.A_fun_batch, self.mask = self.inpaint_center(box_size)
        elif function =='inpaint_pixel':
            self.A_fun, self.A_fun_batch, self.mask = self.inpaint_pixel(drop_ratio)
        elif function == 'inpaint_block':
            box_size = int(max(1.0,drop_ratio * data_shape[1]/total_box))
            self.A_fun, self.A_fun_batch, self.mask = self.inpaint_block(box_size, total_box = total_box)
        elif function == 'compressive_sensing':
            self.A_fun, self.A_fun_batch, self.mask = self.compressive_sensing(compress_ratio = drop_ratio)
        else:
            raise NotImplementedError

    def inpaint_center(self, box_size):
        x_shape = self.data_shape
        mask = torch.ones(*x_shape, device=self.device)

        idx_row = round(float(x_shape[1]) / 2.0 - float(box_size) / 2.0)
        idx_col = round(float(x_shape[2]) / 2.0 - float(box_size) / 2.0)

        mask[0, idx_row:idx_row + box_size, idx_col:idx_col + box_size] = 0.

        def A_fun(x):
            y = torch.multiply(x, mask)
            return y

        def A_fun_batch(x_batch):
            y = torch.multiply(x_batch, mask.unsqueeze(0))
            return y

        return A_fun, A_fun_batch, mask

    def inpaint_pixel(self, drop_prob = 0.5):
        x_shape = self.data_shape
        mask = torch.rand(*x_shape, device=self.device) > drop_prob

        def A_fun(x):
            y = torch.multiply(x, mask)
            return y

        def A_fun_batch(x_batch):
            y = torch.multiply(x_batch, mask.unsqueeze(0))
            return y

        return A_fun, A_fun_batch, mask

    def inpaint_block(self, box_size, total_box = 1):
        x_shape = self.data_shape
        spare = 0.25 * box_size
        mask = torch.ones(*x_shape, device=self.device)

        for i in range(total_box):
            start_row = spare
            end_row = x_shape[1] - spare - box_size - 1
            start_col = spare
            end_col = x_shape[2] - spare - box_size - 1

            idx_row = int(np.random.rand(1) * (end_row - start_row) + start_row)
            idx_col = int(np.random.rand(1) * (end_col - start_col) + start_col)

            mask[0,idx_row:idx_row+box_size,idx_col:idx_col+box_size] = 0.

        def A_fun(x):
            y = torch.multiply(x, mask)
            return y

        def A_fun_batch(x_batch):
            y = torch.multiply(x_batch, mask.unsqueeze(0))
            return y

        return A_fun, A_fun_batch, mask

    def compressive_sensing(self, compress_ratio):
        x_shape = self.data_shape
        d = np.prod(x_shape).astype(int)
        m = np.round(compress_ratio * d).astype(int)
        A =  torch.rand(m, d, device=self.device)  / np.sqrt(m)

        def A_fun(x):
            x_flat = x.view(d,1)
            y = A @ x_flat
            return y

        def A_fun_batch(x_batch):
            x_flat = x_batch.view(len(x_batch),d)
            y = x_flat @ A.T
            return y

        return A_fun, A_fun_batch, A

    def transform(self, img):
        return self.A_fun(img)

    def transform_batch(self, imgs):
        return self.A_fun_batch(imgs)

    def transform_loss(self, x, y=None,shift_x=False):
        if shift_x:
            x = x.add(1).div(2)
        y_hat  = self.transform_batch(x)
        loss = torch.nn.functional.mse_loss(y, y_hat)
        return loss

    def gradient_descent(self, x, y=None, lr=0.1, n_step=10, shift_x=False):
        prev = torch.is_grad_enabled()
        torch.set_grad_enabled(True)
        for i in range(n_step):
            X = x.clone().detach().requires_grad_(True)
            loss = self.transform_loss(X,y,shift_x=shift_x)
            grad = torch.autograd.grad(loss, X)[0]
            new_X = X - lr * grad
            x = new_X.detach()
        torch.set_grad_enabled(prev)
        return x


def get_gaussian_noisy_img(img, noise_level):
    return img + torch.randn_like(img).cuda() * noise_level

def MeanUpsample(x, scale):
    n, c, h, w = x.shape
    out = torch.zeros(n, c, h, scale, w, scale).to(x.device) + x.view(n,c,h,1,w,1)
    out = out.view(n, c, scale*h, scale*w)
    return out

def color2gray(x):
    coef=1/3
    x = x[:,0,:,:] * coef + x[:,1,:,:]*coef +  x[:,2,:,:]*coef
    #return x
    return x.repeat(1,3,1,1)

def gray2color(x):
    x = x[:,0,:,:]
    coef=1/3
    base = coef**2 + coef**2 + coef**2
    return torch.stack((x*coef/base, x*coef/base, x*coef/base), 1)
# Implementation of A and its pseudo-inverse Ap

def simple_constraint(fn,fn_scale=4,device='cpu', base_mask_dir='store/inp_masks'):
    if fn == 'colorization':
        A = lambda z: color2gray(z)
        Ap = lambda z: gray2color(z)
    elif fn == 'denoising':
        A = lambda z: z
        Ap = A
    elif fn == 'sr_averagepooling':
        scale = round(fn_scale)
        A = torch.nn.AdaptiveAvgPool2d((256 // scale, 256 // scale))
        Ap = lambda z: MeanUpsample(z, scale)
    elif fn == 'inpainting':
        loaded = np.load(os.path.join(base_mask_dir,"mask.npy"))
        mask = torch.from_numpy(loaded).to(device)
        A = lambda z: z * mask
        Ap = A
    elif fn == 'mask_color_sr':
        loaded = np.load(os.path.join(base_mask_dir,"mask.npy"))
        mask = torch.from_numpy(loaded).to(device)
        A1 = lambda z: z * mask
        A1p = A1

        A2 = lambda z: color2gray(z)
        A2p = lambda z: gray2color(z)

        scale = round(fn_scale)
        A3 = torch.nn.AdaptiveAvgPool2d((256 // scale, 256 // scale))
        A3p = lambda z: MeanUpsample(z, scale)

        A = lambda z: A3(A2(A1(z)))
        Ap = lambda z: A1p(A2p(A3p(z)))
    elif fn == 'diy':
        # design your own degradation
        loaded = np.load(os.path.join(base_mask_dir,"mask.npy"))
        mask = torch.from_numpy(loaded).to(device)
        A1 = lambda z: z * mask
        A1p = A1

        A2 = lambda z: color2gray(z)
        A2p = lambda z: gray2color(z)

        scale = fn_scale
        A3 = torch.nn.AdaptiveAvgPool2d((256 // scale, 256 // scale))
        A3p = lambda z: MeanUpsample(z, scale)

        A = lambda z: A3(A2(A1(z)))
        Ap = lambda z: A1p(A2p(A3p(z)))
    else:
        A=None
        Ap=None


    return A, Ap



def svd_constraint(fn,fn_scale=4,device='cpu', base_mask_dir='store/inp_masks', image_size=256,channels=3 ):
    if fn == 'cs_walshhadamard':
        compress_by = round(fn_scale)
        from functions.svd_operators import WalshHadamardCS
        A_funcs = WalshHadamardCS(channels, image_size, compress_by,
                                  torch.randperm(image_size ** 2, device=device), device)
    elif fn == 'cs_blockbased':
        cs_ratio =fn_scale
        from functions.svd_operators import CS
        A_funcs = CS(channels, image_size, cs_ratio, device)
    elif 'inpainting' in fn:
        from functions.svd_operators import Inpainting
        if fn == 'inpainting_lolcat':
            loaded = np.load("inp_masks/lolcat_extra.npy")
            mask = torch.from_numpy(loaded).to(device).reshape(-1)
            missing_r = torch.nonzero(mask == 0).long().reshape(-1) * 3
        elif fn == 'inpainting_lorem':
            loaded = np.load("inp_masks/lorem3.npy")
            mask = torch.from_numpy(loaded).to(device).reshape(-1)
            missing_r = torch.nonzero(mask == 0).long().reshape(-1) * 3
        elif fn == 'inpainting_ddnm':
            loaded = np.load(os.path.join(base_mask_dir,"mask.npy"))
            mask = torch.from_numpy(loaded).to(device).reshape(-1)
            missing_r = torch.nonzero(mask == 0).long().reshape(-1) * 3
        elif fn == 'inpainting_random':
            missing_r = torch.randperm(image_size ** 2)[:image_size ** 2 // 2].to(device).long() * 3
        elif fn == 'inpainting_half':
            loaded = np.load(os.path.join(base_mask_dir,"mask_half.npy"))
            mask = torch.from_numpy(loaded).to(device).reshape(-1)
            missing_r = torch.nonzero(mask == 0).long().reshape(-1) * 3
        else:
            missing_r = torch.load(os.path.join(base_mask_dir, "mask_random.pt")).to(device)
        missing_g = missing_r + 1
        missing_b = missing_g + 1
        missing = torch.cat([missing_r, missing_g, missing_b], dim=0)
        A_funcs = Inpainting(channels, image_size, missing, device)
    elif fn == 'denoising':
        from functions.svd_operators import Denoising
        A_funcs = Denoising(channels, image_size, device)
    elif fn == 'colorization':
        from functions.svd_operators import Colorization
        A_funcs = Colorization(image_size, device)
    elif fn == 'sr_averagepooling':
        blur_by = int(fn_scale)
        from functions.svd_operators import SuperResolution
        A_funcs = SuperResolution(channels, image_size, blur_by, device)
    elif fn == 'sr_bicubic':
        factor = int(fn_scale)
        from functions.svd_operators import SRConv
        def bicubic_kernel(x, a=-0.5):
            if abs(x) <= 1:
                return (a + 2) * abs(x) ** 3 - (a + 3) * abs(x) ** 2 + 1
            elif 1 < abs(x) and abs(x) < 2:
                return a * abs(x) ** 3 - 5 * a * abs(x) ** 2 + 8 * a * abs(x) - 4 * a
            else:
                return 0

        k = np.zeros((factor * 4))
        for i in range(factor * 4):
            x = (1 / factor) * (i - np.floor(factor * 4 / 2) + 0.5)
            k[i] = bicubic_kernel(x)
        k = k / np.sum(k)
        kernel = torch.from_numpy(k).float().to(device)
        A_funcs = SRConv(kernel / kernel.sum(),  channels, image_size, device, stride=factor)
    elif fn == 'deblur_uni':
        from functions.svd_operators import Deblurring
        A_funcs = Deblurring(torch.Tensor([1 / 9] * 9).to(device), channels,
                             image_size, device)
    elif fn == 'deblur_gauss':
        from functions.svd_operators import Deblurring
        sigma = 10
        pdf = lambda x: torch.exp(torch.Tensor([-0.5 * (x / sigma) ** 2]))
        kernel = torch.Tensor([pdf(-2), pdf(-1), pdf(0), pdf(1), pdf(2)]).to(device)
        A_funcs = Deblurring(kernel / kernel.sum(), channels, image_size, device)
    elif fn == 'deblur_aniso':
        from functions.svd_operators import Deblurring2D
        sigma = 20
        pdf = lambda x: torch.exp(torch.Tensor([-0.5 * (x / sigma) ** 2]))
        kernel2 = torch.Tensor([pdf(-4), pdf(-3), pdf(-2), pdf(-1), pdf(0), pdf(1), pdf(2), pdf(3), pdf(4)]).to(
            device)
        sigma = 1
        pdf = lambda x: torch.exp(torch.Tensor([-0.5 * (x / sigma) ** 2]))
        kernel1 = torch.Tensor([pdf(-4), pdf(-3), pdf(-2), pdf(-1), pdf(0), pdf(1), pdf(2), pdf(3), pdf(4)]).to(
            device)
        A_funcs = Deblurring2D(kernel1 / kernel1.sum(), kernel2 / kernel2.sum(), channels,
                               image_size, device)
    else:
        A_funcs = None
    return A_funcs





def svd_constraint_ddrm(fn,fn_scale=4,device='cpu', base_mask_dir='store/inp_masks', image_size=256,channels=3 ):
    if fn == 'cs_walshhadamard':
        compress_by = round(fn_scale)
        from functions.svd_replacement import WalshHadamardCS
        A_funcs = WalshHadamardCS(channels, image_size, compress_by,
                                  torch.randperm(image_size ** 2, device=device), device)
    elif fn == 'cs_blockbased':
        cs_ratio =fn_scale
        from functions.svd_replacement import CS
        A_funcs = CS(channels, image_size, cs_ratio, device)
    elif 'inpainting' in fn:
        from functions.svd_replacement import Inpainting
        if fn == 'inpainting_lolcat':
            loaded = np.load("inp_masks/lolcat_extra.npy")
            mask = torch.from_numpy(loaded).to(device).reshape(-1)
            missing_r = torch.nonzero(mask == 0).long().reshape(-1) * 3
        elif fn == 'inpainting_lorem':
            loaded = np.load("inp_masks/lorem3.npy")
            mask = torch.from_numpy(loaded).to(device).reshape(-1)
            missing_r = torch.nonzero(mask == 0).long().reshape(-1) * 3
        elif fn == 'inpainting_ddnm':
            loaded = np.load(os.path.join(base_mask_dir,"mask.npy"))
            mask = torch.from_numpy(loaded).to(device).reshape(-1)
            missing_r = torch.nonzero(mask == 0).long().reshape(-1) * 3
        elif fn == 'inpainting_random':
            missing_r = torch.randperm(image_size ** 2)[:image_size ** 2 // 2].to(device).long() * 3
        elif fn == 'inpainting_half':
            loaded = np.load(os.path.join(base_mask_dir,"mask_half.npy"))
            mask = torch.from_numpy(loaded).to(device).reshape(-1)
            missing_r = torch.nonzero(mask == 0).long().reshape(-1) * 3
        else:
            missing_r = torch.load(os.path.join(base_mask_dir, "mask_random.pt")).to(device)
        missing_g = missing_r + 1
        missing_b = missing_g + 1
        missing = torch.cat([missing_r, missing_g, missing_b], dim=0)
        A_funcs = Inpainting(channels, image_size, missing, device)
    elif fn == 'denoising':
        from functions.svd_replacement import Denoising
        A_funcs = Denoising(channels, image_size, device)
    elif fn == 'colorization':
        from functions.svd_replacement import Colorization
        A_funcs = Colorization(image_size, device)
    elif fn == 'sr_averagepooling':
        blur_by = int(fn_scale)
        from functions.svd_replacement import SuperResolution
        A_funcs = SuperResolution(channels, image_size, blur_by, device)
    elif fn == 'sr_bicubic':
        factor = int(fn_scale)
        from functions.svd_replacement import SRConv
        def bicubic_kernel(x, a=-0.5):
            if abs(x) <= 1:
                return (a + 2) * abs(x) ** 3 - (a + 3) * abs(x) ** 2 + 1
            elif 1 < abs(x) and abs(x) < 2:
                return a * abs(x) ** 3 - 5 * a * abs(x) ** 2 + 8 * a * abs(x) - 4 * a
            else:
                return 0

        k = np.zeros((factor * 4))
        for i in range(factor * 4):
            x = (1 / factor) * (i - np.floor(factor * 4 / 2) + 0.5)
            k[i] = bicubic_kernel(x)
        k = k / np.sum(k)
        kernel = torch.from_numpy(k).float().to(device)
        A_funcs = SRConv(kernel / kernel.sum(),  channels, image_size, device, stride=factor)
    elif fn == 'deblur_uni':
        from functions.svd_replacement import Deblurring
        A_funcs = Deblurring(torch.Tensor([1 / 9] * 9).to(device), channels,
                             image_size, device)
    elif fn == 'deblur_gauss':
        from functions.svd_replacement import Deblurring
        sigma = 10
        pdf = lambda x: torch.exp(torch.Tensor([-0.5 * (x / sigma) ** 2]))
        kernel = torch.Tensor([pdf(-2), pdf(-1), pdf(0), pdf(1), pdf(2)]).to(device)
        A_funcs = Deblurring(kernel / kernel.sum(), channels, image_size, device)
    elif fn == 'deblur_aniso':
        from functions.svd_replacement import Deblurring2D
        sigma = 20
        pdf = lambda x: torch.exp(torch.Tensor([-0.5 * (x / sigma) ** 2]))
        kernel2 = torch.Tensor([pdf(-4), pdf(-3), pdf(-2), pdf(-1), pdf(0), pdf(1), pdf(2), pdf(3), pdf(4)]).to(
            device)
        sigma = 1
        pdf = lambda x: torch.exp(torch.Tensor([-0.5 * (x / sigma) ** 2]))
        kernel1 = torch.Tensor([pdf(-4), pdf(-3), pdf(-2), pdf(-1), pdf(0), pdf(1), pdf(2), pdf(3), pdf(4)]).to(
            device)
        A_funcs = Deblurring2D(kernel1 / kernel1.sum(), kernel2 / kernel2.sum(), channels,
                               image_size, device)
    else:
        A_funcs = None
    return A_funcs
