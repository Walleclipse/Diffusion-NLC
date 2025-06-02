import os.path

import torch
import cv2
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
import numpy as np
from scipy.signal import gaussian
from torchvision.utils import save_image
from PIL import Image
import torchvision.transforms as tt
import kornia
def Gaussian_Kernel(k=3, mu=0, sigma=1, normalize=True):
  gaussian_1d = np.linspace(-1, 1, k)

  x, y = np.meshgrid(gaussian_1d, gaussian_1d)
  distance = (x ** 2 + y ** 2) ** 0.5
  gaussian_2d = np.exp(-(distance-mu) ** 2 / (2 * sigma ** 2))
  gaussian_2d = gaussian_2d/(2 * np.pi * sigma ** 2)

  if normalize:
    gaussian_2d = gaussian_2d / np.sum(gaussian_2d)

  return gaussian_2d

def NonMax_Supression(start=0, end=360, step=45):
  k_thin = 3
  k_increased = k_thin + 2

  thin_kernel_0 = np.zeros((k_increased, k_increased))
  thin_kernel_0[k_increased // 2, k_increased // 2] = 1
  thin_kernel_0[k_increased // 2, k_increased // 2 + 1:] = -1

  thin_kernels = []
  for angle in range(start, end, step):
    h, w = thin_kernel_0.shape
    center = (w // 2, h // 2)

    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1)
    kernel_angle_increased = cv2.warpAffine(thin_kernel_0, rotation_matrix, (w, h), cv2.INTER_NEAREST)


    kernel_angle = kernel_angle_increased[1:-1, 1:-1]
    is_diag = (abs(kernel_angle) == 1)
    kernel_angle = kernel_angle * is_diag
    thin_kernels.append(kernel_angle)

  return thin_kernels
def Sobel_Kernel(k=3):
  range = np.linspace(-(k // 2), k // 2, k)

  x, y = np.meshgrid(range, range)
  sobel_2d_num = x
  sobel_2d_den = (x ** 2 + y ** 2)
  sobel_2d_den[:, k // 2] = 1
  sobel_2d = sobel_2d_num / sobel_2d_den

  return sobel_2d
class CannyFilter(nn.Module):
    def __init__(self,
                 k_gaussian=3,
                 mu=0,
                 sigma=1,
                 k_sobel=3,
                 use_cuda=False):
        super(CannyFilter, self).__init__()
        # device
        self.device = 'cuda' if use_cuda else 'cpu'

        # gaussian

        gaussian_2D = Gaussian_Kernel(k_gaussian, mu, sigma)
        self.gaussian_filter = nn.Conv2d(in_channels=1,
                                         out_channels=1,
                                         kernel_size=k_gaussian,
                                         padding=k_gaussian // 2,
                                         bias=False)
        self.gaussian_filter.weight[:] = torch.from_numpy(gaussian_2D)

        # sobel

        sobel_2D = Sobel_Kernel(k_sobel)
        self.sobel_filter_x = nn.Conv2d(in_channels=1,
                                        out_channels=1,
                                        kernel_size=k_sobel,
                                        padding=k_sobel // 2,
                                        bias=False)
        self.sobel_filter_x.weight[:] = torch.from_numpy(sobel_2D)


        self.sobel_filter_y = nn.Conv2d(in_channels=1,
                                        out_channels=1,
                                        kernel_size=k_sobel,
                                        padding=k_sobel // 2,
                                        bias=False)
        self.sobel_filter_y.weight[:] = torch.from_numpy(sobel_2D.T)


        # thin

        thin_kernels = NonMax_Supression()
        directional_kernels = np.stack(thin_kernels)

        self.directional_filter = nn.Conv2d(in_channels=1,
                                            out_channels=8,
                                            kernel_size=thin_kernels[0].shape,
                                            padding=thin_kernels[0].shape[-1] // 2,
                                            bias=False)
        self.directional_filter.weight[:, 0] = torch.from_numpy(directional_kernels)

        # hysteresis

        hysteresis = np.ones((3, 3)) + 0.25
        self.hysteresis = nn.Conv2d(in_channels=1,
                                    out_channels=1,
                                    kernel_size=3,
                                    padding=1,
                                    bias=False)
        self.hysteresis.weight[:] = torch.from_numpy(hysteresis)

    def forward(self, img, low_threshold=None, high_threshold=None, hysteresis=False):
        # set the setps tensors
        B, C, H, W = img.shape
        blurred = torch.zeros((B, C, H, W)).to(self.device)
        grad_x = torch.zeros((B, 1, H, W)).to(self.device)
        grad_y = torch.zeros((B, 1, H, W)).to(self.device)
        grad_magnitude = torch.zeros((B, 1, H, W)).to(self.device)
        grad_orientation = torch.zeros((B, 1, H, W)).to(self.device)

        # gaussian

        for c in range(C):
            blurred[:, c:c + 1] = self.gaussian_filter(img[:, c:c + 1])

            grad_x = grad_x + self.sobel_filter_x(blurred[:, c:c + 1])
            grad_y = grad_y + self.sobel_filter_y(blurred[:, c:c + 1])

        # thick edges

        grad_x, grad_y = grad_x / C, grad_y / C
        grad_magnitude = (grad_x ** 2 + grad_y ** 2) ** 0.5
        grad_orientation = torch.atan(grad_y / grad_x)
        grad_orientation = grad_orientation * (360 / np.pi) + 180  # convert to degree
        grad_orientation = torch.round(grad_orientation / 45) * 45  # keep a split by 45

        # thin edges

        directional = self.directional_filter(grad_magnitude)
        # get indices of positive and negative directions
        positive_idx = (grad_orientation / 45) % 8
        negative_idx = ((grad_orientation / 45) + 4) % 8
        thin_edges = grad_magnitude.clone()
        # non maximum suppression direction by direction
        for pos_i in range(4):
            neg_i = pos_i + 4
            # get the oriented grad for the angle
            is_oriented_i = (positive_idx == pos_i) * 1
            is_oriented_i = is_oriented_i + (positive_idx == neg_i) * 1
            pos_directional = directional[:, pos_i]
            neg_directional = directional[:, neg_i]
            selected_direction = torch.stack([pos_directional, neg_directional])

            # get the local maximum pixels for the angle
            is_max = selected_direction.min(dim=0)[0] > 0.0
            is_max = torch.unsqueeze(is_max, dim=1)

            # apply non maximum suppression
            to_remove = (is_max == 0) * 1 * (is_oriented_i) > 0
            thin_edges[to_remove] = 0.0

        # thresholds

        if low_threshold is not None:
            low = thin_edges > low_threshold

            if high_threshold is not None:
                high = thin_edges > high_threshold
                # get black/gray/white only
                thin_edges = low * 0.5 + high * 0.5

                if hysteresis:
                    # get weaks and check if they are high or not
                    weak = (thin_edges == 0.5) * 1
                    weak_is_high = (self.hysteresis(thin_edges) > 1) * weak
                    thin_edges = high * 1 + weak_is_high * 1
            else:
                thin_edges = low * 1

        return blurred, grad_x, grad_y, grad_magnitude, grad_orientation, thin_edges


class Net(nn.Module):
    def __init__(self, threshold=10.0):
        super(Net, self).__init__()

        self.threshold = threshold

        filter_size = 5
        generated_filters = gaussian(filter_size,std=1.0).reshape([1,filter_size])

        self.gaussian_filter_horizontal = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1,filter_size), padding=(0,filter_size//2))
        self.gaussian_filter_horizontal.weight.data.copy_(torch.from_numpy(generated_filters))
        self.gaussian_filter_horizontal.bias.data.copy_(torch.from_numpy(np.array([0.0])))
        self.gaussian_filter_vertical = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(filter_size,1), padding=(filter_size//2,0))
        self.gaussian_filter_vertical.weight.data.copy_(torch.from_numpy(generated_filters.T))
        self.gaussian_filter_vertical.bias.data.copy_(torch.from_numpy(np.array([0.0])))

        sobel_filter = np.array([[1, 0, -1],
                                 [2, 0, -2],
                                 [1, 0, -1]])

        self.sobel_filter_horizontal = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=sobel_filter.shape, padding=sobel_filter.shape[0]//2)
        self.sobel_filter_horizontal.weight.data.copy_(torch.from_numpy(sobel_filter))
        self.sobel_filter_horizontal.bias.data.copy_(torch.from_numpy(np.array([0.0])))
        self.sobel_filter_vertical = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=sobel_filter.shape, padding=sobel_filter.shape[0]//2)
        self.sobel_filter_vertical.weight.data.copy_(torch.from_numpy(sobel_filter.T))
        self.sobel_filter_vertical.bias.data.copy_(torch.from_numpy(np.array([0.0])))

        # filters were flipped manually
        filter_0 = np.array([   [ 0, 0, 0],
                                [ 0, 1, -1],
                                [ 0, 0, 0]])

        filter_45 = np.array([  [0, 0, 0],
                                [ 0, 1, 0],
                                [ 0, 0, -1]])

        filter_90 = np.array([  [ 0, 0, 0],
                                [ 0, 1, 0],
                                [ 0,-1, 0]])

        filter_135 = np.array([ [ 0, 0, 0],
                                [ 0, 1, 0],
                                [-1, 0, 0]])

        filter_180 = np.array([ [ 0, 0, 0],
                                [-1, 1, 0],
                                [ 0, 0, 0]])

        filter_225 = np.array([ [-1, 0, 0],
                                [ 0, 1, 0],
                                [ 0, 0, 0]])

        filter_270 = np.array([ [ 0,-1, 0],
                                [ 0, 1, 0],
                                [ 0, 0, 0]])

        filter_315 = np.array([ [ 0, 0, -1],
                                [ 0, 1, 0],
                                [ 0, 0, 0]])

        all_filters = np.stack([filter_0, filter_45, filter_90, filter_135, filter_180, filter_225, filter_270, filter_315])

        self.directional_filter = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=filter_0.shape, padding=filter_0.shape[-1] // 2)
        self.directional_filter.weight.data.copy_(torch.from_numpy(all_filters[:, None, ...]))
        self.directional_filter.bias.data.copy_(torch.from_numpy(np.zeros(shape=(all_filters.shape[0],))))

    def forward(self, img):
        # img_r = img[:,0:1]
        # img_g = img[:,1:2]
        # img_b = img[:,2:3]
        #
        # blur_horizontal = self.gaussian_filter_horizontal(img_r)
        # blurred_img_r = self.gaussian_filter_vertical(blur_horizontal)
        # blur_horizontal = self.gaussian_filter_horizontal(img_g)
        # blurred_img_g = self.gaussian_filter_vertical(blur_horizontal)
        # blur_horizontal = self.gaussian_filter_horizontal(img_b)
        # blurred_img_b = self.gaussian_filter_vertical(blur_horizontal)
        batch_size = img.shape[0]
        blurred_img_r = self.gaussian_filter_vertical(self.gaussian_filter_horizontal(img[:, 0:1]))
        blurred_img_g = self.gaussian_filter_vertical(self.gaussian_filter_horizontal(img[:, 1:2]))
        blurred_img_b = self.gaussian_filter_vertical(self.gaussian_filter_horizontal(img[:, 2:3]))

        blurred_img = torch.stack([blurred_img_r,blurred_img_g,blurred_img_b],dim=1)
        blurred_img = torch.stack([torch.squeeze(blurred_img)])

        grad_x_r = self.sobel_filter_horizontal(blurred_img_r)
        grad_y_r = self.sobel_filter_vertical(blurred_img_r)
        grad_x_g = self.sobel_filter_horizontal(blurred_img_g)
        grad_y_g = self.sobel_filter_vertical(blurred_img_g)
        grad_x_b = self.sobel_filter_horizontal(blurred_img_b)
        grad_y_b = self.sobel_filter_vertical(blurred_img_b)

        # COMPUTE THICK EDGES

        #grad_mag = torch.sqrt(grad_x_r**2 + grad_y_r**2)
        #grad_mag += torch.sqrt(grad_x_g**2 + grad_y_g**2)
        #grad_mag += torch.sqrt(grad_x_b**2 + grad_y_b**2)
        grad_mag = torch.sqrt(grad_x_r ** 2 + grad_y_r ** 2) + torch.sqrt(grad_x_g ** 2 + grad_y_g ** 2) + torch.sqrt(
            grad_x_b ** 2 + grad_y_b ** 2)
        grad_orientation = (torch.atan2(grad_y_r+grad_y_g+grad_y_b, grad_x_r+grad_x_g+grad_x_b) * (180.0/3.14159)) + 180
        grad_orientation =  torch.round( grad_orientation / 45.0 ) * 45.0

        # THIN EDGES (NON-MAX SUPPRESSION)

        all_filtered = self.directional_filter(grad_mag)

        inidices_positive = (grad_orientation / 45) % 8
        inidices_negative = ((grad_orientation / 45) + 4) % 8

        height = inidices_positive.size()[2]
        width = inidices_positive.size()[3]
        pixel_count = height * width
        pixel_range = torch.FloatTensor([range(pixel_count)]).to(img.device)

        indices_pos = (inidices_positive.view(-1).data * pixel_count + pixel_range).squeeze()
        channel_select_filtered_positive = all_filtered.view(-1)[indices_pos.long()].view(1, height, width)

        indices_neg = (inidices_negative.view(-1).data * pixel_count + pixel_range).squeeze()
        channel_select_filtered_negative = all_filtered.view(-1)[indices_neg.long()].view(1, height, width)

        channel_select_filtered = torch.stack([channel_select_filtered_positive,channel_select_filtered_negative])

        is_max = channel_select_filtered.min(dim=0)[0] > 0.0
        is_max = torch.unsqueeze(is_max, dim=0)

        thin_edges = grad_mag.clone()
        thin_edges[is_max==0] = 0.0

        # THRESHOLD

        thresholded = thin_edges.clone()
        thresholded[thin_edges<self.threshold] = 0.0

        early_threshold = grad_mag.clone()
        early_threshold[grad_mag<self.threshold] = 0.0

        assert grad_mag.size() == grad_orientation.size() == thin_edges.size() == thresholded.size() == early_threshold.size()

        return blurred_img, grad_mag, grad_orientation, thin_edges, thresholded, early_threshold

class DifferentiableCanny(nn.Module):
    def __init__(self, sigma=1.0, threshold=0.2,filter_size=5):
        super(DifferentiableCanny, self).__init__()
        self.sigma = sigma
        self.threshold = threshold
        self.filter_size = filter_size

    def forward(self, x):
        # Apply Gaussian smoothing to the input image
        #x = F.gaussian_blur(x, self.sigma)
        x = torchvision.transforms.functional.gaussian_blur(x,kernel_size=(self.filter_size,self.filter_size), sigma=self.sigma)
        # Calculate gradients in x and y directions
        gradient_x = F.conv2d(x, torch.Tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).view(1, 1, 3, 3).to(x.device))
        gradient_y = F.conv2d(x, torch.Tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).view(1, 1, 3, 3).to(x.device))

        # Compute gradient magnitude
        gradient_mag = torch.sqrt(gradient_x ** 2 + gradient_y ** 2)

        # Apply thresholding to obtain binary edges
        edges = gradient_mag > self.threshold

        return edges.float()

class EdgeDetector(nn.Module):
    r"""Detect edges in a given image using a CNN.

    By default, it uses the method described in :cite:`xsoria2020dexined`.

    Return:
        A tensor of shape :math:`(B,1,H,W)`.

    """

    def __init__(self) -> None:
        super().__init__()
        self.model = kornia.filters.DexiNed(pretrained=False)

    def load(self, path_file: str) -> None:
        #self.model.load_from_file(path_file)
        ckpt = torch.load(path_file)
        self.model.load_state_dict(ckpt, strict=True)
        self.model.eval()

    def preprocess(self, image):
        return image

    def postprocess(self, data):
        # input are intermediate layer -- for inference we need only last.
        return data[-1]  # Bx1xHxW

    def forward(self, image) :
        #KORNIA_CHECK_SHAPE(image, ["B", "3", "H", "W"])
        img = self.preprocess(image)
        out = self.model(img)
        return self.postprocess(out)
def canny(raw_img,out_path, use_cuda=False):
    img = torch.from_numpy(raw_img.transpose((2, 0, 1)))
    batch = torch.stack([img]).float()

    net = Net(threshold=2.0)
    #net = CannyFilter()
    if use_cuda:
        net.cuda()
    net.eval()

    data = Variable(batch)
    if use_cuda:
        data = Variable(batch).cuda()
    data = data.clone().detach().requires_grad_(True)
    blurred_img, grad_mag, grad_orientation, thin_edges, thresholded, early_threshold = net(data)
    y = torch.sigmoid(thresholded) * 2 - 1
    bsz=len(blurred_img)
    for j in range(bsz):
        save_image(grad_mag[j], out_path+f'gradient_magnitude_{j}.png')
        save_image(thresholded[j], out_path + f'thin_edges_{j}.png')
        save_image((thresholded[j]> 0.0).to(torch.float32), out_path + f'final_{j}.png')
        save_image(early_threshold[j], out_path + f'thresholded_{j}.png')
        save_image(blurred_img[j], out_path + f'blurred_img_{j}.png')
        save_image(grad_orientation[j], out_path + f'grad_orientation_{j}.png')
        save_image(thin_edges[j], out_path + f'thin_edges_{j}.png')
        save_image(y[j], out_path + f'y_edges_{j}.png')
    data = data.clone().detach().requires_grad_(True)

    low_threshold = 0.1
    high_threshold = 0.2
    canny_fn = kornia.filters.Canny(low_threshold=low_threshold,high_threshold=high_threshold)
    magnitude, edges = canny_fn(data)
    magnitude = (magnitude-low_threshold)/(high_threshold-low_threshold)
    y = torch.sigmoid(magnitude)*2-1
    y = torch.clamp(y, min=0,max=1)
    for j in range(bsz):
        save_image(edges[j], out_path+f'canny_edge{j}.png')
        save_image(edges[j], out_path + f'canny_magnitude{j}.png')
        save_image(y[j], out_path + f'canny_y{j}.png')
    net = kornia.filters.DexiNed(pretrained=True)
    # sobel_fn = kornia.filters.Sobel()
    # data_g = kornia.color.bgr_to_grayscale(data)
    # sob = sobel_fn(data_g)
    # #sob =1 - sob
    # for j in range(bsz):
    #     save_image(sob[j], out_path + f'sobel_edge{j}.png')
    #
    # detect = EdgeDetector() #kornia.contrib.EdgeDetector()
    # detect.load('../store/helper/10_model.pth')
    # edge = detect(data)
    # for j in range(bsz):
    #     save_image(edge[j], out_path+f'edge_kornia_{j}.png')
    print('done')


class MidasDetector:
    def __init__(self):
        self.model = MiDaSInference(model_type="dpt_hybrid").cuda()

    def __call__(self, input_image, a=np.pi * 2.0, bg_th=0.1):
        assert input_image.ndim == 3
        image_depth = input_image
        with torch.no_grad():
            image_depth = torch.from_numpy(image_depth).float().cuda()
            image_depth = image_depth / 127.5 - 1.0
            image_depth = rearrange(image_depth, 'h w c -> 1 c h w')
            depth = self.model(image_depth)[0]

            depth_pt = depth.clone()
            depth_pt -= torch.min(depth_pt)
            depth_pt /= torch.max(depth_pt)
            depth_pt = depth_pt.cpu().numpy()
            depth_image = (depth_pt * 255.0).clip(0, 255).astype(np.uint8)

            depth_np = depth.cpu().numpy()
            x = cv2.Sobel(depth_np, cv2.CV_32F, 1, 0, ksize=3)
            y = cv2.Sobel(depth_np, cv2.CV_32F, 0, 1, ksize=3)
            z = np.ones_like(x) * a
            x[depth_pt < bg_th] = 0
            y[depth_pt < bg_th] = 0
            normal = np.stack([x, y, z], axis=2)
            normal /= np.sum(normal ** 2.0, axis=2, keepdims=True) ** 0.5
            normal_image = (normal * 127.5 + 127.5).clip(0, 255).astype(np.uint8)

            return depth_image, normal_image


if __name__ == '__main__':
    for k in ['1','2','3']:
        pil_image = Image.open(f'../store/temp/{k}.JPEG')
        pil_image.load()
        pil_image = pil_image.convert("RGB")
        img =  np.array(pil_image).astype(np.float32) / 255.0
        # canny(img, use_cuda=False)
        canny(img,out_path=f'../store/temp/{k}_')




