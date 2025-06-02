import os
import argparse
import json
import random
import numpy as np
import pandas as pd
import torch
import yaml
import torch.distributed as dist
#from torchsummary import summary
import pickle

from src.schedulers import get_sampler
from src.script_util import create_edm_sigma_eps_model
from src.image_dataset import load_data
from src.experiments import EDMImageExperiment
from src.utils import get_model_size
from src import logger, dist_util
import dnnlib



if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
# only for training sigma now
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default='ffhq',choices=['cifar10','ffhq'])

    parser.add_argument("--num_timesteps", type=int, default=40)
    parser.add_argument("--start_sigma", type=float, default=80)
    parser.add_argument("--end_sigma", type=float, default=0.002)
    parser.add_argument("--refine_sigma", type=int, default=0)

    parser.add_argument("--sigma_loss", type=str, default='mse', choices=['mae','mse','huber'])
    parser.add_argument("--sigma_block", type=int, default=2)
    parser.add_argument("--sigma_dropout", type=float, default=0.1)
    parser.add_argument("--use_sigma_fp16", type=int, default=0) #feat_layer
    parser.add_argument("--feat_layer", type=int, default=1)

    parser.add_argument("--lr", type=float, default=3e-4) # 1e-3
    parser.add_argument("--weight_decay", type=float, default=0.0) #ema_rate
    parser.add_argument("--ema_rate", type=float, default=0.999)
    parser.add_argument("--batch_size", type=int, default=128) # cifar 10:128, imagenet: 64
    parser.add_argument("--microbatch", type=int, default=128) # cifar 10:128, imagenet: 16, celeba-hq 32, celeba 64
    parser.add_argument("--epoch", type=int, default=500)
    parser.add_argument("--iter_per_epoch", type=int, default=500) # 500
    parser.add_argument("--eval_per_epoch", type=int, default=100) # 100
    parser.add_argument("--save_per_epoch", type=int, default=100)

    parser.add_argument("--eta1_min", type=float, default=-0.5) # -0.5
    parser.add_argument("--eta1_scale", type=float, default=1) # 1.0
    parser.add_argument("--eta2_min", type=float, default=0.) # 0
    parser.add_argument("--eta2_scale", type=float, default=0.) # 0
    parser.add_argument("--sigma_sampler", type=str, default='edm',choices=['edm','random']) #'results'
    parser.add_argument("--loss_weighted", type=int, default=1)

    parser.add_argument("--result_dir", type=str, default='results',) #'results'
    parser.add_argument("--device", type=str, default='cuda:0') # 'cpu, cuda:0
    parser.add_argument("--dist_train", type=int, default=1) #n_dp
    parser.add_argument("--seed", type=int, default=2023)

    parser.add_argument("--load_folder", type=int, default=None)
    parser.add_argument("--load_eps", type=str, default=None)
    parser.add_argument("--fid_target", type=str, default=None)

    parser.add_argument("--resume_model", type=str, default=None) #None
    parser.add_argument("--resume_ema_model", type=str, default=None)
    parser.add_argument("--resume_optim", type=str, default=None)

    args = parser.parse_args()
    args.result_dir = os.path.join(args.result_dir,args.config)
    args.root_dir = args.result_dir

    with open(os.path.join("store","config", args.config+'.yml'), "r") as f:
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
    new_config.model.feat_layer = args.feat_layer

    def get_default(args):
        if args.config == 'cifar10':
            args.load_eps = 'store/models/edm-cifar10-32x32-uncond-vp.pkl' # 'store/models/ADM_IP_015.pt'
            args.fid_target = 'store/fid/cifar10-32x32.npz'
            args.norm_max =54.70 #54.63
            args.norm_min = 0.0
            #args.n_train_samples = 50000
            #args.sigma_block = 2
        elif args.config == 'ffhq':
            args.load_eps = 'store/models/edm-ffhq-64x64-uncond-vp.pkl'
            args.fid_target = 'store/fid/ffhq-64x64.npz'
            args.norm_max = 102.0  # 102.0 110
            args.norm_min = 0  # 0
        else:
            raise NotImplementedError
        return args

    args = get_default(args)
    return args,new_config


def main(args, config):
    logger.configure(dir='./logs/')
    if args.dist_train:
        begin_device = int(args.device.replace('cuda:',''))
        dist_util.setup_dist(begin_device)
        args.device = dist_util.dev()

    root_dir = args.root_dir
    if args.load_folder is None:
        i=0
        save_dir=os.path.join(root_dir,str(i))
        while os.path.exists(save_dir):
            i+=1
            save_dir = os.path.join(root_dir, str(i))
        if dist.get_rank()==0:
            args.result_dir = save_dir
            os.makedirs(args.result_dir, exist_ok=True)
            with open(os.path.join(args.result_dir, 'args.json'), 'w') as f:
                arg_dict = args.__dict__
                arg_dict['device'] = str(arg_dict['device'])
                json.dump(arg_dict, f)
        else:
            i = i - dist.get_rank()
            save_dir = os.path.join(root_dir, str(i))
            args.result_dir = save_dir
    else:
        args.result_dir =os.path.join(root_dir, args.load_folder)

    print('args:',args)
    print('config:', config)

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    # data
    data_config = config.data
    data = load_data(data_dir=data_config.data_dir, batch_size=args.batch_size, image_size=data_config.image_size,
                     class_cond=False,
                     deterministic=False, random_flip=data_config.random_flip, dist_train=args.dist_train)

    # model
    model_config = config.model
    model, sigma_model, feat_shape = create_edm_sigma_eps_model(**vars(model_config))
    # print('eps model:')
    # print(summary(model, input_size=(data_config.channels,data_config.image_size,data_config.image_size)))
    # print('sigma model:')
    # print(summary(sigma_model, input_size=feat_shape))
    print('eps model size:', get_model_size(model))
    print('sigma model size:', get_model_size(sigma_model))


    with dnnlib.util.open_url(args.load_eps, verbose=True) as f:
        saved_eps = pickle.load(f)['ema']
    ckpt = saved_eps.model.state_dict()
    model.load_state_dict(ckpt)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    if args.resume_model and os.path.exists(args.resume_model):
        #ckpt = torch.load(args.resume_model, map_location='cpu')
        ckpt = dist_util.load_state_dict(
                    args.resume_model, map_location='cpu'
                )
        #ckpt = torch.load(ckpt_file, map_location='cpu')
        sigma_model.load_state_dict(ckpt)

        print('resume sigma model from', args.resume_model)

    model.to(args.device)
    sigma_model.to(args.device)
    dist_util.sync_params(model.parameters())
    dist_util.sync_params(sigma_model.parameters())

    if model_config.use_fp16:
        model.convert_to_fp16()
    if model_config.use_sigma_fp16:
        sigma_model.convert_to_fp16()



    # experiment
    experiment = EDMImageExperiment(model, scheduler=None, batch_size=args.batch_size, data_shape=(data_config.channels,data_config.image_size,data_config.image_size),seed=args.seed,  device=args.device,
                                 save_folder=args.result_dir,dist_train=args.dist_train,
                                    num_timesteps = args.num_timesteps)
    experiment.set_model(model, sigma_model,learn_epsvar=False)
    experiment.set_optimizers(args.lr, args.sigma_loss,
                              weight_decay=args.weight_decay, use_fp16= args.use_sigma_fp16, ema_rate=args.ema_rate,
                              resume_ema_model=args.resume_ema_model, resume_optim=args.resume_optim,reduction='none')
    experiment.set_dataset(data, fid_target=args.fid_target, norm_max = args.norm_max, norm_min = args.norm_min)
    experiment.set_perturb_coefficient(args.eta1_min, args.eta1_scale, args.eta2_min, args.eta2_scale)


    # train
    torch.cuda.empty_cache()
    experiment.train_edm(epoch=args.epoch,iter_per_epoch=args.iter_per_epoch, eval_per_epoch=args.eval_per_epoch, save_per_epoch=args.save_per_epoch,
                         sigma_sampler =args.sigma_sampler, loss_weighted=args.loss_weighted)
    print('train done')
    torch.cuda.empty_cache()


if __name__=='__main__':
    args, config=get_args()
    main(args, config)