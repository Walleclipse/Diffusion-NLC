import os
import shutil
import argparse
import json
import random
import math
import numpy as np
import yaml
import torch
import pickle

from src.script_util import create_edm_sigma_eps_model
from src.experiments import EDMImageExperiment
from src.utils import get_model_size
from src import logger
import dnnlib


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default='cifar10',choices=['cifar10','ffhq'])

    parser.add_argument("--sampler", type=str, default='edm',choices=['edm', 'ddim','euler'])
    parser.add_argument("--sigma_type", type=str, default='pred_partial,pred') # base pred pred_partial pred_partial2 pred_partial3 pred_sigma
    parser.add_argument("--norm_eps", type=str, default='00')
    parser.add_argument("--num_timesteps", type=int, default=49) # cifar 18 ffhq 40
    parser.add_argument("--start_sigma", type=float, default=80) # 143.0 150
    parser.add_argument("--end_sigma", type=float, default=0.002) # 0.01
    parser.add_argument("--sigma_data", type=float, default=0.5)
    parser.add_argument("--sigma_style", type=str, default='EDM',choices=['Linear','EDM',])
    parser.add_argument("--eps_ratio", type=float, default=0.5)
    parser.add_argument("--eps_scale", type=float, default=1.0) # 1.01 None 1.004
    parser.add_argument("--eta", type=float, default=1.0) # 0.85
    parser.add_argument("--refine_sigma", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=200) # cifar 10:500, imagenet: 10
    parser.add_argument("--device", type=str, default='cuda:5') # 'cpu, cuda:0
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--result_dir", type=str, default='results',) #'results'
    parser.add_argument("--test_dir", type=str, default='temp',) #'test_result' 'temp'
    parser.add_argument("--sample_size", type=int, default=5000) # 50000, 100


    parser.add_argument("--save_folder", type=str, default=None) # 50000, 100
    parser.add_argument("--save_flag", type=str, default='0')

    parser.add_argument("--sample_overwrite", type=int, default=0)
    parser.add_argument("--load_folder", type=str, default='6') # None, cifar10:'13', imagenet: '0', celeba '0'  ffhq 6
    parser.add_argument("--load_eps", type=str, default=None)
    parser.add_argument("--load_sigma", type=str, default='results/ffhq/6/ema_sigma_ckpt_100.pt') # results/celeba/0/ema_sigma_ckpt_299.pt #'results/ffhq/6/ema_sigma_ckpt_100.pt' # 'results/cifar10/13/ema_sigma_ckpt_499.pt'
    parser.add_argument("--fid_target", type=str, default=None)

    #parser.add_argument("--method", type=str, default='pred_denoise_base',choices=['default','base','pred_denoise_base','pred_denoise_proj','pred_denoise_proj_arbit','pred_proj'])

    args = parser.parse_args()

    args.result_dir = os.path.join(args.result_dir,args.config)
    args.root_dir = args.result_dir
    args.result_dir = os.path.join(args.root_dir, args.load_folder)
    args.test_dir = os.path.join(args.test_dir, args.config)

    with open(os.path.join(args.result_dir,'args.json'), "r") as f:
        saved_args = json.load(f)
    args.load_eps = saved_args['load_eps']
    args.fid_target = saved_args['fid_target']
    args.sigma_block =  saved_args['sigma_block']
    args.sigma_dropout = saved_args['sigma_dropout']
    args.use_sigma_fp16 = saved_args['use_sigma_fp16']

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
    new_config.model.feat_layer = saved_args['feat_layer']

    def get_default(args):
        if args.config == 'cifar10':
            args.norm_max = 54.63 # 54.63  # fid 15.961
            args.norm_min = 0 # 0
            # args.load_folder = '13'  # or '8'  # base 6.628 5-150 6.856  13-240 6.852 , 13-320 6.8166 13-380 6.8067 13-420 6.838, 13-499 6.76697
            # args.load_sigma = 'results/cifar10/13/ema_sigma_ckpt_499.pt' # or 5/ema_sigma_ckpt_150
        elif args.config == 'ffhq':
            args.load_eps = 'store/models/edm-ffhq-64x64-uncond-vp.pkl'
            args.fid_target = 'store/fid/ffhq-64x64.npz'
            args.norm_max = 102.0  # 102.0 110
            args.norm_min = 0  # 0
            # args.load_folder = '6'  # base 7.3155,  6-260 7.2657 7-440 7.2870
            # args.load_sigma = 'results/ffhq/6/ema_sigma_ckpt_260.pt' # or 7/ema_sigma_ckpt_440
        else:
            args.norm_max = None
            args.norm_min = None

        return args
    args = get_default(args)
    return args,new_config


def main(args, config):
    logger.configure(dir='./logs/')

    if args.save_folder is not None :
        args.test_dir = args.save_folder
        print('save folder', args.save_folder)
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
    model, sigma_model, feat_shape = create_edm_sigma_eps_model(**vars(model_config))
    print('eps model size:', get_model_size(model))
    print('sigma model size:', get_model_size(sigma_model))

    with dnnlib.util.open_url(args.load_eps, verbose=True) as f:
        saved_eps = pickle.load(f)['ema']
    print('load eps model from', args.load_eps)
    ckpt = saved_eps.model.state_dict()
    model.load_state_dict(ckpt)
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

    #model = torch.nn.DataParallel(model)
    #sigma_model = torch.nn.DataParallel(sigma_model)

    # experiment
    data_config=config.data
    experiment = EDMImageExperiment(model,scheduler=None, batch_size=args.batch_size, data_shape=(data_config.channels,data_config.image_size,data_config.image_size),seed=args.seed,  device=args.device,
                                 save_folder=args.test_dir,dist_train=False,  num_timesteps = args.num_timesteps,
                                    sigma_min=args.end_sigma, sigma_max=args.start_sigma, sigma_data=args.sigma_data,
                                    )
    experiment.set_model(model, sigma_model,learn_epsvar= False)
    experiment.fid_helper(args.fid_target)
    experiment.set_norm_maxmin(args.norm_min, args.norm_max)

    # evaluate
    images_dir = os.path.join(args.test_dir,args.save_flag,'images')
    if os.path.exists(images_dir):
        if args.sample_overwrite:
            shutil.rmtree(images_dir)
    os.makedirs(images_dir, exist_ok=True)
    gen = experiment.new_gen()
    log_dict = experiment.evaluate_edm(args.sample_size, images_dir, gen=gen,
                                 style=args.sigma_type, norm_eps=args.norm_eps, refine_prior_sigma=args.refine_sigma,
                                       sigma_scheduler=args.sigma_style,eps_ratio=args.eps_ratio,eps_scale=args.eps_scale,
                                       use_second_order=args.sampler =='edm')
    with open(os.path.join(args.test_dir,args.save_flag, 'results.json'), 'w') as f:
        json.dump(log_dict, f)
    print(log_dict)
    print('evaluate done')
    #torch.cuda.empty_cache()

if __name__=='__main__':
    args, config=get_args()
    main(args, config)