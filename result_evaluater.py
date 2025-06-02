import os
import json
import argparse
import numpy as np
import pandas as pd
import torch
from torchvision.io import read_image
from torchvision.transforms.functional import rgb_to_grayscale
# from torchmetrics.image import StructuralSimilarityIndexMeasure
# from pytorch_msssim import SSIM
# from skimage.metrics import structural_similarity
from basicsr.metrics.psnr_ssim import calculate_ssim

from pytorch_fid.fid_score import calculate_fid_given_paths



parser = argparse.ArgumentParser()
parser.add_argument("--base_dir", type=str, default='results_final_edm')
parser.add_argument("--device", type=str, default='cuda:1')
args = parser.parse_args()
DEVICE = args.device

def evaluate_fid(img_path, fid_target, batch_size=128, dims=2048, device='cpu'):
	paths = [img_path, fid_target]
	fid = calculate_fid_given_paths(paths, batch_size, device, dims, num_workers=1)
	return fid

def evaluate_psnr_ssim(img_path):
	path_list = sorted(os.listdir(img_path))
	path_list = [os.path.join(img_path, p) for p in path_list]
	psnr_list = []
	ssim_list = []
	# ssim_fn = StructuralSimilarityIndexMeasure(data_range=1.0)
	for p in path_list:
		sr = read_image(p)
		ref_p = p.replace('images/', 'transform/orig_')
		if not os.path.exists(ref_p):
			temp_dir = ref_p.split('/')
			temp_dir[-5] = 'inpainting'
			temp_dir[-4] = '0'
			temp_dir[-3] = '0'
			ref_p = '/'.join(temp_dir)
		gt = read_image(ref_p)
		ssim = calculate_ssim(sr, gt, crop_border=0, test_y_channel=False)
		gen_img = sr/255.0
		ref_img = gt/255.0
		mse = torch.mean((gen_img - ref_img) ** 2)
		psnr = 10 * torch.log10(1 / mse)
		#ssim = ssim_fn(img1, img2)
		# img1 = gen_img.numpy()
		# img2 = ref_img.numpy()
		# ssim = structural_similarity(img1, img2, data_range=img2.max() -img2.min(), channel_axis=0,win_size=11 ) #
		psnr_list.append(psnr.item())
		ssim_list.append(ssim)
	return psnr_list, ssim_list


def evaluate(result_dir, device='cuda:0'):
	with open(os.path.join(result_dir, 'args.json'), "r") as f:
		saved_args = json.load(f)
	fid_target = saved_args['fid_target']
	if 'constraint' in saved_args:
		costraint = saved_args['constraint']
		constraint_scale= str(saved_args['constraint_scale'])
	else:
		costraint='none'
		constraint_scale=''
	if 'method' in saved_args:
		method = saved_args['method']
	else:
		method='default'
	result_list = []
	for ids in os.listdir(result_dir):
		img_path = os.path.join(result_dir, ids, 'images')
		if os.path.exists(img_path):
			print('------- evaluation on', img_path, '-------')
			result = {'data': saved_args['config'],
			          'constraint': costraint + constraint_scale,
			          'method': method, 'path': img_path,
			          'fid': -1, 'psnr': -1, 'ssim': -1, }
			print(f"method={result['method']},  data= {result['data']} with {result['constraint']}")
			result['n_samples'] = len(os.listdir(img_path))
			result['fid'] = evaluate_fid(img_path, fid_target, batch_size=128, dims=2048, device=device)
			if costraint and costraint!='none':
				psnr_list, ssim_list = evaluate_psnr_ssim(img_path)
				result['psnr'] = np.mean(psnr_list)
				result['ssim'] = np.mean(ssim_list)
				result['psnr_list'] = psnr_list
				result['ssim_list'] = ssim_list
			print(f"fid={round(result['fid'], 3)}, psnr={round(result['psnr'], 3)}, ssim={round(result['ssim'], 3)}")
			with open(os.path.join(result_dir, ids, 'img_results.json'), 'w') as f:
				json.dump(result, f)
			result_list.append(result)
	return result_list


def main(base_dir='results_final', out_path='full_result_list', device=DEVICE):
	full_result_list = []
	for data in os.listdir(base_dir):
		data_dir = os.path.join(base_dir, data)
		if not os.path.exists(data_dir):
			continue
		for const in os.listdir(data_dir):
			const_dir = os.path.join(data_dir, const)
			print('---------------------------------------------')
			print(f'Evaluation on {data} with constraint {const}')
			if 'args.json' in os.listdir(const_dir):
				try:
					result_list = evaluate(const_dir, device=device)
					full_result_list += result_list
				except:
					print('!!!! Error in', const_dir)
					pass
			else:
				for ids in os.listdir(const_dir):
					result_dir = os.path.join(const_dir, ids)
					try:
						result_list = evaluate(result_dir, device=device)
						full_result_list += result_list
					except:
						print('!!!! Error in', result_dir)
						pass
	with open(os.path.join('results_record', out_path + '.json'), 'w') as f:
		json.dump(full_result_list, f)
	df_result_list = []
	for res in full_result_list:
		if 'psnr_list' in res:
			del res['psnr_list']
		if 'ssim_list' in res:
			del res['ssim_list']
		df_result_list.append(res)
	df = pd.DataFrame(df_result_list)
	df.to_csv(os.path.join('results_record', out_path + '.csv'))
	print('eval done')


def temp():
	result_dir = 'results_final/imagenet/sr_bicubic/0'
	evaluate(result_dir, device=DEVICE)


if __name__ == '__main__':
	base_dir =args.base_dir #'results_final_edm'
	out_path = 'full_' + base_dir
	main(base_dir, out_path,device=args.device)
	#temp()
