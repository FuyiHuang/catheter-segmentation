from networks.DUnet import *
import os
import torch
import numpy as np
import torch.nn as nn
import argparse
import copy
import matplotlib.image as mimg
import pandas as pd
import matplotlib.pyplot as plt
from trainer_hessian_involve import *
from networks.BB_Unet import BB_Unet, BB_Unet_with_hessian_stack,BB_Unet_with_hessian_mul_prior


parser = argparse.ArgumentParser()


parser.add_argument('--model_name', type=str,
                    default='UNet',required=True, help='Name of model to train')

parser.add_argument('--splits', type=int,
                    default=6,  help='two splits by T')

parser.add_argument('--setting', type=str,
                    default='scratch',  help='trainsetting scratch/pt_ft/pt_ft_all_T/scratch_dilated')

parser.add_argument('--pt_steps', type=int,
                    default=2e4, help='maximum steps of pretrain')

parser.add_argument('--pt_lr', type=int,
                    default=1e-4, help='learning rate of prerain')

parser.add_argument('--pt_bs', type=int,
                    default=4, help='batch_size in pretrain')

parser.add_argument('--ft_steps', type=int,
                    default=1e4, help='maximum steps of train')

parser.add_argument('--ft_bs', type=int,
                    default=4, help='batch_size in train')

parser.add_argument('--ft_lr', type=float,  default=1e-4,
                    help='finetune learning rate')
parser.add_argument('--n_folds', type=int,
                    default=20, help='n folds cross validation')
parser.add_argument('--n_cal_folds', type=int,
                    default=5, help='n folds cross validation')
parser.add_argument('--n_round', type=int,
                    default=5, help='n round self training')

parser.add_argument('--losses', type=str,
                    default='dice',
                    help='losses that used')

parser.add_argument('--pretrain_folder', type=str,
                    default='/home/baixiang/catheter/Miccai_experiments_bx_fy_aligned/catheter_data/phantom_data',
                    help='folders of where pretrain data are at')
parser.add_argument('--hessian_pt_folder', type=str,
                    default='/home/baixiang/catheter/Miccai_experiments_bx_fy_aligned/catheter_data/enhanced_pt',
                    help='folders of where enhanced phantom data are at')

parser.add_argument('--hessian_real_folder', type=str,
                    default='/home/baixiang/catheter/Miccai_experiments_bx_fy_aligned/catheter_data/enhanced_real',
                    help='folders of where enhanced finetune/scratch train data are at')

parser.add_argument('--finetune_folder', type=str,
                    default='/home/baixiang/catheter/Miccai_experiments_bx_fy_aligned/catheter_data/real_data',
                    help='folders of where finetune/scratch train data are at')

parser.add_argument('--results_output_folder', type=str,
                    default='/home/baixiang/catheter/Miccai_experiments_bx_fy_aligned/outputs',
                    help='folders of where finetune/scratch train data are at')

parser.add_argument('--real_subsets', type=str,
                    default='T6',
                    help='which is real subsets is used as train scratch or finetune')

parser.add_argument('--save_finetune_model', action='store_false',help='whether save finetuned models')
parser.add_argument('--use_adabn', action='store_true',help='if use adabn')
parser.add_argument('--use_tsne', action='store_true',help='if use tsne')
args = parser.parse_args()




if __name__ == '__main__':
    
    # os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
    use_cuda = torch.cuda.is_available()    
    # train setting

    
    model_scratch = BB_Unet_with_hessian_mul_prior()
    if use_cuda:
        model_scratch.cuda()       
    
    input_model_name = args.model_name
    
    args.model_name = input_model_name + args.real_subsets + 'n_folds' + str(args.n_folds) + args.losses +args.setting

    if 'dilated' in args.setting:
        args.hessian_real_folder = '/home/baixiang/catheter/Miccai_experiments_bx_fy_aligned/catheter_data/dilated_enhanced_real'
        
    
    
    
    
    MODEL_RESULT_PATH = os.path.join(args.results_output_folder, args.model_name+ '_results')
    if not os.path.exists(MODEL_RESULT_PATH):
        os.makedirs(MODEL_RESULT_PATH)

    VIS_RESULT_PATH= os.path.join(args.results_output_folder, args.model_name+ '_results_vis') 
    if not os.path.exists(VIS_RESULT_PATH):
        os.makedirs(VIS_RESULT_PATH)
    
    
    args.MODEL_RESULT_PATH = MODEL_RESULT_PATH
    args.VIS_RESULT_PATH = VIS_RESULT_PATH
    args.use_cuda = use_cuda
    
    



    if args.setting == 'pt_ft':
        mean_dices_all = train_pt_ft_on_real_with_hessian_oneT(model_scratch, args)
    
    
        print(mean_dices_all) 
        np.save(os.path.join(MODEL_RESULT_PATH,str(np.mean(mean_dices_all))+'.npy'), mean_dices_all )
    elif args.setting == 'pt_ft_all_T':
        all_means = train_pt_ft_on_real_with_hessian(model_scratch, args)
        np.save(os.path.join(MODEL_RESULT_PATH,'all_mean'+str(np.mean(all_means))+'.npy'), all_means )
    elif args.setting == 'scratch' or args.setting ==  'scratch_dilated':
        mean_dices_all =  train_real_with_hessian_main(model_scratch, args)
        print(mean_dices_all) 
        np.save(os.path.join(MODEL_RESULT_PATH,str(np.mean(mean_dices_all))+'.npy'), mean_dices_all )
    elif args.setting =='train_on_real_all':
        all_means  =  train_on_real_with_hessian(model_scratch, args)
        np.save(os.path.join(MODEL_RESULT_PATH,'all_mean'+str(np.mean(all_means))+'.npy'), all_means )
    

    
    