import os
from utils.binary_dice_loss import loss_dice,one_hot, cal_dice,get_grad_loss,cal_hd_loss
from utils.focal_loss import BinaryFocalLoss
from utils.datasets import *
from utils.swa import *
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import random_split,DataLoader,ConcatDataset
import copy
import matplotlib.image as mimg
import pandas as pd
import matplotlib.pyplot as plt
from utils.swa import *
import glob
from tqdm import tqdm
from networks.DUnet import  *


def vis_predict(vis_results_name_part_,dice_score_numeber,pred_):

    dice_str = str(dice_score_numeber)
    dice_str_ = dice_str.replace('.','_')
    pred_for_view = np.round(pred_*255)
    pred_for_view = pred_for_view[0,0,:,:]
    
    npy_name_ = vis_results_name_part_.replace('image','predict')
    

    np.save(npy_name_+'.npy', pred_[0,0,:,:])

    mimg.imsave((npy_name_+'_'+dice_str_ +'.jpg'),
                pred_for_view.astype(np.uint8),cmap='gray'
    )

    
        
def test_dice_with_prior(teset_data_loader_,model_to_test_in_, shape_prior,args,which_cv_fold):
    
    c_shape = torch.zeros((1, 1, 256, 256)).cuda() 
    torch_shape = torch.from_numpy(shape_prior)
    c_shape[0,0,:,:] = torch_shape    
    
    model_to_test_ = copy.deepcopy(model_to_test_in_)
    which_data_subset = args.real_subsets
    #print('see len of trainloader',len(teset_data_loader_))
    model_to_test_.eval()
    test_ls = []
    results = []
    VIS_RESULTS_PATH_ = os.path.join(args.VIS_RESULT_PATH,which_data_subset,which_cv_fold)
    if not os.path.exists(VIS_RESULTS_PATH_):
        os.makedirs(VIS_RESULTS_PATH_, exist_ok=True)
        
    mimg.imsave(os.path.join(VIS_RESULTS_PATH_,'shape_'+which_data_subset+'.jpg'),
                shape_prior,cmap='gray'
    )

    for id, (image, label, hessian,img_name) in enumerate(teset_data_loader_):
        with torch.no_grad():
            

            mimg.imsave(os.path.join(VIS_RESULTS_PATH_,'hessian_'+which_data_subset+'.jpg'),
                        hessian[0][0].numpy(),cmap='gray'
            )
            print('hessian.shape', hessian[0][0].shape)
            print('shape_prior.shape', shape_prior.shape)
            mimg.imsave(os.path.join(VIS_RESULTS_PATH_,'hessian_shapep_'+which_data_subset+'.jpg'),
                        (hessian[0][0]*shape_prior).numpy(),cmap='gray'
            )
   
            c_shape_expanded = c_shape.expand(image.shape[0], -1, -1, -1).cuda()
            pred = model_to_test_(image.cuda(),hessian.cuda(),c_shape_expanded)
            pred_np = pred.detach().cpu().numpy()

        #print('output shape', pred_np.shape)
        #print('max and min of output', np.max(pred_np), np.min(pred_np))
            
            
            dice_score_ = cal_dice(pred, label.cuda())
            dice_np = dice_score_.detach().cpu().numpy()
            


            
            output_name = os.path.join(VIS_RESULTS_PATH_,img_name[0])




  
            vis_predict(output_name,dice_np,pred_np)
            results.append(dice_np)    
    
    mean_dice = np.mean(results)
    #print('output digits shape ',digits_outputs.shape)
    return mean_dice,test_ls


def vis_train_data(vis_results_name_part_, label_):
    
    #print('vis_results_name_part_',vis_results_name_part_)
    npy_name_ = vis_results_name_part_.replace('image')
    #print('npy_name_',npy_name_)

    label_np = label_[0].numpy()
    label_for_view = np.round(label_np*255)



    np.save(npy_name_+'.npy', label_np[0])

    mimg.imsave((npy_name_+ '.jpg'),
                label_for_view.astype(np.uint8),cmap='gray'
    )
    
    
    
    

    
    
    

def train_one_time_with_hessian(model_in_, dataset_in_,shape_prior, args, notes = 'real'):
    # model_finetune_ = torch.load(os.path.join(pre_train_path_,model_name_+'_saved_model_pretrain_on_phantom'))
    model_to_train_ = copy.deepcopy(model_in_)
    
    if 'pt_ft' in args.setting:
        for name, param in model_to_train_.named_parameters():
            if param.requires_grad and 'bottleneck' in name:
                param.requires_grad = False 
    
    
    VIS_RESULTS_PATH_ = args.VIS_RESULT_PATH
    if not os.path.exists(VIS_RESULTS_PATH_):
        os.makedirs(VIS_RESULTS_PATH_, exist_ok=True)
    
    if 'real' in notes:
        train_steps_ = args.ft_steps
        lr_ = args.ft_lr
        batch_size = args.ft_bs
    elif 'pt' in notes:
        train_steps_ = args.pt_steps
        lr_ = args.pt_lr
        batch_size = args.pt_bs
    use_cuda_ = args.use_cuda
    
    dataloader_in_ = DataLoader(dataset_in_, batch_size=batch_size, shuffle=True, drop_last = True)
    
    if use_cuda_:
        model_to_train_.cuda()

    freq_print = 100  # in steps 2e4
    total_steps = train_steps_
    step = 0
    
    loss_record_ = []
    step_record_ = []

    train_ls_ = []

    optimizer = torch.optim.Adam(model_to_train_.parameters(), lr= lr_)
    
    c_shape = torch.zeros((1, 1, 256, 256)).cuda() if use_cuda_ else torch.zeros((1, 1, 256, 256))

    torch_shape = torch.from_numpy(shape_prior)
    c_shape[0,0,:,:] = torch_shape

    loss = 0.0
    while step < total_steps:
        for ii, (images, labels,hessians,img_name) in enumerate(dataloader_in_):

            step += 1
            
  
            
            if use_cuda_:
                images, labels,hessians = images.cuda(), labels.cuda(), hessians.cuda()


            

            optimizer.zero_grad()
            c_shape_expanded = c_shape.expand(images.shape[0], -1, -1, -1).cuda()
            preds = model_to_train_(images,hessians,c_shape_expanded)

            if 'dice' in args.losses and 'grad' in args.losses and 'hd' in args.losses:
                loss = 0.7*loss_dice(preds, labels) + 0.1*get_grad_loss(hessians,preds) + 0.2*cal_hd_loss(preds,labels)
            elif 'dice' in args.losses and 'grad' in args.losses:
                loss = 0.8*loss_dice(preds, labels) + 0.2*get_grad_loss(hessians,preds)
            elif 'dice' in args.losses and 'hd' in args.losses:
                print('losses used')
                loss = 0.7*loss_dice(preds, labels) + 0.3*cal_hd_loss(preds,labels)
            elif 'dice' in args.losses:
                loss = loss_dice(preds, labels)
            # loss = 0.33*loss_dice(preds, labels) + 0.33*FL(preds, labels) + 0.33*nn.BCEWithLogitsLoss()(preds, labels)

            # loss = custom_loss(preds, labels)
            # loss = nn.BCEWithLogitsLoss()(preds, labels)
            loss.backward()
            optimizer.step()
            del images, labels, preds

            # Compute and print loss
            if (step % freq_print) == 0:    # print every freq_print mini-batches
                print('Step %d loss: %.5f' % (step,loss.item()))
                loss_record_.append(loss.item())
                step_record_.append(step)

            del loss    
    
    return model_to_train_,train_ls_,loss_record_,step_record_    

    
def _load_npy(folder_name,filename):
        filename = os.path.join(folder_name, filename)
        return torch.unsqueeze(torch.tensor(np.float32(np.load(filename))),dim=0)    

def load_one_T_data_with_hessian_in_tensor(data_folder_name, hessian_folder_name, data_prefix):
    
    # just for get length
    filepaths = glob.glob(os.path.join(data_folder_name,(data_prefix + '*image*test*.npy')))
    
    keys = ['img_data','label_data', 'img_path', 'label_path' ]  # Specify the keys

    list_of_data = [dict.fromkeys(keys) for _ in range(len(filepaths))]


    for idx in range(len(filepaths)):
            image_path = data_prefix + "_image_test%04d.npy" % idx
            label_path = data_prefix + "_label_test%04d.npy" % idx
            hessian_path = image_path
            image = _load_npy(data_folder_name, image_path)
            label = _load_npy(data_folder_name, label_path)
            hessian = _load_npy(hessian_folder_name, hessian_path)
            list_of_data[idx]['img_data'] = image
            list_of_data[idx]['label_data'] = label
            list_of_data[idx]['hessian_data'] = hessian
            list_of_data[idx]['img_path'] = image_path
            list_of_data[idx]['label_path'] = label_path
            list_of_data[idx]['hessian_path'] = hessian_path
        
    return list_of_data

from sklearn.model_selection import train_test_split
from utils.datasets_self_train import *


from utils.create_shape import * 

def train_real_with_hessian_main(model_in_, args):
    data_subset = args.real_subsets
    
    finetune_folder = args.finetune_folder 
    hessian_real_folder = args.hessian_real_folder
    list_of_data = load_one_T_data_with_hessian_in_tensor(finetune_folder,hessian_real_folder, data_subset)
    
    args.VIS_RESULT_PATH = os.path.join(args.VIS_RESULT_PATH,data_subset) 

    
    n_folds = args.n_folds
    n_cal_folds = args.n_cal_folds
    
    mean_dices_all = np.zeros((n_cal_folds))

    
    len_list_ = ([1/ n_folds * len(list_of_data)] *  n_folds)
    len_list_ = list(map(int, len_list_))
    len_list_[-1] = len(list_of_data) - sum(len_list_[0: n_folds-1])
    
    datals_splits = random_split(list_of_data, len_list_, generator=torch.Generator().manual_seed(42))
    
    for cv_ind in range(n_cal_folds):
        train_datals = datals_splits[cv_ind]

        test_subls = datals_splits[:cv_ind] + datals_splits[cv_ind+1:]
        test_datals = ConcatDataset(test_subls)

    
    # the only one with true label 
        train_Dataset_in_ = NPyDataset_with_hessian(train_datals)
    
    # test setting here 
        test_Dataset_in_ = NPyDataset_with_hessian(test_datals) 
        testloader_in_ = DataLoader(test_Dataset_in_, batch_size=1, shuffle=True, drop_last = True)
    
    # test one  
        shape = create_shape_from_list(train_datals,args)
        model_got_,_,_,_ = train_one_time_with_hessian(model_in_, train_Dataset_in_,shape, args, 'real')
        mean_dice_one,test_ls_one = test_dice_with_prior(testloader_in_,model_got_,shape, args, 'cv_'+str(cv_ind))
        mean_dices_all[cv_ind]  = mean_dice_one
        print(mean_dices_all)


    np.save(os.path.join(args.MODEL_RESULT_PATH,str(np.mean(mean_dices_all))+'.npy'), mean_dices_all )
    return mean_dices_all
        

def load_phantom_data_with_hessian_in_tensor(data_folder_name, hessian_folder_name):
    
    # just for get length
    filepaths = glob.glob(os.path.join(data_folder_name,'*image*train*.npy'))
    
    keys = ['img_data','label_data', 'img_path', 'label_path' ]  # Specify the keys

    list_of_data = [dict.fromkeys(keys) for _ in range(len(filepaths))]


    for idx in range(len(filepaths)):
            image_path =   "image_train%04d.npy" % idx
            label_path =  "label_train%04d.npy" % idx
            hessian_path = image_path
            image = _load_npy(data_folder_name, image_path)
            label = _load_npy(data_folder_name, label_path)
            hessian = _load_npy(hessian_folder_name, hessian_path)
            list_of_data[idx]['img_data'] = image
            list_of_data[idx]['label_data'] = label
            list_of_data[idx]['hessian_data'] = hessian
            list_of_data[idx]['img_path'] = image_path
            list_of_data[idx]['label_path'] = label_path
            list_of_data[idx]['hessian_path'] = hessian_path
        
    return list_of_data


def train_pt_with_hessian_main(model_in_, args):
    
    
    pt_folder = args.pretrain_folder 
    hessian_pt_folder = args.hessian_pt_folder
    list_of_data = load_phantom_data_with_hessian_in_tensor(pt_folder,hessian_pt_folder)
    
    args.VIS_RESULT_PATH = os.path.join(args.VIS_RESULT_PATH,'pt') 
    

    train_Dataset_in_ = NPyDataset_with_hessian(list_of_data)
    

 
    shape = create_shape_from_list(list_of_data,args)
    model_got_,_,_,_ = train_one_time_with_hessian(model_in_, train_Dataset_in_,shape, args, 'pt')
    
    return model_got_




def train_pt_ft_on_real_with_hessian(model_in_, args):
    

    model_out_ = train_pt_with_hessian_main(model_in_, args)
    
    all_means = np.zeros((args.n_cal_folds,6))
 
    for T in range(1,7):
        args.real_subsets = 'T' + str(T)    
        model_in_ft = copy.deepcopy(model_out_)
        
        mean_dices_all = train_real_with_hessian_main(model_in_ft, args)

        all_means[:,T-1] = mean_dices_all
        
        print('all_means',np.mean(all_means))
    return all_means

def train_on_real_with_hessian(model_in_, args):
    all_means = np.zeros((args.n_cal_folds,6))
    for T in range(1,7):
        model_train = copy.deepcopy(model_in_)
        args.real_subsets = 'T' + str(T)    
        mean_dices_all = train_real_with_hessian_main(model_train, args)

        all_means[:,T-1] = mean_dices_all

    non_zero =  all_means[all_means!=0]    
    print('all_means',np.mean(non_zero))
    return all_means    


def train_pt_ft_on_real_with_hessian_oneT(model_in_, args):
    

    model_out_ = train_pt_with_hessian_main(model_in_, args)
    

 

    model_in_ft = copy.deepcopy(model_out_)
        
    mean_dices_all = train_real_with_hessian_main(model_in_ft, args)

    print(mean_dices_all)
    return mean_dices_all