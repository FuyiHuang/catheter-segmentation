import torch
import numpy as np
import torch.nn as nn
import os
import glob
import pandas as pd 


class NPyDataset_self_train(torch.utils.data.Dataset):
    def __init__(self, data_dict_list, label_flag):
        self.data_dict_list = data_dict_list
        self.label_flag = label_flag
        

    def __len__(self):

        return (len(self.data_dict_list))



    def __getitem__(self, idx):

        image = self.data_dict_list[idx]['img_data']
        
        image_name = self.__split_path__(self.data_dict_list[idx]['img_path'])
        if self.label_flag == 'true':
            label = self.data_dict_list[idx]['label_data']
        elif self.label_flag == 'pseduo':
            label = self.data_dict_list[idx]['pesudo_label']

        return image, label,image_name
    
    def __split_path__(self,filepath_):
        directory, filename_with_ext = os.path.split(filepath_)
        filename, file_extension = os.path.splitext(filename_with_ext)
        

        return filename
    
    
class NPyDataset_with_hessian(torch.utils.data.Dataset):
    def __init__(self, data_dict_list):
        self.data_dict_list = data_dict_list

        

    def __len__(self):

        return (len(self.data_dict_list))



    def __getitem__(self, idx):

        image = self.data_dict_list[idx]['img_data']
        
        image_name = self.__split_path__(self.data_dict_list[idx]['img_path'])

        label = self.data_dict_list[idx]['label_data']
        
        hessian = self.data_dict_list[idx]['hessian_data']


        return image, label,hessian,image_name
    
    def __split_path__(self,filepath_):
        directory, filename_with_ext = os.path.split(filepath_)
        filename, file_extension = os.path.splitext(filename_with_ext)
        

        return filename
        
        
        

