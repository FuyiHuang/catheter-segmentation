import torch
import numpy as np
import torch.nn as nn
import os
import glob
import pandas as pd 

## data loader
class NPyDataset_pretrain(torch.utils.data.Dataset):
    def __init__(self, folder_name, is_train=True):
        self.folder_name = folder_name
        self.is_train = is_train
        self.filepaths_train = glob.glob(os.path.join(folder_name,'image*train*.npy'))
        self.filepaths_test = glob.glob(os.path.join(folder_name,'image*test*.npy'))
        

    def __len__(self):
        if self.is_train:
            return (len(self.filepaths_train))
        else:
            return (len(self.filepaths_test))


    def __getitem__(self, idx):
        if self.is_train:
            image = self._load_npy("image_train%04d.npy" % idx)
            label = self._load_npy("label_train%04d.npy" % idx)
            image_name = "image_train%04d" % idx
            return image, label, image_name
        else:
            image = self._load_npy("image_test%04d.npy" % idx)
            label = self._load_npy("label_test%04d.npy" % idx)
            image_name = "image_test%04d" % idx
            return image, label, image_name

    def _load_npy(self, filename):
        filename = os.path.join(self.folder_name, filename)
        return torch.unsqueeze(torch.tensor(np.float32(np.load(filename))),dim=0)
    
    
class NPyDataset_fineTune(torch.utils.data.Dataset):
    def __init__(self, folder_name, data_prefix, is_train=True):
        self.folder_name = folder_name
        self.is_train = is_train
        self.data_prefix = data_prefix
        self.filepaths = glob.glob(os.path.join(folder_name,(self.data_prefix + '*image*test*.npy')))

    def __len__(self):
        return (len(self.filepaths))

    def __getitem__(self, idx):
        if self.is_train:
            image = self._load_npy(self.data_prefix + "_image_test%04d.npy" % idx)
            label = self._load_npy(self.data_prefix + "_label_test%04d.npy" % idx)
            image_name = self.data_prefix + "_image_test%04d" % idx
            return image, label, image_name
        else:
            image = self._load_npy(self.data_prefix + "_image_test%04d.npy" % idx)
            label = self._load_npy(self.data_prefix + "_label_test%04d.npy" % idx)
            image_name = self.data_prefix + "_image_test%04d" % idx
            return image, label, image_name

    def _load_npy(self, filename):
        filename = os.path.join(self.folder_name, filename)
        return torch.unsqueeze(torch.tensor(np.float32(np.load(filename))),dim=0)
    

class NPyDataset_fineTune_all_from_csv(torch.utils.data.Dataset):
    def __init__(self, folder_name, csv_file_path, is_train=True):
        self.folder_name = folder_name
        self.is_train = is_train
        self.csv_file_path = csv_file_path

    def __len__(self):
        return (len(self._load_csv()))

    def __getitem__(self, idx):
        if self.is_train:
            files_ls = self._load_csv()
            image_name  = files_ls[idx]
            label_name = image_name.replace('image','label')
            image = self._load_npy(image_name + '.npy')
            label = self._load_npy(label_name + '.npy')
            return image, label, image_name
        else:
            files_ls = self._load_csv()
            image_name  = files_ls[idx]
            label_name = image_name.replace('image','label')
            image = self._load_npy(image_name + '.npy')
            label = self._load_npy(label_name + '.npy')
            return image, label, image_name

    def _load_npy(self, filename):
        filename = os.path.join(self.folder_name, filename)
        return torch.unsqueeze(torch.tensor(np.float32(np.load(filename))),dim=0)   
    
    def _load_csv(self):
        if self.is_train:
            the_file = pd.read_csv(os.path.join(self.csv_file_path))
            train_img_ls = the_file['train_list_all'].values
            return train_img_ls
        else:
            the_file = pd.read_csv(os.path.join(self.csv_file_path))
            train_img_ls = the_file['test_list_all'].values
            return train_img_ls

    
class NPyDataset_synPt(torch.utils.data.Dataset):
    def __init__(self, folder_name, is_train=True):
        self.folder_name = folder_name
        self.is_train = is_train
        self.filepaths = glob.glob(os.path.join(folder_name,'syn_phantom_image_*.npy'))

    def __len__(self):
        return (len(self.filepaths))

    def __getitem__(self, idx):
        if self.is_train:
            image = self._load_npy("syn_phantom_image_%04d.npy" % idx)
            label = self._load_npy("syn_phantom_label_%04d.npy" % idx)
            image_name = "syn_phantom_image_%04d" % idx
            return image, label, image_name
        else:
            image = self._load_npy("syn_phantom_image_%04d.npy" % idx)
            label = self._load_npy("syn_phantom_image_%04d.npy" % idx)
            image_name =  "syn_phantom_image_%04d" % idx
            return image, label, image_name

    def _load_npy(self, filename):
        filename = os.path.join(self.folder_name, filename)
        return torch.unsqueeze(torch.tensor(np.float32(np.load(filename))),dim=0)
    
class NPyDataset_enhanced_pt(torch.utils.data.Dataset):
    def __init__(self, folder_name,folder_name_enhanced, is_train=True):
        self.folder_name = folder_name
        self.folder_name_enhanced = folder_name_enhanced
        self.is_train = is_train
        self.filepaths_train = glob.glob(os.path.join(folder_name,'image*train*.npy'))
        self.filepaths_test = glob.glob(os.path.join(folder_name,'image*test*.npy'))

    def __len__(self):
        if self.is_train:
            return (len(self.filepaths_train))
        else:
            return (len(self.filepaths_test))


    def __getitem__(self, idx):
        if self.is_train:
            image = self._load_npy("image_train%04d.npy" % idx)
            image_enhanced = self._load_enhanced_npy("image_train%04d.npy" % idx)
            label = self._load_npy("label_train%04d.npy" % idx)
            image_name = "image_train%04d" % idx
            return image,image_enhanced, label, image_name
        else:
            image = self._load_npy("image_test%04d.npy" % idx)
            label = self._load_npy("label_test%04d.npy" % idx)
            image_enhanced = self._load_enhanced_npy("image_test%04d.npy" % idx)
            image_name =  "image_test%04d" % idx
            return image, image_enhanced,label, image_name

    def _load_npy(self, filename):
        filename = os.path.join(self.folder_name, filename)
        return torch.unsqueeze(torch.tensor(np.float32(np.load(filename))),dim=0)
    
    def _load_enhanced_npy(self, filename):
        filename = os.path.join(self.folder_name_enhanced, filename)
        return torch.unsqueeze(torch.tensor(np.float32(np.load(filename))),dim=0)
    
    
class NPyDataset_enhanced_real(torch.utils.data.Dataset):
    def __init__(self, folder_name,folder_name_enhanced,data_prefix, is_train=True):
        self.folder_name = folder_name
        self.folder_name_enhanced = folder_name_enhanced
        self.is_train = is_train
        self.data_prefix = data_prefix
        self.filepaths = glob.glob(os.path.join(folder_name,(self.data_prefix + '*image*test*.npy')))        

    def get_file_paths(self):
        return self.filepaths

    def __len__(self):

        return (len(self.filepaths))



    def __getitem__(self, idx):
        if self.is_train:
            image = self._load_npy(self.data_prefix + "_image_test%04d.npy" % idx)
            image_enhanced = self._load_enhanced_npy(self.data_prefix + "_image_test%04d.npy" % idx)
            label = self._load_npy(self.data_prefix + "_label_test%04d.npy" % idx)
            image_name = self.data_prefix + "_image_test%04d" % idx
            return image,image_enhanced, label, image_name
        else:
            image = self._load_npy(self.data_prefix + "_image_test%04d.npy" % idx)
            label = self._load_npy(self.data_prefix + "_label_test%04d.npy" % idx)
            image_enhanced = self._load_enhanced_npy(self.data_prefix + "_image_test%04d.npy" % idx)
            image_name =  self.data_prefix + "_image_test%04d" % idx
            return image, image_enhanced,label, image_name

    def _load_npy(self, filename):
        filename = os.path.join(self.folder_name, filename)
        return torch.unsqueeze(torch.tensor(np.float32(np.load(filename))),dim=0)
    
    def _load_enhanced_npy(self, filename):
        filename = os.path.join(self.folder_name_enhanced, filename)
        return torch.unsqueeze(torch.tensor(np.float32(np.load(filename))),dim=0)
    