from hessian import Hessian2D
from frangiFilter2D import FrangiFilter2D, eig2image
import numpy as np
import os
import glob
import matplotlib.image as mimg  
from scipy.ndimage import binary_dilation

def get_pt_save_ehanced(s_path, t_path):

    imgpaths = glob.glob(os.path.join(s_path,'image*train*.npy'))

    for imgpath in imgpaths:
        only_file_name = os.path.basename(imgpath) 

        
        img = np.load(imgpath)
        
        outIm, whatScale, direction = FrangiFilter2D(img, FrangiScaleRange=np.array([1, 3]), FrangiScaleRatio=3,
                   FrangiBetaOne=0.5, FrangiBetaTwo=5, verbose=False, BlackWhite=True)
        #
        out_jpg_name = only_file_name.replace('.npy','.jpg')
        out_npy_name = only_file_name

        outIm = np.where(outIm>0.1, 1, 0)
        
        structuring_element = np.ones((4, 4), dtype=bool)
        
        outIm =binary_dilation(outIm, structure=structuring_element)
        

        mimg.imsave(os.path.join(t_path,out_jpg_name), 
                    (255*outIm).astype(np.uint8),cmap='gray'  )
        np.save(os.path.join(t_path,out_npy_name),outIm)

            
def get_rl_save_ehanced(s_path, t_path):

    imgpaths = glob.glob(os.path.join(s_path,'*image*test*.npy'))

    for imgpath in imgpaths:
        only_file_name = os.path.basename(imgpath) 

        
        img = np.load(imgpath)
        
        outIm, whatScale, direction = FrangiFilter2D(img, FrangiScaleRange=np.array([1, 3]), FrangiScaleRatio=3,
                   FrangiBetaOne=0.5, FrangiBetaTwo=5, verbose=False, BlackWhite=True)
        
        
        outIm = np.where(outIm>0.1, 1, 0)
        
        structuring_element = np.ones((4, 4), dtype=bool)
        
        outIm =binary_dilation(outIm, structure=structuring_element)
        
        
        
        out_jpg_name = only_file_name.replace('.npy','.jpg')
        out_npy_name = only_file_name

        mimg.imsave(os.path.join(t_path,out_jpg_name), 
                    (255*outIm).astype(np.uint8),cmap='gray'  )
        np.save(os.path.join(t_path,out_npy_name),outIm)



pt_s_path = '/home/baixiang/catheter/Miccai_experiments_bx_fy_aligned/catheter_data/phantom_data'
rl_s_path ='/home/baixiang/catheter/Miccai_experiments_bx_fy_aligned/catheter_data/real_data'


dilated_pt_t_path = '/home/baixiang/catheter/Miccai_experiments_bx_fy_aligned/catheter_data/dilated_enhanced_pt'
dilated_rl_t_path = '/home/baixiang/catheter/Miccai_experiments_bx_fy_aligned/catheter_data/dilated_enhanced_real'

if not os.path.exists(dilated_pt_t_path):
    os.makedirs(dilated_pt_t_path)

if not os.path.exists(dilated_rl_t_path):
    os.makedirs(dilated_rl_t_path)


get_pt_save_ehanced(pt_s_path,dilated_pt_t_path)

get_rl_save_ehanced(rl_s_path,dilated_rl_t_path)
