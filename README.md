# shape prior unet

Here is the implementation code using shape prior unet.<br>


# requirements
Cuda 10.1<br>
Python 3.8<br>
torch 1.12.0<br>
torchvision 0.13.0<br>

## create env

```
pip install -r requirements.txt
```

## train and test all 6 video file
```
nohup python -u train_involve_hessian.py --model_name bb_hessian_unet --setting train_on_real_all --n_folds 5 > train_unet_hessain_shape_5folds.out
```


# result
dice coefficients for different models

 Methods | 5folds | 10folds | 20folds
 ------------------- | ----- | -----  |-----
 U-Net  | 0.7582 | 0.7157 | 0.6642 
 U-Net++ | 0.7535 | 0.7187 | 0.6712
 GSCNN | 0.7114 | 0.6896 | 0.6423
 Attention Unet | 0.7514 | 0.7129 | 0.6745
 **U-Net+shape*hessian** | **0.7768** | **0.7429** | **0.7026** 
