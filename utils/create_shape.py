import numpy as np
import skfmm
# from scipy.ndimage.morphology import distance_transform_edt
from scipy.ndimage import distance_transform_cdt
import os
import matplotlib.pyplot as plt
import cv2
import torch


def generate_shape(labels):

    accumulative_intensity_map = np.zeros_like(labels[0], dtype=np.float64)
    count = 0
    for label in labels:
        # print(count)
        count = count+1
        # target_points = np.where(label == 1)

        if np.min(label) != 0 or np.max(label) != 1:
            print("Unexpected label range, should be 0-1: ", np.min(label), np.max(label))
        elif np.sum(label == 0) == 0:
            print("Label does not contain any zero values.")
        else:
            distance_map = skfmm.distance(label, dx =1)

            accumulative_intensity_map += distance_map
        # plt.imshow(label)
        # plt.xticks([])
        # plt.yticks([])
        # plt.colorbar()
        # plt.show()
        # plt.savefig('label_show_one.png')    

        # plt.figure()
        # # Plot results
        # plt.imshow(distance_map)
        # plt.colorbar()
        # plt.show()
        # plt.savefig('distance_map_one.png')
        
        # print()


        # # binary_image = 1 - label
        # # for y, x in zip(*target_points):
        # distance_map = skfmm.distance(label, dx = 1)
        # # distance_map = distance_transform_cdt(binary_image, metric='taxicab')
        # # intensity_map = 1 / np.exp(distance_map)
        # accumulative_intensity_map += distance_map

        # plt.imshow(label)
        # plt.xticks([])
        # plt.yticks([])
        # plt.colorbar()
        # plt.show()
        # plt.savefig('label_show.png')

    # plt.figure()
    # # Plot results
    # plt.imshow(accumulative_intensity_map)
    # plt.colorbar()
    # plt.show()
    # plt.savefig('distance_map.png')

    # plt.close()

    return accumulative_intensity_map


# def gaussian_blur_shape(labels):
#     accumulative_intensity_map = np.zeros_like(labels[0], dtype=np.float64)
#     ksize = (5, 5)
#     sigmaX = 5
#     for label in labels:
#         blurred_arr = cv2.GaussianBlur(label, ksize, sigmaX)
#         accumulative_intensity_map += blurred_arr

#     return accumulative_intensity_map    
# shape = gaussian_blur_shape(label_list)

# label_dir = "/home/fuyi/Project/data"
# label_filenames = [f"label_train{i:04d}.npy" for i in range(2000)]
# label_list = [np.load(os.path.join(label_dir, filename)) for filename in label_filenames]

# shape = generate_shape(label_list)

# normalized_shape = (shape - np.min(shape)) / (np.max(shape) - np.min(shape))

# plt.imsave('shape2.png', normalized_shape, cmap='gray')





# label_dir = '/home/fuyi/Project/finetune_data'
# fine_tune_train_list = [('T4_image_test0056', 'T2_image_test0064', 'T6_image_test0048', 'T6_image_test0186'), ('T1_image_test0175', 'T5_image_test0023', 'T6_image_test0110', 'T1_image_test0210'), ('T5_image_test0057', 'T1_image_test0090', 'T5_image_test0010', 'T4_image_test0004'), ('T4_image_test0174', 'T3_image_test0198', 'T3_image_test0077', 'T1_image_test0109'), ('T1_image_test0144', 'T3_image_test0069', 'T6_image_test0216', 'T1_image_test0226'), ('T2_image_test0030', 'T2_image_test0052', 'T2_image_test0130', 'T1_image_test0115'), ('T1_image_test0086', 'T4_image_test0142', 'T3_image_test0056', 'T6_image_test0142'), ('T5_image_test0123', 'T2_image_test0042', 'T4_image_test0198', 'T3_image_test0047'), ('T6_image_test0038', 'T3_image_test0004', 'T3_image_test0055', 'T2_image_test0035'), ('T1_image_test0033', 'T4_image_test0050', 'T6_image_test0012', 'T5_image_test0083'), ('T5_image_test0002', 'T3_image_test0142', 'T4_image_test0055', 'T1_image_test0186'), ('T1_image_test0058', 'T4_image_test0077', 'T5_image_test0014', 'T4_image_test0047'), ('T3_image_test0050', 'T1_image_test0284', 'T3_image_test0174', 'T1_image_test0198'), ('T4_image_test0069', 'T2_image_test0095', 'T6_image_test0196', 'T6_image_test0078')]

# labels_dict = {"T1": [], "T2": [], "T3": [], "T4": [], "T5": [], "T6": []}
# for tuple in fine_tune_train_list:
#     for label_name in tuple:
#         prefix = label_name.split("_")[0]  # 获取前缀（例如 "T1"）
#         label_name = label_name.replace('image','label')
#         label = np.load(os.path.join(label_dir, label_name + ".npy"))
#         labels_dict[prefix].append(label)

# # 为每个前缀生成一个形状
# for prefix in labels_dict.keys():
#     shape = generate_shape(labels_dict[prefix])
#     normalized_shape = (shape - np.min(shape)) / (np.max(shape) - np.min(shape))
#     plt.imsave(f'{prefix}_shape.png', normalized_shape, cmap='gray')  # 将形状保存为 PNG 文件


def phantom_shape(phantom_data_folder_):
    squre = True
    label_dir = phantom_data_folder_
    label_filenames = [f"label_train{i:04d}.npy" for i in range(2000)]
    label_list = [np.load(os.path.join(label_dir, filename)) for filename in label_filenames]
    shape = generate_shape(label_list)
    normalized_shape = (shape - np.min(shape)) / (np.max(shape) - np.min(shape))
    output = normalized_shape
    save_name = 'phantom_shape.png'

    if squre == True:
        shape_2 = normalized_shape*normalized_shape
        normalized_shape_2 = (shape_2 - np.min(shape_2)) / (np.max(shape_2) - np.min(shape_2))
        output = normalized_shape_2
        save_name = 'phantom_square_shape.png'
    
    plt.imshow(output, cmap='gray')
    plt.colorbar()
    plt.savefig(save_name)
    return normalized_shape

# phantom_shape()




def finetune_shape(fine_tune_train_list, cv, target_label, n_folds, square_flag = False):
    label_dir = '/home/fuyi/Project/finetune_data'
    labels_dict = {f"T{target_label}": []}
    for tuple in fine_tune_train_list:
        for label_name in tuple:
            prefix = label_name.split("_")[0]  
            label_name = label_name.replace('image','label')
            label = np.load(os.path.join(label_dir, label_name + ".npy"))
            labels_dict[prefix].append(label)


    for prefix in labels_dict.keys():
        shape = generate_shape(labels_dict[prefix])
        normalized_shape = (shape - np.min(shape)) / (np.max(shape) - np.min(shape))
        output = normalized_shape
        save_name = f'{prefix}_shape_{n_folds}folds_fold{cv}.png'

        if square_flag == True:
            shape_2 = normalized_shape*normalized_shape
            normalized_shape_2 = (shape_2 - np.min(shape_2)) / (np.max(shape_2) - np.min(shape_2))
            output = normalized_shape_2
            save_name = f'{prefix}_squre_shape_{n_folds}folds_fold{cv}.png'

        
        # plt.imsave(f'{prefix}_shape_fold{cv}.png', normalized_shape, cmap='gray')  
        plt.imshow(output, cmap='gray')
        plt.colorbar()
        plt.savefig(save_name)
        plt.close()
    return output

def create_shape_from_list(datalist,args):
    
    accumulative_intensity_map = np.zeros_like(datalist[0]['label_data'][0], dtype=np.float64)

    for count, data in enumerate(datalist):
        # print(count)
  
        label = torch.squeeze(data['label_data']).numpy()
        # target_points = np.where(label == 1)


        
        if np.min(label) != 0 or np.max(label) != 1:
            print("Unexpected label range, should be 0-1: ", np.min(label), np.max(label))
        elif np.sum(label == 0) == 0:
            print("Label does not contain any zero values.")
        else:
            #distance_map = skfmm.distance(label, dx = 1)
            distance_map = skfmm.distance(label-0.5)
            accumulative_intensity_map += distance_map

        # # binary_image = 1 - label
        # # for y, x in zip(*target_points):
        # distance_map = skfmm.distance(label, dx = 1)
        # # distance_map = distance_transform_cdt(binary_image, metric='taxicab')
        # # intensity_map = 1 / np.exp(distance_map)
        # accumulative_intensity_map += distance_map

        # plt.imshow(label)
        # plt.xticks([])
        # plt.yticks([])
        # plt.colorbar()
        # plt.show()
        # plt.savefig('label_show.png')

    # plt.figure()
    # # Plot results
    # plt.imshow(accumulative_intensity_map)
    # plt.colorbar()
    # plt.show()
    # plt.savefig('distance_map.png')

    # plt.close()
    normalized_shape = (accumulative_intensity_map - np.min(accumulative_intensity_map)) / (np.max(accumulative_intensity_map) - np.min(accumulative_intensity_map))
    
    plt.imshow(normalized_shape, cmap='gray')
    plt.colorbar()
    
    prefix = args.real_subsets
    save_name = f'{prefix}_shape_.png'
    plt.savefig(save_name)
    plt.close()
    
    
    
    return normalized_shape

    





