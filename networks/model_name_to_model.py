import networks.DUnet as DU


def get_model_by_name(model_to_get_):
    if model_to_get_ == 'DU_TD_UNet':
        return DU.TD_UNet(1,1) 
    if model_to_get_ == 'DU_TD_UNet_light':
        return DU.TD_UNet_light(1,1) 
    if model_to_get_ == 'DU_UNet_1TD':
        return DU.UNet_1TD(1,1)
    if model_to_get_ == 'DU_UNet_2TD':
        return DU.UNet_2TD(1,1)
    if model_to_get_ == 'DU_UNet_3TD':
        return DU.UNet_3TD(1,1)
    if model_to_get_ == 'DU_UNet':
        return DU.UNet(1,1)
    if model_to_get_ == 'DU_UNet_Lighter':
        return DU.UNet_Lighter(1,1)
    if model_to_get_ == 'DU_Dual_Branch_UNet':
        return DU.Dual_Branch_UNet(1,1)
    
    return print('this model is not included')