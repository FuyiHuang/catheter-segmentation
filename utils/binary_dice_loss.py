import torch
import numpy


def loss_dice(y_pred, y_true, eps=1e-6):
    '''
    y_pred, y_true -> [N, C=1, D, H, W]
    '''
    numerator = torch.sum(y_true*y_pred, dim=(2,3)) * 2
    denominator = torch.sum(y_true, dim=(2,3)) + torch.sum(y_pred, dim=(2,3)) + eps
    return torch.mean(1. - (numerator / denominator))


def one_hot(data, threshold=0.5):
    data[data>threshold] = 1
    data[data<=threshold] = 0
    return data

def cal_dice(y_pred, y_true, eps=1e-6):
    '''
    y_pred, y_true -> [N, C=1, D, H, W]
    '''
    y_pred = one_hot(y_pred)
    numerator = torch.sum(y_true*y_pred, dim=(2,3)) * 2
    denominator = torch.sum(y_true, dim=(2,3)) + torch.sum(y_pred, dim=(2,3)) + eps
    return torch.mean(numerator / denominator)    

def get_grad_loss(hessian, pred):
	
	
	pt_grads = pred*hessian
	# grad_loss = -pt_grads.mean(1)
	grad_loss = -pt_grads.mean()
	return grad_loss


def cdist(x, y):
    '''
    Input: x is a Nxd Tensor
           y is a Mxd Tensor
    Output: dist is a NxM matrix where dist[i,j] is the norm
           between x[i,:] and y[j,:]
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||
    '''
    differences = x.unsqueeze(1) - y.unsqueeze(0)
    distances = torch.sum(differences**2, -1).sqrt()
    return distances

def AveragedHausdorffLoss(set1, set2):
        d2_matrix = cdist(set1, set2)
        term_1 = torch.mean(torch.min(d2_matrix, 1)[0])
        term_2 = torch.mean(torch.min(d2_matrix, 0)[0])

        res = term_1 + term_2

        return res
    
def cal_hd_loss_one_pair(y_pred,y_true):
    y_predpoints = torch.nonzero(y_true)
    y_truepoints = torch.nonzero(y_pred)
    
    res = AveragedHausdorffLoss(y_predpoints, y_truepoints)
    
    return res

    
    
def cal_hd_loss(y_pred,y_true):
    
    batch_size = y_pred.shape[0]
    


    terms = []
    
    for b in range(batch_size):
        y_pred_cur = y_pred[b][0]
        y_true_cur = y_true[b][0]
        res_cur = cal_hd_loss_one_pair(y_pred_cur, y_true_cur)
        terms.append(res_cur)
        
    terms = torch.stack(terms)
    
    res = terms.mean()
    
    return res

