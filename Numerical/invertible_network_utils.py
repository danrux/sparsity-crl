
"""Create invertible mixing networks."""

import numpy as np
import torch
from torch import nn
from scipy.stats import ortho_group
from typing import Union
from typing_extensions import Literal
import os

def construct_invertible_mlp(n: int = 20, n_layers: int = 2, n_iter_cond_thresh: int = 10000,
                             cond_thresh_ratio: float = 0.25,
                             weight_matrix_init: Union[Literal["pcl"], Literal["rvs"]] = 'pcl',
                             act_fct: Union[Literal["relu"], Literal["leaky_relu"], Literal["elu"],
                                            Literal["smooth_leaky_relu"], Literal["softplus"]] = 'leaky_relu'
                             ):
    """
    Create an (approximately) invertible mixing network based on an MLP.
    Based on the mixing code by Hyvarinen et al.

    Args:
        n: Dimensionality of the input and output data
        n_layers: Number of layers in the MLP.
        n_iter_cond_thresh: How many random matrices to use as a pool to find weights.
        cond_thresh_ratio: Relative threshold how much the invertibility
            (based on the condition number) can be violated in each layer.
        weight_matrix_init: How to initialize the weight matrices.
        act_fct: Activation function for hidden layers.
    """

    class SmoothLeakyReLU(nn.Module):
        def __init__(self, alpha=0.2):
            super().__init__()
            self.alpha = alpha

        def forward(self, x):
            return self.alpha * x + (1 - self.alpha) * torch.log(1 + torch.exp(x))

    def get_act_fct(act_fct):
        if act_fct == 'relu':
            return torch.nn.ReLU, {}, 1
        if act_fct == 'leaky_relu':
            return torch.nn.LeakyReLU, {'negative_slope': 0.2}, 1
        elif act_fct == 'elu':
            return torch.nn.ELU, {'alpha': 1.0}, 1
        elif act_fct == 'max_out':
            raise NotImplemented()
        elif act_fct == 'smooth_leaky_relu':
            return SmoothLeakyReLU, {'alpha': 0.2}, 1
        elif act_fct == 'softplus':
            return torch.nn.Softplus, {'beta': 1}, 1
        else:
            raise Exception(f'activation function {act_fct} not defined.')

    layers = []
    act_fct, act_kwargs, act_fac = get_act_fct(act_fct)

    # Subfuction to normalize mixing matrix
    def l2_normalize(Amat, axis=0):
        # axis: 0=column-normalization, 1=row-normalization
        l2norm = np.sqrt(np.sum(Amat * Amat, axis))
        Amat = Amat / l2norm
        return Amat

    condList = np.zeros([n_iter_cond_thresh])
    if weight_matrix_init == 'pcl':
        for i in range(n_iter_cond_thresh):
            A = np.random.uniform(-1, 1, [n, n])
            A = l2_normalize(A, axis=0)
            condList[i] = np.linalg.cond(A)
        condList.sort()  # Ascending order
    condThresh = condList[int(n_iter_cond_thresh * cond_thresh_ratio)]
    #print("condition number threshold: {0:f}".format(condThresh))

    for i in range(n_layers):

        lin_layer = nn.Linear(n, n, bias=False)

        if weight_matrix_init == 'pcl':
            condA = condThresh + 1
            while condA > condThresh:
                weight_matrix = np.random.uniform(-1, 1, (n, n))
                weight_matrix = l2_normalize(weight_matrix, axis=0)

                condA = np.linalg.cond(weight_matrix)
                
            lin_layer.weight.data = torch.tensor(weight_matrix, dtype=torch.float32)

        elif weight_matrix_init == 'rvs':
            weight_matrix = ortho_group.rvs(n)
            lin_layer.weight.data = torch.tensor(weight_matrix, dtype=torch.float32)
        elif weight_matrix_init == 'expand':
            pass
        else:
            raise Exception(f'weight matrix {weight_matrix_init} not implemented')

        layers.append(lin_layer)

        if i < n_layers - 1:
            layers.append(act_fct(**act_kwargs))

    mixing_net = nn.Sequential(*layers)

    # fix parameters
    for p in mixing_net.parameters():
        p.requires_grad = False

    return mixing_net


# this function is modified based on https://github.com/slachapelle/disentanglement_via_mechanism_sparsity
def get_decoder(x_dim, z_dim, seed, n_layers, load_f,save_dir, manifold='nn', smooth=False):
    rng_data_gen=np.random.default_rng(seed)
    
    
        
    if manifold == "nn":
        
        
        # NOTE: injectivity requires z_dim <= h_dim <= x_dim
        h_dim = x_dim
        neg_slope = 0.2
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        
        if load_f is not None:
            dd = np.load(load_f)
            W0 = [dd[f] for f in dd]
        else:
            W0 = []
            for l in range(n_layers):
                Wi = rng_data_gen.normal(size=(h_dim, x_dim))
                Wi = np.linalg.qr(Wi.T)[0].T
                # print("distance to identity:", np.max(np.abs(np.matmul(W4, W4.T) - np.eye(h_dim))))
                Wi *= np.sqrt(2 / (1 + neg_slope ** 2)) * np.sqrt(2. / (x_dim + h_dim))
                W0.append(Wi)
            
            save_path = os.path.join(save_dir, 'f.npz')
            np.savez(save_path,*W0)
        
        W=[]
        for l in range(n_layers):
            Wi = W0[l]
            Wi = torch.Tensor(Wi).to(device)
            Wi.requires_grad = False
            W.append(Wi)
        
        
            
            
        
        
        
        
        

        # note that this decoder is almost surely invertible WHEN dim <= h_dim <= x_dim
        # since Wx is injective
        # when columns are linearly indep, which happens almost surely,
        # plus, composition of injective functions is injective.
        def decoder(z):
            with torch.no_grad():
                
                z = torch.Tensor(z).to(device)
                h = torch.matmul(z, W[0])
                if n_layers>1:
                    for l in range(n_layers-1):
                        if smooth:
                            h = neg_slope * h + (1 - neg_slope) * torch.log(1 + torch.exp(h))
                        else:
                            h = torch.maximum(neg_slope * h, h)  # leaky relu
                           
                        h = torch.matmul(h, W[l+1])
               
            return h

        #noise_std = 0.01
    else:
        raise NotImplementedError(f"The manifold {manifold} is not implemented.")

    return decoder
