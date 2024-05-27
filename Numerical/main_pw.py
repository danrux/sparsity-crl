import numpy as np
import torch
import argparse
import invertible_network_utils
import random
import os
import encoders
import csv
from torch import nn
from evaluation import MCC, reorder
import cooper
import utils_latent as ut
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.optimize import linear_sum_assignment
use_cuda = torch.cuda.is_available()
if use_cuda:
    device = "cuda"
else:
    device = "cpu"

print("device:", device)




def main():

    args, parser = parse_args()
    if args.n_mixing_layer==1:
        mix_type = 'linear'
    else:
        mix_type = 'pw'
    args.model_dir = f'{mix_type}_masksize_{args.mask_size}_{args.oracle}'
    heat_path_est = os.path.join(args.model_dir,'heatmaps','est')
    heat_path_true = os.path.join(args.model_dir,'heatmaps','true')
    heat_path_indep = os.path.join(args.model_dir,'heatmaps','indep')
    c_path =  os.path.join(args.model_dir,'learn_graph', 'c')
    c_hat_path = os.path.join(args.model_dir, 'learn_graph', 'chat')
    
    if not os.path.exists(heat_path_est):
        os.makedirs(heat_path_est)
    if not os.path.exists(heat_path_true):
        os.makedirs(heat_path_true)
    if not os.path.exists(heat_path_indep):
        os.makedirs(heat_path_indep)
    if not os.path.exists(c_path):
        os.makedirs(c_path)
    if not os.path.exists(c_hat_path):
        os.makedirs(c_hat_path)
        
        
    mcc_scores=[]
    mcc_indep_scores=[]
    
    for args.seed in args.seeds:
        
        # By default set the dimension of representations to be the same as z
        if args.nn==None:
            args.nn=args.z_n
        
        args.save_dir = os.path.join(args.model_dir, f'{mix_type}{args.n_mixing_layer}_d{args.distance}_n{args.z_n}_nn{args.nn}_M{args.mask_dense}_G{args.DAG_dense}_rs{args.seed}')
        
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
        
        results_file = os.path.join(args.save_dir, 'results.csv')
        
        B_file = os.path.join(args.save_dir, 'B.csv')
        W_file = os.path.join(args.save_dir, 'W.csv')
        mask_values_file = os.path.join(args.save_dir, 'mask_values.csv')
        Corr_file = os.path.join(args.save_dir, 'Corr_est.csv')
        Corr_true = os.path.join(args.save_dir, 'Corr_true.csv')
        heatmap_file_est =  os.path.join(heat_path_est, f'{mix_type}{args.n_mixing_layer}_d{args.distance}_n{args.z_n}_nn{args.nn}_M{args.mask_dense}_G{args.DAG_dense}_rs{args.seed}_Corr_heatmap.pdf')
        heatmap_file_true =  os.path.join(heat_path_true, f'{mix_type}{args.n_mixing_layer}_d{args.distance}_n{args.z_n}_nn{args.nn}_M{args.mask_dense}_G{args.DAG_dense}_rs{args.seed}_Corr_heatmap.pdf')
        heatmap_file_indep =  os.path.join(heat_path_indep, f'{mix_type}{args.n_mixing_layer}_d{args.distance}_n{args.z_n}_nn{args.nn}_M{args.mask_dense}_G{args.DAG_dense}_rs{args.seed}_Corr_heatmap.pdf')

            
        if args.evaluate:
            args.load_g = os.path.join(args.save_dir, 'g.pth')
            args.load_f = os.path.join(args.save_dir, 'f.npz')
            args.load_f_hat = os.path.join(args.save_dir, 'f_hat.pth')
            args.n_steps = 1
            
        global device
        if args.no_cuda:
            device = "cpu"
            print("Using cpu")
        if args.seed is not None:
            np.random.seed(args.seed)
            random.seed(args.seed)
            torch.manual_seed(args.seed)
        
        
        if not args.evaluate:
            B_ori = ut.simulate_dag(args.z_n, args.z_n*args.DAG_dense, args.graph_type)
            np.savetxt(B_file, B_ori, delimiter=',')
            W_ori = ut.simulate_parameter(B_ori)   
            np.savetxt(W_file, W_ori, delimiter=',')
        else:
            B_ori = np.loadtxt(B_file, delimiter=',')
            W_ori = np.loadtxt(W_file, delimiter=',')
            
        if not args.evaluate:
            z = ut.simulate_linear_sem(W_ori, 5000, 'gauss')
            Sigma_z = np.cov(z.T)
            Mean = np.mean(z, axis=0)
            sigma = np.sqrt(Sigma_z.diagonal())   
            mask_values = args.distance*sigma+Mean
            np.savetxt(mask_values_file, mask_values, delimiter=',')
        else:
            mask_values = np.loadtxt(mask_values_file, delimiter=',')


        def generate_rhohot_batch(batch_size, vector_dimension, rho):
            if vector_dimension < 2:
                raise ValueError("Vector dimension must be at least 2.")
            
            # Create a batch array with zeros
            batch_data = np.zeros((batch_size, vector_dimension), dtype=int)
            
            for i in range(batch_size):
                # Generate indices and shuffle them
                indices = np.arange(vector_dimension)
                np.random.shuffle(indices)
                # Set the first 5 indices to 1
                batch_data[i, indices[:rho]] = 1
            
            return batch_data      
        
        if args.mask_dense==1:
            ac=1
        elif args.mask_dense==50:
            ac = int(args.z_n/2)
        elif args.mask_dense==75:
            ac = int(args.z_n*0.75)
        
        
        if args.mask_size>1:   
            masks = generate_rhohot_batch(args.mask_size*args.z_n, args.z_n, ac)
        else:
            #when mask size is relatively low, artificially design masks to ensure assumption 2.2
            masks = np.ones(args.z_n)
            masks = (np.tril(masks,-args.z_n-1+ac)+np.tril(np.triu(masks),ac-1)).tolist()
        
        def sample_whole_latent(size, indep=False, Mask=True, device=device):
            
            if indep:
                Diag_B = ut.simulate_dag(args.z_n, 0, args.graph_type)
                z = ut.simulate_linear_sem(Diag_B, size, 'gauss')
            else:
                z = ut.simulate_linear_sem(W_ori, size, 'gauss')
            
            if not Mask:
                return z
            
            z = torch.tensor(z)
            
            
            mini_batch = size // len(masks)
            for k in range(len(masks)):
                mask = np.array(masks[k])
                if k==len(masks)-1:
                    z[k*mini_batch:,:]=z[k*mini_batch:,:]*mask
                else:
                    
                    z[k*mini_batch:(k+1)*mini_batch,:]=z[k*mini_batch:(k+1)*mini_batch,:]*mask
            
            for i in range(args.z_n):
                z[z[:,i]==0,i]=mask_values[i]
            
            z = z.float()
            z = z.to(device)
            
            return z
           
        
        f=invertible_network_utils.get_decoder(args.x_n, args.z_n, args.seed, args.n_mixing_layer, args.load_f, args.save_dir, smooth=False)
        
            
                    
        # calculate the scale value such that x is around 10 and variance around 1 thereof
        pre_data = f(sample_whole_latent(4000))
        scale = 1/torch.std(pre_data, dim=0).mean()
        
        
        
        
        class Constrained_DE(cooper.ConstrainedMinimizationProblem):
            def __init__(self,latent_dim=args.z_n, n_classes=len(masks)):
                self.criterion = nn.MSELoss(reduction='sum')
                
                
                super().__init__(is_constrained=True)
            def compute_loss(self, x, x_hat, mu):
                
                mini_batch = args.batch_size // len(masks)
                
                log_p_x_given_z = self.criterion(x_hat, x)
                
                stat_loss = 0
                
                if args.oracle=='oracle':
                    log_var_prior = torch.tensor(masks).to(device)
                    log_var_prior[log_var_prior==0]=-10
                    
                mu_prior = torch.ones(args.nn).to(device)*2
                
                
                for k in range(len(masks)):
                    if k==len(masks)-1:
                        
                        mu_k = mu[k*mini_batch:,:]
                        
                    else:
                        
                        mu_k=mu[k*mini_batch:(k+1)*mini_batch,:]
                        
                    
                    
                    diffs = mu_k - mu_prior
                    if args.oracle=='oracle':
                        std = (log_var_prior[k,:]/2).exp()
                    else:
                        var = torch.mean(torch.pow(diffs, 2.0))
                        std = torch.pow(var, 0.5)
                    zscores = diffs / (std+1e-9)
                    skews = torch.mean(torch.pow(zscores, 3.0),dim=0)
                    kurtoses = torch.mean(torch.pow(zscores, 4.0),dim=0) - 3.0
                    
                    
                    stat_loss += torch.mean(torch.abs(skews))+torch.mean(torch.abs(kurtoses))
                    
                
                return log_p_x_given_z,stat_loss
            
            def closure(self,  inputs):
                x = f(inputs)
                x = x*scale #scale up the observed data
                z_mu = g(x)
                
                z_hat = z_mu
                x_hat = f_hat(z_hat)
                
                loss,stat_loss = self.compute_loss(x, x_hat, z_mu)
                ineq_defect = torch.sum(torch.abs(z_hat))/args.batch_size/args.z_n - args.sparse_level
        
                return cooper.CMPState(loss=loss+stat_loss, ineq_defect=ineq_defect, eq_defect=None)
                
        
        g = encoders.get_mlp(
            n_in=args.x_n,
            n_out=args.nn,
            layers=[
                 
                 (args.nn) * 10,
                 (args.nn) * 50,
                 (args.nn) * 50,
                 (args.nn) * 50,
                 (args.nn) * 50,
                 (args.nn) * 10,
                 
                 ],
            output_normalization = "bn",
             )   
        
        
        
        f_hat = encoders.get_mlp(
                n_in=(args.nn),
                n_out=args.x_n,
                layers=[
                     
                     (args.nn) * 10,
                     (args.nn) * 50,
                     (args.nn) * 50,
                     (args.nn) * 50,
                     (args.nn) * 50,
                     (args.nn) * 10,
                     
                     ],
            output_normalization = "bn",
             ) 
        
            
        if args.load_g is not None:
            g.load_state_dict(torch.load(args.load_g))
        
        
        if args.load_f_hat is not None:
            f_hat.load_state_dict(torch.load(args.load_f_hat))
        
        g = g.to(device)
        f_hat = f_hat.to(device)
        
        
       
        
        params = list(g.parameters())+list(f_hat.parameters())
        
       
        
        ############### VaDE ###################
        cmp_vade = Constrained_DE()
        formulation = cooper.LagrangianFormulation(cmp_vade, args.aug_lag_coefficient)
        primal_optimizer = cooper.optim.ExtraAdam(params, lr=args.lr)
        dual_optimizer = cooper.optim.partial_optimizer(cooper.optim.ExtraAdam, lr=args.lr/2)
        
        coop_vade = cooper.ConstrainedOptimizer(
            formulation=formulation,
            primal_optimizer=primal_optimizer,
            dual_optimizer=dual_optimizer,
        )
        




        
            
        total_loss_values = []
        global_step = len(total_loss_values) + 1
        
        while (
            global_step <= args.n_steps
        ):
            if not args.evaluate:
                
                g.train()
                f_hat.train()
                
                
                data = sample_whole_latent(size=args.batch_size)
                
                coop_vade.zero_grad()
                lagrangian = formulation.composite_objective(
                        cmp_vade.closure,data
                        )
                formulation.custom_backward(lagrangian)
                coop_vade.step(cmp_vade.closure, data)
                    
                    
                
            if global_step % args.n_log_steps == 1 or global_step == args.n_steps:
                    f_hat.eval()
                    g.eval()
                    
                    
                    z_disentanglement = sample_whole_latent(5000)  #4096
                        
                    hz_disentanglement = f(z_disentanglement)*scale
                    
                    hz_disentanglement = g(hz_disentanglement)
                    mcc, cor_m = MCC(z_disentanglement, hz_disentanglement, args.z_n)
                    mcc = mcc/args.z_n
                    mind = linear_sum_assignment(-1*cor_m)[1] 
                    
                    if not args.evaluate:
                        fileobj=open(results_file,'a+')
                        writer = csv.writer(fileobj)
                        wri = ['MCC', mcc]
                        writer.writerow(wri)
                        print(global_step)  
                        print('estimate_mcc')
                        print(mcc)
                       
                        save_path = os.path.join(args.save_dir, 'g.pth')
                        torch.save(g.state_dict(), save_path)
                        save_path = os.path.join(args.save_dir, 'f_hat.pth')
                        torch.save(f_hat.state_dict(), save_path)
                    
            global_step += 1
            if mcc>0.995:
                break
        
        
        fileobj=open(args.model_dir+'.csv','a+')
        writer = csv.writer(fileobj)
        wri = [args.distance, args.n_mixing_layer, args.z_n, args.nn, args.mask_dense, args.DAG_dense, args.seed, mcc]
        writer.writerow(wri)
        fileobj.close()
        mcc_scores.append(mcc)
        
        mcc_true,cor_true = MCC(z_disentanglement, z_disentanglement, args.z_n)
        np.savetxt(Corr_file, cor_m, delimiter=',')
        np.savetxt(Corr_true, cor_true, delimiter=',')
        
        
        # draw heatmaps for truth
        sns.set(rc={"figure.dpi":100, 'savefig.dpi':900})
        fig, axes = plt.subplots(1, 1, figsize=(4, 4))
        gap1 = args.z_n//10
        if gap1==0:
            gap1=1
        list1 = list(range(0,args.z_n,gap1))+[args.z_n-1]
        z_label = ['']*args.z_n
        for i in list1:
            kk=i+1
            z_label[i]=r'$\mathbf{z}$'+f'$_{{{kk}}}$'
        cor_true=pd.DataFrame(cor_true, index=z_label, columns=z_label)
        sns.heatmap(cor_true, xticklabels=True, yticklabels=True, annot=False, cmap="Blues", ax=axes, cbar=False, fmt=".2f", vmin=0, vmax=1)  
        axes.set_title(fr'{mix_type} n={args.z_n} m={args.n_mixing_layer} $\delta$={args.distance}$\sigma$ $\rho$={args.mask_dense} k={args.DAG_dense}',fontsize=15)
        plt.savefig(heatmap_file_true, format="pdf", bbox_inches='tight')
        
        # draw heatmaps for est 
        cor_m = reorder(cor_m,args.z_n)
        sns.set(rc={"figure.dpi":100, 'savefig.dpi':900})
        fig, axes = plt.subplots(1, 1, figsize=(4*args.nn/args.z_n, 4))
        gap2 = args.nn//10
        if gap2==0:
            gap2=1
        list2 = list(range(0,args.nn,gap1))+[args.nn-1]
        z_hat_label = ['']*args.nn
        for i in list2:
            kk=i+1
            z_hat_label[i]=r'$\widehat{\mathbf{z}}$'+f'$_{{{kk}}}$'
        cor_m=pd.DataFrame(cor_m, index=z_label, columns=z_hat_label)
        sns.heatmap( cor_m, xticklabels=True, yticklabels=True, annot=False, cmap="Blues", ax=axes, cbar=False, fmt=".2f", vmin=0, vmax=1)  
        if args.nn==args.z_n:
            axes.set_title(fr'{mix_type} n={args.z_n} m={args.n_mixing_layer} $\delta$={args.distance}$\sigma$ $\rho$={args.mask_dense} k={args.DAG_dense}',fontsize=15)
        else: 
            axes.set_title(fr'{mix_type} n={args.z_n} nn={args.nn} m={args.n_mixing_layer} $\delta$={args.distance}$\sigma$ $\rho$={args.mask_dense} k={args.DAG_dense}',fontsize=15)
            
        plt.savefig(heatmap_file_est, format="pdf", bbox_inches='tight')
        
        
        
        # testing on indep
        z_indep = sample_whole_latent(5000, indep=True)
        hz_indep = f(z_indep*scale)
        hz_indep = g(hz_indep)
        
        mcc_indep, cor_indep = MCC(z_indep, hz_indep, args.z_n)
        mcc_indep = mcc_indep/args.z_n
        
        fileobj=open(args.model_dir+'_independent_test.csv','a+')
        writer = csv.writer(fileobj)
        wri = [args.distance, args.n_mixing_layer, args.z_n, args.nn, args.mask_dense, args.DAG_dense, args.seed, mcc_indep]
        writer.writerow(wri)
        fileobj.close()
        
        mcc_indep_scores.append(mcc_indep)
        
        # draw heatmaps for indep
        cor_indep = reorder(cor_indep,args.z_n)
        sns.set(rc={"figure.dpi":100, 'savefig.dpi':900})
        fig, axes = plt.subplots(1, 1, figsize=(4*args.nn/args.z_n, 4))
        cor_indep=pd.DataFrame(cor_indep, index=z_label, columns=z_hat_label)
        sns.heatmap( cor_indep, xticklabels=True, yticklabels=True, annot=False, cmap="Blues", ax=axes, cbar=False, fmt=".2f", vmin=0, vmax=1)  
        if args.nn==args.z_n:
            axes.set_title(fr'{mix_type} n={args.z_n} m={args.n_mixing_layer} $\delta$={args.distance}$\sigma$ $\rho$={args.mask_dense} k={args.DAG_dense}',fontsize=15)
        else: 
            axes.set_title(fr'{mix_type} n={args.z_n} nn={args.nn} m={args.n_mixing_layer} $\delta$={args.distance}$\sigma$ $\rho$={args.mask_dense} k={args.DAG_dense}',fontsize=15)
           
        plt.savefig(heatmap_file_indep, format="pdf", bbox_inches='tight')
        
        
        # store unmask z and z_hat for graph learning
        c = sample_whole_latent(size=5000, Mask=False)
        c_hat = g(f(c)*scale)
        c_hat=c_hat.detach().cpu().numpy()
        c_file = os.path.join(c_path, f'{mix_type}{args.n_mixing_layer}_d{args.distance}_n{args.z_n}_M{args.mask_dense}_G{args.DAG_dense}_rs{args.seed}')
        c_hat_file = os.path.join(c_hat_path, f'{mix_type}{args.n_mixing_layer}_d{args.distance}_n{args.z_n}_M{args.mask_dense}_G{args.DAG_dense}_rs{args.seed}')
        np.savetxt(c_hat_file+'.csv', c_hat[:,mind], delimiter=',')
        np.savetxt(c_file+'.csv', c, delimiter=',')
        print('finished one random seeds!')

    fileobj=open('SUM_'+args.model_dir+'.csv','a+')
    writer = csv.writer(fileobj)
    wri = [args.distance, args.n_mixing_layer, args.z_n, args.nn, args.mask_dense, args.DAG_dense, np.mean(mcc_scores),np.std(mcc_scores)]
    writer.writerow(wri)
    fileobj.close()

    fileobj=open('SUM_INDE_'+args.model_dir+'.csv','a+')
    writer = csv.writer(fileobj)
    wri = [args.distance, args.n_mixing_layer, args.z_n, args.nn, args.mask_dense, args.DAG_dense, np.mean(mcc_indep_scores),np.std(mcc_indep_scores)]
    writer.writerow(wri)
    fileobj.close()
        


    
    
    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--z-n", type=int, default=10, choices=[3, 5, 10, 20, 40])
    parser.add_argument("--x-n", type=int, default=10, choices=[3, 5, 10, 20, 40])
    parser.add_argument("--distance", type=float, default=2.0, choices=[0, 1, 2, 3, 5, 10])
    parser.add_argument("--DAG-dense", type=int, default=1, choices=[0, 1, 2, 3])
    parser.add_argument("--mask-dense", type=int, default=50, choices=[1, 50, 75, 100])
    parser.add_argument("--n-mixing-layer", type=int, default=10, choices=[1, 3, 10, 20]) #larger means more complicated piecewise
    parser.add_argument("--mask-size", type=int, default=5)
    parser.add_argument("--nn", type=int) 
    parser.add_argument("--oracle", type=str, default='oracle', choices=['oracle', 'nonoracle'])
    
    
    parser.add_argument("--evaluate", action='store_true')
    parser.add_argument("--causal", action="store_false")
    parser.add_argument("--seeds", type=int, 
                        default=[2])
    parser.add_argument("--seed", type=int, default=222)
    parser.add_argument("--scm-type", type=str, default='linear', choices=['linear', 'nonlinear'])
    parser.add_argument("--noise-type", type=str, default="gauss", choices=['gauss', 'exp', 'gumbel'])
    
    
    parser.add_argument("--lr", type=float, default=5e-5) 
    parser.add_argument("--batch-size", type=int, default=10000) 
    parser.add_argument("--n-log-steps", type=int, default=250) 
    parser.add_argument("--n-steps", type=int, default=10001) 
    
    
    parser.add_argument("--load-f", default=None)
    parser.add_argument("--load-g", default=None)
    parser.add_argument("--load-f-hat", default=None)
    parser.add_argument("--graph-type", type=str, default="ER")
    parser.add_argument("--no-cuda", action="store_true")
    parser.add_argument("--model-dir", type=str, default="")
    parser.add_argument("--save-dir", type=str, default="")
    
    parser.add_argument("--aug-lag-coefficient", type=float, default=0.00)
    parser.add_argument("--sparse-level", type=float, default=0.01)
    
    
    
    args = parser.parse_args()
    
    return args, parser    

if __name__ == "__main__":
    
    main()