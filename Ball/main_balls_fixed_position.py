import numpy as np
import torch
import argparse
import random
import os
import csv
from torch import nn
from evaluation import MCC
import cooper
from collections import OrderedDict
from generate_balls_dataset import generate_ball_dataset
import encoders_image as encoders
from torchvision.models import resnet18
use_cuda = torch.cuda.is_available()
if use_cuda:
    device = "cuda"
else:
    device = "cpu"

print("device:", device)






def main():
    args, parser = parse_args()
    
    if not args.evaluate:
        args.save_dir = os.path.join(args.model_dir, f'balls{args.n_balls}_rs{args.seed}')
            
        
        
        if not os.path.exists(args.save_dir):
                os.makedirs(args.save_dir)
        results_file = os.path.join(args.save_dir, 'results.csv')
        
        
    else:
        args.load_g = os.path.join(args.model_dir, f'balls{args.n_balls}_rs{args.seed}','g.pth')
        args.load_f_hat = os.path.join(args.model_dir,f'balls{args.n_balls}_rs{args.seed}', 'f_hat.pth')
        args.n_steps = 1
        
    
    global device
    if args.no_cuda:
        device = "cpu"
        print("Using cpu")
    if args.seed is not None:
        np.random.seed(args.seed)
        random.seed(args.seed)
        torch.manual_seed(args.seed)
    
    
    class Constrained_DE(cooper.ConstrainedMinimizationProblem):
        def __init__(self,latent_dim=args.n_balls*2, n_classes=len(args.masks)):
            self.criterion = nn.MSELoss(reduction='sum')
            
            super().__init__(is_constrained=True)
        
        def compute_loss(self, x, x_hat, mu):
            
            mini_batch = args.group_batch_size
            log_p_x_given_z = self.criterion(x_hat, x)
            
            stat_loss = 0
            log_var_prior = torch.tensor(np.array(args.masks).reshape((len(args.masks),-1))).to(device)
            
            log_var_prior[log_var_prior==0]=-20
            mu_prior = torch.ones(args.n_balls*2).to(device)*2
            
            for k in range(len(args.masks)):
                if k==len(args.masks)-1:
                    
                    mu_k = mu[k*mini_batch:,:]
                    
                else:
                    
                    mu_k=mu[k*mini_batch:(k+1)*mini_batch,:]
                   
                
                
                mean = mu_prior
                diffs = mu_k - mean
                std = (log_var_prior[k,:]/2).exp()
                zscores = diffs / (std+1e-9)
                skews = torch.mean(torch.pow(zscores, 3.0),dim=0)
                kurtoses = torch.mean(torch.pow(zscores, 4.0),dim=0) - 3.0
                
                
                stat_loss += torch.mean(torch.abs(skews))+torch.mean(torch.abs(kurtoses))
                
            
            return log_p_x_given_z, stat_loss
        
        def closure(self,  inputs):
            x = inputs
            
            z_hat = g(x)
            x_hat = f_hat(z_hat)
            
            loss1,stat_loss = self.compute_loss(x, x_hat, z_hat)
            ineq_defect = torch.sum(torch.abs(z_hat))/args.group_batch_size/args.n_balls/2/len(args.masks) - args.sparse_level
    
            return cooper.CMPState(loss=loss1+stat_loss, ineq_defect=ineq_defect, eq_defect=None)
            
    
    g = encoders.ResNetEnc(resnet18, hidden_size=200, encoding_size=args.n_balls*2, output_normalization="bn")
    f_hat = encoders.ResNetDec(z_dim=args.n_balls*2)

    if args.load_g is not None:
        g.load_state_dict(torch.load(args.load_g, map_location=device))

    
      
    if args.load_f_hat is not None:
        f_hat.load_state_dict(torch.load(args.load_f_hat))
    
    g = g.to(device)
    f_hat = f_hat.to(device)
    
    params = list(g.parameters())+list(f_hat.parameters())
    
    
    
    
    ############### DE ###################
    cmp_vade = Constrained_DE()
    formulation = cooper.LagrangianFormulation(cmp_vade, args.aug_lag_coefficient)
    primal_optimizer = cooper.optim.ExtraAdam(params, lr=args.lr)
    dual_optimizer = cooper.optim.partial_optimizer(cooper.optim.ExtraAdam, lr=args.lr/2)
    
    coop_vade = cooper.ConstrainedOptimizer(
        formulation=formulation,
        primal_optimizer=primal_optimizer,
        dual_optimizer=dual_optimizer,
    )
    

    means = np.random.uniform(low=0.4, high=0.6, size=(args.n_balls, 2))
   
   

    if (
        "total_loss_values" in locals() and not args.resume_training
    ) or "total_loss_values" not in locals():
        
        total_loss_values = []
        mse_values=[]
        norm_values=[]
    global_step = len(total_loss_values) + 1
    last_save_at_step = 0
    state_history = OrderedDict()
    while (
        global_step <= args.n_steps
    ):
        if not args.evaluate:
            
            g.train()
            f_hat.train()
           
            data_z, data_x = generate_ball_dataset(n_balls=args.n_balls, means=means, Sigma=args.Sigma,  masks=args.masks, mask_value=args.mask_value, sample_num_per_group=args.group_batch_size)
            data_z = data_z.to(device)
            data_x = data_x.to(device)
            coop_vade.zero_grad()
            lagrangian = formulation.composite_objective(
                    cmp_vade.closure,data_x
                    )
            formulation.custom_backward(lagrangian)
            coop_vade.step(cmp_vade.closure, data_x)
                
                
            
        if global_step % args.n_log_steps == 1 or global_step == args.n_steps:
                
                mcc_scores=[]
                mcc_G=[]
                Cor_M=[]
                f_hat.eval()
                g.eval()
               
                
                z_disentanglement, x_disentanglement = generate_ball_dataset(n_balls=args.n_balls,  means=means, Sigma=args.Sigma, masks=args.masks, mask_value=args.mask_value, sample_num_per_group=50)  
                
                z_disentanglement = z_disentanglement.to(device)
                x_disentanglement = x_disentanglement.to(device)
                    
                    
                    
                hz_disentanglement = g(x_disentanglement)
                    
                mcc, cor_m = MCC(z_disentanglement, hz_disentanglement, args.n_balls*2)
                   
                    
                mcc = mcc/args.n_balls/2
                mcc_scores.append(mcc)
                Cor_M.append(cor_m)
                     
                if not args.evaluate:
                    
                    fileobj=open(results_file,'a+')
                    writer = csv.writer(fileobj)
                    wri = ['MCC', np.mean(mcc_scores), np.std( mcc_scores)]
                    writer.writerow(wri)
                    fileobj.close()
                
                print('estimate_mcc')
                print(np.mean(mcc_scores))
               
                
                
                
               
                if not args.evaluate and (global_step % args.n_log_steps == 1 or global_step == args.n_steps):
                    
                    print(
                        f"Step: {global_step} \t",
                        f"Loss: {cmp_vade.state.loss.item():.4f} \t"
                        )
                if args.save_dir:
                    if not os.path.exists(args.save_dir):
                        os.makedirs(args.save_dir)
                   
                    save_path = os.path.join(args.save_dir, 'g.pth')
                    torch.save(g.state_dict(), save_path)
                    save_path = os.path.join(args.save_dir, 'f_hat.pth')
                    torch.save(f_hat.state_dict(), save_path)
                    
                   
                   
                
                            

        global_step += 1
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-balls", type=int, default=2)
    
    parser.add_argument("--masks", type=int, default=[[[1,0],[1,0]],  [[1,0],[0,1]], [[0,1],[0,1]], [[0,1],[1,0]]])  # 0 is masked part
    parser.add_argument("--mask-value", type=int, default=[[0.05,0.05],[0.95,0.95]])
    parser.add_argument("--Sigma", type=float, default=[[0.01,0.005],[0.005,0.01]])
    
    
    parser.add_argument("--evaluate", action='store_true')
    
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--model-dir", type=str, default="ball_fixed")
    parser.add_argument("--encoder", default="rn18", choices=("rn18", "rn50", "rn101", "rn151",
                                                              "ccrn18", "ccrn50", "ccrn101", "ccrn152"))
    
    
    parser.add_argument("--save-dir", type=str, default="")
    
    
   
   
    parser.add_argument("--lr", type=float, default=1e-5) 
    parser.add_argument("--no-cuda", action="store_true")
    
    
    parser.add_argument("--group-batch-size", type=int, default=50) 
    parser.add_argument("--n-log-steps", type=int, default=50)
    parser.add_argument("--n-steps", type=int, default=50001) 

    parser.add_argument("--load-g", default=None)
    parser.add_argument("--load-f-hat", default=None)
    parser.add_argument("--aug-lag-coefficient", type=float, default=0.00)
    parser.add_argument("--sparse-level", type=float, default=0.01)
    
    
    
    
    
    
    args = parser.parse_args()
    
    return args, parser

if __name__ == "__main__":
    
    main()