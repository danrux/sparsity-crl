import numpy as np
import torch
import argparse
import random
import os
import encoders
from sklearn.preprocessing import StandardScaler
import csv
from torch import nn
from evaluation import MCC
import matplotlib.pyplot as plt
import cooper
from collections import OrderedDict
from torch.utils.data import DataLoader
from torchvision import transforms
import datasets
from infinite_iterator import InfiniteIterator
from torchvision.models import resnet18
import faiss.contrib.torch_utils

def reg_scheduler(step, start=1.0):
    return start + np.log(step)


# define your own when running the code
DATAPATH = " "


use_cuda = torch.cuda.is_available()
if use_cuda:
    device = "cuda"
else:
    device = "cpu"

print("device:", device)




def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataroot", type=str, default=DATAPATH)
    parser.add_argument(
        "--dataset-name",
        type=str,
        # required=True,
        default="causal3di"
    )
    parser.add_argument(
        "--masks",
        type=int,
        default=torch.triu(torch.ones(13, 10), diagonal=-1),  # torch.eye(10)
    )  # replace 10 with latent_dim # 0 is masked part
    parser.add_argument("--mask-value", type=int, default=torch.Tensor([1.0] * 10))
    parser.add_argument(
        "--encoder",
        default="rn18",
        choices=(
            "rn18",
            "rn50",
            "rn101",
            "rn151",
            "ccrn18",
            "ccrn50",
            "ccrn101",
            "ccrn152",
        ),
    )
    parser.add_argument("--encoding-size", type=int, default=10)
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--model-dir", type=str, default="causal3di")

    parser.add_argument("--save-dir", type=str, default="")

    parser.add_argument("--seed", type=int, default=np.random.randint(32**2 - 1))
    parser.add_argument(
        "--lr", type=float, default=1e-4
    )  
    parser.add_argument("--no-cuda", action="store_true")

    parser.add_argument("--group-batch-size", type=int, default=13)  
    parser.add_argument("--n-log-steps", type=int, default=250)  
    parser.add_argument("--n-steps", type=int, default=100001)  
    parser.add_argument("--resume-training", action="store_true")

    parser.add_argument("--load-f", default=None)
    parser.add_argument("--load-g", default=None)
    parser.add_argument("--load-f-hat", default=None)
    parser.add_argument("--mask-prob", type=float, default=0.5)
    parser.add_argument(
        "--aug_lag_coefficient", type=float, default=0.0
    )  
    parser.add_argument("--sparse-level", type=float, default=0.01)
    parser.add_argument("--workers", type=int, default=12)

    args = parser.parse_args(args=[])
    
    return args, parser


def main():
    args, parser = parse_args()
    setattr(args, "batch_size", args.group_batch_size * len(args.masks))
    setattr(args, "latent_dim", 10)  # ground truth latent dimension
    if not args.evaluate:
        args.save_dir = os.path.join(args.model_dir)
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
    else:
        args.load_g = os.path.join(
            args.model_dir, "g.pth"
        )  
        args.load_f_hat = os.path.join(args.model_dir, "f_hat.pth")
        args.n_steps = 1
    
    global device
    if args.no_cuda:
        device = "cpu"
        print("Using cpu")
    if args.seed is not None:
        np.random.seed(args.seed)
        random.seed(args.seed)
        torch.manual_seed(args.seed)

    ############ load dataset ############
    if args.dataset_name == "causal3di":
        args.DATASETCLASS = datasets.Causal3DIdent
    else:
        raise f"{args.dataset_name=} not supported."

    args.datapath = os.path.join(args.dataroot, args.dataset_name)
    
    # loading dataset
    # define kwargs
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                args.DATASETCLASS.mean_per_channel, args.DATASETCLASS.std_per_channel
            ),
            transforms.Resize(64),
        ]
    )
    dataset_kwargs = {
        "transform": transform,
        "class_idx": 6
        
    }  # NOTE: control variability of samples here
    dataset_kwargs["sigma"] = 0.1

    train_dataset = args.DATASETCLASS(
        data_dir=args.datapath,
        mode="train",
        mask_prob=args.mask_prob,
        **dataset_kwargs,
    )
    collate_fn = lambda batch: train_dataset.collate_fn(
        args.masks, args.mask_value, batch
    )

    # define datasets and dataloaders
    # if args.evaluate:
    test_dataset = args.DATASETCLASS(
        data_dir=args.datapath,
        mode="test",
        mask_prob=0.5,
        **dataset_kwargs,
    )

    dataloader_kwargs = {
        "batch_size": args.group_batch_size,
        "shuffle": True,
        "drop_last": True,
        "num_workers": args.workers,
        "pin_memory": True,
    }

    test_loader = DataLoader(test_dataset, collate_fn=collate_fn, **dataloader_kwargs)
    test_iterator = InfiniteIterator(test_loader)

    train_loader = DataLoader(train_dataset, collate_fn=collate_fn, **dataloader_kwargs)
    train_iterator = InfiniteIterator(train_loader)

    

    # constrained ae
    class Constrained_AE(cooper.ConstrainedMinimizationProblem):
        def __init__(self, latent_dim=args.encoding_size, n_classes=len(args.masks)):
            self.criterion = nn.MSELoss(reduction="sum")
            super().__init__(is_constrained=True)

        

        def compute_loss(self, x, x_hat, mu):
            mini_batch = args.group_batch_size

            log_p_x_given_z = self.criterion(x_hat, x)

            stat_loss = 0

            log_var_prior = torch.tensor(
                np.array(args.masks).reshape((len(args.masks), -1))
            ).to(device)
            
            log_var_prior[log_var_prior == 0] = -20
            mu_prior = torch.ones(args.encoding_size).to(device)*2
            
            
            for k in range(len(args.masks)):
                if k == len(args.masks) - 1:
                    mu_k = mu[k * mini_batch :, :]

                else:
                    mu_k = mu[k * mini_batch : (k + 1) * mini_batch, :]

                mean = mu_prior
                diffs = mu_k - mean
                std = (log_var_prior[k, :] / 2).exp()
                zscores = diffs / (std + 1e-9)
                skews = torch.mean(torch.pow(zscores, 3.0), dim=0)
                kurtoses = torch.mean(torch.pow(zscores, 4.0), dim=0) - 3.0

                stat_loss += torch.mean(torch.abs(skews)) + torch.mean(
                    torch.abs(kurtoses)
                )

            return log_p_x_given_z, stat_loss

        def closure(self, inputs):
            x = inputs
            z_hat = g(x)

            x_hat = f_hat(z_hat)

            loss1, stat_loss = self.compute_loss(x, x_hat, z_hat)

            ineq_defect = (
                torch.sum(torch.abs(z_hat))
                / args.group_batch_size
                / args.latent_dim
                / len(args.masks)
                - args.sparse_level
            )

            return cooper.CMPState(
                loss=loss1 + stat_loss,
                ineq_defect=ineq_defect,
                eq_defect=None,
                misc={"recon_loss": loss1, "stat_loss": stat_loss},
            )

    ######################## Define models #####################
    
    g = encoders.ResNetEnc(
        resnet18,
        hidden_size=100,
        encoding_size=args.encoding_size,
        output_normalization="bn",
    )
    f_hat = encoders.ResNetDec(z_dim=args.encoding_size)

    if args.load_g is not None:
        g.load_state_dict(torch.load(args.load_g))

    if args.load_f_hat is not None:
        f_hat.load_state_dict(torch.load(args.load_f_hat))

    g = g.to(device)
    f_hat = f_hat.to(device)
    params = list(g.parameters()) + list(f_hat.parameters())
    
    
    ############### Constrained_AE ###################
    cmp_vade = Constrained_AE()
    
    formulation = cooper.LagrangianFormulation(cmp_vade, args.aug_lag_coefficient)
    primal_optimizer = cooper.optim.ExtraAdam(params, lr=args.lr)
    dual_optimizer = cooper.optim.partial_optimizer(
        cooper.optim.ExtraAdam, lr=args.lr / 2
    )

    coop_vade = cooper.ConstrainedOptimizer(
        formulation=formulation,
        primal_optimizer=primal_optimizer,
        dual_optimizer=dual_optimizer,
    )

    if (
        "total_loss_values" in locals() and not args.resume_training
    ) or "total_loss_values" not in locals():
        total_loss_values = []
        mse_values = []
        norm_values = []
    global_step = len(total_loss_values) + 1
    last_save_at_step = 0
    state_history = OrderedDict()
    while global_step <= args.n_steps:
        if not args.evaluate:
            # training
            g.train()
            f_hat.train()

            data = next(train_iterator)
            data_z, data_x = data
            data_z = data_z.to(device)
            data_x = data_x.to(device)
            coop_vade.zero_grad()
            lagrangian = formulation.composite_objective(cmp_vade.closure, data_x)
            formulation.custom_backward(lagrangian)
            coop_vade.step(cmp_vade.closure, data_x)

        if global_step % args.n_log_steps == 1 or global_step == args.n_steps:
            mcc_scores = []
            Cor_M = []

            test_batch = next(test_iterator)
            z_true = test_batch[0]  # latent
            z_pred = g(test_batch[-1].to(device))  # image
            scaler_hz = StandardScaler()
            scaler_z = StandardScaler()
            hz = scaler_hz.fit_transform(z_pred.detach().cpu().numpy())
            z_true = scaler_z.fit_transform(z_true.detach().cpu().numpy())

            mcc, cor_m = MCC(z_true, hz, args.encoding_size)
            mcc = mcc / args.encoding_size
            mcc_scores.append(mcc)
            Cor_M.append(cor_m)

            fileobj = open(
                os.path.join(args.save_dir, f"{args.dataset_name}.csv"), "a+"
            )
            writer = csv.writer(fileobj)
            wri = ["MCC", mcc]
            writer.writerow(wri)
            np.save(file=os.path.join(args.save_dir, "m3di_cor.npy"), arr=cor_m)
            # wri = ['cor_mean', cor_m.flatten()]
            # writer.writerow(wri)

            fileobj.close()
            print(np.mean(mcc_scores))
            if not args.evaluate:
                print(
                    f"Step: {global_step} \t",
                    f"Recon: {cmp_vade.state.misc['recon_loss']:.4f} \t",
                    f"Stat: {cmp_vade.state.misc['stat_loss']:.4f} \t",
                    f"Loss: {cmp_vade.state.loss.item():.4f} \t",
                    f"ineq constraint: {cmp_vade.state.ineq_defect.item():.4f}",
                )

            if args.save_dir:
                if not os.path.exists(args.save_dir):
                    os.makedirs(args.save_dir)

                save_path = os.path.join(args.save_dir, "g.pth")
                torch.save(g.state_dict(), save_path)
                save_path = os.path.join(args.save_dir, "f_hat.pth")
                torch.save(f_hat.state_dict(), save_path)
        global_step += 1
    plt.plot(mse_values, color="blue", label="MSE")
    plt.show()
    plt.plot(norm_values, color="red", label="Penalty Term")
    plt.show()


if __name__ == "__main__":
    main()