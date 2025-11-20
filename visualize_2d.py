import os
import sys
import json
import argparse

import torch
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import animation

from wave_equation import Wave
from utils import DEVICE

def main(args):
    
    if args.experiment_num < 1:
        experiment_num = str(max([int(x) for x in os.listdir('experiments')]))
    else:
        experiment_num = str(args.experiment_num)
    path = os.path.join('experiments', experiment_num)
    
    sys.path.append(path)
    
    from f import u

    hparams = json.load(
        open(os.path.join(path, 'hparams.json'), 'r', encoding='utf-8')
    )

    model = Wave(
        width=hparams['width'],
        c=hparams['c'],
        input_dim=hparams['input_dim'],
        output_dim=hparams['output_dim']
    ).to(DEVICE)

    model.load_state_dict(torch.load(
        os.path.join(path, 'model.pth')
    ))

    mins = hparams['mins']
    maxes = hparams['maxes']
    tr = args.temporal_resolution
    sr = args.spatial_resolution
    
    input_cols = tuple([torch.arange(mins[0], maxes[0], (maxes[0] - mins[0]) / tr)] + [
        torch.arange(start, end, (end - start) / sr)
        for start, end in zip(mins[1:], maxes[1:])
    ])

    X, Y = torch.meshgrid(input_cols[1], input_cols[2])
    X_flat = X.flatten().to(DEVICE)
    Y_flat = Y.flatten().to(DEVICE)
    X_np = X.cpu().numpy()
    Y_np = Y.cpu().numpy()
    
    for plot_true_sol in (True, False):
        
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        artists = []

        for i in range(len(input_cols[0])):
            
            t = input_cols[0][i].expand(X_flat.shape[0]).unsqueeze(1).to(DEVICE)
            pos = torch.stack((X_flat, Y_flat)).transpose(-1, 0)
            
            if plot_true_sol:
                Z = u(
                    t,
                    pos[:, 0].unsqueeze(1),
                    pos[:, 1].unsqueeze(1)
                )
            else:
                Z = model(t, pos)
            
            Z_np = Z.detach().reshape(X.shape).cpu().numpy()
            
            surf = ax.plot_surface(
                X_np, Y_np, Z_np,
                cmap=cm.coolwarm,
                linewidth=0,
                antialiased=False,
                vmax=args.height
            )
            
            artists.append([surf])
            
        ani = animation.ArtistAnimation(
            fig=fig,
            artists=artists,
            interval=tr
        )
        
        fname = 'animation_true.gif' if plot_true_sol else 'animation.gif'
        ani.save(
            filename=os.path.join(path, fname),
            writer="pillow"
        )

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--experiment_num',
        type=int,
        default=-1,
        help='Assumes there is an experiment with this number in the experiments directory. Default: most recent.'
    )
    parser.add_argument(
        '--temporal_resolution',
        type=float,
        default=100,
        help='Number of time steps to evaluate at per unit time. Default 100.'
    )
    parser.add_argument(
        '--spatial_resolution',
        type=float,
        default=100,
        help='Number of space steps to evaluate at per unit space. Default 100.'
    )
    parser.add_argument(
        '--height',
        type=float,
        default=2,
        help='z-value maximum height for the visualization.'
    )

    args = parser.parse_args()

    main(args)