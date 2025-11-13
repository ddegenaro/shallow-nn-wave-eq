import os
import json
import argparse
from inspect import signature

import torch
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import animation

from wave_equation import Wave
from function import u
from utils import DEVICE

def main(args):
    
    path = os.path.join('experiments', str(args.experiment_num))

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

    mins = args.mins
    maxes = args.maxes
    tr = args.temporal_resolution
    sr = args.spatial_resolution
    
    input_cols = tuple([torch.arange(mins[0], maxes[0], (maxes[0] - mins[0]) / tr)] + [
        torch.arange(start, end, (end - start) / sr)
        for start, end in zip(mins[1:], maxes[1:])
    ])

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    artists = []

    X, Y = torch.meshgrid(input_cols[1], input_cols[2])
    X_flat = X.flatten().to(DEVICE)
    Y_flat = Y.flatten().to(DEVICE)
    X_np = X.cpu().numpy()
    Y_np = Y.cpu().numpy()

    for i in range(len(input_cols[0])):
        
        t = input_cols[0][i].expand(X_flat.shape[0]).unsqueeze(1).to(DEVICE)
        pos = torch.stack((X_flat, Y_flat)).transpose(-1, 0)
        Z_np = model(t, pos).detach().reshape(X.shape).cpu().numpy()
        
        # breakpoint()
        
        surf = ax.plot_surface(
            X_np, Y_np, Z_np,
            cmap=cm.coolwarm,
            linewidth=0,
            antialiased=False
        )
        
        artists.append([surf])
        
    ani = animation.ArtistAnimation(
        fig=fig,
        artists=artists,
        interval=tr
    )
    ani.save(
        filename=os.path.join(path, 'animation.gif'),
        writer="pillow"
    )

def validate(args):
    path = os.path.join('experiments', str(args.experiment_num))
    assert os.path.exists(path)
    assert sorted(os.listdir(path)) == [
        'hparams.json', 'model.pth', 'mse.tsv', 'time.tsv'
    ]

if __name__ == "__main__":

    # inferring dimension from f
    input_dim = len(signature(u).parameters) - 1

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--experiment_num',
        type=int,
        default=1,
        help='Assumes there is an experiment with this number in the experiments directory.'
    )
    parser.add_argument(
        '--mins',
        type=float,
        nargs='+',
        default=[0.] * (input_dim + 1),
        help='Time and position to begin visualization. First dimension interpreted as time.'
    )
    parser.add_argument(
        '--maxes',
        type=float,
        nargs='+',
        default=[1.] * (input_dim + 1),
        help='Time and position to end visualization. First dimension interpreted as time.'
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

    args = parser.parse_args()

    validate(args)

    main(args)