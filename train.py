import os
import json
import argparse
from time import time

from tqdm import tqdm
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from data_sampler import random_data
from function import f
from wave_equation import Wave

def train_epoch(
    train_loader: DataLoader,
    val_loader: DataLoader,
    model: nn.Module,
    optimizer: torch.optim.AdamW,
    loss_fn: torch.nn.MSELoss,
    epoch: int,
    epochs: int,
    log_freq: int,
    verbose: bool
) -> float:
    
    es = epoch + 1
    print(f'Epoch {es}/{epochs}...', end='\r')
    model.train()
    total_loss_train = 0.

    if verbose:
        enum_train_loader = enumerate(tqdm(train_loader))
        enum_val_loader = enumerate(tqdm(val_loader))
    else:
        enum_train_loader = enumerate(train_loader)
        enum_val_loader = enumerate(val_loader)

    training_start = time()
    for i, (inputs, targets) in enum_train_loader:

        optimizer.zero_grad()
        outputs = model(inputs[:, 0].unsqueeze(1), inputs[:, 1:])
        loss = loss_fn(outputs, targets.unsqueeze(1))
        loss.backward()
        optimizer.step()

        lv = loss.item()
        total_loss_train += lv
        s = i + 1
        if s % log_freq == 0:
            al = total_loss_train / s
            print(f'Epoch: {es:02d} - Step: {s:04d} - Loss: {lv:.4f} - Avg: {al:.4f}')
    training_time = time() - training_start

    model.eval()
    total_loss_val = 0.
    num_examples = 0

    val_start = time()
    for i, (inputs, targets) in enum_val_loader:
        
        outputs = model(inputs[:, 0].unsqueeze(1), inputs[:, 1:])
        loss = loss_fn(outputs, targets.unsqueeze(1))

        total_loss_val += loss.item() * targets.shape[0]
        num_examples += targets.shape[0]
    val_time = time() - val_start

    return (total_loss_val / num_examples) / (len(val_loader)), training_time, val_time

def main(args):

    print(f'Beginning training.')

    train_loader = DataLoader(
        TensorDataset(
            *random_data(
                num_samples=args.n_train,
                spatial_dim=args.input_dim,
                mins=args.train_mins,
                maxes=args.train_maxes,
                f=f
            )
        ),
        batch_size=args.batch_size,
        shuffle=True
    )

    val_loader = DataLoader(
        TensorDataset(
            *random_data(
                num_samples=args.n_val,
                spatial_dim=args.input_dim,
                mins=args.val_mins,
                maxes=args.val_maxes,
                f=f
            )
        ),
        batch_size=args.batch_size,
        shuffle=True
    )

    model = Wave(
        width=args.width,
        c=args.c,
        input_dim=args.input_dim,
        output_dim=args.output_dim
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.wd
    )

    loss_fn = torch.nn.MSELoss()
    
    os.makedirs('experiments', exist_ok=True)
    this_experiment = str(1 + max(
        [int(d) for d in os.listdir('experiments')] + [0]
    ))
    os.makedirs(os.path.join('experiments', this_experiment))

    hparams = vars(args)
    hparams['param_count'] = sum([p.numel() for p in model.parameters()])
    json.dump(
        hparams,
        open(
            os.path.join('experiments', this_experiment, 'hparams.json'),
            'w+', encoding='utf-8'
        ),
        indent=4
    )

    with open(
        os.path.join('experiments', this_experiment, 'mse.tsv'),
        'w+', encoding='utf-8'
    ) as fp:
        fp.write('epoch\tmse\n')

    with open(
            os.path.join('experiments', this_experiment, 'time.tsv'),
            'w+', encoding='utf-8'
        ) as fp:
            fp.write('epoch\ttraining_time\tval_time\n')

    for epoch in range(args.epochs):
        mse, training_time, val_time = train_epoch(
            train_loader=train_loader,
            val_loader=val_loader,
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            epoch=epoch,
            epochs=args.epochs,
            log_freq=args.log_freq,
            verbose=args.verbose
        )

        with open(
            os.path.join('experiments', this_experiment, 'mse.tsv'),
            'a', encoding='utf-8'
        ) as fp:
            fp.write(f'{epoch+1}\t{mse}\n')

        with open(
            os.path.join('experiments', this_experiment, 'time.tsv'),
            'a', encoding='utf-8'
        ) as fp:
            fp.write(f'{epoch+1}\t{training_time}\t{val_time}\n')

        torch.save(
            model.state_dict(),
            os.path.join('experiments', this_experiment, 'model.pth')
        )

        if mse < args.tol: # must improve to break
            print(f'Stopping early at epoch {epoch+1}.')
            break

    print('Done.')
    print(f'Results can be found at {os.path.join('experiments', this_experiment)}.')

def validate(args):
    assert args.n_train > 0
    assert args.n_val > 0
    assert args.batch_size > 0
    assert args.width > 0
    assert args.c > 0
    assert args.input_dim > 0
    assert args.output_dim > 0
    assert args.epochs > 0
    assert args.lr > 0
    assert args.wd > 0
    assert args.log_freq > 0
    assert args.tol > 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--n_train',
        type=int,
        default=100_000,
        help='Number of samples to generate for training. Default 1,000.'
    )
    parser.add_argument(
        '--n_val',
        type=int,
        default=10_000,
        help='Number of samples to generate for validation. Default 10,000.'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=1_000,
        help='Batch size for training and validation. Default 1_000.'
    )
    parser.add_argument(
        '--width',
        type=int,
        default=1000,
        help='Width of the hidden layer of the neural network. Default 10.'
    )
    parser.add_argument(
        '--c',
        type=float,
        default=1.,
        help='Speed of propagation in the medium. Default 1.0.'
    )
    parser.add_argument(
        '--input_dim',
        type=int,
        default=2,
        help='Number of spatial dimensions to be input. Default 2.'
    )
    parser.add_argument(
        '--train_mins',
        type=float,
        nargs='+',
        default=[0., 0., 0.],
        help='Minimum value for each dimension (training data). First dimension interpreted as time.'
    )
    parser.add_argument(
        '--train_maxes',
        type=float,
        nargs='+',
        default=[1., 1., 1.],
        help='Maximum value for each dimension (training data). First dimension interpreted as time.'
    )
    parser.add_argument(
        '--val_mins',
        type=float,
        nargs='+',
        default=[0., 0., 0.],
        help='Minimum value for each dimension (validation data). First dimension interpreted as time.'
    )
    parser.add_argument(
        '--val_maxes',
        type=float,
        nargs='+',
        default=[1., 1., 1.],
        help='Maximum value for each dimension (validation data). First dimension interpreted as time.'
    )
    parser.add_argument(
        '--output_dim',
        type=int,
        default=1,
        help='Number of wave displacement dimensions to be output. Default 1.'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=1_000,
        help='Number of times to show the data to the model.'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=1e-1,
        help='Learning rate.'
    )
    parser.add_argument(
        '--wd',
        type=float,
        default=1e-3,
        help='Weight decay.'
    )
    parser.add_argument(
        '--log_freq',
        type=int,
        default=100,
        help='Show a log message every log_freq steps during training.'
    )
    parser.add_argument(
        '--verbose',
        type=bool,
        default=False,
        help='Log with tqdm.'
    )
    parser.add_argument(
        '--tol',
        type=float,
        default=8e-5,
        help='Stop training if MSE is less than this tolerance.'
    )

    args = parser.parse_args()
    validate(args)
    main(args)