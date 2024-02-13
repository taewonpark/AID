#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import logging
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from task.systematic_associative_recall import SARtask

from models.AID import Network

parser = argparse.ArgumentParser(description='Running SAR Task')

# model parameters
parser.add_argument('-hidden_size', type=int, default=256, help='')
parser.add_argument('-num_hidden', type=int, default=1, help='')
parser.add_argument('-num_input', type=int, default=3, help='')
parser.add_argument('-slot_size', type=int, default=32, help='')
parser.add_argument('-mlp_hidden_size', type=int, default=64, help='')
parser.add_argument('-num_iter', type=int, default=2, help='number of iterations')
parser.add_argument('-mem_size', type=int, default=32, help='')
parser.add_argument('-read_heads', type=int, default=1, help='')
parser.add_argument('-input_proj_size', type=int, default=32, help='')

# task parameters
parser.add_argument('-batch_size', type=int, default=64, metavar='N', help='batch size')
parser.add_argument('-num_sample', type=int, default=100)
parser.add_argument('-ratio', type=float, default=0.0)
parser.add_argument('-embedding_size', type=int, default=50)

# training parameters
parser.add_argument('-lr', type=float, default=1e-3, help='initial learning rate')
parser.add_argument('-optim', type=str, default='adam', help='learning rule, supports adam|rmsprop')
parser.add_argument('-clip', type=float, default=50, help='gradient clipping')

parser.add_argument('-seed', type=int, default=0000, help='')

parser.add_argument('-iterations', type=int, default=30000, metavar='N', help='total number of iteration')
parser.add_argument('-summarize_freq', type=int, default=100, metavar='N', help='summarize frequency')
parser.add_argument('-check_freq', type=int, default=500, metavar='N', help='check point frequency')
parser.add_argument('-log_dir', type=str, default='AID', help='directory to store log data')
parser.add_argument("--logging-level", type=str, metavar='LEVEL', default=20, choices=range(10, 51, 10),
                    help="Logging level (default: 20)")


if __name__ == '__main__':

    args = parser.parse_args()
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=args.logging_level)
    writer = SummaryWriter(log_dir=os.path.join('log', args.log_dir))

    with open(os.path.join('log', args.log_dir, 'config.json'), 'w') as f:
        json.dump(dict(args._get_kwargs()), f, indent=4)

    logging.info(f"\nRunning SAR Task!\n")
    logging.info(f"\nLog name: {args.log_dir}\n")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    random.seed(args.seed)
    np.random.seed(args.seed)
    os.environ["PYTHONHASHSEED"] = str(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = True  # type: ignore

    # Decide task.
    dataloader = SARtask(
        batch_size=args.batch_size,
        num_sample=args.num_sample,
        ratio=args.ratio
    )

    model = Network(
        input_size=args.embedding_size * 2 + 2,
        hidden_size=args.hidden_size,
        output_size=dataloader.output_size,
        embedding_size=args.embedding_size,
        vocab_size=len(dataloader.vocab),
        input_proj_size=args.input_proj_size,
        num_input=args.num_input,
        num_hidden=args.num_hidden,
        slot_size=args.slot_size,
        mlp_hidden_size=args.mlp_hidden_size,
        num_iterations=args.num_iter,
        mem_size=args.mem_size,
        n_read=args.read_heads,
        batch_first=True,
    )

    model = model.to(device)

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"\nThe number of parameters: {pytorch_total_params:d}\n")

    # Optimizer
    if args.optim == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, eps=1e-9, betas=(0.9, 0.98))
    elif args.optim == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters(), lr=args.lr, momentum=0.9, eps=1e-10)
    # Loss function
    loss_fn = nn.CrossEntropyLoss(reduction='none', ignore_index=0)

    # Test samples
    test_enc_inputs, test_dec_inputs, test_outputs = dataloader.get_test_samples()

    fl, f, r = test_enc_inputs
    test_enc_inputs = fl.to(device), f.to(device), r.to(device)
    fl, f, r = test_dec_inputs
    test_dec_inputs = fl.to(device), f.to(device), r.to(device)
    test_outputs = test_outputs.to(device)

    max_valid_acc = 0

    # Validation
    model.eval()
    with torch.no_grad():
        logits, _ = model(
            test_enc_inputs,
            test_dec_inputs,
        )
        logits = logits[:, -test_outputs.shape[1]:]
        valid_acc = (torch.argmax(logits[:, 1:], dim=-1) == test_outputs[:, 1:]).float().mean().item()
        valid_loss = loss_fn(logits.transpose(1, 2), test_outputs).mean().item()

    writer.add_scalar("valid_acc", valid_acc * 100, 0)
    max_valid_acc = valid_acc if max_valid_acc < valid_acc else max_valid_acc

    logging.info(f"\nIteration: {0:5d}"
                 f"\nValid accuracy: {valid_acc * 100:.2f}%, loss: {valid_loss:.3f}\n")


    for epoch in range(int(args.iterations / args.check_freq)):

        # Training
        model.train()
        train_loss = 0

        for n_iter in tqdm(range(args.check_freq)):

            enc_inputs, dec_inputs, outputs = next(dataloader)

            fl, f, r = enc_inputs
            enc_inputs = fl.to(device), f.to(device), r.to(device)
            fl, f, r = dec_inputs
            dec_inputs = fl.to(device), f.to(device), r.to(device)

            optimizer.zero_grad()
            logits, logs = model(
                enc_inputs,
                dec_inputs,
            )
            outputs = outputs.to(device)

            logits = logits[:, -outputs.shape[1]:]

            loss = loss_fn(logits.transpose(1, 2), outputs)
            train_loss += loss.sum().item()
            loss = loss.mean()
            
            train_acc = (torch.argmax(logits, dim=-1)[outputs != 0] == outputs[outputs != 0]).float().sum().item()
            train_num = (outputs != 0).sum().item()

            writer.add_scalar("train_loss", loss.item(), epoch * args.check_freq + n_iter)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.clip)

            optimizer.step()

        train_loss /= args.check_freq * args.batch_size

        # Validation
        model.eval()
        with torch.no_grad():
            logits, _ = model(
                test_enc_inputs,
                test_dec_inputs,
            )
            logits = logits[:, -test_outputs.shape[1]:]
            valid_acc = (torch.argmax(logits[:, 1:], dim=-1) == test_outputs[:, 1:]).float().mean().item()
            valid_loss = loss_fn(logits.transpose(1, 2), test_outputs).mean().item()

        writer.add_scalar("train_acc", train_acc / train_num * 100, (epoch + 1) * args.check_freq)
        writer.add_scalar("valid_acc", valid_acc * 100, (epoch + 1) * args.check_freq)

        if max_valid_acc < valid_acc:
            torch.save(model.state_dict(), os.path.join('log', args.log_dir, 'model.pt'))
            max_valid_acc = valid_acc

        logging.info(f"\nIteration: {(epoch + 1) * args.check_freq:5d} | Train accuracy: {train_acc / train_num * 100:.2f}%, loss: {train_loss:.3f}"
                     f"\nValid accuracy: {valid_acc * 100:.2f}%, loss: {valid_loss:.3f}"
                     f"\nMax valid accuracy: {max_valid_acc * 100:.2f}%\n")
        
    writer.close()

    logging.info(f"\nLog name: {args.log_dir}\n")
