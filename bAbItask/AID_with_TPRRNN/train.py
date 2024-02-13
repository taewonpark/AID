#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import random
import json
import time
import logging
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from babi.data_preprocess.preprocess import parse
from models.utils import WarmupScheduler

from models.AID import TprRnn

import wandb

parser = argparse.ArgumentParser(description='Running bAbIQA Task')

# task parameters
parser.add_argument('-batch_size', type=int, default=128, metavar='N', help='batch size')

# training parameters
parser.add_argument('-lr', type=float, default=1e-3, help='initial learning rate')
parser.add_argument('-optim', type=str, default='adam', help='learning rule, supports adam|rmsprop')
parser.add_argument('-scheduler', type=bool, default=True, help='scheduler')
parser.add_argument('-clip', type=float, default=50, help='gradient clipping')

# added parameters for sentence-level
parser.add_argument('-epochs', type=int, default=100, metavar='N', help='total number of epochs')
parser.add_argument('-data_path', type=str, default='./babi/data/en-valid-10k', help='data path')
parser.add_argument('-warm_up_steps', type=int, default=1, metavar='N', help='warm_up_steps')
parser.add_argument('-warm_up_factor', type=float, default=0.1, help='warm_up_factor')
parser.add_argument('-decay', type=bool, default=True, help='decay')
parser.add_argument('-decay_thr', type=float, default=0.1, help='decay_thr')
parser.add_argument('-decay_factor', type=float, default=0.5, help='decay_factor')

parser.add_argument('-reconstruction', type=bool, default=True, help='')
parser.add_argument('-orthogonality', type=bool, default=False, help='')
parser.add_argument('-hsic', type=bool, default=False, help='')

parser.add_argument('-seed', type=int, default=0000, help='')

parser.add_argument('-iterations', type=int, default=60000, metavar='N', help='total number of iteration')
parser.add_argument('-log_dir', type=str, default='AID', help='directory to store log data')
parser.add_argument("--logging-level", type=str, metavar='LEVEL', default=20, choices=range(10, 51, 10),
                    help="Logging level (default: 20)")


if __name__ == '__main__':

    args = parser.parse_args()
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=args.logging_level)
    writer = SummaryWriter(log_dir=os.path.join('log', args.log_dir))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    random.seed(args.seed)
    np.random.seed(args.seed)
    os.environ["PYTHONHASHSEED"] = str(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = True  # type: ignore

    with open(os.path.join('log', args.log_dir, 'config.json'), 'w') as f:
        json.dump(dict(args._get_kwargs()), f, indent=4)

    logging.info(f"\nRunning bAbIQA Task!\n")

    task_ids = range(1, 21)
    word2id = None
    train_data_loaders = {}
    valid_data_loaders = {}
    test_data_loaders = {}

    # Data loader
    num_train_batches = num_valid_batches = num_test_batches = 0
    max_seq = 0
    for i in task_ids:
        train_raw_data, valid_raw_data, test_raw_data, word2id = parse(args.data_path,
                                                                       str(i), word2id=word2id,
                                                                       use_cache=True, cache_dir_ext="")
        train_epoch_size = train_raw_data[0].shape[0]
        valid_epoch_size = valid_raw_data[0].shape[0]
        test_epoch_size = test_raw_data[0].shape[0]

        max_story_length = np.max(train_raw_data[1])
        max_sentences = train_raw_data[0].shape[1]
        max_seq = max(max_seq, train_raw_data[0].shape[2])
        max_q = train_raw_data[0].shape[1]
        valid_batch_size = valid_epoch_size // 73  # like in the original implementation
        test_batch_size = test_epoch_size // 73

        train_dataset = TensorDataset(*[torch.LongTensor(a) for a in train_raw_data[:-1]])
        valid_dataset = TensorDataset(*[torch.LongTensor(a) for a in valid_raw_data[:-1]])
        test_dataset = TensorDataset(*[torch.LongTensor(a) for a in test_raw_data[:-1]])

        train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        valid_data_loader = DataLoader(valid_dataset, batch_size=valid_batch_size)
        test_data_loader = DataLoader(test_dataset, batch_size=test_batch_size)

        train_data_loaders[i] = [iter(train_data_loader), train_data_loader]
        valid_data_loaders[i] = valid_data_loader
        test_data_loaders[i] = test_data_loader

        num_train_batches += len(train_data_loader)
        num_valid_batches += len(valid_data_loader)
        num_test_batches += len(test_data_loader)

    logging.info(f"\ntotal train data: {num_train_batches*args.batch_size}"
                 f"total valid data: {num_valid_batches * valid_batch_size}"
                 f"vocab size {len(word2id)}")

    # model parameters
    model_config = {
        "entity_size": 90,
        "hidden_size": 40,
        "role_size": 20,
        "init_limit": 0.10,
        "LN": True,
        "vocab_size": len(word2id),
        "max_seq": max_seq,
        "symbol_size": len(word2id),
    }
    model = TprRnn(model_config).to(device)

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"\nThe number of parameters: {pytorch_total_params}\n")

    # Optimizer
    if args.optim == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, eps=1e-9, betas=(0.9, 0.99))
    elif args.optim == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters(), lr=args.lr, momentum=0.9, eps=1e-10)

    scheduler = WarmupScheduler(optimizer=optimizer,
                                steps=args.warm_up_steps if args.scheduler else 0,
                                multiplier=args.warm_up_factor if args.scheduler else 1)

    # Loss function
    loss_fn = nn.CrossEntropyLoss(reduction='none', ignore_index=0)
    recon_fn = nn.MultiLabelSoftMarginLoss(reduction='none')

    if os.path.exists(os.path.join('log', args.log_dir, 'model.pt')):
        model.load_state_dict(torch.load(os.path.join('log', args.log_dir, 'model.pt')))

    max_valid_acc = 0

    # Validation
    model.eval()
    correct = 0
    valid_loss = 0
    with torch.no_grad():
        total_valid_samples = 0
        valid_acc_per_task = {}
        for k, va in valid_data_loaders.items():
            valid_data_loader = va
            task_acc = 0
            single_valid_samples = 0
            for story, story_length, query, answer in valid_data_loader:
                logits = model(story.to(device), query.to(device))
                answer = answer.to(device)
                correct_batch = (torch.argmax(logits, dim=-1) == answer).sum()
                correct += correct_batch.item()
                loss = loss_fn(logits, answer)
                valid_loss += loss.sum().item()
                task_acc += correct_batch.item()
                total_valid_samples += story.shape[0]
                single_valid_samples += story.shape[0]

            valid_acc_per_task[k] = task_acc/single_valid_samples

        valid_acc = correct / total_valid_samples
        valid_loss = valid_loss / total_valid_samples
        writer.add_scalar("valid_acc", valid_acc * 100, 0)
        writer.add_scalar("valid_loss", valid_loss, 0)

        logging_msg = ""
        for t_id in task_ids:
            logging_msg += f"\nvalidate acc task {t_id}: {valid_acc_per_task[t_id]}"
        logging.info(logging_msg)

    decay_done = False
    for i in range(1, args.epochs + 1):
        logging.info(f"##### EPOCH: {i} #####")
        scheduler.step()
        model.train()
        correct = 0
        train_loss = 0
        for _ in tqdm(range(num_train_batches)):
            loader_i = random.randint(0, len(train_data_loaders) - 1) + 1
            try:
                story, story_length, query, answer = next(train_data_loaders[loader_i][0])
            except StopIteration:
                train_data_loaders[loader_i][0] = iter(train_data_loaders[loader_i][1])
                story, story_length, query, answer = next(train_data_loaders[loader_i][0])
            optimizer.zero_grad()
            
            if args.reconstruction:
                logits, auxiliary = model(story.to(device), query.to(device), args.reconstruction)
            else:
                logits = model(story.to(device), query.to(device))

            answer = answer.to(device)
            correct_batch = (torch.argmax(logits, dim=-1) == answer).sum()
            correct += correct_batch.item()

            loss = loss_fn(logits, answer)
            train_loss += loss.sum().item()
            loss = loss.mean()

            if args.reconstruction:
                loss += auxiliary['recon_loss'] * 1e-2
        
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.clip)

            optimizer.step()

        train_acc = correct / (num_train_batches * args.batch_size)
        train_loss = train_loss / (num_train_batches * args.batch_size)

        writer.add_scalar("train_acc", train_acc * 100, i)
        writer.add_scalar("train_loss", train_loss, i)

        if args.decay and valid_loss < args.decay_thr and not decay_done:
            scheduler.decay_lr(args.decay_factor)
            decay_done = True

        model.eval()
        correct = 0
        valid_loss = 0
        with torch.no_grad():
            total_valid_samples = 0
            valid_acc_per_task = {}
            for k, va in valid_data_loaders.items():
                valid_data_loader = va
                task_acc = 0
                single_valid_samples = 0
                for story, story_length, query, answer in valid_data_loader:
                    logits = model(story.to(device), query.to(device))
                    answer = answer.to(device)
                    correct_batch = (torch.argmax(logits, dim=-1) == answer).sum()
                    correct += correct_batch.item()
                    loss = loss_fn(logits, answer)
                    valid_loss += loss.sum().item()
                    task_acc += correct_batch.item()
                    total_valid_samples += story.shape[0]
                    single_valid_samples += story.shape[0]

                valid_acc_per_task[k] = task_acc / single_valid_samples

            valid_acc = correct / total_valid_samples
            valid_loss = valid_loss / total_valid_samples
            writer.add_scalar("valid_acc", valid_acc * 100, i)
            writer.add_scalar("valid_loss", valid_loss, i)

            logging_msg = ""
            for t_id in task_ids:
                logging_msg += f"\nvalidate acc task {t_id}: {valid_acc_per_task[t_id]}"
            logging.info(logging_msg)

            if max_valid_acc < valid_acc:
                torch.save(model.state_dict(), os.path.join('log', args.log_dir, 'model.pt'))
                max_valid_acc = valid_acc

        logging.info(f"\nEpoch: {i:4d} | Train accuracy: {train_acc * 100:.2f}%, loss: {train_loss:.3f}"
                     f"\nMax valid accuracy: {max_valid_acc * 100:.2f}% | cur valid accuracy: {valid_acc * 100:.2f}%\n")
    
    writer.close()
