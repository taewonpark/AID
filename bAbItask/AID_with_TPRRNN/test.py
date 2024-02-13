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

from models.AID import TprRnn

parser = argparse.ArgumentParser(description='Running bAbIQA Task')

# task parameters
parser.add_argument('-batch_size', type=int, default=32, metavar='N', help='batch size')

parser.add_argument('-data_path', type=str, default='./babi/data/en-valid-10k', help='data path')
parser.add_argument('-sg_data_path', type=str, default='./babi/data/revised', help='data path')
parser.add_argument('-log_dir', type=str, default='AID', help='directory to store log data')
parser.add_argument("--logging-level", type=str, metavar='LEVEL', default=20, choices=range(10, 51, 10),
                    help="Logging level (default: 20)")


if __name__ == '__main__':

    args = parser.parse_args()
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=args.logging_level)
    writer = SummaryWriter(log_dir=os.path.join('log', args.log_dir))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with open(os.path.join('log', args.log_dir, 'config.json'), 'w') as f:
        json.dump(dict(args._get_kwargs()), f, indent=4)

    logging.info(f"\nTesting bAbIQA Task!\n")
    logging.info(f"\n{args.log_dir}\n")

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
        test_batch_size = test_epoch_size // 10

        test_dataset = TensorDataset(*[torch.LongTensor(a) for a in test_raw_data[:-1]])
        test_data_loader = DataLoader(test_dataset, batch_size=test_batch_size)

        test_data_loaders[i] = test_data_loader

        num_test_batches += len(test_data_loader)

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

    if os.path.exists(os.path.join('log', args.log_dir, 'model.pt')):
        model.load_state_dict(torch.load(os.path.join('log', args.log_dir, 'model.pt')))
    else:
        import sys
        sys.exit()

    loss_fn = nn.CrossEntropyLoss(reduction='sum')

    # Validation
    model.eval()
    correct = 0
    test_loss = 0
    with torch.no_grad():
        total_test_samples = 0
        test_result = {}
        for k, te in test_data_loaders.items():
            if not int(k) in [1, 2, 3, 6, 7, 8, 9, 10, 11, 12, 13, 14]:
                continue
            test_data_loader = te
            task_acc = 0
            single_test_samples = 0
            for story, story_length, query, answer in test_data_loader:
                logits = model(story.to(device), query.to(device))
                answer = answer.to(device)
                correct_batch = (torch.argmax(logits, dim=-1) == answer).sum()
                correct += correct_batch.item()
                loss = loss_fn(logits, answer)
                test_loss += loss.item()
                task_acc += correct_batch.item()
                total_test_samples += story.shape[0]
                single_test_samples += story.shape[0]

            test_result[k] = task_acc/single_test_samples

        test_result['acc'] = correct / total_test_samples
        test_result['loss'] = correct / total_test_samples



    test_data_loaders = {}

    # Data loader
    num_train_batches = num_valid_batches = num_test_batches = 0
    max_seq = 0
    for i in task_ids:
        train_raw_data, valid_raw_data, test_raw_data, word2id = parse(args.sg_data_path,
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
        test_batch_size = test_epoch_size // 10

        test_dataset = TensorDataset(*[torch.LongTensor(a) for a in test_raw_data[:-1]])
        test_data_loader = DataLoader(test_dataset, batch_size=test_batch_size)

        test_data_loaders[i] = test_data_loader

        num_test_batches += len(test_data_loader)

    correct = 0
    test_loss = 0
    with torch.no_grad():
        total_test_samples = 0
        test_sg_result = {}
        for k, te in test_data_loaders.items():
            if not int(k) in [1, 2, 3, 6, 7, 8, 9, 10, 11, 12, 13, 14]:
                continue
            test_data_loader = te
            task_acc = 0
            single_test_samples = 0
            for story, story_length, query, answer in test_data_loader:
                logits = model(story.to(device), query.to(device))
                answer = answer.to(device)
                correct_batch = (torch.argmax(logits, dim=-1) == answer).sum()
                correct += correct_batch.item()
                loss = loss_fn(logits, answer)
                test_loss += loss.item()
                task_acc += correct_batch.item()
                total_test_samples += story.shape[0]
                single_test_samples += story.shape[0]

            test_sg_result[k] = task_acc/single_test_samples

        test_sg_result['acc'] = correct / total_test_samples
        test_sg_result['loss'] = correct / total_test_samples

    logging_msg = ""
    for t_id in [1, 2, 3, 6, 7, 8, 9, 10, 11, 12, 13, 14]:
        logging_msg += f"\ntask {t_id:2d}: {(1 - test_result[t_id]) * 100:7.2f} {(1 - test_sg_result[t_id]) * 100:7.2f}"
    logging_msg += f"\nmean err: {(1 - test_result['acc']) * 100:6.2f} {(1 - test_sg_result['acc']) * 100:7.2f}"
    logging.info(logging_msg)
