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
from transformers import get_linear_schedule_with_warmup

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from task.bAbIQA import BAbIBatchGenerator, BAbITestBatchGenerator, BAbIValidBatchGenerator
from models.AID import Network


parser = argparse.ArgumentParser(description='Running bAbIQA Task')

# model parameters
parser.add_argument('-hidden_size', type=int, default=256, help='')
parser.add_argument('-num_hidden', type=int, default=1, help='')
parser.add_argument('-num_input', type=int, default=6, help='')
parser.add_argument('-slot_size', type=int, default=32, help='')
parser.add_argument('-mlp_hidden_size', type=int, default=64, help='')
parser.add_argument('-num_iter', type=int, default=3, help='number of iterations')
parser.add_argument('-mem_size', type=int, default=32, help='')
parser.add_argument('-read_heads', type=int, default=3, help='')
parser.add_argument('-input_proj_size', type=int, default=32, help='')

# task parameters
parser.add_argument('-batch_size', type=int, default=64, metavar='N', help='batch size')
parser.add_argument('-embedding_size', type=float, default=256, help='word embedding size')
parser.add_argument('-reconstruction', type=bool, default=True, help='')

# training parameters
parser.add_argument('-lr', type=float, default=1e-3, help='initial learning rate')
parser.add_argument('-optim', type=str, default='adam', help='learning rule, supports adam|rmsprop')
parser.add_argument('-clip', type=float, default=5, help='gradient clipping')

parser.add_argument('-seed', type=int, default=0000, help='')

parser.add_argument('-iterations', type=int, default=60000, metavar='N', help='total number of iteration')
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

    # Decide task.
    train_loader = BAbIBatchGenerator(batch_size=args.batch_size)
    valid_loader = BAbIValidBatchGenerator(batch_size=300)
    test_loader = BAbITestBatchGenerator()

    model = Network(
        input_size=args.embedding_size,
        hidden_size=args.hidden_size,
        output_size=train_loader.output_size,
        vocab_size=train_loader.input_size,
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
    logging.info(f"\nThe number of parameters: {pytorch_total_params}\n")

    # Optimizer
    if args.optim == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, eps=1e-9, betas=(0.9, 0.98))
    elif args.optim == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters(), lr=args.lr, momentum=0.9, eps=1e-10)

    # Loss function
    loss_fn = nn.CrossEntropyLoss(reduction='none', ignore_index=0)

    if os.path.exists(os.path.join('log', args.log_dir, 'model.pt')):
        model.load_state_dict(torch.load(os.path.join('log', args.log_dir, 'model.pt')))

    max_same_valid_acc = 0
    max_diff_valid_acc = 0

    # Validation
    model.eval()
    with torch.no_grad():
        start = time.time()
        valid_loader.reset()
        valid_acc = 0
        valid_size = 0
        for _ in range(valid_loader.limit):
            input_sequence, target_mask, answer, seq_len = next(valid_loader)
            logits = model(
                input_sequence.to(device),
            )

            logits = torch.masked_select(torch.argmax(logits, dim=-1), target_mask.to(device))
            answer = torch.masked_select(answer.to(device), target_mask.to(device))
            valid_acc += (logits == answer).float().sum().item()
            valid_size += target_mask.sum().item()

        valid_acc /= valid_size

        writer.add_scalar("valid_acc", valid_acc * 100, 0)
        max_valid_acc = valid_acc

        logging.info(f"\nIteration: {0:5d} | Valid accuracy: {valid_acc * 100:.2f}% | Time consumption: {(time.time() - start) / 60:.2f}min\n")

    for epoch in range(int(args.iterations / args.check_freq)):
        # Training
        model.train()
        train_loss, train_acc, train_size = 0, 0, 0
        reconstruction_loss, recon_size = 0, 1e-6

        start_train = time.time()
        for n_iter in tqdm(range(args.check_freq)):

            input_sequence, target_mask, answer, seq_len = next(train_loader)

            optimizer.zero_grad()

            if args.reconstruction:
                logits, recon_loss = model(
                    input_sequence.to(device),
                    reconstruction=args.reconstruction,
                )
            else:
                logits = model(
                    input_sequence.to(device),
                )
            answer = answer.to(device)

            loss = loss_fn(logits.transpose(1, 2), answer)

            writer.add_scalar("train_loss", loss.sum().item() / target_mask.sum().item(), epoch * args.check_freq + n_iter)

            train_loss += loss.sum().item()
            loss = loss.mean()

            logits = torch.masked_select(torch.argmax(logits, dim=-1), target_mask.to(device))
            answer = torch.masked_select(answer, target_mask.to(device))
            train_acc += (logits == answer).float().sum().item()
            train_size += target_mask.sum().item()

            if args.reconstruction:
                loss += recon_loss * 1e-2

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.clip)

            optimizer.step()

        end_train = time.time()

        train_acc /= train_size
        train_loss /= args.check_freq * args.batch_size
        writer.add_scalar("train_acc", train_acc * 100, (epoch + 1) * args.check_freq)

        # Validation
        model.eval()
        with torch.no_grad():
            start = time.time()
            valid_loader.reset()
            valid_acc = 0
            valid_size = 0
            for _ in range(valid_loader.limit):
                input_sequence, target_mask, answer, seq_len = next(valid_loader)
                logits = model(
                    input_sequence.to(device),
                )

                logits = torch.masked_select(torch.argmax(logits, dim=-1), target_mask.to(device))
                answer = torch.masked_select(answer.to(device), target_mask.to(device))
                valid_acc += (logits == answer).float().sum().item()
                valid_size += target_mask.sum().item()

            valid_acc /= valid_size

        if max_valid_acc < valid_acc:
            torch.save(model.state_dict(), os.path.join('log', args.log_dir, 'model.pt'))
            max_valid_acc = valid_acc

        writer.add_scalar("valid_acc", valid_acc * 100, (epoch + 1) * args.check_freq)

        logging.info(f"\nIteration: {(epoch + 1) * args.check_freq:5d} | Train accuracy: {train_acc * 100:.2f}%, loss: {train_loss:.3f} | Time consumption: {(end_train - start_train) / 60:.2f}min"
                     f"\nMax valid accuracy: {max_valid_acc * 100:.2f}% | cur valid accuracy: {valid_acc * 100:.2f}% | Time consumption: {(time.time() - start) / 60:.2f}min\n")
    
        # Test
        if (epoch + 1) % 20 == 0:
            model.eval()
            with torch.no_grad():
                tasks_results = {}
                start = time.time()
                for t in os.listdir(test_loader.test_data_dir):

                    task_number, test_size = test_loader.feed_data(t)
                    test_acc, counter = 0, 0
                    results = []

                    test_batch = 50
                    test_loader.feed_batch_size(test_batch)
                    for idx in range(int(test_size / test_batch) + 1):

                        if idx == int(test_size / test_batch):
                            if test_size % test_batch == 0:
                                break
                            test_loader.feed_batch_size(test_size % test_batch)

                        input_sequence, target_mask, answer, seq_len = next(test_loader)
                        logits = model(
                            input_sequence.to(device),
                        )

                        logits = torch.masked_select(torch.argmax(logits, dim=-1), target_mask.to(device))
                        answer = torch.masked_select(answer.to(device), target_mask.to(device))
                        test_acc += (logits == answer).float().sum().item()
                        counter += target_mask.sum().item()

                    test_acc /= counter
                    error_rate = 1. - test_acc
                    tasks_results[task_number] = error_rate

                logging_msg = ""
                str_task = "Task"
                str_result = "Result"

                logging_msg += f"\n\n{str_task:27s}{str_result:s}"
                logging_msg += f"\n-----------------------------------"
                for k in range(20):
                    task_id = str(k + 1)
                    task_result = str("%.2f%%" % (tasks_results[task_id] * 100))
                    logging_msg += f"\n{task_id:27s}{task_result:s}"

                all_tasks_results = [v for _, v in tasks_results.items()]
                results_mean = str("%.2f%%" % (np.mean(all_tasks_results) * 100))
                failed_count = str("%d" % (np.sum(np.array(all_tasks_results) > 0.05)))

                str_mean_err = "Mean Err."
                str_failed = "Failed (err. > 5%)"
                logging_msg += f"\n{str_mean_err:27s}{results_mean:s}"
                logging_msg += f"\n{str_failed:27s}{failed_count:s}"
                logging_msg += f"\n-----------------------------------\n\n"
                logging.info(logging_msg)

    writer.close()
