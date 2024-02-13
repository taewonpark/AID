#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
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
from task.bAbIQA import BAbITestBatchGenerator, BAbIDiffTestBatchGenerator
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

    logging.info(f"\nRunning bAbIQA Task!\n")

    # Decide task.
    test_loader = BAbITestBatchGenerator()
    test_diff_loader = BAbIDiffTestBatchGenerator()

    model = Network(
        input_size=args.embedding_size,
        hidden_size=args.hidden_size,
        output_size=test_loader.output_size,
        vocab_size=test_loader.input_size,
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

    if os.path.exists(os.path.join('log', args.log_dir, 'model.pt')):
        model.load_state_dict(torch.load(os.path.join('log', args.log_dir, 'model.pt')))
    else:
        sys.exit()

    max_same_valid_acc = 0
    max_diff_valid_acc = 0

    task_numbers = [1, 2, 3, 6, 7, 8, 9, 10, 11, 12, 13, 14]

    model.eval()
    with torch.no_grad():
        # Test
        tasks_results = {}
        start = time.time()
        for t in os.listdir(test_loader.test_data_dir):

            task_number, test_size = test_loader.feed_data(t)
            test_acc, counter = 0, 0
            results = []

            if int(task_number) in task_numbers:
                test_batch = 100
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

        tasks_diff_results = {}
        start = time.time()
        for t in os.listdir(test_diff_loader.test_data_dir):

            task_number, test_size = test_diff_loader.feed_data(t)
            test_acc, counter = 0, 0
            results = []

            if int(task_number) in task_numbers:
                test_batch = 100
                test_diff_loader.feed_batch_size(test_batch)
                for idx in range(int(test_size / test_batch) + 1):

                    if idx == int(test_size / test_batch):
                        if test_size % test_batch == 0:
                            break
                        test_diff_loader.feed_batch_size(test_size % test_batch)

                    input_sequence, target_mask, answer, seq_len = next(test_diff_loader)

                    logits = model(
                        input_sequence.to(device),
                    )

                    logits = torch.masked_select(torch.argmax(logits, dim=-1), target_mask.to(device))
                    answer = torch.masked_select(answer.to(device), target_mask.to(device))
                    test_acc += (logits == answer).float().sum().item()
                    counter += target_mask.sum().item()

                test_acc /= counter
                error_rate = 1. - test_acc
                tasks_diff_results[task_number] = error_rate

        logging_msg = ""
        str_task = "Task"
        str_result = "Result"

        logging_msg += f"\n\n{str_task:27s}{str_result:s}"
        logging_msg += f"\n-----------------------------------"

        for k in task_numbers:
            task_id = str(k)
            task_result = str("%.2f%%" % (tasks_results[task_id] * 100))
            task_diff_result = str("%.2f%%" % (tasks_diff_results[task_id] * 100))
            logging_msg += f"\n{task_id:27s}{task_result:10s}{task_diff_result:10s}"

        all_tasks_results = [v for _, v in tasks_results.items()]
        results_mean = str("%.2f%%" % (np.mean(all_tasks_results) * 100))
        failed_count = str("%d" % (np.sum(np.array(all_tasks_results) > 0.05)))

        all_tasks_results = [v for _, v in tasks_diff_results.items()]
        diff_results_mean = str("%.2f%%" % (np.mean(all_tasks_results) * 100))
        diff_failed_count = str("%d" % (np.sum(np.array(all_tasks_results) > 0.05)))

        str_mean_err = "Mean Err."
        str_failed = "Failed (err. > 5%)"
        logging_msg += f"\n{str_mean_err:27s}{results_mean:10s}{diff_results_mean:10s}"
        logging_msg += f"\n{str_failed:27s}{failed_count:10s}{diff_failed_count:10s}"
        logging_msg += f"\n-----------------------------------\n\n"
        logging.info(logging_msg)

        print((time.time() - start) / 60)
