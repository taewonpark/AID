#!/usr/bin/env bash

for ratio in 0.0 0.5 1.0; do
	for seed in 0000 1111 2222 3333 4444 5555 6666 7777 8888 9999; do
		python train.py -ratio=${ratio} -seed=${seed} -log_dir=model/${ratio}/${seed}
done