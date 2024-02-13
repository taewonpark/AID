#!/usr/bin/env bash
for seed in 0000 1111 2222 3333 4444 5555 6666 7777 8888 9999; do
	python test.py -log_dir=model/${seed}
done