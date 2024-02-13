#!/usr/bin/env bash
python sort_of_clevr_generator.py
for seed in 0000 1111 2222 3333 4444 5555 6666 7777 8888 9999; do
	./run.sh AID 4 256 4 32 3 ${seed} --dot
done