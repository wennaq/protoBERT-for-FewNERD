#!/bin/sh
#SBATCH -p cluster121
#SBATCH -N 1
#SBATCH --gres=gpu:2
#SBATCH --job-name=intra-5-5
#SBATCH --output=%j.out
#SBATCH --error=%j.err

python3 train-demo.py \
	--batch_size 2 \
	--train /data/private/qinwenna/FewNERD/train-intra.txt \
	--val /data/private/qinwenna/FewNERD/val-intra.txt \
	--test /data/private/qinwenna/FewNERD/test-intra.txt \
	--trainN 5 --N 5 --K 5 --Q 5 >trainlog/intra-5-5-2gpu.log
