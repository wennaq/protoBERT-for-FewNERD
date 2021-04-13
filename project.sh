#!/bin/sh
#SBATCH -p cluster121
#SBATCH -N 1 
#SBATCH --job-name=tsne-intra
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --mem=64GB
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-type=fail         # send email if job fails
#SBATCH --mail-user=wennaq@andrew.cmu.edu

python3 projection.py \
	--K 50 \
	--ckpt_file /proto-bert-train-inter.txt-val-inter.txt-5-5.pth.tar \
        --data_path /data/private/qinwenna/FewNERD/test-inter.txt	
