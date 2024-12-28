#!/bin/bash
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --job-name=mindeye_inference
#SBATCH --account=plgstabilityai-gpu-a100
#SBATCH --time=00:40:00          # total run time limit (HH:MM:SS)
#SBATCH -e slurms/%j.err        # first create a "slurms" folder in current directory to store logs
#SBATCH -o slurms/%j.out
#SBATCH --comment=medarc
#SBATCH --mem=100G
#SBATCH --gres=gpu

source ~/.bashrc
cd /net/pr2/projects/plgrid/plggrai/kzrobek/MindEyeV2/
source fmri/bin/activate 
cd src


jupyter nbconvert recon_inference.ipynb --to python
#jupyter nbconvert enhanced_recon_inference.ipynb --to python

model_name="fmri_init_use_difussion_prior_2GPUS_1SUBJ"

export XDG_CACHE_HOME=/net/pr2/projects/plgrid/plggrai/kzrobek/MindEyeV2/.cache

srun python recon_inference.py \
--model_name=${model_name} \
--data_path=/net/pr2/projects/plgrid/plggrai/kzrobek/MindEyeV2/data  \
--subj=1 \
--n_blocks=4 \
--hidden_dim=1024 \
--batch_size=4 \
--ckpt_path=../train_logs/${model_name}/last.pth