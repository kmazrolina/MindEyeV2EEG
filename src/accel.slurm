#!/bin/bash
#SBATCH --account=plgstabilityai-gpu-a100
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --gres=gpu:2
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 4
#SBATCH --job=mindeye_no_fmri_use_prior_1subj   
#SBATCH --time=48:00:00          # total run time limit (HH:MM:SS)
#SBATCH -e slurms/%j.err        # first create a "slurms" folder in current directory to store logs
#SBATCH -o slurms/%j.out
#SBATCH --mem=300GB



# Make sure you activate your fmri environment created from src/setup.sh
cd /net/pr2/projects/plgrid/plggrai/kzrobek/MindEyeV2/src
source ../fmri/bin/activate

# The following line converts your jupyter notebook into a python script runnable with Slurm
jupyter nbconvert Train.ipynb --to python

export NUM_GPUS=2 # Set to equal gres=gpu:#!
export BATCH_SIZE=4
export GLOBAL_BATCH_SIZE=$((BATCH_SIZE * NUM_GPUS))

# Make sure another job doesnt use same port, here using random number
export MASTER_PORT=$((RANDOM % (19000 - 11000 + 1) + 11000)) 
export HOSTNAMES=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export COUNT_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l)
export ACCELERATE_LOG_LEVEL=debug

echo MASTER_ADDR=${MASTER_ADDR}
echo MASTER_PORT=${MASTER_PORT}
echo WORLD_SIZE=${COUNT_NODE}

# multisubject pretraining
model_name="eeg_only_use_difussion_prior_2GPUS_1SUBJ"
echo model_name=${model_name}

data_path="/net/pr2/projects/plgrid/plggrai/kzrobek/MindEyeV2/data"
cache_path="/net/pr2/projects/plgrid/plggrai/kzrobek/MindEyeV2/.cache"
export XDG_CACHE_HOME=${cache_path}


accelerate launch \
--num_processes=$(($NUM_GPUS * $COUNT_NODE)) \
--num_machines=$COUNT_NODE \
--main_process_ip=$MASTER_ADDR \
--main_process_port=$MASTER_PORT \
--mixed_precision=bf16 \
Train.py \
--data_path=${data_path} \
--model_name=${model_name} \
--subj=1 \
--batch_size=${BATCH_SIZE} \
--max_lr=3e-4 \
--mixup_pct=.33 \
--num_epochs=100 \
--use_prior \
--prior_scale=30 \
--clip_scale=1 \
--blur_scale=.5 \
--use_image_aug \
--n_blocks=4 \
--hidden_dim=1024 \
--ckpt_interval=1 \
--ckpt_saving \
--wandb_log \
--preprocessing_method=THINGSEEG2_17ch_200ms #\
#--fmri_ckpt=/net/pr2/projects/plgrid/plggrai/kzrobek/MindEyeV2/train_logs/multisubject_subj01_1024hid_nolow_300ep

# singlesubject finetuning
#model_name="finetuned_subj01_40sess"
#echo model_name=${model_name}
#accelerate launch --num_processes=$(($NUM_GPUS * $COUNT_NODE)) --num_machines=$COUNT_NODE --main_process_ip=$MASTER_ADDR --main_process_port=$MASTER_PORT --mixed_precision=fp16 Train.py --data_path=/weka/proj-fmri/shared/mindeyev2_dataset --cache_dir=/weka/proj-fmri/shared/cache --model_name=${model_name} --no-multi_subject --subj=1 --batch_size=${BATCH_SIZE} --max_lr=3e-4 --mixup_pct=.33 --num_epochs=150 --use_prior --prior_scale=30 --clip_scale=1 --no-blurry_recon --blur_scale=.5 --no-use_image_aug --n_blocks=4 --hidden_dim=1024 --num_sessions=40 --ckpt_interval=999 --ckpt_saving --wandb_log --multisubject_ckpt=../train_logs/multisubject_excludingsubj01_40sess
