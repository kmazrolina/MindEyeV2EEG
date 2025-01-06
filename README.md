# MindEye2 for EEG data

Utilizing [MindEye2 model](https://github.com/MedARC-AI/MindEyeV2/tree/main?tab=readme-ov-file) for image reconstruction from EEG data.

## Installation

1. Git clone this repository:

```
git clone https://github.com/MedARC-AI/MindEyeV2EEG.git
cd MindEyeV2EEG/src
```

2. Run ```. setup.sh``` to install a new virtual environment. Make sure the virtual environment is activated with "source venv/bin/activate".

3. Download Data

To train the model you need to download opensource [THINGSEEG2 Dataset](https://osf.io/3jk45/).
You can either download preprocessed data from [PreprocessedEEG direcotry](https://osf.io/anp5v/) or download [RawEEG](https://osf.io/crxs4/) and preprocess the data using `src/data_preprocessing/load_eeg.py` script for custom preprocessing method. 

You can donwload raw data using the following script:
```
cd MindeyeV2EEG/src/data_preprocessing
# # Download THINGSEEG2 raw data for all subjects 1-10
python download_data.py --data_path $data_dir
```


4. Data preparation
With virtual environment activated and inside `src/data_preprocessing` run following commands:
```
# absolute path to the directory where you've downloaded THINGSEEG Raw/Preprocessed data.
data_dir=/<path to>/MindEyeV2/data

# # Prep image path info for model training 
python image_paths.py --data_path $data_dir

# Generate image captions (for evaluation purpouses only)
python image_captions.py --data_path $data_dir 

#optional
# # Preprocess eeg (see preprocessing methods in load_eeg.py)
python load_eeg.py ${data_dir}/RawEEG \
--savePath /net/pr2/projects/plgrid/plggrai/kzrobek/MindEyeV2/data/PreprocessedEEG_THINGS_500ms_rep \
--subjId 2  --sfreq 100 --repMean --tbef -0.2 --taft 0.8 --cutoff1 0 --cutoff2 0.5 --check 2 --mnvdim epochs

```

4. Training
To train the model use of MULTI-GPU setup is recomended. Use `src/accel.slurm` to run the training job using accelerate or run training in jupyter using `src/Train.ipynb` directly.

- You can train the model with initialization of latent space with pretrained fmri model or on eeg data only. You can download fmri pretrained weights from: https://huggingface.co/datasets/pscotti/mindeyev2/tree/main/train_logs/multisubject_subj01_1024hid_nolow_300ep 
- You can train the model on all eeg subjects except one (for evaluation) or on one subject only. 

Example training parameters config (one subject, fmri initialization):
```
--data_path=${data_path} \ #path to PreprocessedEEG PARENT dir
--model_name=${model_name} \ #name the model you will be training 
--subj=2 \ #train model on one subject (subject id is 2 in this case)
--batch_size=${BATCH_SIZE} \ 
--max_lr=3e-5 \
--mixup_pct=.33 \
--num_epochs=150 \
--use_prior \  #use diffusion prior (see MindEye2 paper)
--prior_scale=30 \
--clip_scale=1 \
--blur_scale=.5 \
--use_image_aug \
--n_blocks=4 \
--hidden_dim=1024 \
--ckpt_interval=2 \
--ckpt_saving \
--wandb_log \
--preprocessed_eeg_dir=PreprocessedEEG \
--preprocessing_method=THINGSEEG_200ms_rep_17ch \
--fmri_ckpt=train_logs/multisubject_subj01_1024hid_nolow_300ep #this directory should store predownloded `last.pth` file
```

Example training parameters config (multisubject, training from scratch on eeg only):
```
--data_path=${data_path} \ #path to PreprocessedEEG PARENT dir
--model_name=${model_name} \ #name the model you will be training 
--multi_subject \ #train model on all-except-one subjects
--subj=1 \ # leave subject no. 1 for evaluation purpouses
--batch_size=${BATCH_SIZE} \ 
--max_lr=3e-5 \
--mixup_pct=.33 \
--num_epochs=150 \
--use_prior \  #use diffusion prior (see MindEye2 paper)
--prior_scale=30 \
--clip_scale=1 \
--blur_scale=.5 \
--use_image_aug \
--n_blocks=4 \
--hidden_dim=1024 \
--ckpt_interval=2 \
--ckpt_saving \
--wandb_log \
--preprocessed_eeg_dir=PreprocessedEEG \
--preprocessing_method=THINGSEEG_200ms_rep_17ch 
```

6. Inference
After training run `recon_inference.ipynb` in jupyter. Adjust args to match the model you've trained.
8. Evaluation
Evaluation can be run only after running `recon_inference.ipynb`. Run `src/final_evaluations.ipynb` to evaluate the model. Adjust args to match the model you've trained.


### References
[Gifford, Alessandro T., et al. "A large and rich EEG dataset for modeling human visual object recognition." NeuroImage 264 (2022): 119754.](https://doi.org/10.1016/j.neuroimage.2022.119754)
[Scotti, Paul S., et al. "MindEye2: Shared-Subject Models Enable fMRI-To-Image With 1 Hour of Data." arXiv preprint arXiv:2403.11207 (2024).](https://arxiv.org/abs/2403.11207)
