#!/bin/bash
#SBATCH -N 1
#SBATCH -p gpu
#SBATCH --gpus=1
#SBATCH -C a100-80gb
#SBATCH -c 12
#SBATCH --mem=128GB
#SBATCH --time=2-00:00:00
pwd;hostname;date;

source ~/.bashrc
source ~/venvs/general/bin/activate

dataset_path=/mnt/home/atanelus/ceph/neurips_datasets/audio/sologerbil-4m-e1_audio.h5
config_path=/mnt/home/atanelus/configs/4mic_mc.json5

echo "Running training"
python -u -m vocalocatorssl  \
    --data $dataset_path \
    --config $config_path \
    --save-path ~/ceph/gerbilizer/gcontrastive

date;