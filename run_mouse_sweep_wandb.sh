#!/bin/bash

sweep_save_dir=/mnt/home/atanelus/ceph/experiments/mouse_model_size_sweep
ceph_dataset_path=/mnt/home/atanelus/ceph/datasets/mouse_mf_ff
data_basename=$(basename $ceph_dataset_path)
data_path=/tmp/$data_basename

source ~/.bashrc
source ~/venvs/new/bin/activate

hostname; date;

# Copy data to local storage
if [ ! -d ${data_path} ]; then
    echo "Copying data to ${data_path}"
    rsync -aPh ${ceph_dataset_path}/ /tmp/$data_basename
else
    if [ ! $(python /mnt/home/atanelus/ceph/experiments/entropy_sweep_ssl/wait_until_dir_is_copied.py ${ceph_dataset_path} ${data_path}) ]; then
        rsync -aPh ${ceph_dataset_path}/ /tmp/$data_basename
    fi
fi

# Run the sweep
sweep_id="aramist/vocalocator-ssl/x9huwros"
export SWEEP_SAVE_DIR=${sweep_save_dir}
export DATA_PATH=${data_path}
wandb agent $sweep_id
