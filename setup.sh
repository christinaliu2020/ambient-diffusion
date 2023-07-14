#!/bin/bash

sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt-get install python3.8
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1

apt-get install python3-pip
apt-get install rsync
sudo apt-get install vim

git clone https://github.com/christinaliu2020/ambient-diffusion.git

pip3 install git+https://github.com/huggingface/diffusers.git

pip3 install -r requirements.txt

python dataset_tool.py --source=downloads/cifar10/cifar-10-python.tar.gz \
    --dest=datasets/cifar10-32x32.zip

torchrun --standalone --nproc_per_node=1 train.py --outdir=result --experiment_name=ambient-diffusion --dump=200 --cond=0 --arch=ddpmpp --precond=ambient --cres=1,2,2,2 --lr=2e-4 --dropout=0.1 --augment=0.15 --data=dataset --norm=2 --max_grad_norm=1.0 --mask_full_rgb=True --corruption_probability=0.4 --delta_probability=0.1 --batch=256 --max_size=30000