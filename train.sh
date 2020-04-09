#!/usr/local_rwth/bin/zsh

### #SBATCH directives need to be in the first part of the jobscript
#SBATCH --time=1
#SBATCH --gres=gpu:1
#SBATCH --output=log_itc.log

### your code goes here, the second part of the jobscript

source $HOME/setup_itc.sh
# python3 train.py Net_0 voc12 --epochs 1000 --batch_size 2 --gpu 0 1
python3 train.py ResNet50_16s voc12 --epochs 2 --batch_size 8 --gpu 0
# python3 train.py UNET_EMADL voc12 --epochs 10 --batch_size 4 --gpu 0 1
# python3 train.py FCN_EMADL voc12 --epochs 2 --batch_size 2 --gpu 0 1
# python3 train.py ResNet50_16s voc12 --epochs 300