#!/usr/local_rwth/bin/zsh

### #SBATCH directives need to be in the first part of the jobscript
#SBATCH --time=45
#SBATCH --gres=gpu:1
#SBATCH --output=log_itc.log

### your code goes here, the second part of the jobscript

source $HOME/setup_itc.sh
python3 train.py ResNet50_16s voc12