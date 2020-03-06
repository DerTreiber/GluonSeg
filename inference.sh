source $HOME/setup_itc.sh
IMAGE=("$HOME/git/GluonSeg/resources/images/2007_000061")
GPU=("1")

python3 inference.py ResNet50_16s voc12 --image_names $IMAGE --gpu $GPU