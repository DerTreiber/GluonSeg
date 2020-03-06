from __future__ import absolute_import
import os

def get_dataset_path(dataset):
    if dataset == 'voc12':
        train_file_path = os.path.join(os.path.expanduser("~"),  '.mxnet/datasets/voc/VOC2012/ImageSets/Segmentation/train.txt')
        val_file_path   = os.path.join(os.path.expanduser("~"),  '.mxnet/datasets/voc/VOC2012/ImageSets/Segmentation/val.txt')
        data_dir        = os.path.join(os.path.expanduser("~"),  '.mxnet/datasets/voc/VOC2012/JPEGImages')
        label_dir       = os.path.join(os.path.expanduser("~"),  '.mxnet/datasets/voc/VOC2012/SegmentationClass')

    return train_file_path, val_file_path, data_dir, label_dir

def get_dataset_classes(dataset):
    if dataset == 'voc12':
        return 21

def get_dataset_suffix(dataset):
    if dataset == 'voc12':
        return '.jpg', '.png'
