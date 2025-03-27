""" configurations for this project

author baiyu
"""
import os
from datetime import datetime

# Path to the CIFAR-100 dataset
CIFAR100_PATH = './data/cifar100'  # Update this path as needed

# Mean and std of CIFAR-100 dataset
CIFAR100_TRAIN_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
CIFAR100_TRAIN_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)

# Directory to save weights file
CHECKPOINT_PATH = 'checkpoint'

# Total training epochs
EPOCH = 200
MILESTONES = [60, 120, 160]

# Date format for logging
DATE_FORMAT = '%A_%d_%B_%Y_%Hh_%Mm_%Ss'
TIME_NOW = datetime.now().strftime(DATE_FORMAT)

# TensorBoard log directory
LOG_DIR = 'runs'

# Save weights file every SAVE_EPOCH epochs
SAVE_EPOCH = 10
