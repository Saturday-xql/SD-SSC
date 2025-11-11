import torch
import time
from tqdm import tqdm
from utils.metrics import Accuracy, MIoU, CompletionIoU
from utils.misc import get_run
# from utils.data import sample2dev
import torch.nn.functional as F
from torch import optim
import sys
import logging
import os
import random
import numpy as np
nyu_classes = ["ceil", "floor", "wall", "window", "chair", "bed", "sofa", "table", "tvs", "furniture", "objects",
               "empty"]

def setup_logger(log_file):
    if os.path.exists(log_file):
        os.remove(log_file)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file), 
            logging.StreamHandler()         
        ]
    )
    return logging.getLogger()

def seed_torch(seed=2025):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
