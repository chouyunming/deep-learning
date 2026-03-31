import os
import random
import numpy as np
import torch


def seeding(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def epoch_time(start_time, end_time):
    elapsed = end_time - start_time
    return int(elapsed / 60), int(elapsed % 60)
