import os
import random
import torch
import numpy as np

SEED = 30
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def mkdirifnotexists(in_pths, chdir=False):
    def _mksingledir(in_pth):
        os.makedirs(in_pth, exist_ok=True)
        return in_pth
