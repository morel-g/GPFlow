import torch
import random
import numpy as np


def set_seed_data(data, set_seed=False, seed=12):
    data.seed = seed
    data.set_seed = set_seed
    if data.set_seed:
        print("Seed fixed to ", data.seed)
        set_seed_fun(data.seed)


def set_seed_fun(seed_int):
    torch.manual_seed(seed_int)
    torch.cuda.manual_seed_all(seed_int)
    np.random.seed(seed_int)
    random.seed(seed_int)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # Only with cpu
    torch.use_deterministic_algorithms(True)
