import random
import numpy 
import os

seed = 1314


numpy.random.seed(seed)
random.seed(seed)

try:
    import torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
except ImportError:
    pass

try:
    import tensorflow
    tensorflow.random.set_seed(seed)
    os.environ["TF_DETERMINISTIC_OPS"] = str(seed)
    os.environ["PYTHONASHSEED"] = str(seed)
except ImportError:
    pass


