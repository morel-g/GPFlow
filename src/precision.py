import torch
import numpy as np

# torch_float_precision = torch.float16
# np_float_precision = np.float16
# np_float_precision = np.float32
torch_float_precision = torch.float  # double
# torch_float_precision = torch.double
# np_float_precision = np.float64
if torch_float_precision == torch.double:
    eps_precision = np.finfo(np.float64).eps
else:
    eps_precision = np.finfo(np.float32).eps
