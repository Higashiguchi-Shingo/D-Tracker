import numpy as np
import pandas as pd
import tensorly as tl
from tensorly.decomposition import non_negative_tucker, non_negative_tucker_hals, parafac
import time
from tensorly.metrics.regression import RMSE
import matplotlib.pyplot as plt

from tensorly.decomposition import tucker



#------------------------------------------------------------
# Nonnegative Tucker Decomposition
# input:
#   tensor: 3rd-order original tensor
#   rank:   rank of NTD (list)
#
# output:
#   core, factor1, factor2, factor3, reconst error
#------------------------------------------------------------
def nonnegative_tucker_decomp(tensor, rank):
    if rank[0] > rank[1]*rank[2]: rank[0] = rank[1]*rank[2] - 1
    
    tensor_mu, error_mu = non_negative_tucker_hals(tensor, rank=rank, tol=1e-12, n_iter_max=100, return_errors=True)
    return tensor_mu[0], tensor_mu[1][0], tensor_mu[1][1], tensor_mu[1][2], RMSE(tensor, tl.tucker_to_tensor(tensor_mu))


def parafac_decomp(tensor, rank):
    factors = parafac(tensor, rank=rank)
    return factors[1][0], factors[1][1], factors[1][2], tl.cp_to_tensor(factors)





