import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import shutil
import time
import argparse
import psutil

import dtracker

from statsmodels.tsa.seasonal import STL
from tensorly.metrics.regression import RMSE

import sys
import MDL
sys.path.append('..')
import utils
import tucker


def costT(Xc, dk, dl, ds, seasonal_period, stl_period, ablation_diffusion=False, ablation_seasonal=False):
    print("Searching for (dk, dl, ds) = " + str((dk, dl, ds)))
    lc = Xc.shape[0]

    model = dtracker.DTrackerModel(Xc, dk=dk, dl=dl, ds=ds, seasonal_period=seasonal_period, stl_period=stl_period, \
                                    ablation_seasonal=ablation_seasonal, ablation_diffusion=ablation_diffusion)
    model.ModelEstimation()

    Xc_hat = model.gen_forecast(lc)
    cost = MDL.costM(model) + MDL.costE(Xc, Xc_hat)

    print("Cost:", cost)
    return cost


# input
#   rank: current rank (dk, dl, ds)
#   X: input tensor
#   current_cost
#   seasonal_period, stl_period

# output
#   updated rank_d (dk, dl, ds)

def RankUpdate(dk, dl, ds, Xc, current_cost, seasonal_period, stl_period, max_dk=6, max_dl=6, max_ds=6, min_dk=2, min_dl=2, min_ds=0, \
                ablation_seasonal=False, ablation_diffusion=False):
    print("Updating dk, dl and ds ...")

    best_dk, best_dl, best_ds = dk, dl, ds
    best_cost = current_cost

    print("Current cost:", current_cost)

    ### Update the rank for keyword ###
    # search for "dk + 1"
    if dk < max_dk:
        cost = costT(Xc, dk+1, dl, ds, seasonal_period=seasonal_period, stl_period=stl_period)
        if best_cost > cost:
            best_dk, best_dl, best_ds = dk+1, dl, ds
            best_cost = cost

    # search for "dk - 1"
    if dk > min_dk:
        cost = costT(Xc, dk-1, dl, ds, seasonal_period=seasonal_period, stl_period=stl_period)
        if best_cost > cost:
            best_dk, best_dl, best_ds = dk-1, dl, ds
            best_cost = cost

    
    ### Update the rank for location ###
    # search for "dl + 1"
    if dl < max_dl:
        cost = costT(Xc, dk, dl+1, ds, seasonal_period=seasonal_period, stl_period=stl_period)
        if best_cost > cost:
            best_dk, best_dl, best_ds = dk, dl+1, ds
            best_cost = cost

    # search for "dl - 1"
    if dl > min_dl:
        cost = costT(Xc, dk, dl-1, ds, seasonal_period=seasonal_period, stl_period=stl_period)
        if best_cost > cost:
            best_dk, best_dl, best_ds = dk, dl-1, ds
            best_cost = cost


    ### Update the rank for seasonality ###
    # search for "ds + 1"
    if ds < max_ds:
        cost = costT(Xc, dk, dl, ds+1, seasonal_period=seasonal_period, stl_period=stl_period)
        if best_cost > cost:
            best_dk, best_dl, best_ds = dk, dl, ds+1
            best_cost = cost

    # search for "ds - 1"
    if ds > min_ds:
        cost = costT(Xc, dk, dl, ds-1, seasonal_period=seasonal_period, stl_period=stl_period)
        if best_cost > cost:
            best_dk, best_dl, best_ds = dk, dl, ds-1
            best_cost = cost
    
    print("best cost:", best_cost)
    print("updated dk, dl, ds =", best_dk, best_dl, best_ds)

    return best_dk, best_dl, best_ds