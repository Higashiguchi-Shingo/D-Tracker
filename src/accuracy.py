import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorly as tl
import seaborn as sns
import argparse
import os

from tensorly.metrics.regression import RMSE
from utils import MAE


def concat(X, X_dt):
    if len(X)==0:
        return X_dt
    else:
        return np.concatenate([X, X_dt], axis=0)

# ArgumentParser
parser = argparse.ArgumentParser()

# add
parser.add_argument('--dataset', type=str, help='the name of dataset', default="device")
parser.add_argument("--lc", type=int, default=104)
parser.add_argument("--lf", type=str, default=None)
parser.add_argument("--start_timestep", type=int, default=312)
parser.add_argument("--end_timestep", type=int, default=636)
parser.add_argument("--stl_period", type=int, default=26)
parser.add_argument("--rankupdate", type=int, default=0)
parser.add_argument("--ablation_seasonal", type=int, default=0)
parser.add_argument("--ablation_diffusion", type=int, default=0)
parser.add_argument("--outlier", type=int, default=0)

args = parser.parse_args()

lf = [int(i) for i in args.lf.split("/")]

# I/O setup
root = "../result"   # root
dir = os.path.join(root, args.dataset, "lc=" + str(args.lc), "stl=" + str(args.stl_period), "rankupdate=" + str(args.rankupdate), \
                        "ablation_s=" + str(args.ablation_seasonal), "ablation_d=" + str(args.ablation_diffusion), "outlier=" + str(args.outlier), "forecast")

outdir = os.path.join(root, args.dataset, "lc=" + str(args.lc), "stl=" + str(args.stl_period), "rankupdate=" + str(args.rankupdate), \
                        "ablation_s=" + str(args.ablation_seasonal), "ablation_d=" + str(args.ablation_diffusion), "outlier=" + str(args.outlier), "log")

dt = 4


# accuracy
for l in lf:
    X = np.array([])
    X_hat = np.array([])

    for t in range(args.start_timestep, args.end_timestep, dt):
        path = dir + "/t=" + str(t)
        Xf = np.load(path + "/Xf.npy")[l-dt:l]
        Xf_hat = np.load(path + "/Xf_hat.npy")[l-dt:l]

        X = concat(X, Xf)
        X_hat = concat(X_hat, Xf_hat)
    
    print("lf=" + str(l))
    print("MAE:", MAE(X, X_hat))
    print("RMSE:", RMSE(X, X_hat))

    np.save(outdir + "/Xf.npy", X)
    np.save(outdir + "/Xf_hat.npy", X_hat)

    plt.figure(figsize=(15,3))
    plt.plot(X[:,:,-3], color="lightgrey", linewidth=3)
    plt.plot(X_hat[:,:,-3], linewidth=3)
    plt.savefig(outdir + "/forecast" + str(l) + ".jpg")

    with open(outdir + "/accurcy" + str(l) + ".txt", 'w', encoding='utf-8') as file:
        file.write("MAE: " + str(MAE(X, X_hat)))
        file.write("\n")
        file.write("RMSE: " + str(RMSE(X, X_hat)))

