import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import shutil
import time
import argparse
import psutil

import utils
from dtracker import DTrackerModel

# ArgumentParser
parser = argparse.ArgumentParser()

# add
parser.add_argument('--dataset', type=str, help='the name of dataset', default="device")
parser.add_argument("--start_date", type=str, default="2010-01-01")
parser.add_argument("--end_date", type=str, default="2022-12-31")
parser.add_argument("--dk", type=int, default=4)
parser.add_argument("--dl", type=int, default=4)
parser.add_argument("--ds", type=int, default=4)
parser.add_argument("--seasonal_period", type=int, default=52)
parser.add_argument("--stl_period", type=int, default=26)
parser.add_argument("--ablation_seasonal", type=int, default=0)
parser.add_argument("--ablation_diffusion", type=int, default=0)
parser.add_argument("--outlier", type=int, default=0)
parser.add_argument("--ts", type=int, default=0)
parser.add_argument("--te", type=int, default=104)

args = parser.parse_args()

# load dataset
start_date = args.start_date
end_date = args.end_date
dataset = args.dataset
data_path = "../data/" + dataset + ".csv.gz"
outlier = bool(args.outlier)
ablation_seasonal = bool(args.ablation_seasonal)
ablation_diffusion = bool(args.ablation_diffusion)

ts, te = args.ts, args.te
lc = te - ts

X = utils.load_tensor(data_path, time_key="date", facets=["query","geo"], values="volume", start_date=start_date, end_date=end_date, scale="full")

# I/O setup
outdir = "../decompose"   # root
outdir = os.path.join(outdir, args.dataset, "ts=" + str(ts), "te=" + str(te), "stl=" + str(args.stl_period), \
                        "ablation_s=" + str(args.ablation_seasonal), "ablation_d=" + str(args.ablation_diffusion), "outlier=" + str(args.outlier))

if os.path.exists(outdir): shutil.rmtree(outdir)
os.makedirs(outdir, exist_ok=True)

Xc = X[ts:te,:,:]

model = DTrackerModel(Xc, dk=args.dk, dl=args.dl, ds=args.ds, seasonal_period=args.seasonal_period, stl_period=args.stl_period, \
                            ablation_seasonal=ablation_seasonal, ablation_diffusion=ablation_diffusion, \
                            outlier=args.outlier)
model.ModelEstimation()
model.save_params(outdir, ts)