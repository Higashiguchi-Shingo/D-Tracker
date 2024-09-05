import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os, sys
import shutil
import time
import argparse
import psutil

import dtracker
import utils




# ArgumentParser
parser = argparse.ArgumentParser()

# add
parser.add_argument('--dataset', type=str, help='the name of dataset', default="device")
parser.add_argument("--lf", type=int, default=39)
parser.add_argument("--lc", type=int, default=104)
parser.add_argument("--start_date", type=str, default="2010-01-01")
parser.add_argument("--end_date", type=str, default="2022-12-31")
parser.add_argument("--init", type=int, default=0)
parser.add_argument("--maxdk", type=int, default=4)
parser.add_argument("--maxdl", type=int, default=4)
parser.add_argument("--maxds", type=int, default=2)
parser.add_argument("--seasonal_period", type=int, default=52)
parser.add_argument("--stl_period", type=int, default=26)
parser.add_argument("--rankupdate", type=int, default=0)
parser.add_argument("--out_lf", type=str, default=None)
parser.add_argument("--ablation_seasonal", type=int, default=0)
parser.add_argument("--ablation_diffusion", type=int, default=0)
parser.add_argument("--outlier", type=int, default=0)

args = parser.parse_args()


# load dataset
start_date = args.start_date
end_date = args.end_date
dataset = args.dataset
data_path = "../data/" + dataset + ".csv.gz"

if dataset == "covid":
    X = utils.load_tensor(data_path, time_key="date", facets=["key","location_key"], values="value", start_date=start_date, end_date=end_date, scale="full")
else:
    X = utils.load_tensor(data_path, time_key="date", facets=["query","geo"], values="volume", start_date=start_date, end_date=end_date, scale="full")


# I/O setup
outdir = "../result"   # root
outdir = os.path.join(outdir, args.dataset, "lc=" + str(args.lc), "stl=" + str(args.stl_period), "rankupdate=" + str(args.rankupdate), \
                        "ablation_s=" + str(args.ablation_seasonal), "ablation_d=" + str(args.ablation_diffusion), "outlier=" + str(args.outlier))

if os.path.exists(outdir): shutil.rmtree(outdir)
os.makedirs(outdir, exist_ok=True)

if args.out_lf:
    out_lf = [int(lf) for lf in args.out_lf.split("/")]
else:
    out_lf = args.out_lf


# D-tracker
dtracker = dtracker.DTracker(X=X, lc=args.lc, lf=args.lf, init=args.init, \
                            max_dk=args.maxdk, max_dl=args.maxdl, max_ds=args.maxds, \
                            seasonal_period=args.seasonal_period, stl_period=args.stl_period, \
                            outdir=outdir, rankupdate=args.rankupdate, out_lf=out_lf, \
                            ablation_seasonal=args.ablation_seasonal, ablation_diffusion=args.ablation_diffusion, \
                            outlier=args.outlier)

dtracker.initialization()
dtracker.run()
dtracker.save_result()