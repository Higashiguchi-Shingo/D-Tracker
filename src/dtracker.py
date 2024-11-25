import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorly as tl
import seaborn as sns
import lmfit
import time, sys, os
from scipy.integrate import odeint
from itertools import product, combinations, permutations
from copy import deepcopy
from tensorly.tenalg import multi_mode_dot
from tensorly.metrics.regression import RMSE
from tensorly.tenalg.svd import svd_interface
from tensorly.decomposition import parafac
import time

import utils
from utils import MAE
import tucker
from rank import RankUpdate
from MDL import costDB, costE, costM


# input:
#   Xc          : input tensor
#   dk, dl, ds  : number of latent dynamics
class DTrackerModel:
    def __init__(self, Xc, dk, dl, ds, seasonal_period, stl_period, ablation_seasonal, ablation_diffusion, outlier=False):
        self.epsilon = 1e-4
        # input tensor
        self.Xc = Xc

        # dimension
        self.dk, self.dl, self.ds = dk, dl, ds      # dim of latent dynamics
        self.lc = Xc.shape[0]                       # length of input tensor
        self.k, self.l = Xc.shape[1], Xc.shape[2]   # dim of input tensor

        # inital params of reaction-diffusion system
        self.A = np.random.rand(self.dk, self.dl).clip(min=self.epsilon,max=0.001)
        self.X = np.random.rand(self.dk, self.dl).clip(min=self.epsilon,max=0.01)
        self.D = np.zeros((self.dk, self.dl, self.dl))

        self.n_p = seasonal_period
        self.stl_period = stl_period
        self.ablation_seasonal = ablation_seasonal
        self.ablation_diffusion = ablation_diffusion
        self.outlier = outlier
        self.Xo = np.zeros(Xc.shape)

    # ----------------------- #
    #    Initialization
    # ----------------------- #
    def initialization(self, init_D=True):
        ### Seaasonal-Trend Decomposition ###
        Xd, Xs, _ = utils.ST_decomp(self.Xc, self.stl_period)
        """if self.ablation_seasonal:
            Xd = self.Xc"""
        if self.ds==0:
            Xd = self.Xc

        ### Initialization of "theta_d" = {A, D, X, Wkey, Wloc} ###
        rank = [self.lc, self.dk, self.dl]
        self.core, self.Wtime, self.Wkey, self.Wloc, _ = tucker.nonnegative_tucker_decomp(Xd, rank=rank)
        self.Wcore = np.tensordot(self.Wtime, self.core, axes=(1,0))

        self.init_rds() # initialization of Reaction Diffusion System (i.e., A, D, X)
        self.gen_Wcore = self.gen_latent(self.lc)
        self.Xd_hat = self.gen_trend(self.lc)

        ### Initialization of "theta_s" = {Stime, skey, Sloc} ###
        if self.ds>=1:
            self.Stime, self.Skey, self.Sloc, self.Xs_hat = tucker.parafac_decomp(Xs, rank=self.ds)
            #avg = (self.Stime[:52] + self.Stime[52:]) / 2
            #self.Stime = np.concatenate([avg, avg], axis=0)
        else:
            self.Stime, self.Skey, self.Sloc, self.Xs_hat = tucker.parafac_decomp(Xs, rank=1)
            self.Stime = np.zeros(self.Stime.shape)
            self.Skey = np.zeros(self.Skey.shape)
            self.Sloc = np.zeros(self.Sloc.shape)
        """if self.ablation_seasonal:
            self.Stime = np.zeros(self.Stime.shape)
            self.Skey = np.zeros(self.Skey.shape)
            self.Sloc = np.zeros(self.Sloc.shape)"""
        self.Xs_hat = self.gen_seasonal(self.lc)


    def init_rds(self, A=None, D=None, X=None, t_eval=None):
        if t_eval is None:
            time_array = np.linspace(0, 1, self.lc)
        else:
            time_array = t_eval

        for i in range(self.dk):
            self.A[i], self.D[i], self.X[i] = self.slice_fit(index=i)


    # ---------------------------- #
    #   Model Estimation
    # ---------------------------- #
    def ModelEstimation(self, init_D=True):
        # initialization
        self.initialization(init_D=init_D)

        pre_error = 1
        iterration = 1

        while True:
            tic = time.time()

            # update "theta_d"
            Xd_hat = self.Xc - self.Xs_hat - self.Xo
            self.update_rds(self.A, self.D, self.X, Xd_hat) # Reaction Diffusion System
            self.gen_Wcore = self.gen_latent(self.lc) # Wcore
            self.update_Wkey(Xd_hat) # Wkey
            self.update_Wloc(Xd_hat) # Wloc
            self.Xd_hat = self.gen_trend(self.lc)

            # update "theta_s"
            #if not self.ablation_seasonal:
            Xs_hat = self.Xc - self.Xd_hat - self.Xo
            if self.ds>=1:
                self.Stime, self.Skey, self.Sloc = self.update_seasonal_factors(Xs_hat)
                self.Xs_hat = self.gen_seasonal(self.lc)

            if self.outlier:
                self.Xo = self.sparse(self.Xc, self.Xd_hat + self.Xs_hat)

            toc = time.time() - tic
            #print("Iter", iterration, "Runtime (s):", toc)

            error = self.reconstruction_error()
            if pre_error - error < self.epsilon: break
            pre_error = error
            iterration = iterration + 1
        
        print("# of outliers:", np.count_nonzero(self.Xo))
        print("# of iterrations:", iterration)


    # --------------------------------- #
    #  single lmfit functions for init
    # --------------------------------- #
    def slice_fit(self, a=None, D=None, x=None, t_eval=None, index=0):
        if t_eval is None:
            time_array = np.linspace(0, 1, self.lc)
        else:
            time_array = t_eval
        
        minimizer = lmfit.Minimizer(self.residual, 
                                    self.slice_init_params(a=a, D=D, x=x, index=index), 
                                    fcn_args=(self.Wcore[:,index,:], time_array))
        out = minimizer.leastsq()
        return self.slice_params2ndarray(out.params)
    
    def slice_init_params(self, a=None, D=None, x=None, index=0):
        params = lmfit.Parameters()

        # growth rate: A
        if a is None: a = self.A[index]
        for i in range(self.dl):
            params.add(f"a{i}", value=a[i], vary=True, min=-2, max=2)
        
        # diffusion strength: D
        if D is None: D = self.D[index]
        if self.ablation_diffusion:
            vary = np.full((self.dl, self.dl), False)
        else:
            vary = np.full((self.dl, self.dl), True)
            for i in range(self.dl):
                vary[i,i] = False
        for j in range(self.dl):
            for k in range(self.dl):
                params.add(f'D{j}{k}', value=D[j][k], vary=vary[j][k], min=0, max=1)

        # initial value: X
        if x is None: x = self.X[index]
        for i in range(self.dl):
            params.add(f'x{i}', value=x[i], vary=True, min=self.epsilon, max=0.15)

        return params
    
    def dxdt(self, X, t, A, D):
        diff = np.zeros(self.dl)
        for i in range(self.dl):
            diff[i] = A[i] * X[i] + np.dot(D[i], tl.clip(X-X[i], a_min=0))
        return diff

    def generate(self, X, time_array, A, D):
        data = odeint(self.dxdt, X, time_array, args=(A, D))
        return data
    
    def residual(self, params, data, time_array):
        # Update parameters
        a, D, x = self.slice_params2ndarray(params)
        pred = self.generate(x, time_array, a, D)
        pred = pred.reshape(self.lc, self.dl)
        return (data - pred).ravel()
    
    def slice_params2ndarray(self, params):
        tmp_a = np.zeros((self.dl, ))
        tmp_D = np.zeros((self.dl, self.dl))
        tmp_x = np.zeros((self.dl, ))

        # "a" and "x"
        for i in range(self.dl):
            tmp_a[i] = params[f"a{i}"]
            tmp_x[i] = params[f"x{i}"]
        # "D"
        for i in range(self.dl):
            for j in range(self.dl):
                tmp_D[i, j] = params[f"D{i}{j}"]

        return tmp_a, tmp_D, tmp_x
    
    # generate latent dynamics with reaction diffusion system
    def gen_latent(self, length):
        ratio = length / self.lc
        time_array = np.linspace(0, 1 * ratio, length)

        Wcore = np.zeros((length, self.dk, self.dl))
        for i in range(self.dk):
            Wcore[:,i,:] = self.generate(self.X[i], time_array, self.A[i], self.D[i])

        return Wcore

    # generate future values of the Trend tensor Xd
    def gen_trend(self, length):
        return multi_mode_dot(self.gen_latent(length), [self.Wkey, self.Wloc], modes=[1,2])
    
    # generate future values of the Seasonal tensor  Xs
    def gen_seasonal(self, length):
        Xs_hat = tl.cp_to_tensor((None, [self.Stime, self.Skey, self.Sloc]))
        ret = np.zeros((length, self.k, self.l))

        if length > self.lc:
            for t in range(0, self.lc):
                ret[t,:,:] = Xs_hat[t,:,:]
            for t in range(self.lc, length):
                ret[t,:,:] = ret[t-self.n_p,:,:]
        else:
            for t in range(0, length):
                ret[t,:,:] = Xs_hat[t,:,:]

        return ret
    
    def gen_outlier(self, length):
        ret = np.zeros((length, self.k, self.l))

        if length > self.lc:
            for t in range(0, self.lc):
                ret[t,:,:] = self.Xo[t,:,:]
        else:
            for t in range(0, length):
                ret[t,:,:] = self.Xo[t,:,:]
        
        return ret
        
    def gen_forecast(self, length):
            if self.ablation_seasonal:
                return self.gen_trend(length) + self.gen_outlier(length)
            else:
                return self.gen_trend(length) + self.gen_seasonal(length) + self.gen_outlier(length)
    

    # --------------------------------------- #
    #  ALS update methods for Trend tensor
    # --------------------------------------- #
    def update_rds(self, A, D, X, Xd_hat, t_eval=None):
        if t_eval is None:
            time_array = np.linspace(0, 1, self.lc)
        else:
            time_array = t_eval
        
        minimizer = lmfit.Minimizer(self.residual_block, 
                                    self.init_params_block(A=A, D=D, X=X), 
                                    fcn_args=(time_array, self.Wkey, self.Wloc, Xd_hat)) 
        out = minimizer.leastsq()

        self.A, self.D, self.X = self.params2ndarray_block(out.params)
    
    def update_Wkey(self, Xd_hat):
        S_key = multi_mode_dot(self.gen_Wcore, [self.Wkey, self.Wloc], modes=[1,2], skip=0)
        S_key = tl.transpose(self.unfold(S_key, 1))

        epsilon = self.epsilon

        numerator = tl.dot(self.unfold(Xd_hat, 1), S_key)
        numerator = tl.clip(numerator, a_min=epsilon, a_max=None)

        denominator = tl.dot(self.Wkey, tl.dot(tl.transpose(S_key), S_key))
        denominator = tl.clip(denominator, a_min=epsilon, a_max=None)

        self.Wkey *= numerator / denominator
    
    def update_Wloc(self, Xd_hat):
        S_loc = multi_mode_dot(self.gen_Wcore, [self.Wkey, self.Wloc], modes=[1,2], skip=1)
        S_loc = tl.transpose(self.unfold(S_loc, 2))

        epsilon = self.epsilon

        numerator = tl.dot(self.unfold(Xd_hat, 2), S_loc)
        numerator = tl.clip(numerator, a_min=epsilon, a_max=None)

        denominator = tl.dot(self.Wloc, tl.dot(tl.transpose(S_loc), S_loc))
        denominator = tl.clip(denominator, a_min=epsilon, a_max=None)

        self.Wloc *= numerator / denominator

    def unfold(self, tensor, mode):
        return tl.reshape(tl.moveaxis(tensor, mode, 0), (tensor.shape[mode], -1))

    def reconstruction_error(self):
        X_hat = self.gen_forecast(self.lc)
        return RMSE(X_hat, self.Xc)
    

    # ------------------------------------------- #
    #  ALS update methods for Seasonal tensor 
    # ------------------------------------------- #
    def update_seasonal_factors(self, Xs_hat):
        old_cp = [None, [self.Stime, self.Skey, self.Sloc]]
        new_cp = parafac(Xs_hat, rank=self.ds, init=old_cp, n_iter_max=1)
        return new_cp[1][0], new_cp[1][1], new_cp[1][2]  

    
    # --------------------------------- #
    # block lmfit methods
    # --------------------------------- #
    def dxdt_block(self, X, t, A, D):
        X = X.reshape(self.dk, self.dl) # (keyword, location) の形に直す
        diff = np.zeros((self.dk, self.dl))
        for i in range(self.dk):
            for j in range(self.dl):
                diff[i,j] = A[i,j] * X[i,j] + np.dot(D[i,j], tl.clip(X[i,:]-X[i,j], a_min=0)) #添字があってるかどうかデバッグ段階で確認する
        return diff.reshape(self.dk * self.dl, )
    
    def generate_block(self, X, time_array, A, D):
        X = X.flatten()
        data = odeint(self.dxdt_block, X, time_array, args=(A, D))  # odeintのパラメータを変える必要あり？(dismoでは色々変更していた)
        return data
    
    def residual_block(self, params, time_array, Wkey, Wloc, Xd_hat):
        # Update parameters
        A, D, X = self.params2ndarray_block(params)
        pred = self.generate_block(X, time_array, A, D)
        pred = pred.reshape(self.lc, self.dk, self.dl)
        reconst = multi_mode_dot(self.gen_latent(self.lc), [Wkey, Wloc], modes=[1,2])
        return (Xd_hat - reconst).ravel()
    
    def params2ndarray_block(self, params):
        tmp_A = np.zeros((self.dk, self.dl))
        tmp_D = np.zeros((self.dk, self.dl, self.dl))
        tmp_X = np.zeros((self.dk, self.dl))
        for i in range(self.dk):
            for j in range(self.dl):
                tmp_A[i,j] = params[f"A{i}{j}"]
                tmp_X[i,j] = params[f"X{i}{j}"]
        for i in range(self.dk):
            for j, k in permutations(range(self.dl), 2):
                tmp_D[i, j, k] = params[f"D{i}{j}{k}"]
        return tmp_A, tmp_D, tmp_X
    
    def init_params_block(self, A=None, D=None, X=None):
        params = lmfit.Parameters()

        #growth rate
        if A is None: A = np.random.rand(self.dk, self.dl)
        for i in range(self.dk):
            for j in range(self.dl):
                params.add(f"A{i}{j}", value=A[i,j], vary=True, min=-2, max=2)

        #diffusion strength
        if D is None: D = np.zeros((self.dk, self.dl, self.dl))  
        if self.ablation_diffusion:
            vary = np.full((self.dk, self.dl, self.dl), False)
        else:
            vary = np.full((self.dk, self.dl, self.dl), True) 
            for i in range(self.dk):
                vary[i][np.diag_indices(self.dl)] = False
        for i in range(self.dk):
            for j, k in permutations(range(self.dl), 2):
                    params.add(f'D{i}{j}{k}', value=D[i][j][k], vary=vary[i][j][k], min=0, max=1)

        #Initial value
        if X is None: X = np.full((self.dk, self.dl), 0.001)
        for i in range(self.dk):
            for j in range(self.dl):
            # max value depends on normalization
                params.add(f'X{i}{j}', value=X[i,j], vary=True, min=self.epsilon, max=0.15)
        
        return params

    
    # --------------------------------- #
    #  methods for saving params
    # --------------------------------- #
    def save_params(self, outdir, ts):
        os.makedirs(outdir + "/model/t=" + str(ts) + "-", exist_ok=True)

        # save the parameters for trend tensor
        np.save(outdir + "/model/t=" + str(ts) + "-/A.npy", self.A)
        np.save(outdir + "/model/t=" + str(ts) + "-/D.npy", self.D)
        np.save(outdir + "/model/t=" + str(ts) + "-/X.npy", self.X)
        np.save(outdir + "/model/t=" + str(ts) + "-/Wkey.npy", self.Wkey)
        np.save(outdir + "/model/t=" + str(ts) + "-/Wloc.npy", self.Wloc)
        np.save(outdir + "/model/t=" + str(ts) + "-/Wcore.npy", self.Wcore)
        np.save(outdir + "/model/t=" + str(ts) + "-/gen_Wcore.npy", self.gen_Wcore)

        # make figures for trend tensor
        plt.figure()
        sns.heatmap(self.A, cmap="bwr", square=True, annot=True, linewidth=1)
        plt.savefig(outdir + "/model/t=" + str(ts) + "-/A.jpg")

        for i in range(self.dk):
            plt.figure()
            sns.heatmap(self.D[i], cmap="bwr", square=True, annot=True, linewidth=1)
            plt.savefig(outdir + "/model/t=" + str(ts) + "-/D" + str(i) + ".jpg")

        plt.figure()
        sns.heatmap(self.X, cmap="bwr", square=True, annot=True, linewidth=1)
        plt.savefig(outdir + "/model/t=" + str(ts) + "-/X.jpg")

        plt.figure()
        sns.heatmap(self.Wkey, cmap="Reds", square=True, linewidth=1)
        plt.savefig(outdir + "/model/t=" + str(ts) + "-/Wkey.jpg")

        plt.figure()
        sns.heatmap(self.Wloc.T, cmap="Reds", square=True, linewidth=1)
        plt.savefig(outdir + "/model/t=" + str(ts) + "-/Wloc.jpg")

        for i in range(self.dk):
            plt.figure()
            plt.plot(self.Wcore[:,i,:], color="lightgrey")
            plt.plot(self.gen_Wcore[:,i,:])
            plt.savefig(outdir + "/model/t=" + str(ts) + "-/fit" + str(i) + ".jpg")
        
        # save parameter for seasonal tensor
        np.save(outdir + "/model/t=" + str(ts) + "-/Stime.npy", self.Stime)
        np.save(outdir + "/model/t=" + str(ts) + "-/Skey.npy", self.Skey)
        np.save(outdir + "/model/t=" + str(ts) + "-/Sloc.npy", self.Sloc)

        # make figures for seasonal tensor
        plt.figure()
        plt.plot(self.Stime)
        plt.savefig(outdir + "/model/t=" + str(ts) + "-/Stime.jpg")

        plt.figure()
        sns.heatmap(self.Skey, cmap="bwr", square=True, linewidth=1)
        plt.savefig(outdir + "/model/t=" + str(ts) + "-/Skey.jpg")

        plt.figure()
        sns.heatmap(self.Sloc.T, cmap="bwr", square=True, linewidth=1)
        plt.savefig(outdir + "/model/t=" + str(ts) + "-/Sloc.jpg")


    # Sparsify the outlier tensor
    def sparse(self, Xc, Xds):
        cost_unit = np.log(Xc.shape[0]) + np.log(Xc.shape[1]) + np.log(Xc.shape[2]) + 32
        current_cost = costE(Xc, Xds)
        Xo = np.zeros(Xc.shape)

        while True:
            resid = Xc - Xds - Xo
            flat_max_idx = np.argmax(resid)
            max_idx = np.unravel_index(flat_max_idx, Xc.shape)
            Xo[max_idx] = resid[max_idx]
            new_cost = costE(Xc, Xds + Xo)
            if current_cost - new_cost < cost_unit:
                Xo[max_idx] = 0
                break
            current_cost = new_cost

        return Xo











class DTracker:
    def __init__(self, X, lc, lf, init, max_dk, max_dl, max_ds, outdir, \
                seasonal_period, stl_period, rankupdate, \
                ablation_seasonal, ablation_diffusion, outlier=0):
        self.X = X                      # data stream
        self.lc, self.lf = lc, lf       # length of input windows and forecasting window
        self.X_hat = np.zeros(X.shape)  # fitting result
        self.t = 0                      # current time-step
        self.n = X.shape[0]             # length of tensor stream

        self.modelDB = []               # model database
        self.rmses = []                 # RMSE record
        self.maes = []                  # MAE record
        self.timelog = []               # runtime record
        self.model_switchlog = []       # model switch time

        self.max_dk, self.max_dl, self.max_ds = \
                                max_dk, max_dl, max_ds  # maximum dimention
        self.seasonal_period = seasonal_period          # seasonal period: n_p
        self.stl_period = stl_period
        self.outdir = outdir

        self.init = bool(init)
        self.rankupdate = bool(rankupdate)
        self.ablation_seasonal = bool(ablation_seasonal)
        self.ablation_diffusion = bool(ablation_diffusion)
        self.outlier = bool(outlier)

    
    #---------------------#
    #   Initialization    #
    #---------------------#
    def initialization(self, outlog=True):
        if self.init:
            print("Initialization -- grid search")
            self.dk, self.dl, self.ds = self.grid_search(outlog=outlog)
        else:
            print("Initialization")
            self.dk, self.dl, self.ds = 2, 2, 0
        
        print("Best dk:", self.dk)
        print("Best dl:", self.dl)
        print("Best ds:", self.ds)

    # grid search
    def grid_search(self, outlog=True):
        Xc = self.X[:self.lc]
        best_costT = 0
        best_dk = 0
        best_dl = 0
        best_ds = 0

        for dk in range(2, self.max_dk+1):
            for dl in range(2, self.max_dl+1):
                for ds in range(0, self.max_ds+1):
                    if outlog: print("rank:", dk, dl, ds)
                    model = DTrackerModel(Xc, dk=dk, dl=dl, ds=ds, seasonal_period=self.seasonal_period, stl_period=self.stl_period, \
                                        ablation_diffusion=self.ablation_diffusion, ablation_seasonal=self.ablation_seasonal, \
                                        outlier=self.outlier)
                    model.ModelEstimation()
                    Xc_hat = model.gen_forecast(self.lc)

                    # total cost
                    _costM = costM(model)
                    _costE = costE(Xc, Xc_hat)
                    costT = _costM + _costE

                    if outlog:
                        print("Model cost:", _costM)
                        print("Encoding cost:", _costE)
                        print("Total cost:", costT)

                    if best_costT > costT:
                        best_costT = costT
                        best_dk = dk
                        best_dl = dl
                        best_ds = ds
        
        print("Best cost:", best_costT)

        return best_dk, best_dl, best_ds


    #---------------------#
    #    Run D-Tracker    #
    #---------------------#
    def run(self, dt=4):
        # first process
        print("\n")
        print("------------------------------------------------")
        print("current window:", "ts=", 0, "te=", self.lc)
        ts = 0
        Xc = self.X[:self.lc]

        model = DTrackerModel(Xc, dk=self.dk, dl=self.dl, ds=self.ds, seasonal_period=self.seasonal_period, stl_period=self.stl_period, \
                            ablation_seasonal=self.ablation_seasonal, ablation_diffusion=self.ablation_diffusion, \
                            outlier=self.outlier)
        model.ModelEstimation()
        model.save_params(self.outdir, ts)

        # stream processing
        for t in range(self.lc + dt, self.n - self.lf, dt):
            t_final = t
            print("\n")
            print("------------------------------------------------")
            print("current window:", "ts=" ,t - self.lc ,"te=" ,t)
            print("(dk, dl, ds) = ({}, {}, {})".format(self.dk, self.dl, self.ds))

            seq = self.X[ts:t]
            Xc = seq[-self.lc:]
            Xf = self.X[t:t+self.lf]

            tic = time.time()

            ### calculate cost when using current model ###
            generated_seq = model.gen_forecast(t-ts)
            cost_keep = costDB(self.modelDB) + costM(model) + costE(self.X[:t], np.concatenate((self.X_hat[:ts], generated_seq), axis=0))


            ### calculate cost when using two models ###
            new_model = DTrackerModel(Xc[-self.lc:], dk=self.dk, dl=self.dl, ds=self.ds, seasonal_period=self.seasonal_period, stl_period=self.stl_period, \
                                        ablation_seasonal=self.ablation_seasonal, ablation_diffusion=self.ablation_diffusion, outlier=self.outlier)
            new_model.ModelEstimation()
            Xc_hat, Xf_hat = np.split(new_model.gen_forecast(self.lc+self.lf), [self.lc])  # forecast future values

            generated_seq = np.concatenate((generated_seq[:-self.lc], Xc_hat), axis=0)
            cost_switch = costDB(self.modelDB) + costM(model) + costM(new_model) + costE(self.X[:t], np.concatenate((self.X_hat[:ts], generated_seq), axis=0))

            print("cost when keep using current model:", cost_keep)
            print("cost when switch to new model:", cost_switch)

            ### update X_hat, i.e., fitting results ###
            self.X_hat[ts:t-self.lc,:,:] = generated_seq[:t-ts-self.lc]

            ### forecasting accuracy ###
            rmse, mae = RMSE(Xf[-dt:], Xf_hat[-dt:]), MAE(Xf[-dt:], Xf_hat[-dt:])
            print("RMSE:", rmse, "MAE:", mae)
            self.rmses.append(rmse)
            self.maes.append(mae)

            self.make_snapshot(self.outdir + "/forecast/t=" + str(t), Xc, Xc_hat, Xf, Xf_hat)

            ### compare ###
            if cost_keep > cost_switch:
                # if the cost with two models is better
                # switch to the new model
                print("Switch to the new model.")

                self.save_model_toDB(model, ts, t)
                ts = t-self.lc
                self.model_switchlog.append(ts)
                model = new_model

                # update rank (dk, dl, ds)
                if self.rankupdate:
                    current_cost = costM(model) + costE(Xc, model.gen_forecast(self.lc))
                    self.dk, self.dl, self.ds = RankUpdate(self.dk, self.dl, self.ds, Xc, current_cost=current_cost, \
                                                            seasonal_period=self.seasonal_period, stl_period=self.stl_period, \
                                                            max_dk=self.max_dk, max_dl=self.max_dl, max_ds=self.max_ds, \
                                                            ablation_seasonal=self.ablation_seasonal, ablation_diffusion=self.ablation_diffusion)

                # save new model
                model.save_params(self.outdir, ts)

            toc = time.time() - tic
            print("Runtime (s):", toc)
            self.timelog.append(toc)
        
        self.X_hat[-self.lc-self.lf:-self.lf,:,:] = generated_seq[-self.lc:]


    # save model to modelDB
    def save_model_toDB(self, model, ts, te):
        log = {"ts": ts, "te": te, "model": model}
        self.modelDB.append(log)

    # make snapshot
    def make_snapshot(self, outdir, Xc, Xc_hat, Xf, Xf_hat):
        # save forecasting result
        os.makedirs(outdir, exist_ok=True)
        np.save(outdir + "/Xc_hat.npy", Xc_hat)
        np.save(outdir + "/Xc.npy", Xc)
        np.save(outdir + "/Xf_hat.npy", Xf_hat)
        np.save(outdir + "/Xf.npy", Xf)

        # make snapshot for US
        plt.figure()
        plt.plot(range(self.lc), Xc[:,:,-3], color="lightgrey", linewidth=3)
        plt.plot(range(self.lc), Xc_hat[:,:,-3], linewidth=3)
        plt.plot(range(self.lc, self.lc+self.lf), Xf[:,:,-3], color="lightgrey", linewidth=3)
        plt.plot(range(self.lc, self.lc+self.lf), Xf_hat[:,:,-3], color="red", linewidth=4)
        plt.savefig(outdir + "/snapshot_US.jpg")

    # save final result
    def save_result(self):
        rmses = np.array(self.rmses)
        maes = np.array(self.maes)
        timelog = np.array(self.timelog)

        os.makedirs(self.outdir + "/log", exist_ok=True)

        np.save(self.outdir + "/log/timelog.npy" , timelog)
        np.save(self.outdir + "/log/X_hat.npy", self.X_hat)
        np.save(self.outdir + "/log/X.npy", self.X)

        self.save_fitting()

        plt.figure()
        plt.plot(timelog)
        plt.savefig(self.outdir + "/log/timelog.jpg")

        with open(self.outdir + "/log/model_switch.txt", 'w') as file:
            for number in self.model_switchlog:
                file.write(f"{number}\n")  

        print("MAE (mean):", maes.mean())
        print("RMSE (mean):", np.sqrt((rmses**2).mean()))

        self.final_MDLcost()
    
    # save fitting
    def save_fitting(self):
        plt.figure(figsize=(15,3))
        plt.plot(self.X[:,:,-3], color="lightgrey", linewidth=3)
        plt.plot(self.X_hat[:,:,-3], linewidth=3)
        for ts in self.model_switchlog:
            plt.vlines(x=ts, color="black", linestyles="dashed", ymax=1, ymin=0)
        plt.savefig(self.outdir + "/log/fitting.jpg")

    # total modeling cost
    def final_MDLcost(self):
        final_cost = costDB(self.modelDB) + costE(self.X[:-self.lf], self.X_hat[:-self.lf])
        with open(self.outdir + "/log/finalcost.txt", 'w', encoding='utf-8') as file:
            file.write(str(final_cost))
        return final_cost