import numpy as np
from sklearn import preprocessing
from scipy.stats import norm

"MDL cost of diffusion model"

# modelDBを入力として、全モデルのtotal costを計算する関数の実装
# ablationの時も同様の関数でコスト計算できるように修正
# 

def trend_cost(A, D, X0, W_key, W_loc, float_cost=32, tol=1e-3):
    cost = 0

    # cost of A
    nonzero_A = np.count_nonzero(np.logical_or(A < -tol, tol < A))
    #nonzero_A = A.shape[0] * A.shape[1]
    cost += nonzero_A * (np.log(A.shape[0]) + np.log(A.shape[1]) + float_cost)

    # cost of D
    if D is not None:
        nonzero_D = np.count_nonzero(np.logical_or(D < -tol, tol < D))
        #nonzero_D = D.shape[0] * D.shape[1] * D.shape[2]
        cost += nonzero_D * (np.log(D.shape[0]) + np.log(D.shape[1]) + np.log(D.shape[2]) + float_cost)

    # cost of X0
    nonzero_X0 = np.count_nonzero(np.logical_or(X0 < -tol, tol < X0))
    #nonzero_X0 = X0.shape[0] * X0.shape[1]
    cost += nonzero_X0 * (np.log(X0.shape[0]) + np.log(X0.shape[1]) + float_cost)

    # cost of W_key
    nonzero_W_key = np.count_nonzero(np.logical_or(W_key < -tol, tol < W_key))
    #nonzero_W_key = W_key.shape[0] * W_key.shape[1]
    cost += nonzero_W_key * (np.log(W_key.shape[0]) + np.log(W_key.shape[1]) + float_cost)

    # cost of W_loc
    nonzero_W_loc = np.count_nonzero(np.logical_or(W_loc < -tol, tol < W_loc))
    #nonzero_W_loc = W_loc.shape[0] * W_loc.shape[1]
    cost += nonzero_W_loc * (np.log(W_loc.shape[0]) + np.log(W_loc.shape[1]) + float_cost)

    return cost


def seasonal_cost(Stime, Skey, Sloc, float_cost=32, tol=1e-3):
    cost = 0
    # cost of Stime
    #nonzero_Stime = Stime.shape[0] * Stime.shape[1]
    nonzero_Stime = np.count_nonzero(np.logical_or(Stime < -tol, tol < Stime))
    cost += nonzero_Stime * (np.log(Stime.shape[0]) + np.log(Stime.shape[1]) + float_cost)

    # cost of Skey
    #nonzero_Skey = Skey.shape[0] * Skey.shape[1]
    nonzero_Skey = np.count_nonzero(np.logical_or(Skey < -tol, tol < Skey))
    cost += nonzero_Skey * (np.log(Skey.shape[0]) + np.log(Skey.shape[1]) + float_cost)
    # cost of Sloc
    #nonzero_Sloc = Sloc.shape[0] * Sloc.shape[1]
    nonzero_Sloc = np.count_nonzero(np.logical_or(Sloc < -tol, tol < Sloc))
    cost += nonzero_Sloc * (np.log(Sloc.shape[0]) + np.log(Sloc.shape[1]) + float_cost)

    return cost

def costXo(Xo, float_cost=32, tol=1e-3):
    nonzero_Xo = np.count_nonzero(np.logical_or(Xo < -tol, tol < Xo))
    return nonzero_Xo * (np.log(Xo.shape[0]) + np.log(Xo.shape[1]) + np.log(Xo.shape[2]) + float_cost)

def costM(model):
    #print("costM", trend_cost(model.A, model.D, model.X, model.Wkey, model.Wloc) + seasonal_cost(model.Stime, model.Skey, model.Sloc))
    return trend_cost(model.A, model.D, model.X, model.Wkey, model.Wloc) + seasonal_cost(model.Stime, model.Skey, model.Sloc) + costXo(model.Xo)


def costE(X, Y, avoid_negative_score=True):
    """
        X: input data (matrix/tensor)
        Y: reconstruction of X
    """
    diff = (X - Y).ravel()
    prob = norm.pdf(diff, loc=diff.mean(), scale=diff.std())

    # if avoid_negative_score == True:
    #     prob[prob > 1] = 1.

    # print(-1 * np.log2(prob).sum())
    #print("costE", -1 * np.log2(np.maximum(prob, 1e-10)).sum())
    return -1 * np.log2(np.maximum(prob, 1e-10)).sum()

def costDB(modelDB):
    total_cost = 0
    for model in modelDB:
        total_cost += costM(model["model"])
        
    return total_cost