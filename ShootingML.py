# -*- coding: utf-8 -*-
"""
Author: Nicholas T. Smith
Date:   September 1st 2020
Desc:   An implementation of the shooting regressor described here:
        https://arxiv.org/pdf/2009.06172.pdf
"""
from   joblib          import Parallel, parallel_backend, delayed
import numpy           as     np
import pandas          as     pd
from   scipy.optimize  import minimize
from   sklearn.metrics import r2_score
from   sklearn.tree    import DecisionTreeRegressor

def LSGrad(Y, YH):
    return Y - YH

def FastCor(Czz, Czi, Czj, Cij, Cii, Cjj, s):
    return (Czz - s * Czi - s * Czj + s * s * Cij)   / \
           (np.sqrt(s * s * Cii - 2 * s * Czi + Czz) * \
            np.sqrt(s * s * Cjj - 2 * s * Czj + Czz))

def Fit(MF, A, Y, W, ss, bs):
    if (ss is not None) and (0 <= ss <= 1):
        ss = int(np.round(ss * A.shape[0]))
        bi = np.random.choice(A.shape[0], ss, bs)
        A = A[bi]
        Y = Y[bi]
        W = W[bi]
    np.random.seed(0)
    return MF().fit(A, Y, sample_weight=W)

def Pred(M, A, IP):
    I = (A @ IP)
    T = M.predict(A)
    if I.shape != T.shape:
        T = T.reshape(I.shape)
    return I + T

def PredRaw(M, A):
    return M.predict(A)

class ShootingRegressor:
    
    PARAM_SET = dict(L=1.0,    ne=100,   MF=None,   mpar={},   LF=None,
                     RF=None,  ss=None,  bs=False,  dm=None,   n_jobs=1,
                     norm=True)
    
    def __init__(self, L=1.0,    ne=100,   MF=None,   mpar={},   LF=None,
                       RF=None,  ss=None,  bs=False,  dm=None,   n_jobs=1,
                       norm=True):
        self.MF     = DecisionTreeRegressor if MF is None else MF
        self.mpar   = mpar
        self.L      = L
        self.ne     = ne
        self.LF     = LSGrad if (LF is None) else LF
        self.RF     = RF
        self.ss     = ss
        self.bs     = bs
        self.dm     = dm
        self.n_jobs = n_jobs
        self.IP     = None
        self.norm   = norm

    def fit(self, A, Y, sample_weight=None):
        self.os = [-1] + list(Y.shape[1:])
        if len(Y.shape) <= 1:
            Y = Y.reshape(Y.shape[0], -1)
        W = sample_weight
        
        if self.norm:
            A, Y = self.Normalize(A, Y, W)
        else:
            self.Am = self.Ym = 0.
            self.As = self.Ys = 1.
        
        self.Ridge(A, Y, W)
        self.GenerateIP()

        # Initial position offsets
        Ev  = self.IP - self.Mv
        MC  = A @ self.Mv
        EC  = A @ Ev
        Z   = Y - MC
        
        # Initial position correlations
        Czz    = np.cov(Z, Z, False)[0, 1]
        CD     = pd.DataFrame(EC).cov().values
        Ri, Ci = np.triu_indices(self.ne)
        # Mean to initial position correlations
        ZD = np.zeros(self.ne)
        for i in range(self.ne):
            ZD[i] = np.cov(Z, EC[:, i], False)[0, 1]
        # Upper trianglulr indices for correlation matrix
        I, J = np.triu_indices(self.ne, k=1)
        emv  = np.square(EC).mean()

        # Nu estimation objective function
        def OF(xi):
            s   = xi[0]
            gmi = np.abs(s) * emv
            cmi = np.abs(FastCor(Czz, ZD[I], ZD[J], CD[I, J], CD[I, I], CD[J, J], s)).mean()
            return gmi + cmi
        
        mr = minimize(OF, np.array([1.]))
        self.sf = mr.x[0]
        # Scale initial positions by sf
        self.IP = self.Mv + self.sf * Ev
        
        # Fit gradient estimator
        self.FitTrees(A, Y, W)

        self.MW = np.full(self.ne, 1 / self.ne)
        return self

    def FitTrees(self, A, Y, W):
        NG = self.LF(Y, A @ self.IP)     
        MF = lambda : self.MF(**self.mpar)
        if self.n_jobs > 1:
            with parallel_backend('threading', n_jobs=self.n_jobs):
                self.EL = Parallel()(delayed(Fit)(MF, A, NG[:, [i]], W, self.ss, self.bs) for i in range(self.ne))
        else:
            self.EL = [Fit(MF, A, NG[:, [i]], W, self.ss, self.bs) for i in range(self.ne)]
        return self
    
    def GenerateIP(self):
        self.IP = np.random.multivariate_normal(self.Mv[:, 0], self.CM, self.ne).T
        return self

    def get_params(self, deep=True):
        return {k: getattr(self, k, None) for k in ShootingRegressor.PARAM_SET}

    def Normalize(self, A, Y, W):
        self.Am = np.average(A, 0, W)[None, :]
        self.As = A.std(0, keepdims=True)
        self.Ym = np.average(Y, 0, W)[None, :]
        self.Ys = Y.std(0, keepdims=True)
        A = (A - self.Am) / self.As
        Y = (Y - self.Ym) / self.Ys
        return A, Y

    def predict(self, A):
        return (self.Ym + self.Ys * sum(i * j for i, j in zip(self.Predict(A), self.MW))).reshape(self.os)

    def Predict(self, A):
        A = (A - self.Am) / self.As
        with parallel_backend('threading', n_jobs=self.n_jobs):
            PA = Parallel()(delayed(Pred)(Mi, A, self.IP[:, [i]]) for i, Mi in enumerate(self.EL))
        return PA
    
    def PredictAll(self, A):
        return self.Ym + self.Ys * np.hstack([i.reshape(i.shape[0], -1) for i in self.Predict(A)])

    def PredictRaw(self, A):
        with parallel_backend('threading', n_jobs=self.n_jobs):
            PA = Parallel()(delayed(PredRaw)(Mi, A) for i, Mi in enumerate(self.EL))
        return np.hstack([i.reshape(i.shape[0], -1) for i in PA])
    
    def Ridge(self, A, Y, W):
        # Perform a ridge regression
        if W is not None:
            W  = np.sqrt(W)[:, None]
            A *= W
            Y *= W
        U, D, VT = np.linalg.svd(A, False)
        k   = next(iter(np.nonzero(D <= 1e-15)[0]), D.shape[0])
        DK  = D[:k]
        VK  = VT[:k].T
        UTK = U.T[:k]
        np.divide(DK, np.square(DK) + self.L * self.L, out=DK)
        if VK.size > UTK.size: #Use order with fewest operations
            np.multiply(UTK, DK, out=UTK)
        else:
            np.multiply(VK, DK, out=VK)
        PI = np.dot(VK, UTK)
        self.Mv = np.dot(PI, Y)
        #self.Mv = np.divide(Mv, self.As.T, out=Mv)
    
        # Compute covariance of regression coefficients
        YH       = (A @ self.Mv)
        Rv       = Y - YH
        self.CM  = np.dot(PI, np.square(Rv) * PI.T)
        return self
    
    def score(self, A, Y, sample_weight=None):
        YH = self.predict(A)
        return r2_score(Y, YH.reshape(Y.shape), sample_weight=sample_weight)

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self
