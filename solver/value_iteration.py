import imp


import numpy as np
from typing import Dict, Any, Union


class VITrainer():
    def __init__(self, config: Dict[str, Any] = None, env=None):
        if config is None:
            config = {}
        self.config = config
        self.context = env.context
        self.action_space = env.action_space

        self.config.setdefault('grid_nums', np.array([50, 20]))
        self.config.setdefault('ccrit', 0.5)
        self.config['psi_fn'] = env.env_config['psi_fn']
        self.config['pi_fn'] = env.env_config['pi_fn']
        self.config.setdefault('modeldefs_fn', lambda k, kp, delta: kp-(1-delta)*k)

    def rouwen(self, mu, step, num):
        gamma, delta, theta, rho, sigma = self.context

        # discrete state space
        dscSp = np.linspace(mu - (num-1)/2*step, mu + (num-1)/2*step, num).T

        # transition probability matrix
        q = p = (rho + 1)/2.
        transP = np.array([[p**2, p*(1-q), (1-q)**2],
                           [2*p*(1-p), p*q+(1-p)*(1-q), 2*q*(1-q)],
                           [(1-p)**2, (1-p)*q, q**2]]).T

        while transP.shape[0] <= num - 1:

            # see Rouwenhorst 1995
            len_P = transP.shape[0]
            transP = p * np.vstack((np.hstack((transP, np.zeros((len_P, 1)))), np.zeros((1, len_P+1)))) \
                + (1 - p) * np.vstack((np.hstack((np.zeros((len_P, 1)), transP)), np.zeros((1, len_P+1)))) \
                + (1 - q) * np.vstack((np.zeros((1, len_P+1)), np.hstack((transP, np.zeros((len_P, 1)))))) \
                + q * np.vstack((np.zeros((1, len_P+1)),
                                np.hstack((np.zeros((len_P, 1)), transP))))

            transP[1:-1] /= 2.

        # ensure columns sum to 1
        if np.max(np.abs(np.sum(transP, axis=1) - np.ones(transP.shape))) >= 1e-12:
            print('Problem in rouwen routine!')
            return None
        else:
            return transP.T, dscSp

    def train(self):
        gamma, delta, theta, rho, sigma = self.context
        knpts, znpts = self.config['grid_nums']

        r = 1/gamma - 1
        k_st = (((r+delta)*(1+0.01*delta)-0.005 *
                (delta**2))/theta) ** (1/(theta-1))
        klow = 0.0
        khigh = 2 * k_st
        kgrid = np.linspace(klow, khigh, num=knpts)
        zstep = 2 * 2 * sigma / (znpts - 1)
        Pimat, lnzgrid = self.rouwen(0., zstep, znpts)
        zgrid = np.exp(lnzgrid)

        VF = np.zeros((knpts, znpts))
        VFnew = np.zeros((knpts, znpts))
        PF = np.zeros((knpts, znpts))

        ccrit = self.config['ccrit']
        maxit = 1000
        damp = 1.
        dist = 1.0E+99
        iters = 0

        while (dist > ccrit) and (iters < maxit):
            VFnew.fill(0.0)
            iters = iters + 1
            for i in range(0, knpts):
                for j in range(0, znpts):
                    maxval = -1.0E+98
                    for m in range(0, knpts):
                        # get current period utility
                        I = self.config['modeldefs_fn'](kgrid[i], kgrid[m], delta)
                        # get expected value
                        val = self.config['pi_fn'](
                            kgrid[i], zgrid[j], theta) - self.config['psi_fn'](I, kgrid[i]) - I
                        for n in range(0, znpts):
                            # sum over all possible value of z(t+1) with Markov probs
                            val = val + Pimat[n, j]*VF[m, n]/(1+r)
                            # if this exceeds previous maximum do replacements
                        if val > maxval:
                            maxval = val
                            VFnew[i, j] = val
                            PF[i, j] = kgrid[m]
            dist = np.amax(np.abs(VF - VFnew))
            
            '''if iters % 5 == 0:
                print('iteration: ', iters, 'distance: ', dist, end='\r')'''
            VF = damp*VFnew + (1-damp)*VF

        '''print('Converged after', iters, 'iterations')
        print('Policy function at (', int((knpts-1)/2), ',', int((znpts-1)/2), ') should be',
              kgrid[int((knpts-1)/2)], 'and is', PF[int((knpts-1)/2), int((znpts-1)/2)])'''

        self.kgrid = kgrid
        self.zgrid = zgrid
        self.PF = PF
        
    def policy(self, k = 0, z = 1):
        gamma, delta, theta, rho, sigma = self.context
        i = np.argmin(abs(self.kgrid-k))
        j = np.argmin(abs(self.zgrid-z))
        k_new = self.PF[i,j]
        invest = self.config['modeldefs_fn'](k, k_new, delta)
        return k_new, invest

    def compute_single_action(self, obs: np.array):
        gamma, delta, theta, rho, sigma = self.context
        k, z = obs[:2]
        i = np.argmin(abs(self.kgrid-k))
        j = np.argmin(abs(self.zgrid-z))
        k_new = self.PF[i,j]
        invest = self.config['modeldefs_fn'](k, k_new, delta)
        action = invest * (self.action_space.n - 1.) / k
        return round(np.clip(action, 0, 19))
