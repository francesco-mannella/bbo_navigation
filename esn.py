#!/usr/bin/env python
"""

The MIT License (MIT)

Copyright (c) 2015 Francesco Mannella <francesco.mannella@gmail.com> 

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the
"Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""

import numpy as np
from numpy.random import RandomState as RNG
import time
import os 


def evaluate_rho(M):
    """ 
    Computes the spectral radius of a squared matrix

    Args:
        M (array): a squared matrix
    
    Returns:
        (float) spectral radius of M
    """

    if M.shape [0] != M.shape[1] :
        raise ValueError( "evaluate_rho(M) requires square matrix" );
    
    # find eigenvalues
    eigenvalues = np.linalg.eigvals(M)
    
    # the spectral radius is the maximum between the 
    # absolute values of eigenvalues.
    return  np.abs(eigenvalues).max();


#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
#-- RESERVOIR ---------------------------------------------------------------
#----------------------------------------------------------------------------
#----------------------------------------------------------------------------

class ESN(object):
    ''' 
    A firing-rate esn.
    Internal weights are normalized following the echo-state algorithm
    plus a correction of the infinitesimal rotation/expansion ratio.
    The output function of all units is a positive-truncated tanh.
    '''

    def __init__(
            self,
            name         = "r",
            stime        = 1000,
            dt           = 0.001,
            tau          = 0.01,
            N            = 100,
            alpha        = 0.5,
            beta         = 0.5,
            epsilon      = .5e-8,
            sparseness   = 1.0,
            weight_mean  = 0.0,
            th           = 0.0,
            amp          = 1.0,
            radius_amp   = 1.0,
            trunk        = False,
            noise        = False,
            noise_std    = 0.1,
            rng          = None
            ) :
        """
        Args:
                name (string): name of the object 
                stime (float): time of storage interval (number of timesteps)
                dt (float): integration step (sec)
                tau (float): decay of leaky units (sec)
                N (int): number of units
                alpha (float): infinitesimal expansion
                beta (float): infinitesimal rotation
                epsilon (float): epsilon - spectral radius between is epsilon and 1
                sparseness (float): sparseness of weights
                weight_mean (float): initial mean of random weights 
                th (float): output threshold
                amp (float): output amplitude
                radius_amp (float): spectral radius amplification after normalization
                trunk (bool): output function trunctation
                noise (bool): internal gaussian noise
                noise_std (float): standard deviation of internal noise
                rng (RandomState): random number generator
        """

        self.rng = rng
        if rng is None:
            self.seed = np.frombuffer(os.urandom(4), dtype=np.uint32)[0]
            self.rng = np.random.RandomState(self.seed) 

        # Consts
        self.NAME = name 
        self.DT = dt 
        self.TAU = tau 
        self.STIME = int(stime)
        self.N = N 
        self.ALPHA = alpha
        self.BETA = beta
        self.TH = th
        self.AMP = amp
        self.RADIUS_AMP = radius_amp
        self.TRUNK = trunk
        self.NOISE = noise
        self.NOISE_STD = noise_std
        self.SPARSENESS = sparseness 
        self.WEIGHT_MEAN = weight_mean

        # Variables
        self.pot = np.zeros(self.N)    # potentials 
        self.out = np.zeros(self.N)    # outputs 
        self.w = np.zeros([self.N, self.N])    # inner weights
        
        self.normalize_to_echo( epsilon) 

        # define labels 
        ( 
            self.pot_lab,
            self.out_lab,
            self.inp_lab,
            self.w_norm_lab,
            self.out_mean_lab
        ) = range(5)   
        
        self.data = dict()    # dictionary of stores 
        
        # initialize to zeros
        self.data[self.pot_lab] = np.zeros([self.N, self.STIME])
        self.data[self.out_lab] = np.zeros([self.N, self.STIME])
        self.data[self.inp_lab] = np.zeros([self.N, self.STIME])
        self.data[self.w_norm_lab] = np.zeros(self.STIME)
        self.data[self.out_mean_lab] = np.zeros(self.STIME)

    def normalize_to_echo(self, epsilon) :
        '''
        find the optimized spectral radius of W so that rho(1-epsilon) < Wd < 1,
        where Wd = (dt/tau)*W+(1-(dt/tau))*eye. See Proposition 2 in Jaeger et al. (2007) http://goo.gl/bqGAJu.
        
        Args: 
            epsilon (float): spectral radius must be between epsilon and 1 
        '''        
        
        self.w = self.randomize()
        
        # normalize W so that rho = 1
        self.w = self.normalize(self.w, 1.0)
        
        # the iteration has to reach this value 
        target = 1.0 - epsilon/2.0
        
        # integration step (dt/tau)
        if np.isscalar(self.TAU): 
            h = self.DT/self.TAU
        else:
            h = self.DT/self.TAU.max()
        
        # convenience alias for the identity matrix
        I = np.eye(self.N,self.N) 
        
        e = np.linalg.eigvals(self.w)
        rho = abs(e).max()
        x = e.real
        y = e.imag
        
        # solve quadratic equations
        a = x**2*h**2 + y**2*h**2
        b = 2*x*h - 2*x*h**2
        c = 1 + h**2 - 2*h - target**2
        # just get the positive solutions
        sol = (-b + np.sqrt(b**2 - 4*a*c))/(2*a)
        # and take the minor amongst them
        effective_rho = sol.min()
        
        self.w *= effective_rho

    def randomize(self) :
        ''' 
        Make a random sparse matrix with modulated infinitesimal rotation and translation 
        
        Returns:
            A random matrix 
        '''
        
        # random sparse weigths 
        M = (self.rng.randn(self.N, self.N) + self.WEIGHT_MEAN )* ( 
                (self.rng.rand(self.N, self.N) < self.SPARSENESS) )
        
        # decompose rotation and translation
        M = self.ALPHA*(M+ M.T) + self.BETA*(M - M.T) 

        return M
  
    def normalize(self, M, rho) :
        '''
        Normalize the matrix M so that the spectral radius is rho

        Args: 
            M (array): matrix to be normalized
            rho (float): spectral radius

        Returns:
            the scaled matrix

        '''
        # normalize to spectral radius 1
        return rho * M / evaluate_rho(M)     

    def reset(self):
        '''
        Reset internal units to the initial state
        '''
        self.pot *= 0
        self.out *= 0

    def step(self, inp):  
        '''
        Activation step 
        Args:
            inp (array): external input 

        Returns:
            units activations
        '''

        self.inp = inp
       
        # Collect inputs
        increment = np.dot(self.w, self.out) + inp 
   
        if  self.NOISE :
            increment +=  self.NOISE_STD*self.rng.rand(self.N)

        # Integrate
        self.pot += (self.DT/self.TAU) * \
                ( 
                - self.pot 
                + increment
                )

        # Transfer function
        self.out = np.tanh(self.AMP*(self.pot-self.TH))
        if self.TRUNK : 
            self.out = np.maximum(0, self.out)

        return self.out

    def store(self, timestep):
        '''
        Store activity 

        Args:
            timestep (int): session timestep

        '''
        win_timestep = timestep%self.STIME
        self.data[self.pot_lab][:, win_timestep] = self.pot
        self.data[self.out_lab][:, win_timestep] = self.out 
        self.data[self.inp_lab][:, win_timestep] = self.inp 
        self.data[self.w_norm_lab][win_timestep] = np.linalg.norm(self.w) 
        self.data[self.out_mean_lab][win_timestep] = self.out.mean()

    def reset_data(self):
        '''
        Reset stores 
        '''
        for k in self.data :
            self.data[k] = self.data[k]*0

if __name__ == "__main__":

    import matplotlib.pyplot as plt

    N = 20
    stime = 200

    echo = ESN(
        N       = N,
        stime   = stime,
        dt      = 1.0,
        tau     = 40.0,
        alpha   = 0.1,
        beta    = 0.9,
        epsilon = 1.0e-5)

    input_weights = np.random.randn(N)
    
    # inputs and target trajectory
    x = np.linspace(0, 2*np.pi, stime)
    input_signal = 0.5*(np.sin(x) + 1)
    target = 0.5*((np.sin(x*2) + 0.3*np.cos(x*5+0.6*np.pi)) + 1)
    
    # get rollout
    rollout = np.zeros([N, stime])
    for t in range(stime):
        rollout[:, t] = echo.step(input_signal[t]*input_weights)
    
    # Ridge regression
    lmbd = 0.001
    X = rollout
    Y = target.reshape(1, -1) 
    out_weights = np.matmul(np.matmul(Y, X.T),
            np.linalg.pinv(np.dot(X, X.T) + lmbd*np.eye(N, N)))
    output = np.matmul(out_weights.reshape(1, -1), rollout)

    plt.subplot(311)
    plt.title("ESN units' activation")
    plt.plot(rollout.T)
    plt.subplot(312)
    plt.title("Input signal")
    plt.plot(input_signal)
    plt.subplot(313)
    plt.title("Output")
    tgt, = plt.plot(target, c="red", lw=3)
    out, = plt.plot(output.reshape(-1), c="black")
    plt.legend([tgt, out], ["target", "output"])
    plt.tight_layout()
    plt.show()
