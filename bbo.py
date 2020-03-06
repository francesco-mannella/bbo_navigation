import sys
import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(precision=3, suppress=True)

def softmax(x, lmb=1):
    e = np.exp((x - np.max(x))/lmb)
    return e/sum(e)

class BBO :

    '''P^2BB: Policy Improvement through Black Vox Optimization
    
        At the begin of each epoch create a bunch of samples of the agent parameters taken from 
        a multivariate gaussian distribution centered on the current set of parameter means.
        
        During the epoch run a bunch of episodes, each using one of the samples of parameters 
        as the current parameters of the agent. For each episode compute the path integral of 
        rewards at each timestep.
        
        Use the path integrals to weight the contribution of each sample and based on that 
        update the parameters mean.

    '''

    def __init__(self, 
            cost_func,
            num_params=10, 
            num_rollouts=20, 
            lmb=0.1, 
            epochs=100, 
            sigma=0.001, 
            sigma_decay_amp=0,
            sigma_decay_period=0.1, 
            softmax=softmax):
        '''

            Args:

                cost_fun (callable): function that gets theta array and returns rews over an episode
                num_params (Int): Number of parameters to optimize 
                num_rollouts (Int): number of rollouts per iteration
                lmb (Float): Temperature of the evaluation softmax
                epochs (Int): Number of iterations
                sigma (Float): Amount of exploration around the mean of parameters
                sigma_decay_amp: Initial additive amplitude of exploration
                sigma_decay_period: Decaying period of additive amplitude of exploration

        '''
       
        self.sigma = sigma
        self.lmb = lmb
        self.num_rollouts = num_rollouts
        self.num_params = num_params
        self.theta = 0*np.random.randn(self.num_params)
        self.Cov = np.eye(self.num_params, self.num_params)
        self.epochs = epochs
        self.decay_amp = sigma_decay_amp
        self.decay_period = sigma_decay_period
        self.epoch = 0

        # define softmax
        self.softmax = softmax        
        # define the cost function 
        self.cost_func = cost_func
 
    def sample(self):
        """ Get num_rollouts samples from the current parameters mean
        """  
        
        Sigma = self.sigma + self.decay_amp*np.exp(
            -self.epoch/(self.epochs * self.decay_period))

        # matrix of deviations from the parameters mean
        self.eps = np.random.multivariate_normal(
            np.zeros(self.num_params), 
            self.Cov * Sigma, self.num_rollouts)
    
    def update(self, Sk):
        ''' Update parameters
        
            Args:

                Sk array(Float): rollout costs in an iteration 

        '''
        # Cost-related probabilities of sampled parameters
        probs = self.softmax(Sk, self.lmb).reshape(self.num_rollouts, 1)
        # update with the weighted average of sampled parameters
        self.theta += np.sum(self.eps * probs, 0)
    
    
    def outcomes(self):
        ''' Compute outcomes for all agents
        
            Returns:

                array of rewards. each row contains rewards over an episode

        '''            
        thetas = self.theta + self.explore*self.eps 
        rews = []
        for theta in thetas:
            rews.append(self.cost_func(theta))           
        return np.vstack(rews)    
    
    def eval(self, rews):
        ''' Evaluate rollouts

            Args:
            
                rews (array): Matrix containing agents' rewards at each timestep (columns) of each rollout (rows) 
            
            Returns:

                overall reward path integral of each rollout 

        '''   
        timesteps = rews.shape[1]
        
        # comute path integrals of rews over each episode
        Sk = np.hstack([np.sum([np.sum(rew[j:-1]) 
            for j in range(timesteps)]) for rew in rews])
        
        return Sk
        
    def iteration(self, explore = True):
        """ Run an iteration

            Args:

                explore (bool): if the iteration is for training (True) or test (False)
            
            Returns: 

                path integrals of iteration episodes
        """
        self.explore = explore
        self.sample()
        rews = self.outcomes()   
        Sk = self.eval(rews)
        self.update(Sk)
        self.epoch += 1

        return Sk
