import numpy as np

from bbo import BBO
from arena_env import Agent, ArenaEnv, Plotter

class Objective:
    ''' Defines a callable object to be used by the BBO object
        for running single episodes
    '''

    def __init__(self, env, agent, stime):
        '''
            Args:
                
                env (ArenaEnv): environment that manages the simulation
                agent (Agent): controller. Computes actions based on info from the env
                stime (int): simulation time in timesteps 
        '''

        self.env = env
        self.agent = agent
        self.stime = stime
        self.plotter = Plotter(self.env)
    
    def __call__(self, theta, plot=False):
        ''' Call operator runnning a single episode.
            
            Args:
                
                theta (array): parameters the agent is set to at beginning of episode
                plot (bool): whether rendering is enabled or not

            Returns:

                rews (array): the reward values at each timestep of the episode
        '''
        rews = np.zeros(self.stime)
        
        self.agent.setParams(theta)
        self.agent.reset()
        status = self.env.reset()
        for t in range(self.stime):
            action = self.agent.step(status)
            status, reward = self.env.step(action)
            rews[t] = reward

            if plot == True:
                self.plotter.update()
            
        return rews

if __name__ == "__main__":
    
    import matplotlib.pyplot as plt

    plt.ion()
    
    num_rollouts = 30
    lmb = 0.0001
    stime = 100
    sigma = 0.001
    epochs = 200
    num_agent_units = 100
    sigma_decay_amp = 0.0
    sigma_decay_period = 0.1


    env = ArenaEnv()

    num_params = env.num_actions * num_agent_units
    agent = Agent(
            num_params=num_params, 
            num_inputs=env.status_size, 
            num_actions=env.num_actions)    

    objective = Objective(env, agent, stime)

    bbo = BBO(num_params=num_params,
            num_rollouts=num_rollouts, lmb=lmb,
            epochs=epochs, sigma=sigma,
            sigma_decay_amp=sigma_decay_amp,
            sigma_decay_period=sigma_decay_period,
            cost_func=objective)
   

    rew_fig = plt.figure()
    ax = rew_fig.add_subplot(111)

    epochs_rews = np.zeros([epochs, 3])
    for e in range(epochs):
        rews = bbo.iteration()
        print((("{:d} ")+("{:.4f} ")*2).format(e, 
            rews.mean(), rews.max()))
        
        epochs_rews[e,:] = np.hstack([np.min(rews), np.mean(rews), np.max(rews)])

        if e%10 == 0:
            objective(bbo.theta, plot=True)
            ax.clear()
            ax.set_title("Rewards")
            ax.fill_between(np.arange(e+1), 
                    epochs_rews[:(e+1), 0],
                    epochs_rews[:(e+1), 2],
                    facecolor=[.8, .8, 1], edgecolor=[.6,.6,1])
            ax.plot(np.arange(e+1), 
                    epochs_rews[:(e+1), 1],
                    lw=3, c=[.4, .4 ,1])
            plt.pause(0.02)

            

