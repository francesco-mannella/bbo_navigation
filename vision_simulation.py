import numpy as np
import matplotlib.pyplot as plt

if __package__ is None or __package__ == "":
    from bbo import BBO
    from arena_env import Agent, ArenaEnv, Plotter
else:
    from bbo_navigation.bbo import BBO
    from bbo_navigation.arena_env import Agent, ArenaEnv, Plotter


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
    
    def __call__(self, theta, show=False, save=False):
        ''' Call operator runnning a single episode.
            
            Args:
                
                theta (array): parameters the agent is set to at beginning of episode
                show (bool): whether rendering is enabled or not
                save (bool): whether saving frames on file

            Returns:

                rews (array): the reward values at each timestep of the episode
        '''
        rews = np.zeros(self.stime)
        
        self.agent.setParams(theta)
        self.agent.reset()
        status = self.env.reset()
        if show == True or save == True: self.plotter = Plotter(self.env, show=show, save=save)
        for t in range(self.stime):
            action = self.agent.step(status)
            status, reward = self.env.step(action)
            rews[t] = reward

            if show == True or save == True: self.plotter.update()
        
        if show == True or save == True: self.plotter.close()
         
        return rews


def reset_dirs(dirs):
    import os, glob

    for d in dirs:
        if not os.path.exists(d):
            os.makedirs(d)
        files = glob.glob(d + os.sep + "*")
        for f in files:
            if(os.path.isfile(f)):
                os.remove(f)


class Simulation:

    def __init__(self):    

        reset_dirs(["frames"])

        self.show = True
        self.save = True

        self.num_rollouts = 30
        self.lmb = 0.0001
        self.stime = 150
        self.sigma = 0.001
        self.epochs = 200
        self.num_agent_units = 200
        self.sigma_decay_amp = 0.0
        self.sigma_decay_period = 0.3
        self.max_rews = 0

        self.env = ArenaEnv()

        self.num_params = self.env.num_actions * self.num_agent_units
        self.agent = Agent( num_params=self.num_params,
                num_inputs=self.env.status_size,
                num_actions=self.env.num_actions)    
        self.objective = Objective(self.env, self.agent, self.stime)
        
        self.bbo = BBO(num_params=self.num_params,
            num_rollouts=self.num_rollouts, lmb=self.lmb,
            epochs=self.epochs, sigma=self.sigma,
            sigma_decay_amp=self.sigma_decay_amp,
            sigma_decay_period=self.sigma_decay_period,
            cost_func=self.objective)

        self.init_plot()

    def init_plot(self):
        self.rew_fig = plt.figure()
        self.ax = self.rew_fig.add_subplot(111)


    def plot_step(self, e, epochs_rews):
        self.ax.clear()
        self.ax.set_title("Rewards")
        self.ax.fill_between(np.arange(e+1), 
                epochs_rews[:(e+1), 0],
                epochs_rews[:(e+1), 2],
                facecolor=[.8, .8, 1], edgecolor=[.6,.6,1])
        self.ax.plot(np.arange(e+1), 
                epochs_rews[:(e+1), 1],
                lw=3, c=[.4, .4 ,1])

    def render(self):
        plt.pause(0.01)

    def run(self):
        
        max_rews = 0
        epochs_rews = np.zeros([self.epochs, 3])
        for e in range(self.epochs):
            rews = self.bbo.iteration()
            print((("{:d} ")+("{:.4f} ")*2).format(e, 
                rews.mean(), rews.max()))
            
            epochs_rews[e,:] = np.hstack([np.min(rews), np.mean(rews), np.max(rews)])
            
            if epochs_rews[e,1] > max_rews:
                max_rews = epochs_rews[e,1]
                self.agent.save("agent.dump")

            if e%10 == 0:
                self.objective(self.bbo.theta, show=self.show, save=self.save)
                self.plot_step(e, epochs_rews)

                if self.show:
                    self.render()
                if self.save:
                    self.rew_fig.savefig("rewards.png")


if __name__ == "__main__":

    import matplotlib.pyplot as plt
    
    sim = Simulation()

    sim.run()
