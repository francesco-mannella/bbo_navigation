if __package__ is None or __package__ == "":
    from esn import ESN
else:
    from bbo_navigation.esn import ESN

import numpy as np
import yaml

def sigmoid(x):
    return 1/(1 + np.exp(-x))

class Agent:

    def __init__(self, num_params, num_inputs, num_actions):

        self.num_units = num_params//num_actions
        self.num_inputs = num_inputs
        self.num_actions = num_actions
        
        self.echo = ESN(
            N       = self.num_units,
            dt      = 1.0,
            tau     = 5.0,
            alpha   = 0.1,
            beta    = 0.9,
            epsilon = 1.0e-10)
        
        self.input_weights = np.random.randn(self.num_inputs, self.num_units)
        self.out_weights = np.random.randn(self.num_units, self.num_actions)

    def save(self, filename):
        with open(filename, "w") as f:
            yaml.dump(self, f)

    @staticmethod
    def load(filename):
        with open(filename, "r") as f:
            return yaml.load(f, Loader=yaml.UnsafeLoader)

    def setParams(self, params):

        self.out_weights = params.reshape(self.num_units, self.num_actions).copy()
    
    def reset(self):

        self.echo.reset()

    def step(self, status):
        
        units_status = np.matmul(status.reshape(1, -1), self.input_weights)
        units = self.echo.step(units_status.ravel())
        action = np.matmul(units, self.out_weights)
        action = np.hstack([ np.tanh(action[0]), sigmoid(action[1])])
        action *= [0.1, 0.5]
        return action

class ArenaEnv:
    ''' 2D arena.

        The agent can move and change direction (the action being [angle, speed]).
        The status of the world is a vector with n pixels dividing the 180° 
        visual field of the agent into cells. The intensity of the pixels is 
        related to the position of the reward.

    '''

    def __init__(self):
        
        self.status_size = 11
        self.num_actions = 2
        self.xlims = np.array([-1, 1])
        self.ylims = np.array([-1, 1])
        self.rew = np.zeros(2)
        self.rew_sigma = 0.6

        self.reset()
        self.t = 0

    def getReward(self):
        dist = np.exp(-0.5*(self.rew_sigma**-2)*np.linalg.norm(self.position - self.rew)**2)

        return dist

    def getStatus(self):

        def correct_angle(a):
            return a if a > 0 else 2*np.pi + a
        
        dist = np.exp(-0.5*np.linalg.norm(self.position - self.rew)**2)

        angle = np.arctan2(*(self.position - self.rew)[::-1])
        angle = correct_angle(angle)
        direction = correct_angle(2*np.pi*self.direction)
        
        rel_angle = correct_angle(direction - angle - np.pi/2) 
        idx = int((rel_angle/np.pi)*10)
        status = np.arange(self.status_size)
        status = np.exp(-0.5*((1e-2+4*dist)**-2)*(status - idx)**2)

        return status

    def reset(self):
        self.t = 0        
        self.rew = np.zeros(2)
        self.position = np.array([
                np.random.uniform(*self.xlims),
                np.random.uniform(*self.ylims)])
        
        self.direction = np.random.rand()

        self.status = np.ones(self.status_size)
        return self.status

    def step(self, action):
        '''
            Args:
                
                action (array(float, float)): [angle, speed]

            Returns:

                status (array): the current retina status
                reward (float): value of the current reward
        '''
        self.rew = 1.*np.hstack([np.cos(self.t/10.0), np.sin(self.t/10.0)])
        self.t += 1

        dang, dlen = action
        self.direction += dang
        self.direction = self.direction % 1
        self.position += dlen*np.array([
            np.cos(2*np.pi*self.direction),
            np.sin(2*np.pi*self.direction)])
        
        self.status = self.getStatus()
        self.reward = self.getReward()
        return self.status, self.reward


import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

class Plotter:

    """ Manages the plotting of the environment
    """

    def __init__(self, env, save=False, show=True):
        """ 
            Args:
                
                env (ArenaEnv): the 2D environment
                save (bool): save frames to file
                show (bool): show frames on screen
        """

        self.save = save
        self.show = show

        self.env = env
        self.figure = plt.figure(figsize=(4,6))
        spec = gridspec.GridSpec(ncols=2, nrows=3, figure=self.figure)
        self.ax = self.figure.add_subplot(spec[:2,:], aspect="equal")
        self.ax.set_title("Environment")
        self.ax.set_xlim([-2, 2])
        self.ax.set_ylim([-2, 2])

        self.rew = self.ax.scatter(*np.zeros(2), c="green", s=300)
        self.pos = self.ax.scatter(*np.zeros(2), c="black", s=150)
        self.dir, = self.ax.plot(*np.zeros(2), c="gray", lw=3)
        self.retina_ax = self.figure.add_subplot(spec[2,:], 
                aspect="equal")
        self.retina_ax.set_title("Visual field")
        self.retina = self.retina_ax.imshow(
                np.zeros([1,self.env.status_size]), 
                cmap=plt.cm.binary, vmin=-0.2, vmax=1)
        self.retina_ax.set_axis_off()

        if self.save:
            self.t = 0

    def update(self):

        self.rew.set_offsets(self.env.rew.reshape(1,-1))        
        self.pos.set_offsets(self.env.position.reshape(1,-1))        
        curr_dir = np.vstack([self.env.position,
            self.env.position + 0.2*np.array([
                np.cos(2*np.pi*self.env.direction), 
                np.sin(2*np.pi*self.env.direction)])])
        self.dir.set_data(*curr_dir.T)
        self.retina.set_array(self.env.status.reshape(1,-1))
        
        if self.save:
            self.figure.savefig("frames/frame{:04d}.png".format(self.t))
            self.t += 1

        if self.show: 
            plt.pause(0.02) 
        

    def close(self):
        plt.close(self.figure)

if __name__ == "__main__":

    plt.ion()
    stime = 100

    env = ArenaEnv()
    agent = Agent(num_params=20, num_inputs=env.status_size, 
            num_actions=env.num_actions)    

    plotter = Plotter(env)
    
    agent.reset()
    status = env.reset()
    for t in range(stime):
        action = agent.step(status)
        status, _ = env.step(action)
        plotter.update()
        plt.pause(0.02)
